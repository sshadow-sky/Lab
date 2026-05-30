import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config.setting import preset_setting, set_setting_by_args
from data_utils.load_data import get_data
from data_utils.split import get_split_index, index_to_data, merge_to_part
from models.Models import Model
from utils.args import get_args_parser
from utils.metric import Metric
from utils.store import make_output_dir, save_state
from utils.utils import result_log, setup_seed, sub_result_log


param_path = "config/model_param/MAET.yaml"


def _load_cfg():
    cfg = {"params": {}, "train": {}}
    try:
        with open(param_path, "r", encoding="utf-8") as fd:
            loaded = yaml.load(fd, Loader=yaml.FullLoader)
            if isinstance(loaded, dict):
                cfg["params"] = loaded.get("params", {}) or {}
                cfg["train"] = loaded.get("train", {}) or {}
    except IOError:
        print(f"{param_path} may not exist or not available")
    return cfg


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1, reduction="mean"):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n_classes = preds.size()[-1]
        log_preds = torch.log_softmax(preds, dim=-1)
        loss = -log_preds.sum(dim=-1)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        nll = torch.nn.functional.nll_loss(log_preds, target, reduction=self.reduction)
        return self.epsilon * (loss / n_classes) + (1 - self.epsilon) * nll


def _label_index(y):
    if y.ndim > 1:
        return y.argmax(dim=1).long()
    return y.long().view(-1)


def _build_subject_independent_train_set(train_data_keep, train_label_keep, train_indexes):
    x_list, y_list, d_list = [], [], []
    for sub_x, sub_y, sub_id in zip(train_data_keep, train_label_keep, train_indexes):
        sx = np.asarray(sub_x, dtype=np.float32)
        sy = np.asarray(sub_y)
        x_list.append(sx)
        y_list.append(sy)
        # Use subject identity as raw domain id, then remap to contiguous labels.
        d_list.append(np.full(shape=(len(sx),), fill_value=int(sub_id), dtype=np.int64))

    x = np.concatenate(x_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0)
    d_raw = np.concatenate(d_list, axis=0)
    _, d = np.unique(d_raw, return_inverse=True)
    return x, y, d.astype(np.int64), int(len(np.unique(d_raw)))


def _build_loader(x, y, d, batch_size, num_workers, shuffle=True):
    x_tensor = torch.from_numpy(np.asarray(x, dtype=np.float32)).float()
    y_tensor = torch.from_numpy(np.asarray(y)).float()
    if d is None:
        ds = TensorDataset(x_tensor, y_tensor)
    else:
        d_tensor = torch.from_numpy(np.asarray(d, dtype=np.int64)).long()
        ds = TensorDataset(x_tensor, y_tensor, d_tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


@torch.no_grad()
def _evaluate(model, data_loader, device, metrics):
    model.eval()
    metric = Metric(metrics)
    for batch in tqdm(data_loader, total=len(data_loader), desc="Evaluating : "):
        samples, targets = batch[0].to(device), batch[1].to(device)
        outputs = model(samples)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        pred = torch.argmax(outputs, dim=1)
        target = _label_index(targets)
        metric.update(pred, target)
    print("\033[34m eval state: " + metric.value())
    return metric.values


def _train_one_round(
    args,
    train_cfg,
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    output_dir,
    use_domain_generalization=False,
    num_domains=1,
):
    model = model.to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", args.lr)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        eps=float(train_cfg.get("eps", 1e-8)),
    )

    best_metric = {m: 0.0 for m in args.metrics}
    metric_key = args.metric_choose if args.metric_choose in args.metrics else args.metrics[0]
    criterion_domain = LabelSmoothingCrossEntropy(epsilon=float(train_cfg.get("label_smoothing_init", 0.1)))

    for epoch in range(args.epochs):
        model.train()
        metric = Metric(args.metrics)

        alpha = 2 / (1 + math.exp(-10 * epoch / max(args.epochs, 1))) - 1
        if use_domain_generalization:
            criterion_domain.epsilon = ((num_domains - 1) / max(num_domains, 1)) * (epoch / max(args.epochs, 1))

        train_bar = tqdm(train_loader, total=len(train_loader), desc=f"Train Epoch {epoch + 1}/{args.epochs}: lr:{optimizer.param_groups[0]['lr']}")
        for batch in train_bar:
            samples = batch[0].to(device)
            targets = batch[1].to(device)
            domain_targets = batch[2].to(device) if use_domain_generalization else None

            outputs = model(samples, alpha_=alpha)
            if isinstance(outputs, tuple):
                logits, domain_output = outputs
            else:
                logits, domain_output = outputs, None

            target = _label_index(targets)
            loss_ce = nn.functional.cross_entropy(input=logits, target=target)
            if use_domain_generalization and domain_output is not None and domain_targets is not None:
                loss_domain = criterion_domain(domain_output, domain_targets)
                loss = loss_ce + float(train_cfg.get("domain_loss_weight", 1.0)) * loss_domain
            else:
                loss = loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.argmax(logits, dim=1)
            metric.update(pred, target, loss.item())
            train_bar.set_postfix_str(f"loss: {loss.item():.4f}")

        print("\033[32m train state: " + metric.value())
        metric_value = _evaluate(model, val_loader, device, args.metrics)
        for m in args.metrics:
            if metric_value[m] > best_metric[m]:
                best_metric[m] = metric_value[m]
                save_state(output_dir, model, optimizer, epoch + 1, metric=m)

    ckpt = output_dir / f"checkpoint-best{metric_key}"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"])

    metric_value = _evaluate(model, test_loader, device, args.metrics)
    for m in args.metrics:
        print(f"best_val_{m}: {best_metric[m]:.2f}")
        print(f"best_test_{m}: {metric_value[m]:.2f}")
    return metric_value


def main(args):
    args.model = "MAET"
    cfg = _load_cfg()
    train_cfg = cfg.get("train", {})

    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)

    setup_seed(args.seed)
    data, label, channels, feature_dim, num_classes = get_data(setting)
    data, label = merge_to_part(data, label, setting)
    device = torch.device(args.device)

    best_metrics = []
    dependent_metrics = [[] for _ in range(len(data))]

    for rridx, (data_i, label_i) in enumerate(zip(data, label), 1):
        tts = get_split_index(data_i, label_i, setting)
        for ridx, (train_indexes, test_indexes, val_indexes) in enumerate(zip(tts["train"], tts["test"], tts["val"]), 1):
            setup_seed(args.seed)
            if val_indexes[0] == -1:
                print(f"train indexes:{train_indexes}, test indexes:{test_indexes}")
            else:
                print(f"train indexes:{train_indexes}, val indexes:{val_indexes}, test indexes:{test_indexes}")

            # Keep DG behavior explicit and deterministic:
            # subject-independent -> enable DG, subject-dependent -> disable DG.
            use_dg = setting.experiment_mode == "subject-independent"

            if use_dg:
                train_data_keep, train_label_keep, _, _, _, _ = index_to_data(
                    data_i,
                    label_i,
                    train_indexes,
                    test_indexes,
                    val_indexes,
                    True,
                )
                train_data, train_label, train_domain, num_domains = _build_subject_independent_train_set(
                    train_data_keep,
                    train_label_keep,
                    train_indexes,
                )
            else:
                train_data, train_label, _, _, _, _ = index_to_data(
                    data_i,
                    label_i,
                    train_indexes,
                    test_indexes,
                    val_indexes,
                    args.keep_dim,
                )
                train_domain = None
                num_domains = 1

            _, _, val_data, val_label, test_data, test_label = index_to_data(
                data_i,
                label_i,
                train_indexes,
                test_indexes,
                val_indexes,
                args.keep_dim,
            )

            if len(val_data) == 0:
                val_data = test_data
                val_label = test_label

            model = Model["MAET"](
                channels,
                feature_dim,
                num_classes,
                domain_generalization=use_dg,
                num_domains=num_domains,
            )

            train_loader = _build_loader(train_data, train_label, train_domain, args.batch_size, args.num_workers, shuffle=True)
            val_loader = _build_loader(val_data, val_label, None, args.batch_size, args.num_workers, shuffle=False)
            test_loader = _build_loader(test_data, test_label, None, args.batch_size, args.num_workers, shuffle=False)

            output_dir = make_output_dir(args, "MAET")
            round_metric = _train_one_round(
                args=args,
                train_cfg=train_cfg,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                output_dir=output_dir,
                use_domain_generalization=use_dg,
                num_domains=num_domains,
            )

            best_metrics.append(round_metric)
            if setting.experiment_mode == "subject-dependent":
                dependent_metrics[rridx - 1].append(round_metric)

    if setting.experiment_mode == "subject-dependent":
        sub_result_log(args, dependent_metrics)
    else:
        result_log(args, best_metrics)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
