import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml

from config.setting import preset_setting, set_setting_by_args
from data_utils.split import get_split_index, index_to_data, merge_to_part
from models.PCLAdversarial import DAANLoss
from models.PCLTDGCN import DomainAdaptationModel, Discriminator
from utils.metric import Metric
from utils.reproduction import (
    concatenate_parts,
    normalize_and_concatenate_parts,
    normalize_labeled_subject_parts,
    resolve_domain_protocol,
)
from utils.store import make_output_dir, save_state
from utils.utils import result_log, setup_seed


param_path = "config/model_param/PCLTDGCN.yaml"
REFERENCE_CLI_DEFAULTS = {"batch_size": 48, "epochs": 100, "seed": 0}
REFERENCE_DEAP_BANDS = [[0.5, 4], [4, 8], [8, 14], [14, 30], [30, 50]]


def _num_training_batches(source_loader, target_loader):
    return min(len(source_loader), len(target_loader))


def _should_evaluate(epoch, max_epochs, interval):
    return epoch % interval == 0 or epoch == max_epochs - 1


def _is_better_checkpoint(acc, loss, best_acc, best_loss):
    return acc > best_acc or (math.isclose(acc, best_acc) and loss < best_loss)


def _build_fold_arrays(
    train_data_parts,
    train_label_parts,
    val_data_parts,
    val_label_parts,
    test_data_parts,
    test_label_parts,
    protocol,
):
    source_data = normalize_and_concatenate_parts(train_data_parts)
    source_label = concatenate_parts(train_label_parts)
    test_data = normalize_and_concatenate_parts(test_data_parts)
    test_label = concatenate_parts(test_label_parts)
    target_data, target_label = test_data, test_label

    if protocol.selection_role == "val":
        selection_data = normalize_and_concatenate_parts(val_data_parts)
        selection_label = concatenate_parts(val_label_parts)
    else:
        selection_data, selection_label = target_data, target_label

    return {
        "source_data": source_data,
        "source_label": source_label,
        "target_data": target_data,
        "target_label": target_label,
        "selection_data": selection_data,
        "selection_label": selection_label,
        "test_data": test_data,
        "test_label": test_label,
    }


def _load_cfg():
    try:
        with open(param_path, "r", encoding="utf-8") as fd:
            return yaml.load(fd, Loader=yaml.FullLoader)
    except IOError:
        print("\n{} may not exist or not available".format(param_path))
        return {}


def _load_train_param():
    return _load_cfg().get("train", {})


class StepwiseLR_GRL:
    def __init__(self, optimizer, init_lr=0.01, gamma=10.0, decay_rate=0.75, max_iter=1000):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0
        self.max_iter = max_iter

    def get_lr(self):
        return self.init_lr / (1.0 + self.gamma * (self.iter_num / self.max_iter)) ** self.decay_rate

    def step(self):
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group.setdefault("lr_mult", 1.0)
            param_group["lr"] = lr * param_group["lr_mult"]
        self.iter_num += 1


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, classes=3, epsilon=0.0005):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon

    def forward(self, input_data, target):
        log_prob = F.log_softmax(input_data, dim=-1)
        weight = input_data.new_ones(input_data.size()) * self.epsilon / (input_data.size(-1) - 1.0)
        weight.scatter_(-1, target.unsqueeze(-1), (1.0 - self.epsilon))
        return (-weight * log_prob).sum(dim=-1).mean()


def _label_to_index(y):
    if len(y.shape) > 1 and y.shape[-1] > 1:
        return torch.argmax(y, dim=1).long()
    return y.view(-1).long()


def _build_loader(feature, label, batch_size, shuffle, device):
    x = torch.tensor(feature, dtype=torch.float32)
    y = torch.tensor(label)
    y = _label_to_index(y)
    idx = torch.arange(x.shape[0]).long()
    dataset = TensorDataset(x, idx, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=(device.type == "cuda"))


@torch.no_grad()
def _init_banks(source_loader, target_loader, model, device):
    model.eval()
    for src_x, src_idx, _ in source_loader:
        model.get_init_banks(src_x.to(device), src_idx.to(device))
    for tar_x, tar_idx, _ in target_loader:
        model.get_init_banks_tgt(tar_x.to(device), tar_idx.to(device))


@torch.no_grad()
def _evaluate(model, test_loader, metrics, device, criterion):
    model.eval()
    metric = Metric(metrics)
    total_loss = 0.0
    count = 0
    for x, _, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        prob = model.target_predict(x)
        loss = criterion(prob, y)
        pred = torch.argmax(prob, dim=1)
        metric.update(pred, y, loss.item())
        total_loss += loss.item()
        count += 1
    print("\033[34m eval state: " + metric.value())
    values = metric.values
    values["loss"] = total_loss / max(count, 1)
    return values


def _train_one_epoch(model, dann_loss, criterion, optimizer, source_loader, target_loader, epoch, max_epochs, train_cfg, device):
    model.train()
    dann_loss.train()
    metric = Metric(["acc"])

    src_iter = iter(source_loader)
    tar_iter = iter(target_loader)
    num_batches = _num_training_batches(source_loader, target_loader)
    if num_batches == 0:
        return {"acc": 0.0, "loss": 0.0}

    conf_threshold = float(train_cfg.get("conf_threshold", 0.7))
    consistency_weight = float(train_cfg.get("consistency_weight", 0.2))
    noise_std = float(train_cfg.get("noise_std", 0.005))
    boost_scale = float(train_cfg.get("boost_scale", 2.0))

    for _ in range(num_batches):
        try:
            src_x, src_idx, src_y = next(src_iter)
            tar_x, tar_idx, _ = next(tar_iter)
        except StopIteration:
            break

        src_x = src_x.to(device)
        tar_x = tar_x.to(device)
        src_idx = src_idx.to(device)
        tar_idx = tar_idx.to(device)
        src_y = src_y.to(device)

        (
            src_out,
            src_feat,
            tar_out,
            tar_feat,
            _source_att,
            _target_att,
            _src_sim,
            tgt_sim,
            tgt_cluster_label,
            s2t_pro,
            t2s_pro,
            s2s_pro,
            t2t_pro,
        ) = model(src_x, tar_x, src_y, src_idx, tar_idx, epoch, max_epochs)

        cls_loss = criterion(src_out, src_y)

        src_prob = F.softmax(src_out, dim=1)
        src_max_prob, _ = src_prob.max(dim=1)
        src_mask = src_max_prob > conf_threshold
        source_loss = criterion(src_prob[src_mask], src_y[src_mask]) if src_mask.any() else torch.tensor(0.0, device=device)

        target_loss = criterion(tgt_sim, tgt_cluster_label.long())

        src_noise = src_feat + noise_std * torch.randn_like(src_feat)
        tar_noise = tar_feat + noise_std * torch.randn_like(tar_feat)
        global_transfer_loss = dann_loss(src_noise, tar_noise, src_prob, tar_out)

        s2t_entropy = -torch.sum(s2t_pro * torch.log(s2t_pro + 1e-10), dim=1).mean()
        t2s_entropy = -torch.sum(t2s_pro * torch.log(t2s_pro + 1e-10), dim=1).mean()
        cross_domain_loss = s2t_entropy + t2s_entropy

        s2s_entropy = -torch.sum(s2s_pro * torch.log(s2s_pro + 1e-10), dim=1).mean()
        t2t_entropy = -torch.sum(t2t_pro * torch.log(t2t_pro + 1e-10), dim=1).mean()
        in_domain_loss = s2s_entropy + t2t_entropy

        boost_factor = boost_scale * (2.0 / (1.0 + math.exp(-epoch / 1000.0)) - 1.0)
        loss = cls_loss + global_transfer_loss + source_loss + boost_factor * target_loss + consistency_weight * (
            cross_domain_loss + in_domain_loss
        )

        if torch.isnan(loss).any():
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = torch.argmax(src_prob, dim=1)
        metric.update(pred, src_y, loss.item())

    print("\033[32m train state: " + metric.value())
    return {"acc": metric.values.get("acc", 0.0), "loss": sum(metric.losses) / max(len(metric.losses), 1)}


def main(args):
    from data_utils.load_data import get_data

    args.model = "PCLTDGCN"
    cfg = _load_cfg()
    params_cfg = cfg.get("params", {})
    train_cfg = cfg.get("train", {})

    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)

    protocol = resolve_domain_protocol(
        setting.experiment_mode,
        setting.split_type,
        [0] if setting.split_type == "train-val-test" else [-1],
    )
    if setting.dataset.startswith("deap"):
        setting.extract_bands = [band[:] for band in REFERENCE_DEAP_BANDS]

    args.reproduction_protocol = protocol.split_type
    args.reproduction_target_role = protocol.target_role
    args.reproduction_selection_role = protocol.selection_role
    args.reproduction_test_leak = protocol.test_leak
    args.reproduction_graph_dropout = float(params_cfg.get("graph_dropout", 0.1))
    args.reproduction_classifier_dropout = float(
        params_cfg.get("classifier_dropout", params_cfg.get("dropout", 0.25))
    )
    args.reproduction_patience = int(train_cfg.get("patience", 10))
    args.reproduction_weight_decay = float(train_cfg.get("weight_decay", 0.001))
    args.reproduction_conf_threshold = float(train_cfg.get("conf_threshold", 0.7))
    args.reproduction_consistency_weight = float(
        train_cfg.get("consistency_weight", 0.2)
    )
    args.reproduction_noise_std = float(train_cfg.get("noise_std", 0.005))
    args.reproduction_boost_scale = float(train_cfg.get("boost_scale", 2.0))
    args.reproduction_eval_interval = int(train_cfg.get("eval_interval", 10))

    setup_seed(args.seed)
    data, label, channels, feature_dim, num_classes = get_data(setting)
    data, label = merge_to_part(data, label, setting)

    device = torch.device(args.device)
    best_metrics = []
    base_output_dir = make_output_dir(args, "PCLTDGCN")

    for rridx, (data_i, label_i) in enumerate(zip(data, label), 1):
        tts = get_split_index(data_i, label_i, setting)
        for ridx, (train_indexes, test_indexes, val_indexes) in enumerate(
            zip(tts['train'], tts['test'], tts['val']), 1
        ):
            setup_seed(args.seed)
            protocol = resolve_domain_protocol(
                setting.experiment_mode, setting.split_type, val_indexes
            )
            if val_indexes[0] == -1:
                print(f"train indexes:{train_indexes}, test indexes:{test_indexes}")
                print(
                    "WARNING: PCL-TDGCN unstrict LOSO uses the test split for "
                    "adaptation, checkpoint selection, and final evaluation."
                )
            else:
                print(f"train indexes:{train_indexes}, val indexes:{val_indexes}, test indexes:{test_indexes}")

            (
                train_data_parts,
                train_label_parts,
                val_data_parts,
                val_label_parts,
                test_data_parts,
                test_label_parts,
            ) = index_to_data(
                data_i, label_i, train_indexes, test_indexes, val_indexes, True
            )
            fold = _build_fold_arrays(
                train_data_parts=train_data_parts,
                train_label_parts=train_label_parts,
                val_data_parts=val_data_parts,
                val_label_parts=val_label_parts,
                test_data_parts=test_data_parts,
                test_label_parts=test_label_parts,
                protocol=protocol,
            )
            test_subject_parts = normalize_labeled_subject_parts(
                test_indexes, test_data_parts, test_label_parts
            )

            source_loader = _build_loader(
                fold["source_data"], fold["source_label"],
                args.batch_size, True, device,
            )
            target_loader = _build_loader(
                fold["target_data"], fold["target_label"],
                args.batch_size, True, device,
            )
            selection_loader = _build_loader(
                fold["selection_data"], fold["selection_label"],
                args.batch_size, False, device,
            )
            test_subject_loaders = [
                (
                    subject_index,
                    _build_loader(
                        subject_data,
                        subject_label,
                        args.batch_size,
                        False,
                        device,
                    ),
                )
                for subject_index, subject_data, subject_label in test_subject_parts
            ]

            source_num = len(source_loader.dataset)
            target_num = len(target_loader.dataset)

            model = DomainAdaptationModel(
                in_planes=(int(feature_dim), int(channels)),
                num_of_class=int(num_classes),
                device=str(device),
                source_num=source_num,
                target_num=target_num,
            ).to(device)
            domain_discriminator = Discriminator(model.hidden_2).to(device)
            criterion = LabelSmoothingCrossEntropy(classes=num_classes).to(device)
            dann_loss = DAANLoss(
                domain_discriminator, num_class=int(num_classes)
            ).to(device)

            optimizer = optim.RMSprop(
                list(model.parameters()) + list(domain_discriminator.parameters()),
                lr=args.lr,
                weight_decay=float(train_cfg.get("weight_decay", 0.001)),
            )
            lr_scheduler = StepwiseLR_GRL(
                optimizer,
                init_lr=args.lr,
                gamma=float(train_cfg.get("lr_gamma", 10.0)),
                decay_rate=float(train_cfg.get("lr_decay_rate", 0.75)),
                max_iter=args.epochs,
            )

            _init_banks(source_loader, target_loader, model, device)

            eval_interval = int(train_cfg.get("eval_interval", 10))
            patience_limit = int(train_cfg.get("patience", 10))
            patience_counter = 0
            best_val_acc = -1.0
            best_val_loss = float("inf")
            selection_metrics = list(dict.fromkeys([*args.metrics, "acc"]))
            round_output_dir = (
                base_output_dir / f"round_{rridx}" / f"fold_{ridx}"
            )

            for epoch in range(args.epochs):
                _train_one_epoch(
                    model=model,
                    dann_loss=dann_loss,
                    criterion=criterion,
                    optimizer=optimizer,
                    source_loader=source_loader,
                    target_loader=target_loader,
                    epoch=epoch,
                    max_epochs=args.epochs,
                    train_cfg=train_cfg,
                    device=device,
                )
                lr_scheduler.step()

                if not _should_evaluate(epoch, args.epochs, eval_interval):
                    continue
                selection_metric = _evaluate(
                    model,
                    selection_loader,
                    selection_metrics,
                    device,
                    criterion,
                )
                if _is_better_checkpoint(
                    selection_metric["acc"],
                    selection_metric["loss"],
                    best_val_acc,
                    best_val_loss,
                ):
                    best_val_acc = selection_metric["acc"]
                    best_val_loss = selection_metric["loss"]
                    patience_counter = 0
                    save_state(
                        round_output_dir,
                        model,
                        optimizer,
                        epoch + 1,
                        metric="acc",
                    )
                else:
                    patience_counter += 1
                if patience_counter >= patience_limit:
                    break

            ckpt_path = round_output_dir / "checkpoint-bestacc"
            if not ckpt_path.exists():
                raise RuntimeError(
                    "PCL-TDGCN did not produce a selection checkpoint."
                )
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state["model"])
            round_metrics = []
            for subject_index, test_loader in test_subject_loaders:
                print(f"final test subject index:{subject_index}")
                round_metrics.append(
                    _evaluate(
                        model,
                        test_loader,
                        args.metrics,
                        device,
                        criterion,
                    )
                )

            best_metrics.extend(round_metrics)
    result_log(args, best_metrics)


if __name__ == '__main__':
    from utils.args import get_args_parser

    parser = get_args_parser()
    parser.set_defaults(**REFERENCE_CLI_DEFAULTS)
    args = parser.parse_args()
    main(args)
