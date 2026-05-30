import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
import yaml
from sklearn.preprocessing import MinMaxScaler

from config.setting import preset_setting, set_setting_by_args
from data_utils.load_data import get_data
from data_utils.split import get_split_index, index_to_data, merge_to_part
from models.PCLAdversarial import DAANLoss
from models.PCLTDGCN import DomainAdaptationModel, Discriminator
from utils.args import get_args_parser
from utils.metric import Metric
from utils.store import make_output_dir, save_state
from utils.utils import result_log, setup_seed, sub_result_log


param_path = "config/model_param/PCLTDGCN.yaml"


def _load_train_param():
    try:
        with open(param_path, "r", encoding="utf-8") as fd:
            cfg = yaml.load(fd, Loader=yaml.FullLoader)
        return cfg.get("train", {})
    except IOError:
        print("\n{} may not exist or not available".format(param_path))
        return {}


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


def _minmax_to_minus1_1(feature):
    x = torch.tensor(feature, dtype=torch.float32).cpu().numpy()
    if x.size == 0:
        return x
    shape = x.shape
    x2 = x.reshape(shape[0], -1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    x2 = scaler.fit_transform(x2).astype("float32")
    return x2.reshape(shape)


@torch.no_grad()
def _init_banks(source_loader, target_loader, model, device):
    model.eval()
    for src_x, src_idx, _ in source_loader:
        model.get_init_banks(src_x.to(device), src_idx.to(device))
    for tar_x, tar_idx, _ in target_loader:
        model.get_init_banks_tgt(tar_x.to(device), tar_idx.to(device))


@torch.no_grad()
def _evaluate(model, test_loader, metrics, device):
    model.eval()
    metric = Metric(metrics)
    total_loss = 0.0
    count = 0
    for x, _, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        prob = model.target_predict(x)
        loss = F.nll_loss(torch.log(prob + 1e-8), y)
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
    num_batches = len(target_loader.dataset) // max(source_loader.batch_size, 1)
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
        global_transfer_loss = dann_loss(src_noise, tar_noise, src_prob, F.softmax(tar_out, dim=1))

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
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)

    setup_seed(args.seed)
    data, label, channels, feature_dim, num_classes = get_data(setting)
    data, label = merge_to_part(data, label, setting)

    train_cfg = _load_train_param()
    device = torch.device(args.device)
    best_metrics = []
    subjects_metrics = [[] for _ in range(len(data))]

    for rridx, (data_i, label_i) in enumerate(zip(data, label), 1):
        tts = get_split_index(data_i, label_i, setting)
        for ridx, (train_indexes, test_indexes, val_indexes) in enumerate(
            zip(tts['train'], tts['test'], tts['val']), 1
        ):
            setup_seed(args.seed)
            if val_indexes[0] == -1:
                print(f"train indexes:{train_indexes}, test indexes:{test_indexes}")
            else:
                print(f"train indexes:{train_indexes}, val indexes:{val_indexes}, test indexes:{test_indexes}")

            train_data, train_label, val_data, val_label, test_data, test_label = index_to_data(
                data_i, label_i, train_indexes, test_indexes, val_indexes, args.keep_dim
            )

            # Keep the same input normalization behavior as original PCL-TDGCN pipeline.
            train_data = _minmax_to_minus1_1(train_data)
            test_data = _minmax_to_minus1_1(test_data)
            if len(val_data) != 0:
                val_data = _minmax_to_minus1_1(val_data)

            if len(val_data) == 0:
                print("skip one split because val split is empty under strict no-test-leak protocol")
                continue

            output_dir = make_output_dir(args, "PCLTDGCN")

            target_data = val_data
            target_label = val_label

            source_loader = _build_loader(train_data, train_label, args.batch_size, True, device)
            target_loader = _build_loader(target_data, target_label, args.batch_size, True, device)
            val_loader = _build_loader(val_data, val_label, args.batch_size, False, device)
            test_loader = _build_loader(test_data, test_label, args.batch_size, False, device)

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
            dann_loss = DAANLoss(domain_discriminator).to(device)

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
            patience_limit = int(train_cfg.get("patience", 40))
            patience_counter = 0
            best_metric_choose = -1.0

            for epoch in range(args.epochs):
                if epoch % eval_interval == 0:
                    eval_metric = _evaluate(model, val_loader, args.metrics, device)
                    key = args.metric_choose if args.metric_choose in eval_metric else args.metrics[0]
                    if eval_metric[key] > best_metric_choose:
                        best_metric_choose = eval_metric[key]
                        patience_counter = 0
                        save_state(output_dir, model, optimizer, epoch + 1, metric=key)
                    else:
                        patience_counter += 1

                    if eval_metric[key] >= 1.0:
                        break
                    if patience_counter >= patience_limit:
                        break

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

            metric_key = args.metric_choose if args.metric_choose in args.metrics else args.metrics[0]
            ckpt_path = output_dir / f"checkpoint-best{metric_key}"
            if ckpt_path.exists():
                state = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(state["model"])
            round_metric = _evaluate(model, test_loader, args.metrics, device)

            best_metrics.append(round_metric)
            if setting.experiment_mode == "subject-dependent":
                subjects_metrics[rridx - 1].append(round_metric)

    if setting.experiment_mode == "subject-dependent":
        sub_result_log(args, subjects_metrics)
    else:
        result_log(args, best_metrics)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
