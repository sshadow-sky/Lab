import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config.setting import preset_setting, set_setting_by_args
from data_utils.load_data import get_data
from data_utils.split import get_split_index, index_to_data, merge_to_part
from models.DSAGC import SemiGCL
from utils.args import get_args_parser
from utils.metric import Metric
from utils.store import make_output_dir, save_state
from utils.utils import result_log, setup_seed, sub_result_log


param_path = "config/model_param/DSAGC.yaml"


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


def _weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.3)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.03)
        if m.bias is not None:
            m.bias.data.zero_()


def _ensure_onehot(label: np.ndarray, num_classes: int) -> np.ndarray:
    if label.ndim > 1 and label.shape[-1] == num_classes:
        return label.astype(np.float32)
    label = label.reshape(-1).astype(np.int64)
    one_hot = np.eye(num_classes, dtype=np.float32)[label]
    return one_hot


def _minmax_scale_samples(data: np.ndarray) -> np.ndarray:
    if data.size == 0:
        return data.astype(np.float32)
    shape = data.shape
    x2 = data.reshape(shape[0], -1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    x2 = scaler.fit_transform(x2).astype(np.float32)
    return x2.reshape(shape)


def _concat_subjects(subject_data_list):
    if len(subject_data_list) == 0:
        return np.array([], dtype=np.float32)
    return np.vstack(subject_data_list)


def _split_labeled_unlabeled(train_indexes, train_data_keep, train_label_keep, target_idx, total_subjects, num_of_u):
    unlabeled_ids = []
    for offset in range(int(num_of_u)):
        cand = (target_idx + offset + 1) % total_subjects
        if cand in train_indexes and cand not in unlabeled_ids:
            unlabeled_ids.append(cand)

    if len(unlabeled_ids) < int(num_of_u):
        for idx in train_indexes:
            if idx not in unlabeled_ids:
                unlabeled_ids.append(idx)
            if len(unlabeled_ids) >= int(num_of_u):
                break

    source_labeled_x, source_labeled_y = [], []
    source_unlabeled_x, source_unlabeled_y = [], []

    for sub_idx, sub_x, sub_y in zip(train_indexes, train_data_keep, train_label_keep):
        sub_x = _minmax_scale_samples(np.array(sub_x, dtype=np.float32))
        sub_y = np.array(sub_y)
        if sub_idx in unlabeled_ids:
            source_unlabeled_x.append(sub_x)
            source_unlabeled_y.append(sub_y)
        else:
            source_labeled_x.append(sub_x)
            source_labeled_y.append(sub_y)

    if len(source_unlabeled_x) == 0:
        source_unlabeled_x, source_unlabeled_y = source_labeled_x, source_labeled_y

    return (
        _concat_subjects(source_labeled_x),
        _concat_subjects(source_labeled_y),
        _concat_subjects(source_unlabeled_x),
        _concat_subjects(source_unlabeled_y),
    )


def _build_loader(feature, label, batch_size, shuffle, drop_last):
    dataset = TensorDataset(torch.from_numpy(feature).float(), torch.from_numpy(label).float())
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def _label_index(y: torch.Tensor) -> torch.Tensor:
    if y.ndim > 1:
        return y.argmax(dim=1)
    return y.long().view(-1)


def _evaluate(model, target_dataset: TensorDataset, metrics, device, batch_size=64):
    model.eval()
    metric = Metric(metrics)
    with torch.no_grad():
        eval_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        for data_target, labels_target in eval_loader:
            data_target = data_target.to(device)
            labels_target = labels_target.to(device)
            pred_prob = model.predict(data_target)
            pred = pred_prob.argmax(dim=1)
            target = _label_index(labels_target)
            metric.update(pred, target)
    print("\033[34m eval state: " + metric.value())
    return metric.values


def _next_cycle(loader_iter, loader):
    try:
        return next(loader_iter), loader_iter
    except StopIteration:
        loader_iter = iter(loader)
        return next(loader_iter), loader_iter


def _train_one_round(args, net_params, train_cfg, source_labeled_loader, source_unlabeled_loader, target_loader, val_eval_dataset, test_eval_dataset, output_dir, device):
    seed_fix = int(train_cfg.get("seed_fix", 20))
    setup_seed(seed_fix)
    model = SemiGCL(net_params).to(device)
    setup_seed(seed_fix)
    model.apply(_weight_init)

    init_lr = float(train_cfg.get("init_lr", args.lr))
    weight_decay = float(train_cfg.get("weight_decay", 1e-5))
    optimizer = optim.RMSprop(model.parameters(), lr=init_lr, weight_decay=weight_decay)

    threshold = int(train_cfg.get("threshold", 30))
    best_metric = -1.0
    metric_key = args.metric_choose if args.metric_choose in args.metrics else args.metrics[0]
    eval_interval = max(1, int(train_cfg.get("eval_interval", 1)))

    for epoch in range(args.epochs):
        setup_seed(seed_fix)
        model.train()

        total_loss = 0.0
        total_num = 0
        source_acc_total = 0.0
        target_acc_total = 0.0

        source_labeled_iter = iter(source_labeled_loader)
        source_unlabeled_iter = iter(source_unlabeled_loader)
        target_bar = tqdm(target_loader, desc=f"Train Epoch {epoch + 1}/{args.epochs}")

        for data_target, label_target in target_bar:
            (data_source, labels_source), source_labeled_iter = _next_cycle(source_labeled_iter, source_labeled_loader)
            (x_un, _), source_unlabeled_iter = _next_cycle(source_unlabeled_iter, source_unlabeled_loader)

            x_un = x_un.to(device)
            data_source = data_source.to(device)
            labels_source = labels_source.to(device)
            data_target = data_target.to(device)
            labels_target = label_target.to(device)

            tripleada = 0 if int(net_params.get("T_DANN", 1)) else 1
            if epoch >= threshold:
                cat_x = torch.cat((data_source, x_un, data_target), dim=0)
                pred, domain_loss, adj_loss, contrastive_loss, sim_weight, _ = model(cat_x, tripleada=tripleada, threshold=1)
            else:
                cat_x = torch.cat((data_source, data_target), dim=0)
                pred, domain_loss, adj_loss, contrastive_loss, sim_weight, _ = model(cat_x, tripleada=0, threshold=0)

            source_pred = pred[0 : len(data_source), :]
            target_pred = pred[-len(data_source) :, :]

            if epoch >= threshold:
                log_prob = torch.log_softmax(sim_weight * source_pred, dim=1)
            else:
                log_prob = torch.log_softmax(source_pred, dim=1)

            ce_loss = -torch.sum(log_prob * labels_source) / len(labels_source)
            loss = (
                ce_loss
                + float(train_cfg.get("DANN", 1.0)) * domain_loss
                + float(train_cfg.get("dynamic_adj", 1.0)) * adj_loss
                + float(train_cfg.get("GCL", 1.0)) * contrastive_loss
            )

            source_scores = source_pred.detach().argmax(dim=1)
            source_target = labels_source.argmax(dim=1)
            source_acc = (source_scores == source_target).float().sum().item()
            source_acc_total += source_acc

            target_scores = target_pred.detach().argmax(dim=1)
            target_target = labels_target.argmax(dim=1)
            target_acc = (target_scores == target_target).float().sum().item()
            target_acc_total += target_acc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = len(data_source)
            total_num += bs
            total_loss += loss.item() * bs
            epoch_train_loss = total_loss / max(total_num, 1)
            target_bar.set_postfix_str(
                f"loss:{epoch_train_loss:.4f} source_acc:{source_acc_total / max(total_num, 1) * 100:.2f}% target_acc:{target_acc_total / max(total_num, 1) * 100:.2f}%"
            )

        if (epoch + 1) % eval_interval == 0:
            eval_metric = _evaluate(model, val_eval_dataset, args.metrics, device, batch_size=args.batch_size)
            if eval_metric[metric_key] > best_metric:
                best_metric = eval_metric[metric_key]
                save_state(output_dir, model, optimizer, epoch + 1, metric=metric_key)

    ckpt_path = output_dir / f"checkpoint-best{metric_key}"
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
    return _evaluate(model, test_eval_dataset, args.metrics, device, batch_size=args.batch_size)


def main(args):
    args.model = "DSAGC"
    cfg = _load_cfg()
    params_cfg = cfg.get("params", {})
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
    subjects_metrics = [[] for _ in range(len(data))]

    for rridx, (data_i, label_i) in enumerate(zip(data, label), 1):
        tts = get_split_index(data_i, label_i, setting)
        total_subjects = len(data_i)

        for ridx, (train_indexes, test_indexes, val_indexes) in enumerate(zip(tts["train"], tts["test"], tts["val"]), 1):
            setup_seed(args.seed)
            if val_indexes[0] == -1:
                print(f"train indexes:{train_indexes}, test indexes:{test_indexes}")
            else:
                print(f"train indexes:{train_indexes}, val indexes:{val_indexes}, test indexes:{test_indexes}")

            train_data_keep, train_label_keep, _, _, _, _ = index_to_data(
                data_i, label_i, train_indexes, test_indexes, val_indexes, True
            )
            _, _, val_data, val_label, test_data, test_label = index_to_data(
                data_i, label_i, train_indexes, test_indexes, val_indexes, args.keep_dim
            )

            test_data = _minmax_scale_samples(np.array(test_data, dtype=np.float32))
            test_label = _ensure_onehot(np.array(test_label), num_classes)

            # Strict no-test-leak protocol: val is used for target stream, training-time evaluation and model selection.
            # Test split is only used once after training ends.
            if len(val_data) == 0:
                print("skip one split because val split is empty under strict no-test-leak protocol")
                continue

            target_data = _minmax_scale_samples(np.array(val_data, dtype=np.float32))
            target_label = _ensure_onehot(np.array(val_label), num_classes)
            target_indexes = val_indexes

            target_idx = int(target_indexes[0])
            source_x_l, source_y_l, source_x_u, source_y_u = _split_labeled_unlabeled(
                train_indexes=train_indexes,
                train_data_keep=train_data_keep,
                train_label_keep=train_label_keep,
                target_idx=target_idx,
                total_subjects=total_subjects,
                num_of_u=int(train_cfg.get("num_of_U", 2)),
            )

            source_x_l = np.array(source_x_l, dtype=np.float32)
            source_x_u = np.array(source_x_u, dtype=np.float32)
            source_y_l = _ensure_onehot(np.array(source_y_l), num_classes)
            source_y_u = _ensure_onehot(np.array(source_y_u), num_classes)

            if len(source_x_l) == 0 or len(source_x_u) == 0:
                print("skip one split because source labeled or unlabeled data is empty")
                continue

            source_labeled_loader = _build_loader(source_x_l, source_y_l, args.batch_size, shuffle=True, drop_last=True)
            source_unlabeled_loader = _build_loader(source_x_u, source_y_u, args.batch_size, shuffle=True, drop_last=True)
            target_loader = _build_loader(target_data, target_label, args.batch_size, shuffle=True, drop_last=True)
            val_eval_dataset = TensorDataset(torch.from_numpy(target_data).float(), torch.from_numpy(target_label).float())
            test_eval_dataset = TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(test_label).float())

            net_params = {
                "DEVICE": device,
                "batch_size": args.batch_size,
                "num_of_vertices": int(channels),
                "num_of_features": int(feature_dim),
                "category_number": int(num_classes),
                "GLalpha": float(params_cfg.get("GLalpha", 0.01)),
                "K": int(params_cfg.get("K", 3)),
                "node_feature_hidden1": int(params_cfg.get("node_feature_hidden1", 5)),
                "linearsize": int(params_cfg.get("linearsize", 128)),
                "drop_rate": float(params_cfg.get("drop_rate", 0.8)),
                "Multi_att": int(params_cfg.get("Multi_att", 1)),
                "T_DANN": int(params_cfg.get("T_DANN", 1)),
            }

            output_dir = make_output_dir(args, "DSAGC")
            round_metric = _train_one_round(
                args=args,
                net_params=net_params,
                train_cfg=train_cfg,
                source_labeled_loader=source_labeled_loader,
                source_unlabeled_loader=source_unlabeled_loader,
                target_loader=target_loader,
                val_eval_dataset=val_eval_dataset,
                test_eval_dataset=test_eval_dataset,
                output_dir=output_dir,
                device=device,
            )

            best_metrics.append(round_metric)
            if setting.experiment_mode == "subject-dependent":
                subjects_metrics[rridx - 1].append(round_metric)

    if setting.experiment_mode == "subject-dependent":
        sub_result_log(args, subjects_metrics)
    else:
        result_log(args, best_metrics)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
