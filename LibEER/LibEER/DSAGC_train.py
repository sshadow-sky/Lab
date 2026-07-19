import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config.setting import preset_setting, set_setting_by_args
from data_utils.split import get_split_index, index_to_data, merge_to_part
from models.DSAGC import SemiGCL
from utils.metric import Metric
from utils.reproduction import (
    concatenate_parts,
    minmax_scale_part,
    normalize_and_concatenate_parts,
    normalize_labeled_subject_parts,
    resolve_domain_protocol,
)
from utils.store import make_output_dir, save_state
from utils.utils import result_log, setup_seed, sub_result_log


param_path = "config/model_param/DSAGC.yaml"
REFERENCE_CLI_DEFAULTS = {"batch_size": 48, "epochs": 60}
REFERENCE_MODEL_SEED = 2024
REFERENCE_DEAP_BANDS = [[0.5, 4], [4, 8], [8, 14], [14, 30], [30, 50]]


def _record_reproduction_args(args, protocol, train_cfg):
    args.reproduction_protocol = protocol.split_type
    args.reproduction_target_role = protocol.target_role
    args.reproduction_selection_role = protocol.selection_role
    args.reproduction_test_leak = protocol.test_leak
    args.reproduction_model_seed = int(
        train_cfg.get("seed_fix", REFERENCE_MODEL_SEED)
    )
    args.reproduction_patience = int(train_cfg.get("patience", 15))
    args.reproduction_num_of_u = int(train_cfg.get("num_of_U", 2))
    args.reproduction_weight_decay = float(train_cfg.get("weight_decay", 1e-5))
    args.reproduction_gcl_weight = float(train_cfg.get("GCL", 1.0))
    args.reproduction_dynamic_adj_weight = float(
        train_cfg.get("dynamic_adj", 1.0)
    )
    args.reproduction_dann_weight = float(train_cfg.get("DANN", 1.0))


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


def _weight_init(m, seed=REFERENCE_MODEL_SEED):
    if isinstance(m, nn.Conv2d):
        setup_seed(seed)
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.3)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        setup_seed(seed)
        m.weight.data.normal_(0, 0.03)
        if m.bias is not None:
            m.bias.data.zero_()


def _ensure_onehot(label: np.ndarray, num_classes: int) -> np.ndarray:
    if label.ndim > 1 and label.shape[-1] == num_classes:
        return label.astype(np.float32)
    label = label.reshape(-1).astype(np.int64)
    one_hot = np.eye(num_classes, dtype=np.float32)[label]
    return one_hot


def _concat_subjects(subject_data_list):
    if len(subject_data_list) == 0:
        return np.array([], dtype=np.float32)
    return np.vstack(subject_data_list)


def _select_unlabeled_ids(split_type, train_indexes, test_idx, total_subjects, num_of_u):
    num_of_u = int(num_of_u)
    if num_of_u < 1 or num_of_u >= len(train_indexes):
        raise ValueError(
            "num_of_U must be at least one and smaller than the training part count."
        )
    if split_type == "train-val-test":
        return list(train_indexes[:num_of_u])

    train_set = set(train_indexes)
    ordered = [
        (test_idx + step) % total_subjects
        for step in range(1, total_subjects + 1)
        if (test_idx + step) % total_subjects in train_set
    ]
    return ordered[:num_of_u]


def _split_labeled_unlabeled(
    split_type,
    train_indexes,
    train_data_keep,
    train_label_keep,
    test_idx,
    total_subjects,
    num_of_u,
):
    unlabeled_ids = _select_unlabeled_ids(
        split_type, train_indexes, test_idx, total_subjects, num_of_u
    )

    source_labeled_x, source_labeled_y = [], []
    source_unlabeled_x, source_unlabeled_y = [], []

    for sub_idx, sub_x, sub_y in zip(train_indexes, train_data_keep, train_label_keep):
        sub_x = minmax_scale_part(sub_x)
        sub_y = np.array(sub_y)
        if sub_idx in unlabeled_ids:
            source_unlabeled_x.append(sub_x)
            source_unlabeled_y.append(sub_y)
        else:
            source_labeled_x.append(sub_x)
            source_labeled_y.append(sub_y)

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


def _train_one_round(
    args,
    net_params,
    train_cfg,
    source_labeled_loader,
    source_unlabeled_loader,
    target_loader,
    selection_dataset,
    test_eval_datasets,
    output_dir,
    device,
):
    seed_fix = int(train_cfg.get("seed_fix", REFERENCE_MODEL_SEED))
    setup_seed(seed_fix)
    model = SemiGCL(net_params).to(device)
    model.apply(lambda module: _weight_init(module, seed_fix))

    weight_decay = float(train_cfg.get("weight_decay", 1e-5))
    optimizer = optim.RMSprop(
        model.parameters(), lr=float(args.lr), weight_decay=weight_decay
    )

    threshold = int(train_cfg.get("threshold", 30))
    best_acc = -1.0
    patience = int(train_cfg.get("patience", 15))
    patience_counter = 0
    selection_metrics = list(dict.fromkeys([*args.metrics, "acc"]))

    for epoch in range(args.epochs):
        setup_seed(seed_fix)
        model.train()

        total_loss = 0.0
        total_num = 0
        source_acc_total = 0.0

        source_labeled_iter = iter(source_labeled_loader)
        source_unlabeled_iter = iter(source_unlabeled_loader)
        target_bar = tqdm(target_loader, desc=f"Train Epoch {epoch + 1}/{args.epochs}")

        for data_target, _ in target_bar:
            (data_source, labels_source), source_labeled_iter = _next_cycle(source_labeled_iter, source_labeled_loader)
            (x_un, _), source_unlabeled_iter = _next_cycle(source_unlabeled_iter, source_unlabeled_loader)

            x_un = x_un.to(device)
            data_source = data_source.to(device)
            labels_source = labels_source.to(device)
            data_target = data_target.to(device)

            tripleada = 0 if int(net_params.get("T_DANN", 1)) else 1
            if epoch >= threshold:
                cat_x = torch.cat((data_source, x_un, data_target), dim=0)
                pred, domain_loss, adj_loss, contrastive_loss, sim_weight, _ = model(cat_x, tripleada=tripleada, threshold=1)
            else:
                cat_x = torch.cat((data_source, data_target), dim=0)
                pred, domain_loss, adj_loss, contrastive_loss, sim_weight, _ = model(cat_x, tripleada=0, threshold=0)

            source_pred = pred[0 : len(data_source), :]

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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = len(data_source)
            total_num += bs
            total_loss += loss.item() * bs
            epoch_train_loss = total_loss / max(total_num, 1)
            target_bar.set_postfix_str(
                f"loss:{epoch_train_loss:.4f} "
                f"source_acc:{source_acc_total / max(total_num, 1) * 100:.2f}%"
            )

        selection_metric = _evaluate(
            model,
            selection_dataset,
            selection_metrics,
            device,
            batch_size=args.batch_size,
        )
        if selection_metric["acc"] > best_acc:
            best_acc = selection_metric["acc"]
            patience_counter = 0
            save_state(output_dir, model, optimizer, epoch + 1, metric="acc")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"early stopping after {patience} non-improving evaluations")
            break

    ckpt_path = output_dir / "checkpoint-bestacc"
    if not ckpt_path.exists():
        raise RuntimeError("DS-AGC did not produce a validation-selected checkpoint.")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])

    subject_metrics = []
    for subject_index, test_eval_dataset in test_eval_datasets:
        print(f"final test subject index:{subject_index}")
        subject_metrics.append(
            _evaluate(
                model,
                test_eval_dataset,
                args.metrics,
                device,
                batch_size=args.batch_size,
            )
        )
    return subject_metrics


def main(args):
    args.model = "DSAGC"
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

    _record_reproduction_args(args, protocol, train_cfg)

    from data_utils.load_data import get_data

    setup_seed(args.seed)
    data, label, channels, feature_dim, num_classes = get_data(setting)
    data, label = merge_to_part(data, label, setting)
    device = torch.device(args.device)

    best_metrics = []
    for rridx, (data_i, label_i) in enumerate(zip(data, label), 1):
        tts = get_split_index(data_i, label_i, setting)
        total_subjects = len(data_i)

        for ridx, (train_indexes, test_indexes, val_indexes) in enumerate(zip(tts["train"], tts["test"], tts["val"]), 1):
            setup_seed(args.seed)
            protocol = resolve_domain_protocol(
                setting.experiment_mode, setting.split_type, val_indexes
            )
            if val_indexes[0] == -1:
                print(f"train indexes:{train_indexes}, test indexes:{test_indexes}")
                print(
                    "WARNING: DS-AGC unstrict LOSO uses the test split for "
                    "adaptation, checkpoint selection, and final evaluation."
                )
            else:
                print(f"train indexes:{train_indexes}, val indexes:{val_indexes}, test indexes:{test_indexes}")

            (
                train_data_keep,
                train_label_keep,
                val_data_keep,
                val_label_keep,
                test_data_keep,
                test_label_keep,
            ) = index_to_data(
                data_i, label_i, train_indexes, test_indexes, val_indexes, True
            )

            test_subject_parts = normalize_labeled_subject_parts(
                test_indexes, test_data_keep, test_label_keep
            )
            test_data = concatenate_parts(
                [subject_data for _, subject_data, _ in test_subject_parts]
            )
            test_label = _ensure_onehot(
                concatenate_parts(
                    [subject_label for _, _, subject_label in test_subject_parts]
                ),
                num_classes,
            )
            target_data, target_label = test_data, test_label
            if protocol.selection_role == "val":
                selection_data = normalize_and_concatenate_parts(val_data_keep)
                selection_label = _ensure_onehot(
                    concatenate_parts(val_label_keep), num_classes
                )
            else:
                selection_data, selection_label = target_data, target_label

            source_x_l, source_y_l, source_x_u, source_y_u = _split_labeled_unlabeled(
                split_type=setting.split_type,
                train_indexes=train_indexes,
                train_data_keep=train_data_keep,
                train_label_keep=train_label_keep,
                test_idx=int(test_indexes[0]),
                total_subjects=total_subjects,
                num_of_u=int(train_cfg.get("num_of_U", 2)),
            )

            source_x_l = np.array(source_x_l, dtype=np.float32)
            source_x_u = np.array(source_x_u, dtype=np.float32)
            source_y_l = _ensure_onehot(np.array(source_y_l), num_classes)
            source_y_u = _ensure_onehot(np.array(source_y_u), num_classes)

            source_labeled_loader = _build_loader(source_x_l, source_y_l, args.batch_size, shuffle=True, drop_last=True)
            source_unlabeled_loader = _build_loader(source_x_u, source_y_u, args.batch_size, shuffle=True, drop_last=True)
            target_loader = _build_loader(target_data, target_label, args.batch_size, shuffle=True, drop_last=True)
            selection_dataset = TensorDataset(
                torch.from_numpy(selection_data).float(),
                torch.from_numpy(selection_label).float(),
            )
            test_eval_datasets = [
                (
                    subject_index,
                    TensorDataset(
                        torch.from_numpy(subject_data).float(),
                        torch.from_numpy(
                            _ensure_onehot(subject_label, num_classes)
                        ).float(),
                    ),
                )
                for subject_index, subject_data, subject_label in test_subject_parts
            ]
            if min(
                len(source_labeled_loader),
                len(source_unlabeled_loader),
                len(target_loader),
            ) == 0:
                raise ValueError(
                    "DS-AGC requires at least one complete source, U, and target batch; "
                    "reduce batch_size."
                )

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

            output_dir = (
                make_output_dir(args, "DSAGC")
                / f"round_{rridx}"
                / f"fold_{ridx}"
            )
            round_metrics = _train_one_round(
                args=args,
                net_params=net_params,
                train_cfg=train_cfg,
                source_labeled_loader=source_labeled_loader,
                source_unlabeled_loader=source_unlabeled_loader,
                target_loader=target_loader,
                selection_dataset=selection_dataset,
                test_eval_datasets=test_eval_datasets,
                output_dir=output_dir,
                device=device,
            )

            best_metrics.extend(round_metrics)
    result_log(args, best_metrics)


if __name__ == "__main__":
    from utils.args import get_args_parser

    parser = get_args_parser()
    parser.set_defaults(**REFERENCE_CLI_DEFAULTS)
    args = parser.parse_args()
    main(args)
