from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config.setting import preset_setting, set_setting_by_args
from data_utils.load_data import get_data
from data_utils.split import get_split_index, index_to_data, merge_to_part
from models.FDGCL import FDGCL
from utils.args import get_args_parser
from utils.metric import Metric
from utils.store import make_output_dir, save_state
from utils.utils import result_log, setup_seed, sub_result_log


param_path = "config/model_param/FDGCL.yaml"

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


def _apply_cli_overrides(args, params_cfg, train_cfg):
    param_overrides = {
        "fdgcl_ugfcda_warmup_epochs": "ugfcda_warmup_epochs",
        "fdgcl_ugfcda_eps": "ugfcda_eps",
        "fdgcl_ugfcda_keep_ratio_start": "ugfcda_keep_ratio_start",
        "fdgcl_ugfcda_keep_ratio_end": "ugfcda_keep_ratio_end",
        "fdgcl_ugfcda_keep_ratio_step": "ugfcda_keep_ratio_step",
        "fdgcl_ugfcda_keep_ratio_step_epochs": "ugfcda_keep_ratio_step_epochs",
        "fdgcl_ugfcda_subject_weight": "ugfcda_subject_weight",
        "fdgcl_ugfcda_proto_align_weight": "ugfcda_proto_align_weight",
        "fdgcl_node_drop_rate": "node_drop_rate",
        "fdgcl_edge_drop_rate": "edge_drop_rate",
        "fdgcl_gcl_importance_protect": "gcl_importance_protect",
        "fdgcl_gcl_importance_centrality_weight": "gcl_importance_centrality_weight",
        "fdgcl_gcl_importance_feature_weight": "gcl_importance_feature_weight",
        "fdgcl_gcl_node_sample_temperature": "gcl_node_sample_temperature",
        "fdgcl_gcl_node_sample_eps": "gcl_node_sample_eps",
        "fdgcl_gcl_edge_protect_strength": "gcl_edge_protect_strength",
        "fdgcl_gcl_edge_min_drop_scale": "gcl_edge_min_drop_scale",
        "fdgcl_dropout": "dropout",
        "fdgcl_temperature": "temperature",
        "fdgcl_graph_hidden": "graph_hidden",
        "fdgcl_graph_readout_hidden": "graph_readout_hidden",
        "fdgcl_gcl_readout_hidden": "gcl_readout_hidden",
        "fdgcl_spectral_hidden": "spectral_hidden",
        "fdgcl_disentangle_dim": "disentangle_dim",
        "fdgcl_projection_dim": "projection_dim",
        "fdgcl_cross_scale_heads": "cross_scale_heads",
        "fdgcl_gl_alpha": "GLalpha",
        "fdgcl_cheb_k": "K",
        "fdgcl_ssbn_eps": "ssbn_eps",
        "fdgcl_sin_min_count": "sin_min_count",
        "fdgcl_grl_max_iters": "grl_max_iters",
    }
    train_overrides = {
        "fdgcl_loss_ce": "loss_ce",
        "fdgcl_loss_aj": "loss_aj",
        "fdgcl_loss_gcl": "loss_gcl",
        "fdgcl_loss_align": "loss_align",
        "fdgcl_loss_orth": "loss_orth",
        "fdgcl_loss_subject": "loss_subject",
    }

    for arg_name, cfg_name in param_overrides.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            params_cfg[cfg_name] = value
    for arg_name, cfg_name in train_overrides.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            train_cfg[cfg_name] = value


def _ensure_onehot(label: np.ndarray, num_classes: int) -> np.ndarray:
    if label.ndim > 1 and label.shape[-1] == num_classes:
        return label.astype(np.float32)
    label = label.reshape(-1).astype(np.int64)
    return np.eye(num_classes, dtype=np.float32)[label]


def _label_index(y: torch.Tensor) -> torch.Tensor:
    if y.ndim > 1:
        return y.argmax(dim=1)
    return y.long().view(-1)


def _minmax_scale_samples(data: np.ndarray) -> np.ndarray:
    if data.size == 0:
        return data.astype(np.float32)
    shape = data.shape
    x2 = data.reshape(shape[0], -1)

    x_min = x2.min(axis=1, keepdims=True)
    x_max = x2.max(axis=1, keepdims=True)
    denom = np.maximum(x_max - x_min, 1e-6)
    x2 = 2.0 * (x2 - x_min) / denom - 1.0
    return x2.astype(np.float32).reshape(shape)


def _flatten_time_steps(feature: np.ndarray, label: np.ndarray, sid: np.ndarray):
    if feature.ndim == 3:
        return feature, label, sid
    if feature.ndim != 4:
        raise ValueError(f"FDGCL expects feature shape [N, C, F] or [N, T, C, F], got {feature.shape}")

    num_samples, time_length, num_channels, num_bands = feature.shape
    feature = feature.reshape(num_samples * time_length, num_channels, num_bands)
    label = np.repeat(label, time_length, axis=0)
    sid = np.repeat(sid, time_length, axis=0)
    return feature, label, sid


def _collect_samples(indexes, data_keep, label_keep, num_classes):
    xs, ys, sids = [], [], []
    for sid, sub_x, sub_y in zip(indexes, data_keep, label_keep):
        sub_x = np.array(sub_x, dtype=np.float32)
        sub_y = _ensure_onehot(np.array(sub_y), num_classes)

        if len(sub_x) == 0:
            continue

        sub_sid = np.full((len(sub_x),), int(sid), dtype=np.int64)
        sub_x, sub_y, sub_sid = _flatten_time_steps(sub_x, sub_y, sub_sid)
        sub_x = _minmax_scale_samples(sub_x)

        xs.append(sub_x)
        ys.append(sub_y)
        sids.append(sub_sid)

    if len(xs) == 0:
        return (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.int64),
        )

    return (
        np.vstack(xs).astype(np.float32),
        np.vstack(ys).astype(np.float32),
        np.concatenate(sids).astype(np.int64),
    )


def _build_loader(feature, label, sid, batch_size, shuffle, drop_last):
    dataset = TensorDataset(
        torch.from_numpy(feature).float(),
        torch.from_numpy(label).float(),
        torch.from_numpy(sid).long(),
    )
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def _next_cycle(loader_iter, loader):
    try:
        return next(loader_iter), loader_iter
    except StopIteration:
        loader_iter = iter(loader)
        return next(loader_iter), loader_iter


def _evaluate(model, dataset: TensorDataset, metrics, device, batch_size=64):
    model.eval()
    metric = Metric(metrics)
    with torch.no_grad():
        eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        for batch_x, batch_y, batch_sid in eval_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_sid = batch_sid.to(device)

            pred_prob = model.predict(batch_x, batch_sid)
            pred = pred_prob.argmax(dim=1)
            target = _label_index(batch_y)
            metric.update(pred, target)

    print("\033[34m eval state: " + metric.value())
    return metric.values


def _scalar_from_output(out, key):
    value = out.get(key, None)
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach().item())
    return float(value)


def _format_class_counts(values):
    return "[" + ",".join(str(int(round(v))) for v in values.tolist()) + "]"


def _train_one_round(
    args,
    model_params,
    train_cfg,
    source_loader,
    target_loader,
    val_dataset,
    test_dataset,
    output_dir,
    device,
):
    setup_seed(args.seed)
    model = FDGCL(model_params).to(device)

    learning_rate = float(args.lr)
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    optimizer_name = str(train_cfg.get("optimizer", "adamw")).lower()
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    metric_key = args.metric_choose if args.metric_choose in args.metrics else args.metrics[0]
    best_metric = -1.0
    eval_interval = max(1, int(train_cfg.get("eval_interval", 1)))

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_num = 0
        source_correct = 0.0
        diag_scalar_keys = [
            "total_loss",
            "ce_loss",
            "ajloss",
            "gcl_loss",
            "align_loss",
            "orth_loss",
            "subject_loss",
            "subject_shared_loss",
            "subject_private_loss",
            "shared_subject_acc",
            "private_subject_acc",
            "target_pseudo_conf_mean",
            "target_reliability_mean",
            "target_feature_agreement_mean",
            "target_feature_margin_mean",
            "target_feature_entropy_score_mean",
            "target_scale_consistency_mean",
            "target_subject_invariance_mean",
            "target_align_conf_mean",
            "target_align_coverage",
            "target_align_count",
            "target_keep_ratio",
            "align_active",
        ]
        diag_sums = {key: 0.0 for key in diag_scalar_keys}
        diag_num = 0
        pseudo_class_counts = np.zeros(int(model_params["category_number"]), dtype=np.float64)
        align_class_counts = np.zeros(int(model_params["category_number"]), dtype=np.float64)

        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        step_num = max(len(source_loader), len(target_loader))
        train_bar = tqdm(range(step_num), desc=f"Train Epoch {epoch + 1}/{args.epochs}")

        for _ in train_bar:
            (source_x, source_y, source_sid), source_iter = _next_cycle(source_iter, source_loader)
            (target_x, _, target_sid), target_iter = _next_cycle(target_iter, target_loader)

            source_x = source_x.to(device)
            source_y = source_y.to(device)
            source_sid = source_sid.to(device)
            target_x = target_x.to(device)
            target_sid = target_sid.to(device)

            out = model(
                source_x=source_x,
                target_x=target_x,
                source_subject_ids=source_sid,
                target_subject_ids=target_sid,
                source_y=source_y,
                current_epoch=epoch,
            )
            loss = out["total_loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            source_pred = out["source_logits"].detach().argmax(dim=1)
            source_gt = out.get("source_labels", _label_index(source_y)).detach()
            source_correct += (source_pred == source_gt).float().sum().item()

            bs = len(source_x)
            total_num += bs
            total_loss += loss.item() * bs
            diag_num += bs
            for key in diag_scalar_keys:
                scalar = _scalar_from_output(out, key)
                if scalar is not None:
                    diag_sums[key] += scalar * bs
            pseudo_class_counts += out["target_pseudo_class_counts"].detach().cpu().numpy()
            align_class_counts += out["target_align_class_counts"].detach().cpu().numpy()
            train_bar.set_postfix_str(
                f"loss:{total_loss / max(1, total_num):.4f} source_acc:{source_correct / max(1, total_num) * 100:.2f}%"
            )

        diag_means = {key: value / max(1, diag_num) for key, value in diag_sums.items()}
        print(
            "\033[36m train diag: "
            f"loss={diag_means['total_loss']:.4f} "
            f"ce={diag_means['ce_loss']:.4f} "
            f"aj={diag_means['ajloss']:.4f} "
            f"gcl={diag_means['gcl_loss']:.4f} "
            f"align={diag_means['align_loss']:.4f} "
            f"orth={diag_means['orth_loss']:.4f} "
            f"subj={diag_means['subject_loss']:.4f} "
            f"subj_s={diag_means['subject_shared_loss']:.4f} "
            f"subj_p={diag_means['subject_private_loss']:.4f} "
            f"subj_acc_s={diag_means['shared_subject_acc'] * 100:.2f}% "
            f"subj_acc_p={diag_means['private_subject_acc'] * 100:.2f}% "
            f"pseudo_conf={diag_means['target_pseudo_conf_mean']:.4f} "
            f"rel={diag_means['target_reliability_mean']:.4f} "
            f"feat_rel={diag_means['target_feature_agreement_mean']:.4f} "
            f"margin={diag_means['target_feature_margin_mean']:.4f} "
            f"entropy_score={diag_means['target_feature_entropy_score_mean']:.4f} "
            f"scale_cons={diag_means['target_scale_consistency_mean']:.4f} "
            f"subj_inv={diag_means['target_subject_invariance_mean']:.4f} "
            f"align_conf={diag_means['target_align_conf_mean']:.4f} "
            f"align_cov={diag_means['target_align_coverage'] * 100:.2f}% "
            f"align_count={diag_means['target_align_count']:.2f} "
            f"keep_ratio={diag_means['target_keep_ratio']:.2f} "
            f"align_batch={diag_means['align_active'] * 100:.2f}% "
            f"pseudo_counts={_format_class_counts(pseudo_class_counts)} "
            f"align_counts={_format_class_counts(align_class_counts)}"
        )

        if (epoch + 1) % eval_interval == 0:
            eval_metric = _evaluate(
                model,
                val_dataset,
                args.metrics,
                device,
                batch_size=args.batch_size,
            )
            if eval_metric[metric_key] > best_metric:
                best_metric = eval_metric[metric_key]
                save_state(output_dir, model, optimizer, epoch + 1, metric=metric_key)

    ckpt_path = Path(output_dir) / f"checkpoint-best{metric_key}"
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])

    return _evaluate(
        model,
        test_dataset,
        args.metrics,
        device,
        batch_size=args.batch_size,
    )


def main(args):
    args.model = "FDGCL"
    cfg = _load_cfg()
    params_cfg = cfg.get("params", {})
    train_cfg = cfg.get("train", {})
    _apply_cli_overrides(args, params_cfg, train_cfg)

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

            train_data_keep, train_label_keep, val_data_keep, val_label_keep, test_data_keep, test_label_keep = index_to_data(
                data_i,
                label_i,
                train_indexes,
                test_indexes,
                val_indexes,
                True,
            )

            # Strict no-test-leak protocol: use val split as target stream and for model selection.
            if len(val_data_keep) == 0:
                print("skip one split because val split is empty under strict no-test-leak protocol")
                continue

            source_x, source_y, source_sid = _collect_samples(train_indexes, train_data_keep, train_label_keep, num_classes)
            target_x, target_y, target_sid = _collect_samples(val_indexes, val_data_keep, val_label_keep, num_classes)
            test_x, test_y, test_sid = _collect_samples(test_indexes, test_data_keep, test_label_keep, num_classes)

            if len(source_x) == 0 or len(target_x) == 0 or len(test_x) == 0:
                print("skip one split because source/target/test split is empty")
                continue

            source_loader = _build_loader(source_x, source_y, source_sid, args.batch_size, shuffle=True, drop_last=True)
            target_loader = _build_loader(target_x, target_y, target_sid, args.batch_size, shuffle=True, drop_last=True)

            val_dataset = TensorDataset(
                torch.from_numpy(target_x).float(),
                torch.from_numpy(target_y).float(),
                torch.from_numpy(target_sid).long(),
            )
            test_dataset = TensorDataset(
                torch.from_numpy(test_x).float(),
                torch.from_numpy(test_y).float(),
                torch.from_numpy(test_sid).long(),
            )

            model_params = {
                "DEVICE": device,
                "num_of_vertices": int(channels),
                "num_of_features": int(feature_dim),
                "category_number": int(num_classes),
                "num_subjects": int(total_subjects),
                "graph_hidden": int(params_cfg.get("graph_hidden", 64)),
                "graph_readout_hidden": int(params_cfg.get("graph_readout_hidden", 256)),
                "gcl_readout_hidden": int(params_cfg.get("gcl_readout_hidden", 256)),
                "spectral_hidden": int(params_cfg.get("spectral_hidden", 128)),
                "disentangle_dim": int(params_cfg.get("disentangle_dim", 128)),
                "projection_dim": int(params_cfg.get("projection_dim", 64)),
                "frequency_band_groups": params_cfg.get("frequency_band_groups", None),
                "cross_scale_heads": int(params_cfg.get("cross_scale_heads", 4)),
                "dropout": float(params_cfg.get("dropout", 0.2)),
                "temperature": float(params_cfg.get("temperature", 0.2)),
                "ugfcda_warmup_epochs": int(train_cfg.get("ugfcda_warmup_epochs", params_cfg.get("ugfcda_warmup_epochs", 10))),
                "ugfcda_eps": float(params_cfg.get("ugfcda_eps", 1e-6)),
                "ugfcda_keep_ratio_start": float(params_cfg.get("ugfcda_keep_ratio_start", 0.2)),
                "ugfcda_keep_ratio_end": float(params_cfg.get("ugfcda_keep_ratio_end", 0.6)),
                "ugfcda_keep_ratio_step": float(params_cfg.get("ugfcda_keep_ratio_step", 0.1)),
                "ugfcda_keep_ratio_step_epochs": int(params_cfg.get("ugfcda_keep_ratio_step_epochs", 20)),
                "ugfcda_subject_weight": float(params_cfg.get("ugfcda_subject_weight", 0.5)),
                "ugfcda_proto_align_weight": float(params_cfg.get("ugfcda_proto_align_weight", 0.1)),
                "node_drop_rate": float(params_cfg.get("node_drop_rate", 0.15)),
                "edge_drop_rate": float(params_cfg.get("edge_drop_rate", 0.10)),
                "gcl_importance_protect": params_cfg.get("gcl_importance_protect", True),
                "gcl_importance_centrality_weight": float(params_cfg.get("gcl_importance_centrality_weight", 0.5)),
                "gcl_importance_feature_weight": float(params_cfg.get("gcl_importance_feature_weight", 0.5)),
                "gcl_node_sample_temperature": float(params_cfg.get("gcl_node_sample_temperature", 0.7)),
                "gcl_node_sample_eps": float(params_cfg.get("gcl_node_sample_eps", 1e-6)),
                "gcl_edge_protect_strength": float(params_cfg.get("gcl_edge_protect_strength", 0.7)),
                "gcl_edge_min_drop_scale": float(params_cfg.get("gcl_edge_min_drop_scale", 0.3)),
                "GLalpha": float(params_cfg.get("GLalpha", 0.01)),
                "K": int(params_cfg.get("K", 3)),
                "ssbn_eps": float(params_cfg.get("ssbn_eps", 1e-5)),
                "sin_min_count": int(params_cfg.get("sin_min_count", 2)),
                "grl_max_iters": float(params_cfg.get("grl_max_iters", 2000)),
                "w_ce": float(train_cfg.get("loss_ce", 1.0)),
                "w_aj": float(train_cfg.get("loss_aj", 0.2)),
                "w_gcl": float(train_cfg.get("loss_gcl", 0.3)),
                "w_align": float(train_cfg.get("loss_align", 0.2)),
                "w_orth": float(train_cfg.get("loss_orth", 0.5)),
                "w_subject": float(train_cfg.get("loss_subject", 0.3)),
            }

            output_dir = make_output_dir(args, "FDGCL")
            round_metric = _train_one_round(
                args=args,
                model_params=model_params,
                train_cfg=train_cfg,
                source_loader=source_loader,
                target_loader=target_loader,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
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
