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
from models.DMS_SGPAN import DMS_SGPAN
from utils.args import get_args_parser
from utils.metric import Metric
from utils.store import make_output_dir, save_state
from utils.utils import result_log, setup_seed, sub_result_log


param_path = "config/model_param/DMS_SGPAN.yaml"

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
        "dms_sgpan_ugfcda_warmup_epochs": "ugfcda_warmup_epochs",
        "dms_sgpan_ugfcda_eps": "ugfcda_eps",
        "dms_sgpan_ugfcda_reliability_threshold": "ugfcda_reliability_threshold",
        "dms_sgpan_ugfcda_proto_align_weight": "ugfcda_proto_align_weight",
        "dms_sgpan_node_drop_rate": "node_drop_rate",
        "dms_sgpan_edge_drop_rate": "edge_drop_rate",
        "dms_sgpan_dropout": "dropout",
        "dms_sgpan_temperature": "temperature",
        "dms_sgpan_graph_hidden": "graph_hidden",
        "dms_sgpan_graph_readout_hidden": "graph_readout_hidden",
        "dms_sgpan_gcl_readout_hidden": "gcl_readout_hidden",
        "dms_sgpan_spectral_hidden": "spectral_hidden",
        "dms_sgpan_disentangle_dim": "disentangle_dim",
        "dms_sgpan_projection_dim": "projection_dim",
        "dms_sgpan_cross_scale_heads": "cross_scale_heads",
        "dms_sgpan_gl_alpha": "GLalpha",
        "dms_sgpan_cheb_k": "K",
        "dms_sgpan_ssbn_eps": "ssbn_eps",
        "dms_sgpan_sin_min_count": "sin_min_count",
        "dms_sgpan_grl_max_iters": "grl_max_iters",
    }
    train_overrides = {
        "dms_sgpan_loss_ce": "loss_ce",
        "dms_sgpan_loss_aj": "loss_aj",
        "dms_sgpan_loss_gcl": "loss_gcl",
        "dms_sgpan_loss_align": "loss_align",
        "dms_sgpan_loss_orth": "loss_orth",
        "dms_sgpan_loss_subject": "loss_subject",
    }

    for arg_name, cfg_name in param_overrides.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            params_cfg[cfg_name] = value
        # Special handling for frequency_band_groups passed as JSON string
    fbgs = getattr(args, 'dms_sgpan_frequency_band_groups', None)
    if fbgs is not None:
        import json
        try:
            parsed = json.loads(fbgs)
            # basic validation: should be list/tuple of lists/ints
            if isinstance(parsed, (list, tuple)):
                params_cfg['frequency_band_groups'] = parsed
        except Exception:
            print('Warning: failed to parse -dms_sgpan_frequency_band_groups, ignoring')
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
        raise ValueError(f"DMS_SGPAN expects feature shape [N, C, F] or [N, T, C, F], got {feature.shape}")

    num_samples, time_length, num_channels, num_bands = feature.shape
    feature = feature.reshape(num_samples * time_length, num_channels, num_bands)
    label = np.repeat(label, time_length, axis=0)
    sid = np.repeat(sid, time_length, axis=0)
    return feature, label, sid


def _collect_samples(indexes, data_keep, label_keep, num_classes):
    xs, ys, sids = [], [], []
    for sid, sub_x, sub_y in zip(indexes, data_keep, label_keep):
        try:
            sub_x = np.array(sub_x, dtype=np.float32)
        except ValueError as exc:
            shape_counts = {}
            for sample in sub_x:
                shape = tuple(np.asarray(sample).shape)
                shape_counts[shape] = shape_counts.get(shape, 0) + 1
            shape_text = ", ".join(f"{shape}:{count}" for shape, count in sorted(shape_counts.items())[:8])
            raise ValueError(
                f"DMS_SGPAN collected inconsistent sample shapes for subject {sid}: {shape_text}"
            ) from exc
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


def _format_metric_values(metric, metric_names):
    return " ".join(f"{name}={float(metric[name]):.4f}" for name in metric_names if name in metric)


def _format_subject_indexes(indexes):
    zero_based = ",".join(str(int(idx)) for idx in indexes)
    one_based = ",".join(str(int(idx) + 1) for idx in indexes)
    return f"0-based=[{zero_based}] 1-based=[{one_based}]"


def _print_test_subject_summary(args, test_subject_records):
    if len(test_subject_records) == 0:
        print("FDGCL test subject summary: no finished test subject records")
        return

    print("\nFDGCL test subject summary")
    print(f"FDGCL per-test-subject records: {len(test_subject_records)}")
    header = "|{:^8}|{:^18}|".format("Round", "TestSubject")
    for metric_name in args.metrics:
        header += "{:^15}|".format(metric_name)
    print(header)

    metric_outputs = {metric_name: [] for metric_name in args.metrics}
    for record in test_subject_records:
        subject_text = ",".join(str(int(idx) + 1) for idx in record["test_indexes"])
        row = "|{:^8}|{:^18}|".format(record["round"], subject_text)
        for metric_name in args.metrics:
            value = float(record["metric"][metric_name])
            metric_outputs[metric_name].append(value)
            row += "{:^15.4f}|".format(value)
        print(row)

    row = "|{:^8}|{:^18}|".format("Mean", "-")
    for metric_name in args.metrics:
        values = np.asarray(metric_outputs[metric_name], dtype=np.float64)
        row += "{:^15.4f}|".format(float(np.mean(values)))
    print(row)

    row = "|{:^8}|{:^18}|".format("Std", "-")
    for metric_name in args.metrics:
        values = np.asarray(metric_outputs[metric_name], dtype=np.float64)
        row += "{:^15.4f}|".format(float(np.std(values)))
    print(row)

    for metric_name in args.metrics:
        values = np.asarray(metric_outputs[metric_name], dtype=np.float64)
        print(
            "ALLTestSubject Mean and Std of {} : {:.4f}/{:.4f}".format(
                metric_name,
                float(np.mean(values)),
                float(np.std(values)),
            )
        )


def _subject_records_to_metrics(test_subject_records):
    return [dict(record["metric"]) for record in test_subject_records]


def _train_one_round(
    args,
    model_params,
    train_cfg,
    source_loader,
    target_loader,
    val_dataset,
    test_dataset,
    test_subject_datasets,
    output_dir,
    device,
):
    setup_seed(args.seed)
    model = DMS_SGPAN(model_params).to(device)

    learning_rate = float(args.lr)
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    optimizer_name = str(train_cfg.get("optimizer", "adamw")).lower()
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    metric_key = args.metric_choose if args.metric_choose in args.metrics else args.metrics[0]
    best_metric = -1.0
    best_checkpoint_saved = False
    eval_interval = max(1, int(train_cfg.get("eval_interval", 1)))

    # The model is not eligible for checkpoint selection during UGFCDA warmup.
    # With warmup_epochs=W, epochs 1..W are warmup and selection starts at W+1.
    warmup_epochs = max(0, int(model_params.get("ugfcda_warmup_epochs", 0)))
    selection_start_epoch = warmup_epochs + 1  # one-based epoch number
    if int(args.epochs) < selection_start_epoch:
        raise ValueError(
            f"DMS_SGPAN requires at least {selection_start_epoch} epochs when "
            f"ugfcda_warmup_epochs={warmup_epochs}, but args.epochs={args.epochs}. "
            "No checkpoint would be eligible for selection."
        )
    print(
        f"DMS_SGPAN checkpoint selection: warmup epochs=1..{warmup_epochs}; "
        f"best-model selection starts from epoch {selection_start_epoch}."
    )

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
            "target_align_conf_mean",
            "target_align_coverage",
            "target_align_count",
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
            f"align_conf={diag_means['target_align_conf_mean']:.4f} "
            f"align_cov={diag_means['target_align_coverage'] * 100:.2f}% "
            f"align_count={diag_means['target_align_count']:.2f} "
            f"align_batch={diag_means['align_active'] * 100:.2f}% "
            f"pseudo_counts={_format_class_counts(pseudo_class_counts)} "
            f"align_counts={_format_class_counts(align_class_counts)}"
        )

        epoch_number = epoch + 1
        # Always evaluate the final epoch. This guarantees at least one
        # post-warmup candidate even when eval_interval does not divide epochs.
        should_eval = (
            epoch_number % eval_interval == 0
            or epoch_number == selection_start_epoch
            or epoch_number == int(args.epochs)
        )
        if should_eval:
            eval_metric = _evaluate(
                model,
                val_dataset,
                args.metrics,
                device,
                batch_size=args.batch_size,
            )

            if epoch_number < selection_start_epoch:
                print(
                    f"DMS_SGPAN validation at epoch {epoch_number}: "
                    f"{metric_key}={float(eval_metric[metric_key]):.4f}; "
                    f"warmup is active, so this checkpoint is not eligible "
                    "for best-model selection."
                )
            elif eval_metric[metric_key] > best_metric:
                best_metric = float(eval_metric[metric_key])
                save_state(
                    output_dir,
                    model,
                    optimizer,
                    epoch_number,
                    metric=metric_key,
                )
                best_checkpoint_saved = True
                print(
                    f"DMS_SGPAN selected new best checkpoint at epoch "
                    f"{epoch_number}: {metric_key}={best_metric:.4f}"
                )

    ckpt_path = Path(output_dir) / f"checkpoint-best{metric_key}"
    if best_checkpoint_saved and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        selected_epoch = state.get("epoch", "unknown")
        print(
            f"DMS_SGPAN loading best post-warmup checkpoint: "
            f"epoch={selected_epoch}, {metric_key}={best_metric:.4f}"
        )
    else:
        raise RuntimeError(
            "DMS_SGPAN did not create a post-warmup checkpoint. "
            f"warmup_epochs={warmup_epochs}, epochs={args.epochs}, "
            f"eval_interval={eval_interval}."
        )

    test_metric = _evaluate(
        model,
        test_dataset,
        args.metrics,
        device,
        batch_size=args.batch_size,
    )
    test_subject_metrics = []
    for subject_index, subject_dataset in test_subject_datasets:
        print(f"DMS_SGPAN evaluating test subject {_format_subject_indexes([subject_index])}")
        subject_metric = _evaluate(
            model,
            subject_dataset,
            args.metrics,
            device,
            batch_size=args.batch_size,
        )
        test_subject_metrics.append((subject_index, subject_metric))
    return test_metric, test_subject_metrics


def main(args):
    args.model = "DMS_SGPAN"
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
    test_subject_records = []

    for rridx, (data_i, label_i) in enumerate(zip(data, label), 1):
        tts = get_split_index(data_i, label_i, setting)
        total_subjects = len(data_i)

        for ridx, (train_indexes, test_indexes, val_indexes) in enumerate(zip(tts["train"], tts["test"], tts["val"]), 1):
            setup_seed(args.seed)
            if val_indexes[0] == -1:
                print(f"train indexes:{train_indexes}, test indexes:{test_indexes}")
            else:
                print(f"train indexes:{train_indexes}, val indexes:{val_indexes}, test indexes:{test_indexes}")
            print(
                f"DMS_SGPAN round {ridx}: current test subject(s) "
                f"{_format_subject_indexes(test_indexes)}"
            )

            train_data_keep, train_label_keep, val_data_keep, val_label_keep, test_data_keep, test_label_keep = index_to_data(
                data_i,
                label_i,
                train_indexes,
                test_indexes,
                val_indexes,
                True,
            )

            effective_val_indexes = val_indexes
            # Strict no-test-leak protocol: use val split as target stream and for model selection.
            # The original LibEER leave-one-out split has no validation split. Keep it runnable
            # as the legacy/non-strict protocol by using the test split as target/validation.
            if len(val_data_keep) == 0:
                if setting.split_type == "leave-one-out":
                    effective_val_indexes = test_indexes
                    val_data_keep = test_data_keep
                    val_label_keep = test_label_keep
                    print(
                        "DMS_SGPAN non-strict LOSO compatibility: val split is empty; "
                        "using test split as target/validation for this round (test-leak protocol)."
                    )
                else:
                    print("skip one split because val split is empty under strict no-test-leak protocol")
                    continue

            source_x, source_y, source_sid = _collect_samples(train_indexes, train_data_keep, train_label_keep, num_classes)
            target_x, target_y, target_sid = _collect_samples(effective_val_indexes, val_data_keep, val_label_keep, num_classes)
            test_x, test_y, test_sid = _collect_samples(test_indexes, test_data_keep, test_label_keep, num_classes)

            if len(source_x) == 0 or len(target_x) == 0 or len(test_x) == 0:
                print("skip one split because source/target/test split is empty")
                continue

            source_drop_last = len(source_x) >= args.batch_size
            target_drop_last = False
            source_loader = _build_loader(
                source_x,
                source_y,
                source_sid,
                args.batch_size,
                shuffle=True,
                drop_last=source_drop_last,
            )
            target_loader = _build_loader(
                target_x,
                target_y,
                target_sid,
                args.batch_size,
                shuffle=True,
                drop_last=target_drop_last,
            )
            print(
                f"DMS_SGPAN round {ridx}: samples source={len(source_x)} "
                f"target/val={len(target_x)} test={len(test_x)} "
                f"batch_size={args.batch_size} "
                f"source_batches={len(source_loader)} target_batches={len(target_loader)} "
                f"source_drop_last={source_drop_last} target_drop_last={target_drop_last}"
            )
            if len(source_loader) == 0 or len(target_loader) == 0:
                print("skip one split because source/target loader is empty")
                continue

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
            test_subject_datasets = []
            for subject_index, subject_data, subject_label in zip(test_indexes, test_data_keep, test_label_keep):
                sub_test_x, sub_test_y, sub_test_sid = _collect_samples(
                    [subject_index],
                    [subject_data],
                    [subject_label],
                    num_classes,
                )
                if len(sub_test_x) == 0:
                    continue
                test_subject_datasets.append((
                    int(subject_index),
                    TensorDataset(
                        torch.from_numpy(sub_test_x).float(),
                        torch.from_numpy(sub_test_y).float(),
                        torch.from_numpy(sub_test_sid).long(),
                    ),
                ))

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
            "ugfcda_warmup_epochs": int(train_cfg.get("ugfcda_warmup_epochs", params_cfg.get("ugfcda_warmup_epochs", 15))),
            "ugfcda_eps": float(params_cfg.get("ugfcda_eps", 1e-6)),
            "ugfcda_reliability_threshold": float(params_cfg.get("ugfcda_reliability_threshold", 0.6)),
            "ugfcda_proto_align_weight": float(params_cfg.get("ugfcda_proto_align_weight", 0.1)),
            "node_drop_rate": float(params_cfg.get("node_drop_rate", 0.15)),
            "edge_drop_rate": float(params_cfg.get("edge_drop_rate", 0.10)),
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

            output_dir = make_output_dir(args, "DMS_SGPAN")
            round_metric, round_subject_metrics = _train_one_round(
                args=args,
                model_params=model_params,
                train_cfg=train_cfg,
                source_loader=source_loader,
                target_loader=target_loader,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                test_subject_datasets=test_subject_datasets,
                output_dir=output_dir,
                device=device,
            )

            best_metrics.append(round_metric)
            if len(round_subject_metrics) > 0:
                for subject_index, subject_metric in round_subject_metrics:
                    test_subject_records.append({
                        "round": ridx if len(data) == 1 else f"{rridx}-{ridx}",
                        "test_indexes": [subject_index],
                        "metric": dict(subject_metric),
                    })
            else:
                test_subject_records.append({
                    "round": ridx if len(data) == 1 else f"{rridx}-{ridx}",
                    "test_indexes": list(test_indexes),
                    "metric": dict(round_metric),
                })
            print(
                f"DMS_SGPAN round {ridx}: test subject(s) "
                f"{_format_subject_indexes(test_indexes)} result "
                f"{_format_metric_values(round_metric, args.metrics)}"
            )
            if setting.experiment_mode == "subject-dependent":
                subjects_metrics[rridx - 1].append(round_metric)

    _print_test_subject_summary(args, test_subject_records)

    if len(best_metrics) == 0:
        print("DMS_SGPAN finished without valid round metrics; check split/loader diagnostics above.")
        return

    if setting.experiment_mode == "subject-dependent":
        sub_result_log(args, subjects_metrics)
    else:
        if len(test_subject_records) > 1:
            print(
                "FDGCL final result_log uses per-test-subject metrics "
                "so train-val-test std is computed like LOSO subject std."
            )
            result_log(args, _subject_records_to_metrics(test_subject_records))
        else:
            result_log(args, best_metrics)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
    
