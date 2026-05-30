import copy
import math
from pathlib import Path

import numpy as np
import torch
from utils.metric import Metric

from models.DMMR import DMMRFineTuningModel
from models.DMMR import DMMRPreTrainingModel
from models.DMMR import DMMRTestModel
from models.DMMR import build_correspondence_batch


def _label_to_index(y):
    y = np.asarray(y)
    if y.ndim > 1 and y.shape[-1] > 1:
        return np.argmax(y, axis=1).astype(np.int64)
    return y.reshape(-1).astype(np.int64)


def _to_dmmr_feature_seq(x, time_steps, input_dim):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 4:
        # (sample, time, channel, band) -> (sample, time, channel*band)
        x = x.reshape(x.shape[0], x.shape[1], -1)
    if x.ndim != 3:
        raise ValueError(f"DMMR expects 3D feature tensor [N, T, F], got shape {x.shape}")
    if x.shape[1] != time_steps:
        raise ValueError(f"DMMR expects time_steps={time_steps}, got input shape {x.shape}")
    if x.shape[2] != input_dim:
        raise ValueError(f"DMMR expects input_dim={input_dim}, got input shape {x.shape}")
    return x


def build_dmmr_loaders(
    source_data_list,
    source_label_list,
    target_data,
    target_label,
    batch_size,
    time_steps,
    input_dim,
    num_workers=4,
):
    source_loaders = []
    for one_subject_data, one_subject_label in zip(source_data_list, source_label_list):
        x = _to_dmmr_feature_seq(one_subject_data, time_steps=time_steps, input_dim=input_dim)
        y = _label_to_index(one_subject_label)
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long).unsqueeze(1),
        )
        source_loaders.append(
            torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=True,
            )
        )

    tx = _to_dmmr_feature_seq(target_data, time_steps=time_steps, input_dim=input_dim)
    ty = _label_to_index(target_label)
    target_dataset = torch.utils.data.TensorDataset(
        torch.tensor(tx, dtype=torch.float32),
        torch.tensor(ty, dtype=torch.long).unsqueeze(1),
    )
    target_loader = torch.utils.data.DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        drop_last=True,
    )
    return source_loaders, target_loader


@torch.no_grad()
def _evaluate_target_metrics(test_loader, test_model, device, metrics):
    test_model.eval()
    metric = Metric(metrics)
    for test_input, label in test_loader:
        test_input = test_input.to(device)
        label = label.to(device)
        pred_logits = test_model(test_input)
        pred = torch.argmax(pred_logits, dim=1)
        metric.update(pred, label.squeeze())
    metric.value()
    return metric.values


def train_dmmr(
    source_loaders,
    test_loader,
    args,
    device,
    output_dir,
    optimizer_config,
    subject_id,
):
    iteration = int(args.dmmr_iteration)

    pretrain_model = DMMRPreTrainingModel(
        number_of_source=len(source_loaders),
        number_of_category=args.dmmr_num_classes,
        batch_size=args.batch_size,
        time_steps=args.dmmr_time_steps,
        input_dim=args.dmmr_input_dim,
        hid_dim=args.dmmr_hid_dim,
        n_layers=args.dmmr_n_layers,
    ).to(device)

    source_iters = [iter(loader) for loader in source_loaders]
    optimizer_pretraining = torch.optim.Adam(pretrain_model.parameters(), **optimizer_config)

    for epoch in range(args.dmmr_epoch_pretraining):
        pretrain_model.train()
        for i in range(1, iteration + 1):
            p = float(i + epoch * iteration) / args.dmmr_epoch_pretraining / iteration
            m = 2.0 / (1.0 + np.exp(-10 * p)) - 1

            batch_data_list = []
            batch_label_list = []
            source_batches = []
            source_batch_labels = []
            for j in range(len(source_iters)):
                try:
                    source_data, source_label = next(source_iters[j])
                except Exception:
                    source_iters[j] = iter(source_loaders[j])
                    source_data, source_label = next(source_iters[j])
                source_batches.append(source_data)
                source_batch_labels.append(source_label)
                batch_data_list.append(source_data)
                batch_label_list.append(source_label.squeeze(1))

            for j in range(len(source_iters)):
                source_data = source_batches[j].to(device)
                source_label = source_batch_labels[j].to(device)

                sid = torch.ones(args.batch_size, device=device, dtype=torch.long) * j
                corres_batch_data = build_correspondence_batch(
                    source_batches,
                    [x.squeeze(1) for x in source_batch_labels],
                    source_label.squeeze(1).cpu().tolist(),
                ).to(device)

                optimizer_pretraining.zero_grad()
                rec_loss, sim_loss = pretrain_model(source_data, corres_batch_data, sid, m, mark=j)
                loss_pretrain = rec_loss + args.dmmr_beta * sim_loss
                loss_pretrain.backward()
                optimizer_pretraining.step()

    fine_tune_model = DMMRFineTuningModel(
        base_model=pretrain_model,
        number_of_source=len(source_loaders),
        number_of_category=args.dmmr_num_classes,
        batch_size=args.batch_size,
        time_steps=args.dmmr_time_steps,
    ).to(device)
    source_iters2 = [iter(loader) for loader in source_loaders]
    optimizer_finetuning = torch.optim.Adam(fine_tune_model.parameters(), **optimizer_config)

    metrics = args.metrics if args.metrics is not None else ["acc"]
    metric_choose = args.metric_choose if args.metric_choose in metrics else metrics[0]
    best_choose_score = float("-inf")
    best_metrics = {m: float("nan") for m in metrics}
    best_pretrain_model = copy.deepcopy(pretrain_model.state_dict())
    best_tune_model = copy.deepcopy(fine_tune_model.state_dict())
    best_test_model = copy.deepcopy(DMMRTestModel(fine_tune_model).state_dict())

    for epoch in range(args.dmmr_epoch_finetuning):
        fine_tune_model.train()
        for _ in range(1, iteration + 1):
            source_batches = []
            source_batch_labels = []
            for j in range(len(source_iters2)):
                try:
                    source_data, source_label = next(source_iters2[j])
                except Exception:
                    source_iters2[j] = iter(source_loaders[j])
                    source_data, source_label = next(source_iters2[j])
                source_batches.append(source_data)
                source_batch_labels.append(source_label)

            for j in range(len(source_iters2)):
                source_data = source_batches[j].to(device)
                source_label = source_batch_labels[j].to(device)
                optimizer_finetuning.zero_grad()
                _, _, cls_loss = fine_tune_model(source_data, source_label)
                cls_loss.backward()
                optimizer_finetuning.step()

        test_model = DMMRTestModel(fine_tune_model).to(device)
        metric_values = _evaluate_target_metrics(test_loader, test_model, device, metrics)
        choose_score = metric_values[metric_choose]
        if choose_score > best_choose_score:
            best_choose_score = choose_score
            best_metrics = copy.deepcopy(metric_values)
            best_pretrain_model = copy.deepcopy(pretrain_model.state_dict())
            best_tune_model = copy.deepcopy(fine_tune_model.state_dict())
            best_test_model = copy.deepcopy(test_model.state_dict())

    model_dir = Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_pretrain_model, model_dir / f"subject_{subject_id + 1}_pretrain_model.pth")
    torch.save(best_tune_model, model_dir / f"subject_{subject_id + 1}_tune_model.pth")
    torch.save(best_test_model, model_dir / f"subject_{subject_id + 1}_test_model.pth")
    return best_metrics