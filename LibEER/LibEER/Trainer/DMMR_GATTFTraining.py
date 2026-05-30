import copy
from pathlib import Path

import numpy as np
import torch

from models.DMMR_GATTF import (
    DMMRGATTransformerFineTuningModel,
    DMMRGATTransformerPreTrainingModel,
    DMMRGATTransformerTestModel,
    build_correspondence_batch,
)
from Trainer.DMMRTraining import _evaluate_target_metrics, build_dmmr_loaders
from utils.metric import Metric


@torch.no_grad()
def _evaluate_target_metrics_for_gattf(test_loader, test_model, device, metrics):
    return _evaluate_target_metrics(test_loader, test_model, device, metrics)


def train_dmmr_gattf(
    source_loaders,
    test_loader,
    args,
    device,
    output_dir,
    optimizer_config,
    subject_id,
):
    iteration = int(args.gattf_iteration)

    pretrain_model = DMMRGATTransformerPreTrainingModel(
        number_of_source=len(source_loaders),
        number_of_category=args.gattf_num_classes,
        batch_size=args.batch_size,
        time_steps=args.gattf_time_steps,
        input_dim=args.gattf_input_dim,
        num_channels=args.gattf_num_channels,
        num_bands=args.gattf_num_bands,
        gat_hidden_dim=args.gattf_gat_hidden_dim,
        gat_heads=args.gattf_gat_heads,
        model_dim=args.gattf_model_dim,
        transformer_heads=args.gattf_transformer_heads,
        transformer_layers=args.gattf_transformer_layers,
        decoder_layers=args.gattf_decoder_layers,
        feedforward_dim=args.gattf_feedforward_dim,
        dropout=args.gattf_dropout,
        temperature=args.gattf_temperature,
    ).to(device)

    source_iters = [iter(loader) for loader in source_loaders]
    optimizer_pretraining = torch.optim.Adam(pretrain_model.parameters(), **optimizer_config)

    for epoch in range(args.gattf_epoch_pretraining):
        pretrain_model.train()
        for i in range(1, iteration + 1):
            p = float(i + epoch * iteration) / args.gattf_epoch_pretraining / iteration
            m = 2.0 / (1.0 + np.exp(-10 * p)) - 1

            source_batches = []
            source_batch_labels = []
            batch_data_list = []
            batch_label_list = []
            for j in range(len(source_iters)):
                try:
                    source_data, source_label = next(source_iters[j])
                except Exception:
                    source_iters[j] = iter(source_loaders[j])
                    source_data, source_label = next(source_iters[j])
                source_batches.append(source_data)
                source_batch_labels.append(source_label)
                batch_data_list.append(source_data)
                batch_label_list.append(source_label.view(-1))

            for j in range(len(source_iters)):
                source_data = batch_data_list[j].to(device)
                source_label = batch_label_list[j].to(device)
                sid = torch.ones(args.batch_size, device=device, dtype=torch.long) * j
                corres_batch_data = build_correspondence_batch(
                    source_batches,
                    [x.view(-1) for x in source_batch_labels],
                    source_label.view(-1).cpu().tolist(),
                ).to(device)

                optimizer_pretraining.zero_grad()
                rec_loss, sim_loss, contrast_loss = pretrain_model(source_data, corres_batch_data, sid, source_label, m, mark=j)
                loss_pretrain = rec_loss + args.gattf_beta * sim_loss + args.gattf_gamma * contrast_loss
                loss_pretrain.backward()
                optimizer_pretraining.step()

    fine_tune_model = DMMRGATTransformerFineTuningModel(
        base_model=pretrain_model,
        number_of_source=len(source_loaders),
        number_of_category=args.gattf_num_classes,
        batch_size=args.batch_size,
        time_steps=args.gattf_time_steps,
    ).to(device)

    source_iters2 = [iter(loader) for loader in source_loaders]
    optimizer_finetuning = torch.optim.Adam(fine_tune_model.parameters(), **optimizer_config)

    metrics = args.metrics if args.metrics is not None else ["acc"]
    metric_choose = args.metric_choose if args.metric_choose in metrics else metrics[0]
    best_choose_score = float("-inf")
    best_metrics = {m: float("nan") for m in metrics}
    best_pretrain_model = copy.deepcopy(pretrain_model.state_dict())
    best_tune_model = copy.deepcopy(fine_tune_model.state_dict())
    best_test_model = copy.deepcopy(DMMRGATTransformerTestModel(fine_tune_model).state_dict())

    for epoch in range(args.gattf_epoch_finetuning):
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

        test_model = DMMRGATTransformerTestModel(fine_tune_model).to(device)
        metric_values = _evaluate_target_metrics_for_gattf(test_loader, test_model, device, metrics)
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
