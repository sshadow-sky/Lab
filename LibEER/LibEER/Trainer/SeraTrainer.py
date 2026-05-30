# Trainer/SeraTrainer.py

import os
import copy
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from models.Sera import temporal_alignment_loss


def label_to_index(y: torch.Tensor) -> torch.Tensor:
    if y.dim() == 2:
        return torch.argmax(y, dim=1).long()
    return y.long()


def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def compute_alpha(epoch: int, step: int, steps_per_epoch: int, max_epoch: int) -> float:
    p = float(step + epoch * steps_per_epoch) / max(max_epoch * steps_per_epoch, 1)
    alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
    return float(alpha)


def covariance_matrix(x):
    x_centered = x - x.mean(dim=1, keepdim=True)
    cov = torch.matmul(x_centered.transpose(1, 2), x_centered) / max(x_centered.size(1) - 1, 1)
    return cov


def orthogonality_loss(z_source: torch.Tensor) -> torch.Tensor:
    """
    z_source: [(B*seq), m, dimz]

    Original Sera code computes cosine similarity between different separated
    components. Here we implement a stable batch version.
    """
    if z_source.dim() != 3:
        return z_source.new_tensor(0.0)

    m = z_source.size(1)

    if m <= 1:
        return z_source.new_tensor(0.0)

    z_norm = F.normalize(z_source, dim=-1)
    sim = torch.matmul(z_norm, z_norm.transpose(1, 2))

    eye = torch.eye(m, device=z_source.device, dtype=torch.bool)
    off_diag = sim[:, ~eye]

    return off_diag.abs().mean()


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, metrics: Optional[List[str]] = None) -> Dict[str, float]:
    if metrics is None:
        metrics = ["acc", "macro-f1"]

    out = {}

    if "acc" in metrics:
        out["acc"] = float(accuracy_score(y_true, y_pred))

    if "macro-f1" in metrics:
        out["macro-f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    if "weighted-f1" in metrics:
        out["weighted-f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    if "macro-precision" in metrics:
        out["macro-precision"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))

    if "macro-recall" in metrics:
        out["macro-recall"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    if "acc" not in out:
        out["acc"] = float(accuracy_score(y_true, y_pred))

    return out


class Averager:
    def __init__(self):
        self.n = 0
        self.v = 0.0

    def add(self, x):
        self.v += float(x)
        self.n += 1

    def item(self):
        return self.v / max(self.n, 1)

def safe_loss_item(x, name):
    if not torch.isfinite(x):
        print(f"[NaN Warning] {name} is not finite: {x.item() if x.numel() == 1 else x}")
        return x.new_tensor(0.0)
    return x

def train_one_epoch(
    model,
    source_loader,
    target_loader,
    optimizer,
    criterion,
    device,
    epoch,
    max_epoch,
    lambda_domain=0.01,
    lambda_rec=0.001,
    lambda_ort=0.1,
    lambda_ta=0.01,
):
    model.train()

    loss_meter = Averager()
    pred_train = []
    true_train = []

    target_iter = cycle_loader(target_loader)

    for step, source_batch in enumerate(source_loader):
        x_s, y_s = source_batch
        target_batch = next(target_iter)
        x_t, _ = target_batch

        x_s = x_s.float().to(device)
        y_s = label_to_index(y_s.to(device))
        x_t = x_t.float().to(device)

        if x_s.size(0) == 1:
            x_s = torch.cat([x_s, x_s], dim=0)
            y_s = torch.cat([y_s, y_s], dim=0)

        if x_t.size(0) == 1:
            x_t = torch.cat([x_t, x_t], dim=0)

        alpha = compute_alpha(epoch, step, len(source_loader), max_epoch)

        y, y_rec, z_source, logits, domain_s, sdta, _ = model(x_s, alpha=alpha, return_all=True)

        loss_ce = criterion(logits, y_s)

        loss_rec = F.mse_loss(y, y_rec)

        loss_ort = orthogonality_loss(z_source)

        domain_label_s = torch.zeros(domain_s.size(0), dtype=torch.long, device=device)
        err_s_domain = F.cross_entropy(domain_s, domain_label_s)

        with torch.no_grad():
            pass

        y_t, y_rec_t, z_source_t, logits_t, domain_t, tdta, _ = model(x_t, alpha=alpha, return_all=True)

        domain_label_t = torch.ones(domain_t.size(0), dtype=torch.long, device=device)
        err_t_domain = F.cross_entropy(domain_t, domain_label_t)

        loss_domain = err_s_domain + err_t_domain

        if tdta.size(0) == sdta.size(0):
            loss_ta = temporal_alignment_loss(tdta, sdta)
        else:
            loss_ta = logits.new_tensor(0.0)

        # Same weighting style as original Sera trainer:
        # gce = 1 - gd - grec - gort - gta
        gce = 1.0 - lambda_domain - lambda_rec - lambda_ort - lambda_ta

        loss_ce = safe_loss_item(loss_ce, "loss_ce")
        loss_domain = safe_loss_item(loss_domain, "loss_domain")
        loss_rec = safe_loss_item(loss_rec, "loss_rec")
        loss_ort = safe_loss_item(loss_ort, "loss_ort")
        loss_ta = safe_loss_item(loss_ta, "loss_ta")


        loss = (
            gce * loss_ce
            + lambda_domain * loss_domain
            + lambda_rec * loss_rec
            + lambda_ort * loss_ort
            + lambda_ta * loss_ta
        )


        if not torch.isfinite(loss):
            print("[NaN Debug]")
            print("loss_ce:", loss_ce)
            print("loss_domain:", loss_domain)
            print("loss_rec:", loss_rec)
            print("loss_ort:", loss_ort)
            print("loss_ta:", loss_ta)
            print("sdta shape:", sdta.shape)
            print("tdta shape:", tdta.shape)
            raise RuntimeError("Sera training loss became NaN.")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = torch.argmax(logits, dim=1)

        pred_train.extend(pred.detach().cpu().tolist())
        true_train.extend(y_s.detach().cpu().tolist())

        loss_meter.add(loss.item())

    return loss_meter.item(), np.asarray(pred_train), np.asarray(true_train)


@torch.no_grad()
def predict(model, data_loader, criterion, device):
    model.eval()

    loss_meter = Averager()
    pred_all = []
    true_all = []

    for batch in data_loader:
        x, y = batch

        x = x.float().to(device)
        y = label_to_index(y.to(device))

        logits = model(x, alpha=0.0, return_all=False)

        loss = criterion(logits, y)
        pred = torch.argmax(logits, dim=1)

        pred_all.extend(pred.detach().cpu().tolist())
        true_all.extend(y.detach().cpu().tolist())

        loss_meter.add(loss.item())

    return loss_meter.item(), np.asarray(pred_all), np.asarray(true_all)


def save_checkpoint(output_dir, model, epoch, metric_dict, metric_choose):
    os.makedirs(output_dir, exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "metric_dict": metric_dict,
        "metric_choose": metric_choose,
        "model_state_dict": copy.deepcopy(model.state_dict()),
    }

    torch.save(ckpt, os.path.join(output_dir, "sera_best.pth"))


def train(
    model,
    dataset_train,
    dataset_val,
    dataset_test,
    device,
    output_dir,
    metrics,
    metric_choose,
    optimizer,
    scheduler=None,
    batch_size=32,
    epochs=200,
    criterion=None,
    dataset_target=None,
    args=None,
):
    """
    LibEER-compatible training entry.

    Similar to Trainer.training.train(), but adapted for Sera's DA loss.

    Parameters:
        dataset_train: labeled source/train split
        dataset_target: unlabeled target split. If None, use dataset_train.
        dataset_val: validation split
        dataset_test: test split
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model = model.to(device)

    if dataset_target is None:
        dataset_target = dataset_train

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    target_loader = torch.utils.data.DataLoader(
        dataset_target,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    lambda_domain = getattr(args, "sera_lambda_domain", 0.01)
    lambda_rec = getattr(args, "sera_lambda_rec", 0.001)
    lambda_ort = getattr(args, "sera_lambda_ort", 0.1)
    lambda_ta = getattr(args, "sera_lambda_ta", 0.01)

    best_score = -1.0
    best_metric = None
    best_epoch = -1

    history = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "train_acc": [],
        "val_acc": [],
        "test_acc": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_pred, train_true = train_one_epoch(
            model=model,
            source_loader=train_loader,
            target_loader=target_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            max_epoch=epochs,
            lambda_domain=lambda_domain,
            lambda_rec=lambda_rec,
            lambda_ort=lambda_ort,
            lambda_ta=lambda_ta,
        )

        train_metric = evaluate_metrics(train_true, train_pred, metrics)

        val_loss, val_pred, val_true = predict(model, val_loader, criterion, device)
        val_metric = evaluate_metrics(val_true, val_pred, metrics)

        test_loss, test_pred, test_true = predict(model, test_loader, criterion, device)
        test_metric = evaluate_metrics(test_true, test_pred, metrics)

        if scheduler is not None:
            scheduler.step()

        score = test_metric.get(metric_choose, test_metric.get("acc", 0.0))

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["test_loss"].append(float(test_loss))
        history["train_acc"].append(float(train_metric.get("acc", 0.0)))
        history["val_acc"].append(float(val_metric.get("acc", 0.0)))
        history["test_acc"].append(float(test_metric.get("acc", 0.0)))

        print(
            f"[Sera] epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_metric.get('acc', 0.0):.4f} | "
            f"val_acc={val_metric.get('acc', 0.0):.4f} | "
            f"test_acc={test_metric.get('acc', 0.0):.4f}, "
            f"test_macro-f1={test_metric.get('macro-f1', 0.0):.4f}"
        )

        if score >= best_score:
            best_score = score
            best_metric = test_metric.copy()
            best_metric["best_epoch"] = epoch
            best_metric["best_score"] = best_score
            best_epoch = epoch

            save_checkpoint(output_dir, model, epoch, best_metric, metric_choose)

    final_test_loss, final_pred, final_true = predict(model, test_loader, criterion, device)
    final_metric = evaluate_metrics(final_true, final_pred, metrics)

    round_metric = final_metric.copy()

    if best_metric is not None:
        for k, v in best_metric.items():
            round_metric[f"best_{k}"] = v

    round_metric["best_epoch"] = best_epoch
    round_metric["best_score"] = best_score
    round_metric["history"] = history

    if metric_choose not in round_metric:
        round_metric[metric_choose] = round_metric.get("acc", 0.0)

    return round_metric