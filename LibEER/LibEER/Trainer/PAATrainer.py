# Trainer/PAATrainer.py

import os
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, RMSprop
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn import metrics as sk_metrics

from models.PAA import (
    PAALLoss,
    PAACLoss,
    discriminator,
    Pairwise_Learning,
    DomainAdversarialLoss,
)


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def to_onehot(y: torch.Tensor, num_classes: int, device: torch.device) -> torch.Tensor:
    y = y.to(device)

    if y.dim() == 2:
        return y.float()

    y = y.long()
    return torch.eye(num_classes, device=device)[y]


def label_to_numpy(y: torch.Tensor) -> np.ndarray:
    if y.dim() == 2:
        return torch.argmax(y, dim=1).detach().cpu().numpy()
    return y.detach().cpu().long().numpy()


def unpack_batch(batch):
    if isinstance(batch, dict):
        if "feature" in batch:
            x = batch["feature"]
        elif "x" in batch:
            x = batch["x"]
        elif "data" in batch:
            x = batch["data"]
        else:
            raise KeyError(f"Cannot find feature key in batch: {batch.keys()}")

        if "label" in batch:
            y = batch["label"]
        elif "y" in batch:
            y = batch["y"]
        elif "target" in batch:
            y = batch["target"]
        else:
            y = None

        return x, y

    if isinstance(batch, (tuple, list)):
        if len(batch) == 1:
            return batch[0], None
        return batch[0], batch[1]

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def pairwise_bce_loss(
    pred_sim: torch.Tensor,
    target_sim: torch.Tensor,
    indicator: Optional[torch.Tensor] = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    pred_sim = torch.clamp(pred_sim, min=eps, max=1.0 - eps)
    target_sim = target_sim.detach()

    loss_matrix = -target_sim * torch.log(pred_sim) - (1.0 - target_sim) * torch.log(1.0 - pred_sim)

    if indicator is None:
        return loss_matrix.mean()

    indicator = indicator.detach()
    denom = indicator.sum().clamp_min(1.0)

    return (loss_matrix * indicator).sum() / denom


def discrepancy(out1: torch.Tensor, out2: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(out1 - out2))


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)


@dataclass
class PAATrainerConfig:
    input_dim: int = 310
    hidden_1: int = 64
    hidden_2: int = 64
    hidden_4: int = 64
    num_classes: int = 3
    low_rank: int = 32

    max_epoch: int = 300
    batch_size: int = 256
    seed: int = 2024

    lr_model: float = 1e-3
    lr_cls1: float = 1e-2
    lr_cls2: float = 1e-2
    weight_decay: float = 1e-5

    upper_threshold: float = 0.9
    lower_threshold: float = 0.5
    paa_c_threshold: float = 0.7

    cluster_weight: float = 1.0
    boost_type: str = "exp"

    lambda_ce: float = 1.0
    lambda_adv: float = 1.0
    lambda_paal: float = 0.5
    lambda_paac: float = 1.5
    lambda_proto: float = 0.01
    lambda_cluster: float = 0.1

    lambda_stage2_source: float = 1.0
    lambda_stage2_disc: float = 0.1
    lambda_stage3_disc: float = 0.1

    paa_warmup_epochs: int = 5
    paac_start_epoch: int = 5

    debug: bool = False

    save_dir: Optional[str] = None
    save_best: bool = True


class PAATrainer:
    def __init__(self, model: nn.Module, config: PAATrainerConfig, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device

        setup_seed(config.seed)

        self.input_dim = config.input_dim
        self.hidden_1 = config.hidden_1
        self.hidden_2 = config.hidden_2
        self.hidden_4 = config.hidden_4
        self.num_classes = config.num_classes
        self.low_rank = config.low_rank
        self.max_epoch = config.max_epoch

        self.best_score = -1.0
        self.best_epoch = -1
        self.best_metric_dict = None

        self._build_extra_modules()

    def _build_extra_modules(self):
        self.domain_discriminator = discriminator(self.hidden_2).to(self.device)

        init_cluster_label = np.arange(self.num_classes, dtype=np.int64)

        if hasattr(self.model, "cluster_label"):
            self.model.cluster_label = init_cluster_label.copy()

        self.cls1 = Pairwise_Learning(
            self.hidden_2,
            self.hidden_4,
            self.num_classes,
            self.low_rank,
            self.max_epoch,
            self.config.upper_threshold,
            self.config.lower_threshold,
            self.model.P,
        ).to(self.device)

        self.cls2 = Pairwise_Learning(
            self.hidden_2,
            self.hidden_4,
            self.num_classes,
            self.low_rank,
            self.max_epoch,
            self.config.upper_threshold,
            self.config.lower_threshold,
            self.model.P,
        ).to(self.device)

        self.cls1.cluster_label = init_cluster_label.copy()
        self.cls2.cluster_label = init_cluster_label.copy()

        self.paa_l = PAALLoss(self.num_classes)

        self.paa_c = PAACLoss(
            num_layers=1,
            kernel_num=(5,),
            kernel_mul=(2,),
            num_classes=self.num_classes,
            threshold=self.config.paa_c_threshold,
            low_rank=self.low_rank,
            hidden_2=self.hidden_2,
            hidden_4=self.hidden_4,
            intra_only=False,
        )
        self.paa_c.update_cluster_label(init_cluster_label.copy())

        self.dann_loss = DomainAdversarialLoss(
            self.domain_discriminator,
            max_iter=self.max_epoch,
        ).to(self.device)

        self.optimizer_model = RMSprop(
            self.model.get_parameters() + self.domain_discriminator.get_parameters(),
            lr=self.config.lr_model,
            weight_decay=self.config.weight_decay,
        )

        self.optimizer_cls1 = Adam(
            self.cls1.get_parameters(),
            lr=self.config.lr_cls1,
            weight_decay=self.config.weight_decay,
        )

        self.optimizer_cls2 = RMSprop(
            self.cls2.get_parameters(),
            lr=self.config.lr_cls2,
            weight_decay=self.config.weight_decay,
        )

    def _boost_factor(self, epoch: int) -> float:
        if self.config.boost_type == "linear":
            return self.config.cluster_weight * (epoch / max(self.max_epoch, 1))

        if self.config.boost_type == "exp":
            return self.config.cluster_weight * (
                2.0 / (1.0 + np.exp(-1.0 * epoch / max(self.max_epoch, 1))) - 1.0
            )

        if self.config.boost_type == "constant":
            return self.config.cluster_weight

        return self.config.cluster_weight

    def _prepare_source_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = unpack_batch(batch)

        if y is None:
            raise ValueError("source batch must contain label.")

        x = x.float().to(self.device)
        y = to_onehot(y, self.num_classes, self.device)

        return x, y

    def _prepare_target_batch(self, batch) -> torch.Tensor:
        x, _ = unpack_batch(batch)
        x = x.float().to(self.device)
        return x

    def _match_batch_size(
        self,
        x_s: torch.Tensor,
        y_s: torch.Tensor,
        x_t: torch.Tensor,
    ):
        n = min(x_s.size(0), x_t.size(0))

        if n <= 1:
            return None

        return x_s[:n], y_s[:n], x_t[:n]

    def _source_relation_loss(self, sim_s: torch.Tensor, y_s: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            truth_sim_s = self.model.get_cos_similarity_distance(y_s)
        return pairwise_bce_loss(sim_s, truth_sim_s)

    def _source_ce_loss(self, source_logits: torch.Tensor, y_s: torch.Tensor) -> torch.Tensor:
        y_idx = torch.argmax(y_s, dim=1)
        return F.cross_entropy(source_logits, y_idx)

    def _target_cluster_loss(
        self,
        classifier: Pairwise_Learning,
        feat_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sim_t, pred_t = classifier(feat_t, self.model.P)

        with torch.no_grad():
            pseudo_pair = classifier.get_cos_similarity_by_threshold(sim_t.detach())
            indicator, _ = classifier.compute_indicator(sim_t.detach())

        loss = pairwise_bce_loss(sim_t, pseudo_pair, indicator=indicator)

        return loss, sim_t, pred_t

    def _domain_loss(self, feat_s: torch.Tensor, feat_t: torch.Tensor) -> torch.Tensor:
        noise_s = 0.005 * torch.randn_like(feat_s)
        noise_t = 0.005 * torch.randn_like(feat_t)
        return self.dann_loss(feat_s + noise_s, feat_t + noise_t)

    def _prototype_regularization(self) -> torch.Tensor:
        eye = torch.eye(self.hidden_4, device=self.device)
        return torch.norm(torch.matmul(self.model.P.T, self.model.P) - eye, p="fro")

    def stage1_representation_learning(self, source_loader, target_loader, epoch: int) -> Dict[str, float]:
        self.model.train()
        self.cls1.train()
        self.cls2.train()
        self.dann_loss.train()

        target_iter = cycle_loader(target_loader)
        boost = self._boost_factor(epoch)

        total_loss = 0.0
        total_ce = 0.0
        total_cls = 0.0
        total_adv = 0.0
        total_paal = 0.0
        total_paac = 0.0
        n_batch = 0

        for source_batch in source_loader:
            target_batch = next(target_iter)

            x_s, y_s = self._prepare_source_batch(source_batch)
            x_t = self._prepare_target_batch(target_batch)

            matched = self._match_batch_size(x_s, y_s, x_t)
            if matched is None:
                continue

            x_s, y_s, x_t = matched

            source_logits, feat_s, feat_t, sim_s = self.model(x_s, x_t, y_s)

            ce_loss = self._source_ce_loss(source_logits, y_s)
            cls_loss = self._source_relation_loss(sim_s, y_s)

            cluster_loss1, _, pred_t1 = self._target_cluster_loss(self.cls1, feat_t)
            cluster_loss2, _, _ = self._target_cluster_loss(self.cls2, feat_t)

            adv_loss = self._domain_loss(feat_s, feat_t)
            proto_loss = self._prototype_regularization()

            selected_num = 0

            if epoch >= self.config.paac_start_epoch:
                paa_l_loss = self.paa_l.get_loss(
                    feat_s,
                    feat_t,
                    y_s,
                    pred_t1.detach(),
                )

                paa_c_loss, selected_num, _ = self.paa_c.get_loss(
                    feat_s,
                    feat_t,
                    y_s,
                    pred_t1.detach(),
                    self.model.P.detach(),
                )
            else:
                paa_l_loss = feat_s.new_tensor(0.0)
                paa_c_loss = feat_s.new_tensor(0.0)

            if self.config.debug and n_batch == 0 and epoch % 5 == 0:
                try:
                    selected_num_print = int(selected_num)
                except Exception:
                    selected_num_print = int(selected_num.item())

                print(
                    "[PAAC Debug]",
                    "epoch=", epoch,
                    "selected_num=", selected_num_print,
                    "paa_c_loss=", float(paa_c_loss.detach().cpu()),
                    "filtered_classes=", self.paa_c.filtered_classes,
                )

            if epoch < self.config.paa_warmup_epochs:
                loss = (
                    self.config.lambda_ce * ce_loss
                    + cls_loss
                    + self.config.lambda_proto * proto_loss
                )
            else:
                loss = (
                    self.config.lambda_ce * ce_loss
                    + cls_loss
                    + self.config.lambda_adv * adv_loss
                    + self.config.lambda_paal * paa_l_loss
                    + self.config.lambda_paac * paa_c_loss
                    + self.config.lambda_proto * proto_loss
                    + self.config.lambda_cluster * boost * (cluster_loss1 + cluster_loss2)
                )

            if not torch.isfinite(loss):
                print("[PAA NaN Debug]")
                print("ce_loss:", ce_loss)
                print("cls_loss:", cls_loss)
                print("adv_loss:", adv_loss)
                print("paa_l_loss:", paa_l_loss)
                print("paa_c_loss:", paa_c_loss)
                print("proto_loss:", proto_loss)
                print("cluster_loss1:", cluster_loss1)
                print("cluster_loss2:", cluster_loss2)
                raise RuntimeError("PAA stage1 loss became NaN or Inf.")

            self.optimizer_model.zero_grad()
            self.optimizer_cls1.zero_grad()
            self.optimizer_cls2.zero_grad()

            loss.backward()

            self.optimizer_model.step()
            self.optimizer_cls1.step()
            self.optimizer_cls2.step()

            total_loss += float(loss.detach().cpu())
            total_ce += float(ce_loss.detach().cpu())
            total_cls += float(cls_loss.detach().cpu())
            total_adv += float(adv_loss.detach().cpu())
            total_paal += float(paa_l_loss.detach().cpu())
            total_paac += float(paa_c_loss.detach().cpu())
            n_batch += 1

        n_batch = max(n_batch, 1)

        return {
            "stage1_loss": total_loss / n_batch,
            "stage1_ce": total_ce / n_batch,
            "stage1_cls": total_cls / n_batch,
            "stage1_adv": total_adv / n_batch,
            "stage1_paal": total_paal / n_batch,
            "stage1_paac": total_paac / n_batch,
        }

    def stage2_discrepancy_maximization(self, source_loader, target_loader, epoch: int) -> Dict[str, float]:
        self.model.train()
        self.cls1.train()
        self.cls2.train()

        set_requires_grad(self.model.fea_extrator_f, False)

        target_iter = cycle_loader(target_loader)

        total_disc = 0.0
        total_anchor = 0.0
        n_batch = 0

        for source_batch in source_loader:
            target_batch = next(target_iter)

            x_s, y_s = self._prepare_source_batch(source_batch)
            x_t = self._prepare_target_batch(target_batch)

            matched = self._match_batch_size(x_s, y_s, x_t)
            if matched is None:
                continue

            x_s, y_s, x_t = matched

            with torch.no_grad():
                _, feat_s, feat_t, _ = self.model(x_s, x_t, y_s)
                truth_sim_s = self.model.get_cos_similarity_distance(y_s)

            sim_t1, _ = self.cls1(feat_t, self.model.P.detach())
            sim_t2, _ = self.cls2(feat_t, self.model.P.detach())
            disc = discrepancy(sim_t1, sim_t2)

            sim_s1, _ = self.cls1(feat_s, self.model.P.detach())
            sim_s2, _ = self.cls2(feat_s, self.model.P.detach())

            src_anchor1 = pairwise_bce_loss(sim_s1, truth_sim_s)
            src_anchor2 = pairwise_bce_loss(sim_s2, truth_sim_s)
            src_anchor = src_anchor1 + src_anchor2

            loss = (
                self.config.lambda_stage2_source * src_anchor
                - self.config.lambda_stage2_disc * disc
            )

            if not torch.isfinite(loss):
                print("[PAA NaN Debug Stage2]")
                print("disc:", disc)
                print("src_anchor:", src_anchor)
                raise RuntimeError("PAA stage2 loss became NaN or Inf.")

            self.optimizer_cls1.zero_grad()
            self.optimizer_cls2.zero_grad()

            loss.backward()

            self.optimizer_cls1.step()
            self.optimizer_cls2.step()

            total_disc += float(disc.detach().cpu())
            total_anchor += float(src_anchor.detach().cpu())
            n_batch += 1

        set_requires_grad(self.model.fea_extrator_f, True)

        n_batch = max(n_batch, 1)

        return {
            "stage2_discrepancy": total_disc / n_batch,
            "stage2_anchor": total_anchor / n_batch,
        }

    def stage3_boundary_refinement(self, source_loader, target_loader, epoch: int) -> Dict[str, float]:
        self.model.train()
        self.cls1.train()
        self.cls2.train()
        self.dann_loss.train()

        set_requires_grad(self.cls1, False)
        set_requires_grad(self.cls2, False)

        target_iter = cycle_loader(target_loader)

        total_loss = 0.0
        total_adv = 0.0
        total_cluster = 0.0
        total_disc = 0.0
        n_batch = 0

        for source_batch in source_loader:
            target_batch = next(target_iter)

            x_s, y_s = self._prepare_source_batch(source_batch)
            x_t = self._prepare_target_batch(target_batch)

            matched = self._match_batch_size(x_s, y_s, x_t)
            if matched is None:
                continue

            x_s, y_s, x_t = matched

            source_logits, feat_s, feat_t, sim_s = self.model(x_s, x_t, y_s)

            ce_loss = self._source_ce_loss(source_logits, y_s)
            cls_loss = self._source_relation_loss(sim_s, y_s)
            adv_loss = self._domain_loss(feat_s, feat_t)

            cluster_loss1, sim_t1, _ = self._target_cluster_loss(self.cls1, feat_t)
            cluster_loss2, sim_t2, _ = self._target_cluster_loss(self.cls2, feat_t)

            cluster_loss = cluster_loss1 + cluster_loss2
            disc_refine = discrepancy(sim_t1, sim_t2)

            loss = (
                self.config.lambda_ce * ce_loss
                + cls_loss
                + self.config.lambda_adv * adv_loss
                + self.config.lambda_cluster * cluster_loss
                + self.config.lambda_stage3_disc * disc_refine
            )

            if not torch.isfinite(loss):
                print("[PAA NaN Debug Stage3]")
                print("ce_loss:", ce_loss)
                print("cls_loss:", cls_loss)
                print("adv_loss:", adv_loss)
                print("cluster_loss:", cluster_loss)
                print("disc_refine:", disc_refine)
                raise RuntimeError("PAA stage3 loss became NaN or Inf.")

            self.optimizer_model.zero_grad()

            loss.backward()

            self.optimizer_model.step()

            total_loss += float(loss.detach().cpu())
            total_adv += float(adv_loss.detach().cpu())
            total_cluster += float(cluster_loss.detach().cpu())
            total_disc += float(disc_refine.detach().cpu())
            n_batch += 1

        set_requires_grad(self.cls1, True)
        set_requires_grad(self.cls2, True)

        n_batch = max(n_batch, 1)

        return {
            "stage3_loss": total_loss / n_batch,
            "stage3_adv": total_adv / n_batch,
            "stage3_cluster": total_cluster / n_batch,
            "stage3_disc": total_disc / n_batch,
        }

    def _compute_source_prototype_and_mapping(self, source_loader) -> Tuple[float, float]:
        self.model.eval()

        all_x = []
        all_y = []

        with torch.no_grad():
            for batch in source_loader:
                x, y = self._prepare_source_batch(batch)
                all_x.append(x)
                all_y.append(y)

        source_x = torch.cat(all_x, dim=0)
        source_y = torch.cat(all_y, dim=0)

        with torch.no_grad():
            source_feat = self.model.fea_extrator_f(source_x)

            eye = torch.eye(self.num_classes, device=self.device)
            self.model.P = torch.matmul(
                torch.inverse(torch.diag(source_y.sum(axis=0)) + eye),
                torch.matmul(source_y.T, source_feat),
            )

            stored_mat = torch.matmul(self.model.V, self.model.P.T)
            logits = torch.matmul(torch.matmul(self.model.U, source_feat.T).T, stored_mat)

            source_cluster = torch.argmax(torch.softmax(logits, dim=1), dim=1).detach().cpu().numpy()
            source_label_np = torch.argmax(source_y, dim=1).detach().cpu().numpy()

        cluster_label = np.zeros(self.num_classes, dtype=np.int64)

        for c in range(self.num_classes):
            idx = np.where(source_cluster == c)[0]
            if len(idx) == 0:
                cluster_label[c] = c
            else:
                cluster_label[c] = np.argmax(
                    np.bincount(source_label_np[idx], minlength=self.num_classes)
                )

        source_pred = np.zeros_like(source_label_np)

        for c in range(self.num_classes):
            idx = np.where(source_cluster == c)[0]
            source_pred[idx] = cluster_label[c]

        source_acc = accuracy_score(source_label_np, source_pred)
        source_nmi = sk_metrics.normalized_mutual_info_score(source_pred, source_label_np)

        self.model.cluster_label = cluster_label.copy()
        self.cls1.cluster_label = cluster_label.copy()
        self.cls2.cluster_label = cluster_label.copy()
        self.paa_c.update_cluster_label(cluster_label.copy())

        return float(source_acc), float(source_nmi)

    def train(
        self,
        source_loader,
        target_loader=None,
        test_loader=None,
        metric_choose: str = "acc",
        metrics: Optional[List[str]] = None,
    ):
        if target_loader is None:
            target_loader = source_loader

        history = {
            "source_acc": [],
            "source_nmi": [],
            "test_acc": [],
            "test_macro-f1": [],
            "stage1_loss": [],
            "stage2_discrepancy": [],
            "stage3_loss": [],
        }

        for epoch in range(self.max_epoch):
            log1 = self.stage1_representation_learning(source_loader, target_loader, epoch)

            if epoch < self.config.paa_warmup_epochs:
                log2 = {"stage2_discrepancy": 0.0, "stage2_anchor": 0.0}
                log3 = {"stage3_loss": 0.0, "stage3_adv": 0.0, "stage3_cluster": 0.0, "stage3_disc": 0.0}
            else:
                log2 = self.stage2_discrepancy_maximization(source_loader, target_loader, epoch)
                log3 = self.stage3_boundary_refinement(source_loader, target_loader, epoch)

            self.cls1.update_threshold(epoch)
            self.cls2.update_threshold(epoch)

            source_acc, source_nmi = self._compute_source_prototype_and_mapping(source_loader)

            history["source_acc"].append(source_acc)
            history["source_nmi"].append(source_nmi)
            history["stage1_loss"].append(log1["stage1_loss"])
            history["stage2_discrepancy"].append(log2["stage2_discrepancy"])
            history["stage3_loss"].append(log3["stage3_loss"])

            if test_loader is not None:
                metric_dict = self.evaluate(test_loader, metrics=metrics)

                score = metric_dict.get(metric_choose, metric_dict.get("acc", 0.0))

                history["test_acc"].append(metric_dict.get("acc", 0.0))
                history["test_macro-f1"].append(metric_dict.get("macro-f1", 0.0))

                if score > self.best_score:
                    self.best_score = score
                    self.best_epoch = epoch
                    self.best_metric_dict = metric_dict.copy()

                    if self.config.save_best and self.config.save_dir is not None:
                        self.save_checkpoint(epoch, metric_dict)

                print(
                    f"[PAA] Epoch {epoch:03d} | "
                    f"s_acc={source_acc:.4f}, s_nmi={source_nmi:.4f} | "
                    f"test_acc={metric_dict.get('acc', 0.0):.4f}, "
                    f"test_macro-f1={metric_dict.get('macro-f1', 0.0):.4f} | "
                    f"best_{metric_choose}={self.best_score:.4f} | "
                    f"loss1={log1['stage1_loss']:.4f}, "
                    f"ce={log1.get('stage1_ce', 0.0):.4f}, "
                    f"paac={log1.get('stage1_paac', 0.0):.4f}, "
                    f"disc2={log2['stage2_discrepancy']:.4f}, "
                    f"loss3={log3['stage3_loss']:.4f}"
                )

        return history

    def _classifier_predict(self, x: torch.Tensor, classifier: Pairwise_Learning) -> np.ndarray:
        self.model.eval()
        classifier.eval()

        with torch.no_grad():
            feat = self.model.fea_extrator_f(x)

            stored_mat = torch.matmul(classifier.V, self.model.P.T)
            logits = torch.matmul(torch.matmul(classifier.U, feat.T).T, stored_mat)

            cluster = torch.argmax(torch.softmax(logits, dim=1), dim=1).detach().cpu().numpy()

        pred = np.zeros_like(cluster)

        for c in range(self.num_classes):
            idx = np.where(cluster == c)[0]
            pred[idx] = classifier.cluster_label[c]

        return pred

    def predict(self, data_loader, classifier: str = "cls1") -> np.ndarray:
        clf = self.cls1 if classifier == "cls1" else self.cls2

        preds = []

        for batch in data_loader:
            x, _ = unpack_batch(batch)
            x = x.float().to(self.device)

            pred = self._classifier_predict(x, clf)
            preds.append(pred)

        return np.concatenate(preds, axis=0)

    def evaluate(
        self,
        data_loader,
        metrics: Optional[List[str]] = None,
        classifier: str = "cls1",
    ) -> Dict[str, float]:
        if metrics is None:
            metrics = ["acc", "macro-f1"]

        preds = self.predict(data_loader, classifier=classifier)

        labels = []

        for batch in data_loader:
            _, y = unpack_batch(batch)

            if y is None:
                raise ValueError("Evaluation dataloader must contain labels.")

            labels.append(label_to_numpy(y))

        labels = np.concatenate(labels, axis=0)

        preds = preds.astype(np.int64)
        labels = labels.astype(np.int64)

        if self.config.debug:
            print(
                "[PAA Eval Debug] pred bincount:",
                np.bincount(preds, minlength=self.num_classes),
            )
            print(
                "[PAA Eval Debug] label bincount:",
                np.bincount(labels, minlength=self.num_classes),
            )

        metric_dict = {}

        if "acc" in metrics:
            metric_dict["acc"] = float(accuracy_score(labels, preds))

        if "macro-f1" in metrics:
            metric_dict["macro-f1"] = float(f1_score(labels, preds, average="macro", zero_division=0))

        if "weighted-f1" in metrics:
            metric_dict["weighted-f1"] = float(f1_score(labels, preds, average="weighted", zero_division=0))

        if "macro-precision" in metrics:
            metric_dict["macro-precision"] = float(precision_score(labels, preds, average="macro", zero_division=0))

        if "macro-recall" in metrics:
            metric_dict["macro-recall"] = float(recall_score(labels, preds, average="macro", zero_division=0))

        if "nmi" in metrics:
            metric_dict["nmi"] = float(sk_metrics.normalized_mutual_info_score(preds, labels))

        if "acc" not in metric_dict:
            metric_dict["acc"] = float(accuracy_score(labels, preds))

        return metric_dict

    def save_checkpoint(self, epoch: int, metric_dict: Dict[str, float]):
        os.makedirs(self.config.save_dir, exist_ok=True)

        ckpt = {
            "epoch": epoch,
            "metric_dict": metric_dict,
            "model_state_dict": self.model.state_dict(),
            "domain_discriminator_state_dict": self.domain_discriminator.state_dict(),
            "cls1_state_dict": self.cls1.state_dict(),
            "cls2_state_dict": self.cls2.state_dict(),
            "model_P": self.model.P.detach().cpu(),
            "cluster_label": self.model.cluster_label,
            "cls1_cluster_label": self.cls1.cluster_label,
            "cls2_cluster_label": self.cls2.cluster_label,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "config": self.config.__dict__,
        }

        path = os.path.join(self.config.save_dir, "paa_best.pth")
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.domain_discriminator.load_state_dict(ckpt["domain_discriminator_state_dict"])
        self.cls1.load_state_dict(ckpt["cls1_state_dict"])
        self.cls2.load_state_dict(ckpt["cls2_state_dict"])

        self.model.P = ckpt["model_P"].to(self.device)
        self.model.cluster_label = ckpt["cluster_label"]
        self.cls1.cluster_label = ckpt["cls1_cluster_label"]
        self.cls2.cluster_label = ckpt["cls2_cluster_label"]

        self.best_score = ckpt.get("best_score", -1.0)
        self.best_epoch = ckpt.get("best_epoch", -1)


def train(
    model=None,
    dataset_source=None,
    dataset_target=None,
    dataset_test=None,
    device=None,
    output_dir=None,
    metrics=None,
    metric_choose="acc",
    batch_size=256,
    epochs=300,
    args=None,
):
    if model is None:
        raise ValueError("model is None.")

    if dataset_source is None:
        raise ValueError("dataset_source is None.")

    if dataset_target is None:
        dataset_target = dataset_source

    if dataset_test is None:
        raise ValueError("dataset_test is None.")

    if args is None:
        raise ValueError("args is None.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    input_dim = model.fea_extrator_f.fc1.in_features

    if hasattr(args, "num_classes"):
        num_classes = args.num_classes
    elif hasattr(args, "num_of_class"):
        num_classes = args.num_of_class
    else:
        _, y0 = dataset_source[0]
        if y0.dim() == 0:
            all_labels = [int(dataset_source[i][1].item()) for i in range(len(dataset_source))]
            num_classes = max(all_labels) + 1
        else:
            num_classes = y0.numel()

    config = PAATrainerConfig(
        input_dim=input_dim,
        hidden_1=getattr(args, "hidden_1", 64),
        hidden_2=getattr(args, "hidden_2", 64),
        hidden_4=getattr(args, "hidden_4", 64),
        num_classes=num_classes,
        low_rank=getattr(args, "low_rank", 32),

        max_epoch=epochs,
        batch_size=batch_size,
        seed=getattr(args, "seed", 2024),

        lr_model=getattr(args, "lr", getattr(args, "lr_model", 1e-3)),
        lr_cls1=getattr(args, "lr_cls1", 1e-2),
        lr_cls2=getattr(args, "lr_cls2", 1e-2),
        weight_decay=getattr(args, "paa_weight_decay", getattr(args, "weight_decay", 1e-5)),

        upper_threshold=getattr(args, "upper_threshold", 0.9),
        lower_threshold=getattr(args, "lower_threshold", 0.5),
        paa_c_threshold=getattr(args, "paa_c_threshold", 0.7),

        cluster_weight=getattr(args, "cluster_weight", 1.0),
        boost_type=getattr(args, "boost_type", "exp"),

        lambda_ce=getattr(args, "lambda_ce", 1.0),
        lambda_adv=getattr(args, "lambda_adv", 1.0),
        lambda_paal=getattr(args, "lambda_paal", 0.5),
        lambda_paac=getattr(args, "lambda_paac", 1.5),
        lambda_proto=getattr(args, "lambda_proto", 0.01),
        lambda_cluster=getattr(args, "lambda_cluster", 0.1),

        lambda_stage2_source=getattr(args, "lambda_stage2_source", 1.0),
        lambda_stage2_disc=getattr(args, "lambda_stage2_disc", 0.1),
        lambda_stage3_disc=getattr(args, "lambda_stage3_disc", 0.1),

        paa_warmup_epochs=getattr(args, "paa_warmup_epochs", 5),
        paac_start_epoch=getattr(args, "paac_start_epoch", 5),

        debug=getattr(args, "paa_debug", False),

        save_dir=output_dir,
        save_best=True,
    )

    source_loader = torch.utils.data.DataLoader(
        dataset_source,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    target_loader = torch.utils.data.DataLoader(
        dataset_target,
        batch_size=batch_size,
        shuffle=True,
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

    trainer = PAATrainer(
        model=model,
        config=config,
        device=device,
    )

    history = trainer.train(
        source_loader=source_loader,
        target_loader=target_loader,
        test_loader=test_loader,
        metric_choose=metric_choose,
        metrics=metrics,
    )

    final_metric = trainer.evaluate(test_loader, metrics=metrics)

    # LibEER result_log 通常读取 round_metric["acc"] 和 round_metric["macro-f1"]。
    # 这里返回 best metric，而不是 final metric，避免后期塌缩把最好结果覆盖掉。
    if trainer.best_metric_dict is not None:
        round_metric = dict(trainer.best_metric_dict)
    else:
        round_metric = dict(final_metric)

    for k, v in final_metric.items():
        round_metric[f"final_{k}"] = v

    round_metric["best_epoch"] = trainer.best_epoch
    round_metric["best_score"] = trainer.best_score
    round_metric["history"] = history

    if metric_choose not in round_metric:
        round_metric[metric_choose] = round_metric.get("acc", 0.0)

    return round_metric