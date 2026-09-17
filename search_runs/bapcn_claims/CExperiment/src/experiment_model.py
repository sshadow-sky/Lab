"""BAPCN controls for the agreement factor and the alignment feature space."""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from baseline_model import DMS_SGPAN


class BAPCN(DMS_SGPAN):
    def __init__(self, net_params: Dict, prototype_space: str = "band", ablation: str = "baseline"):
        if prototype_space != "band":
            raise ValueError("C experiments retain band-wise prototypes; prototype_space must be 'band'")
        if ablation not in ("baseline", "no_agreement", "fused_alignment"):
            raise ValueError("ablation must be 'baseline', 'no_agreement', or 'fused_alignment'")
        super().__init__(net_params)
        self.prototype_space = prototype_space
        self.ablation = ablation

    def prototype_inputs(self, encoding):
        return encoding["shared_scales"], encoding["scale_weights"]

    def _reliability_from_factors(self, state):
        confidence = state["feature_agreement"]
        margin = state["feature_margin"]
        entropy_score = state["feature_entropy_score"]
        if self.ablation == "no_agreement":
            return (confidence.pow(1.0 / 3.0) * margin.pow(1.0 / 3.0) *
                    entropy_score.pow(1.0 / 3.0)).clamp(0.0, 1.0)
        return (confidence.pow(0.25) * margin.pow(0.25) * entropy_score.pow(0.25) *
                state["scale_consistency"].pow(0.25)).clamp(0.0, 1.0)

    def _ugfcda_reliability_and_pseudo(
        self, source_scales, target_scales, source_label, target_scale_weights=None,
    ):
        state = super()._ugfcda_reliability_and_pseudo(
            source_scales, target_scales, source_label, target_scale_weights,
        )
        state["reliability"] = self._reliability_from_factors(state)
        return state

    @torch.no_grad()
    def source_prototype_posteriors(
        self, source_features: List[torch.Tensor], source_labels: torch.Tensor,
        target_features: List[torch.Tensor], target_weights: Optional[torch.Tensor] = None,
    ):
        """Export source-derived posteriors using the canonical training operations."""
        if not source_features or len(source_features) != len(target_features):
            raise ValueError("Source and target require the same nonzero number of feature sets")
        source_labels = self._label_index(source_labels)
        prototypes, valid_masks, posteriors = [], [], []
        for source, target in zip(source_features, target_features):
            proto, valid = self._ugfcda_build_batch_prototypes(source, source_labels)
            logits = F.normalize(target.detach(), dim=-1) @ proto.T / max(self.temperature, 1e-6)
            logits = logits.masked_fill(~valid.unsqueeze(0), -1e9)
            prototypes.append(proto)
            valid_masks.append(valid)
            posteriors.append(torch.softmax(logits, dim=1))
        band_posteriors = torch.stack(posteriors, dim=1)
        if target_weights is None or target_weights.numel() == 0:
            weights = torch.full(
                band_posteriors.shape[:2], 1.0 / len(target_features),
                device=band_posteriors.device,
            )
        else:
            weights = target_weights.detach().float().to(band_posteriors.device)
            weights = weights / (weights.sum(dim=1, keepdim=True) + self.ugfcda_eps)
        valid_classes = torch.stack(valid_masks).all(dim=0)
        consensus = (band_posteriors * weights.unsqueeze(-1)).sum(dim=1)
        consensus = consensus.masked_fill(~valid_classes.unsqueeze(0), 0.0)
        pseudo_labels = consensus.argmax(dim=1)
        confidence = torch.gather(consensus, 1, pseudo_labels.view(-1, 1)).squeeze(1)
        top2 = torch.topk(consensus, k=min(2, self.num_classes), dim=1).values
        margin = ((top2[:, 0] - top2[:, 1]) if top2.shape[1] > 1 else top2[:, 0]).clamp(0.0, 1.0)
        distribution = consensus / (consensus.sum(dim=1, keepdim=True) + self.ugfcda_eps)
        entropy = -(distribution.clamp_min(self.ugfcda_eps) *
                    distribution.clamp_min(self.ugfcda_eps).log()).sum(dim=1)
        entropy_score = (1.0 - entropy / float(np.log(max(2, self.num_classes)))).clamp(0.0, 1.0)
        agreement = (band_posteriors.argmax(dim=2) == pseudo_labels.unsqueeze(1)).float().mean(dim=1)
        result = {
            "prototypes": torch.stack(prototypes),
            "valid_classes": valid_classes,
            "band_posteriors": band_posteriors,
            "consensus_posterior": consensus,
            "weights": weights,
            "pseudo_labels": pseudo_labels,
            "feature_agreement": confidence,
            "feature_margin": margin,
            "feature_entropy_score": entropy_score,
            "scale_consistency": agreement,
        }
        result["reliability"] = self._reliability_from_factors(result)
        return result

    # Preserve the canonical training order and intervene only at selection/alignment.
    def forward(
        self,
        source_x: torch.Tensor,
        target_x: torch.Tensor,
        source_subject_ids: torch.Tensor,
        target_subject_ids: torch.Tensor,
        source_y: torch.Tensor,
        source_session_ids: Optional[torch.Tensor] = None,
        target_session_ids: Optional[torch.Tensor] = None,
        current_epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:
        if source_session_ids is None:
            source_session_ids = torch.zeros_like(source_subject_ids, dtype=torch.long)
        if target_session_ids is None:
            target_session_ids = torch.zeros_like(target_subject_ids, dtype=torch.long)

        cat_x = torch.cat([source_x, target_x], dim=0)
        cat_sid = torch.cat([source_subject_ids.long(), target_subject_ids.long()], dim=0)
        cat_sess = torch.cat([source_session_ids.long(), target_session_ids.long()], dim=0)

        source_label = self._label_index(source_y)

        source_count = source_x.shape[0]
        target_count = target_x.shape[0]
        source_mask = torch.cat([
            torch.ones(source_count, dtype=torch.bool, device=cat_x.device),
            torch.zeros(target_count, dtype=torch.bool, device=cat_x.device),
        ], dim=0)

        source_label_all = torch.full((source_count + target_count,), -1, dtype=torch.long, device=cat_x.device)
        source_label_all[:source_count] = source_label.to(cat_x.device)

        mixed_idx = torch.randperm(cat_x.shape[0], device=cat_x.device)
        cat_x = cat_x[mixed_idx]
        cat_sid = cat_sid[mixed_idx]
        cat_sess = cat_sess[mixed_idx]
        source_mask = source_mask[mixed_idx]
        target_mask = ~source_mask
        source_label = source_label_all[mixed_idx][source_mask]

        enc = self._encode_all(cat_x, cat_sid, cat_sess)

        logits_s = enc["logits"][source_mask]
        logits_t = enc["logits"][target_mask]

        ce_loss = F.cross_entropy(logits_s, source_label)

        gcl_loss = self._source_supervised_graph_contrastive_loss(
            enc["graph_feat"][source_mask],
            enc["adj"][source_mask],
            enc["graph_step_feat"][source_mask],
            source_label,
            cat_sid[source_mask],
            cat_sess[source_mask],
        )

        target_prob = torch.softmax(logits_t.detach(), dim=1)
        target_confidence = target_prob.max(dim=1).values

        prototype_features, prototype_weights = self.prototype_inputs(enc)
        source_scales = [feat[source_mask] for feat in prototype_features]
        target_scales = [feat[target_mask] for feat in prototype_features]
        target_scale_weights = prototype_weights[target_mask]

        ugfcda_state = self._ugfcda_reliability_and_pseudo(
            source_scales, target_scales, source_label, target_scale_weights
        )
        target_pseudo = ugfcda_state["pseudo_labels"]
        target_reliability = ugfcda_state["reliability"]
        target_align_mask = (
            target_reliability >= self.ugfcda_reliability_threshold
            if int(current_epoch) >= self.ugfcda_warmup_epochs
            else torch.zeros_like(target_reliability, dtype=torch.bool)
        )
        align_active = bool(target_align_mask.any().item())
        if align_active:
            if self.ablation == "fused_alignment":
                align_source = [enc["final_feat"][source_mask]]
                align_target = [enc["final_feat"][target_mask]]
            else:
                align_source, align_target = source_scales, target_scales
            align_loss = self._ugfcda_alignment_loss(
                align_source, align_target, source_label,
                target_pseudo, target_reliability, target_align_mask
            )
        else:
            align_loss = torch.zeros((), device=logits_s.device)

        subject_loss, subject_shared_loss, subject_private_loss, shared_subject_acc, private_subject_acc = self._subject_loss(
            enc["shared_scales"], enc["private_scales"], cat_sid
        )
        orth_loss = self._cross_covariance_loss(enc["shared_scales"], enc["private_scales"])

        target_total = torch.tensor(float(max(1, target_count)), device=logits_s.device)
        target_align_count = target_align_mask.float().sum()
        target_align_coverage = target_align_count / target_total
        target_pseudo_class_counts = torch.bincount(target_pseudo.detach(), minlength=self.num_classes).float()
        target_align_class_counts = torch.bincount(
            target_pseudo[target_align_mask].detach(), minlength=self.num_classes
        ).float()
        if target_align_mask.any():
            target_align_confidence = target_reliability[target_align_mask].mean()
        else:
            target_align_confidence = torch.zeros((), device=logits_s.device)

        target_feature_agreement = ugfcda_state["feature_agreement"].mean() if target_count > 0 else torch.zeros((), device=logits_s.device)
        target_feature_margin = ugfcda_state["feature_margin"].mean() if target_count > 0 else torch.zeros((), device=logits_s.device)
        target_feature_entropy_score = ugfcda_state["feature_entropy_score"].mean() if target_count > 0 else torch.zeros((), device=logits_s.device)
        target_scale_consistency = ugfcda_state["scale_consistency"].mean() if target_count > 0 else torch.zeros((), device=logits_s.device)

        total_loss = (
            self.w_ce * ce_loss
            + self.w_aj * enc["ajloss"]
            + self.w_gcl * gcl_loss
            + self.w_align * align_loss
            + self.w_orth * orth_loss
            + self.w_subject * subject_loss
        )

        return {
            "total_loss": total_loss,
            "ce_loss": ce_loss,
            "ajloss": enc["ajloss"],
            "gcl_loss": gcl_loss,
            "align_loss": align_loss,
            "orth_loss": orth_loss,
            "subject_loss": subject_loss,
            "subject_shared_loss": subject_shared_loss,
            "subject_private_loss": subject_private_loss,
            "shared_subject_acc": shared_subject_acc,
            "private_subject_acc": private_subject_acc,
            "target_pseudo_conf_mean": target_confidence.mean(),
            "target_reliability_mean": target_reliability.mean(),
            "target_feature_agreement_mean": target_feature_agreement,
            "target_feature_margin_mean": target_feature_margin,
            "target_feature_entropy_score_mean": target_feature_entropy_score,
            "target_scale_consistency_mean": target_scale_consistency,
            "target_align_conf_mean": target_align_confidence,
            "target_align_coverage": target_align_coverage,
            "target_align_count": target_align_count,
            "target_pseudo_class_counts": target_pseudo_class_counts,
            "target_align_class_counts": target_align_class_counts,
            "align_active": torch.tensor(float(align_active), device=logits_s.device),
            "source_logits": logits_s,
            "target_logits": logits_t,
            "source_labels": source_label,
        }
