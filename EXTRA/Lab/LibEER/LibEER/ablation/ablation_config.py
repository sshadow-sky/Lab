"""BAPCN ablation configuration - single source of truth.

Before this file, the ablation setup lived in four places that could drift apart:

  1. ``models/DMS_SGPAN_aba.py``      -> ``DMS_SGPAN_ABLATION_SPECS`` (what each
                                         variant actually switches)
  2. ``models/DMS_SGPAN_ablation.py`` -> 16 thin subclasses + a registry
  3. ``DMS_SGPAN_ablation_train.py``  -> the ``-dms_sgpan_ablation`` CLI
  4. ``search_runs/run_dms_sgpan_ablation_seed{,_iv,_v}.sh``
                                      -> one hard-coded COMMON_ARGS block per
                                         dataset, with the paper's Table II row
                                         names nowhere in the code

This module joins all four. The paper's Table II row labels are the primary
keys; every row records which specification dimension it flips, which code
variant implements it, and whether that variant exists today.

Two Table II rows have **no** implementation in the code at present (see
``STATUS_NEEDS_MODEL`` below); they are listed honestly rather than silently
dropped, and :func:`runnable_variants` excludes them unless
``include_unimplemented=True``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# 1. The paper's ablation table (Table II), group by group
# --------------------------------------------------------------------------

STATUS_READY = "ready"
STATUS_NEEDS_MODEL = "needs-model-change"
STATUS_EXTRA = "code-only"          # exists in code, absent from Table II


@dataclass(frozen=True)
class Variant:
    key: str                 # what goes on the command line
    label: str               # Table II row label, verbatim
    group: str               # Table II group heading
    code_id: str             # key in DMS_SGPAN_ABLATION_SPECS (or "" if unused)
    flips: str               # the specification dimension it changes
    status: str = STATUS_READY
    note: str = ""


GROUPS: Tuple[str, ...] = (
    "Overall",
    "Spectral-Graph Encoding",
    "Shared-Private Representation Learning",
    "Prototype Consensus Adaptation",
    "Removal of the Band Index",
)

VARIANTS: Tuple[Variant, ...] = (
    # ---- Overall ---------------------------------------------------------
    Variant("basic_encoder", "w/ MLP Encoder", GROUPS[0], "basic_encoder",
            "encoder: dual_branch -> basic_mlp",
            note="single representation, single prototype set"),
    Variant("no_disentangle", "w/o Shared-Private Representation Learning",
            GROUPS[0], "no_disentangle",
            "disentanglement: True -> False",
            note="stage removal: the shared/private encoders and the attention "
                 "path are replaced by direct encoders, and both the subject "
                 "predictions and the decorrelation loss switch off with the "
                 "flag, because orthogonal_loss_enabled and subject_loss_enabled "
                 "are gated on disentanglement_enabled. This is the stage-level "
                 "row; 'no_orth' and 'no_subject' isolate the two losses."),
    Variant("no_prototype", "w/o Prototype Consensus Adaptation", GROUPS[0],
            "no_prototype", "prototype_adaptation: True -> False",
            note="equals lambda_align = 0"),

    # ---- Spectral-Graph Encoding ----------------------------------------
    Variant("graph_only", "w/ Graph-Only Encoder", GROUPS[1], "no_spectral",
            "encoder: dual_branch -> graph_only",
            note="code id is 'no_spectral': the row keeps the graph branch. "
                 "Name inversion - do not map by string similarity."),
    Variant("spectral_only", "w/ Spectral-Only Encoder", GROUPS[1], "no_graph",
            "encoder: dual_branch -> spectral_only",
            note="code id is 'no_graph': the row keeps band features and drops "
                 "graph context plus its auxiliary objectives"),
    Variant("no_gcl", "w/o Graph Contrastive Loss", GROUPS[1], "no_gcl",
            "graph_contrastive: True -> False", note="alpha_gcl = 0"),

    # ---- Shared-Private Representation Learning -------------------------
    Variant("no_orth", "w/o Shared-Private Decorrelation Loss", GROUPS[2],
            "no_orth", "orthogonal_loss: True -> False", note="alpha_orth = 0"),
    Variant("no_subject", "w/o Subject Prediction Losses", GROUPS[2],
            "no_subject", "subject_loss: True -> False", note="alpha_sub = 0"),
    Variant("standard_attention", "w/ Standard Cross-Band Self-Attention",
            GROUPS[2], "standard_attention",
            "cross_scale_attention: private_suppression -> standard",
            note="self-attention over shared tokens, no private suppression"),

    # ---- Prototype Consensus Adaptation ---------------------------------
    Variant("no_warmup", "w/o Adaptation Warm-Up", GROUPS[3], "no_warmup",
            "use_warmup: True -> False", note="e_w = 0"),
    Variant("prototype_only", "w/o Sample-Level Alignment", GROUPS[3],
            "prototype_only", "alignment_mode: both -> prototype_only",
            note="code id says what is kept, the row says what is removed"),
    Variant("sample_only", "w/o Prototype-Level Alignment", GROUPS[3],
            "sample_only", "alignment_mode: both -> sample_only",
            note="code id says what is kept, the row says what is removed"),

    # ---- Removal of the Band Index --------------------------------------
    Variant("fused_proto", "at the prototype stage", GROUPS[4], "fused_proto",
            "prototype_space: band -> fused",
            note="class evidence formed on the pooled representation"),
    Variant("pooled_align", "at the alignment stage", GROUPS[4], "pooled_align",
            "alignment_space: band -> pooled",
            note="only the two alignment losses lose the band index: they run on "
                 "the pooled representation z, while the class evidence and the "
                 "cross-band agreement gate stay per-band, so the gate still "
                 "measures agreement across bands."),
    Variant("no_agreement", "w/ Agreement-Free Reliability", GROUPS[4],
            "no_agreement", "use_scale_agreement: True -> False",
            note="drops the agreement factor of Eq. (reliability), i.e. code's "
                 "scale_consistency, the fraction of bands whose argmax equals "
                 "the pseudo-label. The geometric mean keeps an equal weight on "
                 "the remaining three factors (1/3 instead of 1/4)."),

    # ---- present in the code and in past runs, absent from Table II -----
    Variant("standard_norm", "(not in Table II) w/ Standard Normalization",
            "code-only", "standard_norm", "normalization: stratified -> standard",
            status=STATUS_EXTRA, note="source-training global Z-score"),
    Variant("average_fusion", "(not in Table II) w/ Average Band Fusion",
            "code-only", "average_fusion", "scale_fusion: adaptive -> average",
            status=STATUS_EXTRA, note="equal band weights"),
    Variant("classifier_pseudo", "(not in Table II) w/ Classifier Pseudo-Labels",
            "code-only", "classifier_pseudo",
            "pseudo_label_mode: prototype -> classifier",
            status=STATUS_EXTRA,
            note="pseudo-label taken from the classifier instead of the "
                 "prototype consensus. Adjacent to, but not the same as, "
                 "'w/ Agreement-Free Reliability'."),
)

BY_KEY: Dict[str, Variant] = {v.key: v for v in VARIANTS}

#: order used by every runner: Table II group order, baseline first
DEFAULT_VARIANT_ORDER: Tuple[str, ...] = tuple(
    ["baseline"] + [v.key for v in VARIANTS if v.status != STATUS_EXTRA]
)


def runnable_variants(include_unimplemented: bool = False,
                      include_extra: bool = False) -> List[str]:
    out = ["baseline"]
    for v in VARIANTS:
        if v.status == STATUS_NEEDS_MODEL and not include_unimplemented:
            continue
        if v.status == STATUS_EXTRA and not include_extra:
            continue
        out.append(v.key)
    return out


def missing_variants() -> List[Variant]:
    return [v for v in VARIANTS if v.status == STATUS_NEEDS_MODEL]


# --------------------------------------------------------------------------
# 2. Per-dataset protocols and hyper-parameters
# --------------------------------------------------------------------------
#   seed / seediv / seedv: transcribed verbatim from the three existing
#   launchers, so nothing changes for runs already in flight.
#   deap_a / deap_v: NEW. The paper states the selected (lambda_align, delta)
#   pairs only for the three SEED variants; the DEAP values below follow the
#   paper's own trend and are marked unconfirmed until validated.

@dataclass(frozen=True)
class Protocol:
    key: str
    dataset: str                 # -dataset
    setting: str                 # -setting preset
    dataset_path_env: str        # env var holding the dataset root
    default_dataset_path: str
    batch_size: int
    epochs: int
    loss_align: float            # lambda_align
    loss_subject: float
    loss_gcl: float
    loss_orth: float
    dropout: float
    reliability_threshold: float  # delta
    temperature: float
    warmup_epochs: int = 15
    eps: float = 1e-6
    bounds: Tuple[float, ...] = ()   # -bounds: needed to binarize DEAP ratings
    extra_args: Tuple[str, ...] = ()
    search_name: str = ""
    confirmed: bool = True       # False -> runner prints a loud warning
    note: str = ""


PROTOCOLS: Dict[str, Protocol] = {
    "seed": Protocol(
        key="seed", dataset="seed_de_lds",
        setting="seed_sub_independent_train_val_test_setting",
        dataset_path_env="SEED_DATASET_PATH",
        default_dataset_path="/mnt/sdc/sdc1/yangli/yangli/EEG/EEG_Dataset/SEED/SEED_EEG",
        batch_size=5120, epochs=100, loss_align=0.2, loss_subject=0.25, loss_gcl=0.25,
        loss_orth=0.5, dropout=0.4, reliability_threshold=0.95, temperature=0.2,
        search_name="tvt_seed_de_lds_ablation",
    ),
    "seediv": Protocol(
        key="seediv", dataset="seediv_de_lds",
        setting="seediv_sub_independent_train_val_test_setting",
        dataset_path_env="SEEDIV_DATASET_PATH",
        default_dataset_path="/mnt/sdc/sdc1/yangli/yangli/EEG/EEG_Dataset/SEED_IV",
        batch_size=1536, epochs=100, loss_align=0.6, loss_subject=0.4, loss_gcl=0.6,
        loss_orth=0.4, dropout=0.3, reliability_threshold=0.7, temperature=0.1,
        search_name="tvt_seediv_de_lds_ablation",
    ),
    "seedv": Protocol(
        key="seedv", dataset="seedv_de",
        setting="seedv_sub_independent_train_val_test_setting",
        dataset_path_env="SEEDV_DATASET_PATH",
        default_dataset_path="/mnt/sdc/sdc1/yangli/yangli/EEG/EEG_Dataset/SEED-V/SEED-V",
        batch_size=1024, epochs=100, loss_align=0.3, loss_subject=0.6, loss_gcl=0.3,
        loss_orth=0.3, dropout=0.3, reliability_threshold=0.6, temperature=0.1,
        search_name="tvt_seedv_de_ablation",
    ),
    # ---- DEAP -----------------------------------------------------------
    # DEAP-A and DEAP-V are two-class tasks (binarized arousal / valence). Every
    # value below is the one the recorded DEAP-A baseline search run used
    # (search_20260727_162630_765369, job 51/72), so the ablation variants run
    # under the same budget as the baseline they are compared against. Two
    # differences from the SEED family are worth noting: 40 epochs instead of
    # 100, and temperature left at the YAML default 0.2 because that run did not
    # override it.
    "deap_a": Protocol(
        key="deap_a", dataset="deap",
        setting="deap_sub_independent_train_val_test_setting",
        dataset_path_env="DEAP_DATASET_PATH",
        default_dataset_path="/mnt/sdc/sdc1/yangli/yangli/EEG/EEG_Dataset/DEAP/data_preprocessed_python",
        batch_size=2048, epochs=40, loss_align=0.1, loss_subject=0.2,
        loss_gcl=0.2, loss_orth=0.2, dropout=0.3, reliability_threshold=0.9,
        temperature=0.2, warmup_epochs=10,
        bounds=(5.0, 5.0),
        extra_args=("-label_used", "arousal"),
        search_name="tvt_deap_arousal_ablation",
        note="recorded DEAP-A baseline run, job 51/72 of "
             "search_20260727_162630_765369. The same run also passed "
             "-dms_sgpan_ugfcda_keep_ratio_step_epochs 20, which no model or "
             "training script reads, so it is not emitted here.",
    ),
    "deap_v": Protocol(
        key="deap_v", dataset="deap",
        setting="deap_sub_independent_train_val_test_setting",
        dataset_path_env="DEAP_DATASET_PATH",
        default_dataset_path="/mnt/sdc/sdc1/yangli/yangli/EEG/EEG_Dataset/DEAP/data_preprocessed_python",
        batch_size=2048, epochs=40, loss_align=0.1, loss_subject=0.2,
        loss_gcl=0.2, loss_orth=0.2, dropout=0.3, reliability_threshold=0.9,
        temperature=0.2, warmup_epochs=10,
        bounds=(5.0, 5.0),
        extra_args=("-label_used", "valence"),
        search_name="tvt_deap_valence_ablation",
        note="same budget as DEAP-A, changed only in the label axis. The recorded "
             "run supplied for DEAP-A; the valence baseline is assumed to have "
             "shared its setting, which is worth confirming against its own run "
             "record.",
    ),
}

#: arguments that are identical for every dataset and variant
SHARED_ARGS: Tuple[str, ...] = (
    "-model", "DMS_SGPAN",
    "-lr", "0.01",
    "-seed", "2024",
    "-metrics", "acc", "macro-f1",
    "-metric_choose", "macro-f1",
    "-device", "cuda",
    "-feature_type", "de_lds",
    "-time_window", "1",
    "-sample_length", "1",
    "-stride", "1",
    "-onehot",
    "-dms_sgpan_frequency_band_groups", "[[0],[1],[2],[3],[4]]",
)


# --------------------------------------------------------------------------
# 3. Command construction
# --------------------------------------------------------------------------

def build_args(protocol_key: str, variant: str, output_dir: str, log_dir: str,
               num_workers: int = 16, dataset_path: Optional[str] = None,
               model_name: str = "DMS_SGPAN") -> List[str]:
    """Full argument vector for one (dataset, variant) run."""
    p = PROTOCOLS[protocol_key]
    if variant != "baseline" and variant not in BY_KEY:
        raise KeyError("unknown variant %r; known: %s"
                       % (variant, ", ".join(sorted(BY_KEY))))
    path = dataset_path or os.environ.get(p.dataset_path_env,
                                          p.default_dataset_path)
    args: List[str] = []
    args += [a if a != "DMS_SGPAN" else model_name for a in SHARED_ARGS]
    args += ["-batch_size", str(p.batch_size)]
    args += ["-epochs", str(p.epochs)]
    args += ["-dataset", p.dataset]
    args += ["-dataset_path", path]
    args += ["-setting", p.setting]
    args += ["-num_workers", str(num_workers)]
    if p.bounds:
        # label_process binarizes a continuous rating with
        # value <= bounds[0] -> 0 and value >= bounds[1] -> 1, so the DEAP
        # ratings need this or the run stops on a None subscript
        args += ["-bounds"] + [("%g" % b) for b in p.bounds]
    args += list(p.extra_args)
    args += ["-dms_sgpan_loss_align", repr(p.loss_align)]
    args += ["-dms_sgpan_loss_subject", repr(p.loss_subject)]
    args += ["-dms_sgpan_loss_gcl", repr(p.loss_gcl)]
    args += ["-dms_sgpan_loss_orth", repr(p.loss_orth)]
    args += ["-dms_sgpan_dropout", repr(p.dropout)]
    args += ["-dms_sgpan_ugfcda_warmup_epochs", str(p.warmup_epochs)]
    args += ["-dms_sgpan_ugfcda_eps", "%.6f" % p.eps]
    args += ["-dms_sgpan_ugfcda_reliability_threshold", repr(p.reliability_threshold)]
    args += ["-dms_sgpan_temperature", repr(p.temperature)]
    args += ["-output_dir", output_dir, "-log_dir", log_dir]
    if variant != "baseline":
        args += ["-dms_sgpan_ablation", BY_KEY[variant].code_id or variant]
    return args


def entry_script(libeer_dir: str, variant: str,
                 prefix: str = "DMS_SGPAN") -> str:
    """``baseline`` runs the plain entry point; every variant the ablation one."""
    name = ("%s_train.py" % prefix) if variant == "baseline" \
        else ("%s_ablation_train.py" % prefix)
    return os.path.join(libeer_dir, name)


def describe_table() -> str:
    """Human-readable audit of the Table II coverage."""
    tag_of = {"ready": "OK     ",
              "needs-model-change": "MISSING",
              "code-only": "EXTRA  "}
    lines = []
    for g in GROUPS:
        lines.append("[%s]" % g)
        for v in [v for v in VARIANTS if v.group == g]:
            lines.append("  %s  %-46s -> %-20s %s"
                         % (tag_of[v.status], v.label, v.code_id or "-", v.flips))
    extra = [v for v in VARIANTS if v.status == STATUS_EXTRA]
    if extra:
        lines.append("[code-only variants, not in Table II]")
        for v in extra:
            lines.append("  EXTRA    %-46s -> %-20s %s"
                         % (v.label.replace("(not in Table II) ", ""),
                            v.code_id, v.flips))
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_table())
    print()
    print("default run order (%d runs incl. baseline): %s"
          % (len(DEFAULT_VARIANT_ORDER), ", ".join(DEFAULT_VARIANT_ORDER)))
    print("missing from the code: %s"
          % ", ".join(v.key for v in missing_variants()))
