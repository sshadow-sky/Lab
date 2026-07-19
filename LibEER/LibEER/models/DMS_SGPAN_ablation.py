from typing import Dict, Type

from models.DMS_SGPAN_aba import DMS_SGPAN


class _FixedDMS_SGPANAblation(DMS_SGPAN):
    ablation_variant = ""

    def __init__(self, net_params: Dict):
        params = dict(net_params)
        params["ablation_variant"] = self.ablation_variant
        super().__init__(params)


class DMS_SGPAN_StandardNormalization(_FixedDMS_SGPANAblation):
    ablation_variant = "standard_norm"


class DMS_SGPAN_BasicFeatureEncoder(_FixedDMS_SGPANAblation):
    ablation_variant = "basic_encoder"


class DMS_SGPAN_NoDisentanglement(_FixedDMS_SGPANAblation):
    ablation_variant = "no_disentangle"


class DMS_SGPAN_NoPrototypeAdaptation(_FixedDMS_SGPANAblation):
    ablation_variant = "no_prototype"


class DMS_SGPAN_NoSpectralBranch(_FixedDMS_SGPANAblation):
    ablation_variant = "no_spectral"


class DMS_SGPAN_NoGraphBranch(_FixedDMS_SGPANAblation):
    ablation_variant = "no_graph"


class DMS_SGPAN_NoGraphContrastiveRegularization(_FixedDMS_SGPANAblation):
    ablation_variant = "no_gcl"


class DMS_SGPAN_NoOrthogonalLoss(_FixedDMS_SGPANAblation):
    ablation_variant = "no_orth"


class DMS_SGPAN_NoSubjectLoss(_FixedDMS_SGPANAblation):
    ablation_variant = "no_subject"


class DMS_SGPAN_StandardCrossScaleAttention(_FixedDMS_SGPANAblation):
    ablation_variant = "standard_attention"


class DMS_SGPAN_AverageScaleFusion(_FixedDMS_SGPANAblation):
    ablation_variant = "average_fusion"


class DMS_SGPAN_NoWarmup(_FixedDMS_SGPANAblation):
    ablation_variant = "no_warmup"


class DMS_SGPAN_PrototypeLevelOnly(_FixedDMS_SGPANAblation):
    ablation_variant = "prototype_only"


class DMS_SGPAN_SampleLevelOnly(_FixedDMS_SGPANAblation):
    ablation_variant = "sample_only"


class DMS_SGPAN_ClassifierBasedPseudoLabels(_FixedDMS_SGPANAblation):
    ablation_variant = "classifier_pseudo"


DMS_SGPAN_ABLATION_MODELS: Dict[str, Type[DMS_SGPAN]] = {
    "standard_norm": DMS_SGPAN_StandardNormalization,
    "basic_encoder": DMS_SGPAN_BasicFeatureEncoder,
    "no_disentangle": DMS_SGPAN_NoDisentanglement,
    "no_prototype": DMS_SGPAN_NoPrototypeAdaptation,
    "no_spectral": DMS_SGPAN_NoSpectralBranch,
    "no_graph": DMS_SGPAN_NoGraphBranch,
    "no_gcl": DMS_SGPAN_NoGraphContrastiveRegularization,
    "no_orth": DMS_SGPAN_NoOrthogonalLoss,
    "no_subject": DMS_SGPAN_NoSubjectLoss,
    "standard_attention": DMS_SGPAN_StandardCrossScaleAttention,
    "average_fusion": DMS_SGPAN_AverageScaleFusion,
    "no_warmup": DMS_SGPAN_NoWarmup,
    "prototype_only": DMS_SGPAN_PrototypeLevelOnly,
    "sample_only": DMS_SGPAN_SampleLevelOnly,
    "classifier_pseudo": DMS_SGPAN_ClassifierBasedPseudoLabels,
}


__all__ = [
    "DMS_SGPAN_StandardNormalization",
    "DMS_SGPAN_BasicFeatureEncoder",
    "DMS_SGPAN_NoDisentanglement",
    "DMS_SGPAN_NoPrototypeAdaptation",
    "DMS_SGPAN_NoSpectralBranch",
    "DMS_SGPAN_NoGraphBranch",
    "DMS_SGPAN_NoGraphContrastiveRegularization",
    "DMS_SGPAN_NoOrthogonalLoss",
    "DMS_SGPAN_NoSubjectLoss",
    "DMS_SGPAN_StandardCrossScaleAttention",
    "DMS_SGPAN_AverageScaleFusion",
    "DMS_SGPAN_NoWarmup",
    "DMS_SGPAN_PrototypeLevelOnly",
    "DMS_SGPAN_SampleLevelOnly",
    "DMS_SGPAN_ClassifierBasedPseudoLabels",
    "DMS_SGPAN_ABLATION_MODELS",
]
