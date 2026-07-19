def _train_ablation(args, variant):
    from DMS_SGPAN_aba_train import run_experiment
    from models.DMS_SGPAN_ablation import DMS_SGPAN_ABLATION_MODELS

    model_cls = DMS_SGPAN_ABLATION_MODELS[variant]
    run_name = f"DMS_SGPAN_{variant}"
    return run_experiment(
        args,
        model_cls=model_cls,
        run_name=run_name,
        ablation_variant=variant,
    )


def train_standard_norm(args):
    return _train_ablation(args, "standard_norm")


def train_basic_encoder(args):
    return _train_ablation(args, "basic_encoder")


def train_no_disentangle(args):
    return _train_ablation(args, "no_disentangle")


def train_no_prototype(args):
    return _train_ablation(args, "no_prototype")


def train_no_spectral(args):
    return _train_ablation(args, "no_spectral")


def train_no_graph(args):
    return _train_ablation(args, "no_graph")


def train_no_gcl(args):
    return _train_ablation(args, "no_gcl")


def train_no_orth(args):
    return _train_ablation(args, "no_orth")


def train_no_subject(args):
    return _train_ablation(args, "no_subject")


def train_standard_attention(args):
    return _train_ablation(args, "standard_attention")


def train_average_fusion(args):
    return _train_ablation(args, "average_fusion")


def train_no_warmup(args):
    return _train_ablation(args, "no_warmup")


def train_prototype_only(args):
    return _train_ablation(args, "prototype_only")


def train_sample_only(args):
    return _train_ablation(args, "sample_only")


def train_classifier_pseudo(args):
    return _train_ablation(args, "classifier_pseudo")


DMS_SGPAN_ABLATION_TRAINERS = {
    "standard_norm": train_standard_norm,
    "basic_encoder": train_basic_encoder,
    "no_disentangle": train_no_disentangle,
    "no_prototype": train_no_prototype,
    "no_spectral": train_no_spectral,
    "no_graph": train_no_graph,
    "no_gcl": train_no_gcl,
    "no_orth": train_no_orth,
    "no_subject": train_no_subject,
    "standard_attention": train_standard_attention,
    "average_fusion": train_average_fusion,
    "no_warmup": train_no_warmup,
    "prototype_only": train_prototype_only,
    "sample_only": train_sample_only,
    "classifier_pseudo": train_classifier_pseudo,
}


def main(args):
    variant = getattr(args, "dms_sgpan_ablation", None)
    if variant not in DMS_SGPAN_ABLATION_TRAINERS:
        valid = ", ".join(DMS_SGPAN_ABLATION_TRAINERS)
        raise ValueError(f"Select one DMS_SGPAN ablation with -dms_sgpan_ablation. Valid values: {valid}")
    return DMS_SGPAN_ABLATION_TRAINERS[variant](args)


if __name__ == "__main__":
    from utils.args import get_args_parser

    parser = get_args_parser()
    parser.add_argument(
        "-dms_sgpan_ablation",
        required=True,
        choices=list(DMS_SGPAN_ABLATION_TRAINERS),
        help="DMS_SGPAN ablation variant",
    )
    main(parser.parse_args())
