import yaml
import torch
import sys

from config.setting import preset_setting, set_setting_by_args
from data_utils.load_data import get_data
from data_utils.split import get_split_index, index_to_data, merge_to_part
from Trainer.DMMRTraining import build_dmmr_loaders
from Trainer.DMMRTraining import train_dmmr
from utils.args import get_args_parser
from utils.store import make_output_dir
from utils.utils import result_log, setup_seed


param_path = "config/model_param/DMMR.yaml"


def _load_cfg():
    try:
        with open(param_path, "r", encoding="utf-8") as fd:
            return yaml.load(fd, Loader=yaml.FullLoader) or {}
    except IOError:
        print(f"{param_path} may not exist or not available")
        return {}


def _infer_variant(dataset_name):
    name = (dataset_name or "").lower()
    if name == "seediv" or name.startswith("seediv_"):
        return "seediv"
    if name == "seed" or name.startswith("seed_"):
        return "seed"
    raise ValueError(f"DMMR in LibEER currently supports only SEED / SEED-IV datasets, got '{dataset_name}'.")


def _cli_option_provided(*option_names):
    argv = sys.argv[1:]
    for option in option_names:
        if option in argv:
            return True
    return False


def _apply_dmmr_defaults(args, setting, cfg):
    variant = _infer_variant(setting.dataset)
    train_table = cfg.get("train", {})
    train_cfg = train_table.get(variant, {})
    if len(train_cfg) == 0:
        raise ValueError(f"DMMR default train configuration for variant '{variant}' is missing in config/model_param/DMMR.yaml")
    param_cfg = cfg.get("params", {})
    setting_table = cfg.get("setting", {})
    setting_cfg = setting_table.get(variant, setting_table.get("default", {}))

    args.model = "DMMR"

    args.dmmr_input_dim = int(
        args.dmmr_input_dim
        if args.dmmr_input_dim is not None
        else train_cfg.get("input_dim", param_cfg.get("input_dim", 310))
    )
    args.dmmr_hid_dim = int(args.dmmr_hid_dim if args.dmmr_hid_dim is not None else param_cfg.get("hid_dim", 64))
    args.dmmr_n_layers = int(args.dmmr_n_layers if args.dmmr_n_layers is not None else param_cfg.get("n_layers", 1))
    args.dmmr_beta = float(args.dmmr_beta if args.dmmr_beta is not None else param_cfg.get("beta", 0.05))

    args.dmmr_time_steps = int(args.dmmr_time_steps if args.dmmr_time_steps is not None else train_cfg.get("time_steps", 30))
    args.dmmr_num_classes = int(args.dmmr_num_classes if args.dmmr_num_classes is not None else train_cfg.get("num_classes", 3))
    args.dmmr_iteration = int(args.dmmr_iteration if args.dmmr_iteration is not None else train_cfg.get("iteration", 7))
    args.dmmr_epoch_pretraining = int(
        args.dmmr_epoch_pretraining if args.dmmr_epoch_pretraining is not None else train_cfg.get("epoch_pretraining", 300)
    )
    args.dmmr_epoch_finetuning = int(
        args.dmmr_epoch_finetuning if args.dmmr_epoch_finetuning is not None else train_cfg.get("epoch_finetuning", 500)
    )

    if not _cli_option_provided("-batch_size", "--batch_size"):
        args.batch_size = int(train_cfg.get("batch_size", 512))
    if not _cli_option_provided("-lr", "--lr"):
        args.lr = float(train_cfg.get("lr", 0.001))
    if args.dmmr_weight_decay is None:
        args.dmmr_weight_decay = float(train_cfg.get("weight_decay", 0.0005))

    # Apply setting defaults only when the user did not explicitly provide these options.
    if not _cli_option_provided("-experiment_mode", "--experiment_mode"):
        setting.experiment_mode = setting_cfg.get("experiment_mode", "subject-independent")
    if not _cli_option_provided("-split_type", "--split_type"):
        setting.split_type = setting_cfg.get("split_type", "leave-one-out")
    if not _cli_option_provided("-sessions", "--sessions"):
        if setting_cfg.get("sessions") is not None:
            setting.sessions = setting_cfg.get("sessions")
    if not _cli_option_provided("-onehot", "--onehot"):
        if setting_cfg.get("onehot") is not None:
            setting.onehot = setting_cfg.get("onehot")
    if not _cli_option_provided("-feature_type", "--feature_type"):
        setting.feature_type = setting_cfg.get("feature_type", "de_lds")

    # Keep sample_length and dmmr_time_steps consistent.
    sample_length_explicit = _cli_option_provided("-sample_length", "--sample_length")
    time_steps_explicit = _cli_option_provided("-dmmr_time_steps", "--dmmr_time_steps")
    if sample_length_explicit and time_steps_explicit and int(args.sample_length) != int(args.dmmr_time_steps):
        raise ValueError(
            f"sample_length ({args.sample_length}) and dmmr_time_steps ({args.dmmr_time_steps}) are both explicitly set but inconsistent."
        )
    if sample_length_explicit:
        setting.sample_length = int(args.sample_length)
        if not time_steps_explicit:
            args.dmmr_time_steps = int(args.sample_length)
    else:
        setting.sample_length = int(args.dmmr_time_steps)


def main(args):
    cfg = _load_cfg()

    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)

    _apply_dmmr_defaults(args, setting, cfg)
    setup_seed(args.seed)

    data, label, channels, feature_dim, num_classes = get_data(setting)

    inferred_input_dim = None
    if channels is not None and feature_dim is not None:
        inferred_input_dim = int(channels) * int(feature_dim)
    if inferred_input_dim is not None and inferred_input_dim > 0:
        if _cli_option_provided("-dmmr_input_dim", "--dmmr_input_dim"):
            if int(args.dmmr_input_dim) != inferred_input_dim:
                raise ValueError(
                    f"dmmr_input_dim ({args.dmmr_input_dim}) mismatches inferred input_dim ({inferred_input_dim}) from current dataset/features."
                )
        else:
            if args.dmmr_input_dim != inferred_input_dim:
                print(
                    "[DMMR] using inferred input_dim from data shape: "
                    f"{inferred_input_dim} (was {args.dmmr_input_dim})."
                )
            args.dmmr_input_dim = inferred_input_dim

    if num_classes is not None:
        inferred_num_classes = int(num_classes)
        if _cli_option_provided("-dmmr_num_classes", "--dmmr_num_classes"):
            if int(args.dmmr_num_classes) != inferred_num_classes:
                raise ValueError(
                    f"dmmr_num_classes ({args.dmmr_num_classes}) mismatches inferred classes ({inferred_num_classes}) from current labels."
                )
        else:
            if args.dmmr_num_classes != inferred_num_classes:
                print(
                    "[DMMR] using inferred num_classes from labels: "
                    f"{inferred_num_classes} (was {args.dmmr_num_classes})."
                )
            args.dmmr_num_classes = inferred_num_classes
    data, label = merge_to_part(data, label, setting)
    device = torch.device(args.device)

    best_metrics = []
    for data_i, label_i in zip(data, label):
        tts = get_split_index(data_i, label_i, setting)
        for ridx, (train_indexes, test_indexes, val_indexes) in enumerate(
            zip(tts["train"], tts["test"], tts["val"]), 1
        ):
            setup_seed(args.seed)
            if val_indexes[0] == -1:
                print(f"train indexes:{train_indexes}, test indexes:{test_indexes}")
            else:
                print(f"train indexes:{train_indexes}, val indexes:{val_indexes}, test indexes:{test_indexes}")

            train_data, train_label, _, _, test_data, test_label = index_to_data(
                data_i,
                label_i,
                train_indexes,
                test_indexes,
                val_indexes,
                keep_dim=True,
            )

            if len(test_data) != 1:
                raise ValueError("DMMR LOSO expects one target subject in each split round.")

            source_loaders, target_loader = build_dmmr_loaders(
                source_data_list=train_data,
                source_label_list=train_label,
                target_data=test_data[0],
                target_label=test_label[0],
                batch_size=args.batch_size,
                time_steps=args.dmmr_time_steps,
                input_dim=args.dmmr_input_dim,
                num_workers=args.num_workers,
            )

            output_dir = make_output_dir(args, "DMMR") / str(ridx)
            optimizer_config = {"lr": args.lr, "weight_decay": args.dmmr_weight_decay}
            round_metric = train_dmmr(
                source_loaders=source_loaders,
                test_loader=target_loader,
                args=args,
                device=device,
                output_dir=output_dir,
                optimizer_config=optimizer_config,
                subject_id=test_indexes[0],
            )
            best_metrics.append(round_metric)

    result_log(args, best_metrics)


if __name__ == "__main__":
    parser = get_args_parser()
    parser.add_argument("-dmmr_time_steps", default=None, type=int)
    parser.add_argument("-dmmr_num_classes", default=None, type=int)
    parser.add_argument("-dmmr_iteration", default=None, type=int)
    parser.add_argument("-dmmr_epoch_pretraining", default=None, type=int)
    parser.add_argument("-dmmr_epoch_finetuning", default=None, type=int)
    parser.add_argument("-dmmr_input_dim", default=None, type=int)
    parser.add_argument("-dmmr_hid_dim", default=None, type=int)
    parser.add_argument("-dmmr_n_layers", default=None, type=int)
    parser.add_argument("-dmmr_beta", default=None, type=float)
    parser.add_argument("-dmmr_weight_decay", default=None, type=float)
    args = parser.parse_args()
    main(args)