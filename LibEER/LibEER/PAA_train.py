# PAA_train.py

from models.Models import Model
from config.setting import preset_setting, set_setting_by_args
from data_utils.load_data import get_data
from data_utils.split import merge_to_part, index_to_data, get_split_index
from data_utils.preprocess import normalize
from utils.args import get_args_parser
from utils.utils import result_log, setup_seed, sub_result_log
from Trainer.PAATrainer import train as paa_train
from utils.store import make_output_dir

import torch
import numpy as np


def flatten_for_paa(x):
    x = np.asarray(x)

    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)

    return x.astype(np.float32)


def featurewise_standardize(train_data, val_data, test_data, eps=1e-6):
    ori_train_shape = train_data.shape
    ori_val_shape = val_data.shape
    ori_test_shape = test_data.shape

    train_flat = train_data.reshape(train_data.shape[0], -1)
    val_flat = val_data.reshape(val_data.shape[0], -1)
    test_flat = test_data.reshape(test_data.shape[0], -1)

    mean = train_flat.mean(axis=0, keepdims=True)
    std = train_flat.std(axis=0, keepdims=True) + eps

    train_flat = (train_flat - mean) / std
    val_flat = (val_flat - mean) / std
    test_flat = (test_flat - mean) / std

    return (
        train_flat.reshape(ori_train_shape).astype(np.float32),
        val_flat.reshape(ori_val_shape).astype(np.float32),
        test_flat.reshape(ori_test_shape).astype(np.float32),
    )


def main(args):
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)

    if getattr(setting, "dataset", None) in ["deap", "hci"] and getattr(setting, "bounds", None) is None:
        raise ValueError(
            f"{setting.dataset} requires -bounds, for example: -bounds 5 5."
        )

    setup_seed(args.seed)

    data, label, channels, feature_dim, num_classes = get_data(setting)
    data, label = merge_to_part(data, label, setting)

    device = torch.device(args.device)

    best_metrics = []
    subjects_metrics = [[] for _ in range(len(data))]

    for rridx, (data_i, label_i) in enumerate(zip(data, label), 1):
        tts = get_split_index(data_i, label_i, setting)

        for ridx, (train_indexes, test_indexes, val_indexes) in enumerate(
            zip(tts["train"], tts["test"], tts["val"]),
            1,
        ):
            setup_seed(args.seed)

            if val_indexes[0] == -1:
                print(f"train indexes:{train_indexes}, test indexes:{test_indexes}")
            else:
                print(
                    f"train indexes:{train_indexes}, "
                    f"val indexes:{val_indexes}, "
                    f"test indexes:{test_indexes}"
                )

            train_data, train_label, val_data, val_label, test_data, test_label = index_to_data(
                data_i,
                label_i,
                train_indexes,
                test_indexes,
                val_indexes,
                args.keep_dim,
            )

            if len(val_data) == 0:
                val_data = test_data
                val_label = test_label

            print("Before PAA normalization / flatten:")
            print("train_data shape:", train_data.shape)
            print("val_data shape:", val_data.shape)
            print("test_data shape:", test_data.shape)
            print("channels:", channels)
            print("feature_dim:", feature_dim)
            print("num_classes:", num_classes)

            if args.paa_norm == "sample":
                train_data, val_data, test_data = normalize(
                    train_data,
                    val_data,
                    test_data,
                    dim="sample",
                )
            elif args.paa_norm == "feature":
                train_data, val_data, test_data = featurewise_standardize(
                    train_data,
                    val_data,
                    test_data,
                )
            elif args.paa_norm == "none":
                train_data = train_data.astype(np.float32)
                val_data = val_data.astype(np.float32)
                test_data = test_data.astype(np.float32)
            else:
                raise ValueError(f"Unknown paa_norm: {args.paa_norm}")

            train_data = flatten_for_paa(train_data)
            val_data = flatten_for_paa(val_data)
            test_data = flatten_for_paa(test_data)

            input_dim = train_data.shape[1]

            args.input_dim = input_dim
            args.num_classes = num_classes
            args.max_epoch = args.epochs

            print("After flatten:")
            print(f"PAA input_dim: {input_dim}")
            print(f"num_classes: {num_classes}")
            print(f"train_data shape: {train_data.shape}")
            print(f"val_data shape: {val_data.shape}")
            print(f"test_data shape: {test_data.shape}")
            print(f"paa_norm: {args.paa_norm}")

            model = Model["PAA"](
                input_dim=input_dim,
                hidden_1=args.hidden_1,
                hidden_2=args.hidden_2,
                hidden_4=args.hidden_4,
                num_of_class=num_classes,
                low_rank=args.low_rank,
                max_iter=args.epochs,
            )

            dataset_train = torch.utils.data.TensorDataset(
                torch.Tensor(train_data),
                torch.Tensor(train_label),
            )

            dataset_val = torch.utils.data.TensorDataset(
                torch.Tensor(val_data),
                torch.Tensor(val_label),
            )

            dataset_test = torch.utils.data.TensorDataset(
                torch.Tensor(test_data),
                torch.Tensor(test_label),
            )

            if args.paa_target_strategy == "train":
                dataset_target = dataset_train
            elif args.paa_target_strategy == "val":
                dataset_target = dataset_val
            elif args.paa_target_strategy == "test":
                print(
                    "[Warning] You are using test set as unlabeled target data. "
                    "This is transductive and may be unfair for standard benchmark."
                )
                dataset_target = dataset_test
            else:
                raise ValueError(f"Unknown paa_target_strategy: {args.paa_target_strategy}")

            output_dir = make_output_dir(args, "PAA")

            round_metric = paa_train(
                model=model,
                dataset_source=dataset_train,
                dataset_target=dataset_target,
                dataset_test=dataset_test,
                device=device,
                output_dir=output_dir,
                metrics=args.metrics,
                metric_choose=args.metric_choose,
                batch_size=args.batch_size,
                epochs=args.epochs,
                args=args,
            )

            best_metrics.append(round_metric)

            if setting.experiment_mode == "subject-dependent":
                subjects_metrics[rridx - 1].append(round_metric)

    if setting.experiment_mode == "subject-dependent":
        sub_result_log(args, subjects_metrics)
    else:
        result_log(args, best_metrics)


if __name__ == "__main__":
    parser = get_args_parser()

    parser.add_argument("-hidden_1", "--hidden_1", type=int, default=64)
    parser.add_argument("-hidden_2", "--hidden_2", type=int, default=64)
    parser.add_argument("-hidden_4", "--hidden_4", type=int, default=64)
    parser.add_argument("-low_rank", "--low_rank", type=int, default=32)

    parser.add_argument("-upper_threshold", "--upper_threshold", type=float, default=0.9)
    parser.add_argument("-lower_threshold", "--lower_threshold", type=float, default=0.5)
    parser.add_argument("-paa_c_threshold", "--paa_c_threshold", type=float, default=0.7)

    parser.add_argument("-cluster_weight", "--cluster_weight", type=float, default=1.0)
    parser.add_argument(
        "-boost_type",
        "--boost_type",
        type=str,
        default="exp",
        choices=["linear", "exp", "constant"],
    )

    parser.add_argument("-lambda_ce", "--lambda_ce", type=float, default=1.0)
    parser.add_argument("-lambda_adv", "--lambda_adv", type=float, default=1.0)
    parser.add_argument("-lambda_paal", "--lambda_paal", type=float, default=0.5)
    parser.add_argument("-lambda_paac", "--lambda_paac", type=float, default=1.5)
    parser.add_argument("-lambda_proto", "--lambda_proto", type=float, default=0.01)
    parser.add_argument("-lambda_cluster", "--lambda_cluster", type=float, default=0.1)

    parser.add_argument("-lambda_stage2_source", "--lambda_stage2_source", type=float, default=1.0)
    parser.add_argument("-lambda_stage2_disc", "--lambda_stage2_disc", type=float, default=0.1)
    parser.add_argument("-lambda_stage3_disc", "--lambda_stage3_disc", type=float, default=0.1)

    parser.add_argument("-paa_warmup_epochs", "--paa_warmup_epochs", type=int, default=5)
    parser.add_argument("-paac_start_epoch", "--paac_start_epoch", type=int, default=5)

    parser.add_argument("-lr_cls1", "--lr_cls1", type=float, default=1e-2)
    parser.add_argument("-lr_cls2", "--lr_cls2", type=float, default=1e-2)
    parser.add_argument("-paa_weight_decay", "--paa_weight_decay", type=float, default=1e-5)

    parser.add_argument(
        "-paa_target_strategy",
        "--paa_target_strategy",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help=(
            "Which split is used as unlabeled target data for PAA. "
            "train is benchmark-safe. "
            "test is transductive and not recommended for fair comparison."
        ),
    )

    parser.add_argument(
        "-paa_norm",
        "--paa_norm",
        type=str,
        default="feature",
        choices=["none", "sample", "feature"],
        help=(
            "Normalization for PAA DE features. "
            "feature: feature-wise standardization using train statistics. "
            "sample: LibEER sample-wise normalization. "
            "none: no normalization."
        ),
    )

    parser.add_argument(
        "-paa_debug",
        "--paa_debug",
        action="store_true",
        help="Print PAAC selected_num and prediction distribution.",
    )

    args = parser.parse_args()
    main(args)