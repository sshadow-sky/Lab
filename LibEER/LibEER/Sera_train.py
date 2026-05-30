# Sera_train.py

from models.Models import Model
from config.setting import preset_setting, set_setting_by_args
from data_utils.load_data import get_data
from data_utils.split import merge_to_part, index_to_data, get_split_index
from data_utils.preprocess import normalize
from utils.args import get_args_parser
from utils.utils import result_log, setup_seed, sub_result_log
from Trainer.SeraTrainer import train as sera_train
from utils.store import make_output_dir

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import numpy as np


def ensure_4d_for_sera(x, channels=None):
    """
    Convert LibEER data to Sera input shape [N, 1, C, T].

    支持：
        [N, C, T]
        [N, T, C]
        [N, D]
        [N, 1, C, T]
    """
    x = np.asarray(x)

    if x.ndim == 2:
        x = x[:, np.newaxis, np.newaxis, :]

    elif x.ndim == 3:
        if channels is not None:
            if x.shape[1] == channels:
                # already [N, C, T]
                pass
            elif x.shape[2] == channels:
                # [N, T, C] -> [N, C, T]
                x = np.transpose(x, (0, 2, 1))
            else:
                # fallback: assume [N, T, C]
                x = np.transpose(x, (0, 2, 1))
        else:
            x = np.transpose(x, (0, 2, 1))

        x = x[:, np.newaxis, :, :]

    elif x.ndim == 4:
        pass

    else:
        raise ValueError(f"Unsupported data shape for Sera: {x.shape}")

    return x.astype(np.float32)


def auto_patch_size(time_len, pool, preferred_patch_size):
    """
    Sera requires patch_size <= time_len / pool.
    Original code also assumes the reduced time dimension can work with patching.

    This function prevents invalid patch_size for datasets like SEED sample_length=200.
    """
    reduced_t = int(time_len / pool)

    if preferred_patch_size <= reduced_t:
        return preferred_patch_size

    candidates = [32, 24, 20, 16, 12, 10, 8, 5, 4, 2, 1]

    for p in candidates:
        if p <= reduced_t:
            return p

    return 1


def main(args):
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)

    # DEAP / HCI usually require -bounds 5 5
    if getattr(setting, "dataset", None) in ["deap", "hci"] and getattr(setting, "bounds", None) is None:
        raise ValueError(
            f"{setting.dataset} requires -bounds, for example: -bounds 5 5."
        )

    args.seed = 2024
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

            train_data, val_data, test_data = normalize(
                train_data,
                val_data,
                test_data,
                dim="sample",
            )

            train_data = ensure_4d_for_sera(train_data, channels=channels)
            val_data = ensure_4d_for_sera(val_data, channels=channels)
            test_data = ensure_4d_for_sera(test_data, channels=channels)

            _, in_chans, eeg_channels, time_points = train_data.shape
            input_size = (in_chans, eeg_channels, time_points)

            patch_size = args.sera_patch_size
            if args.sera_auto_patch:
                patch_size = auto_patch_size(
                    time_len=time_points,
                    pool=args.sera_pool,
                    preferred_patch_size=args.sera_patch_size,
                )

            patch_stride = args.sera_patch_stride
            if patch_stride <= 0:
                patch_stride = max(1, patch_size // 2)

            print(f"Sera input_size: {input_size}")
            print(f"Sera num_classes: {num_classes}")
            print(f"Sera patch_size={patch_size}, patch_stride={patch_stride}")

            model = Model["Sera"](
                num_classes=num_classes,
                input_size=input_size,
                sampling_rate=args.sera_sampling_rate,
                num_T=args.sera_num_T,
                patch_size=patch_size,
                patch_stride=patch_stride,
                dropout_rate=args.sera_dropout,
                pool=args.sera_pool,
                dimz=args.sera_dimz,
                m=args.sera_m,
                transformer_depth=args.sera_transformer_depth,
                num_head=args.sera_num_head,
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

            if args.sera_target_strategy == "train":
                dataset_target = dataset_train
            elif args.sera_target_strategy == "val":
                dataset_target = dataset_val
            elif args.sera_target_strategy == "test":
                print(
                    "[Warning] You are using test set as unlabeled target data. "
                    "This is transductive and may be unfair for standard benchmark."
                )
                dataset_target = dataset_test
            else:
                raise ValueError(f"Unknown sera_target_strategy: {args.sera_target_strategy}")

            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.sera_weight_decay,
            )

            scheduler = StepLR(
                optimizer,
                gamma=args.sera_scheduler_gamma,
                step_size=args.sera_scheduler_step,
            )

            criterion = nn.CrossEntropyLoss()

            output_dir = make_output_dir(args, "Sera")

            round_metric = sera_train(
                model=model,
                dataset_train=dataset_train,
                dataset_val=dataset_val,
                dataset_test=dataset_test,
                dataset_target=dataset_target,
                device=device,
                output_dir=output_dir,
                metrics=args.metrics,
                metric_choose=args.metric_choose,
                optimizer=optimizer,
                scheduler=scheduler,
                batch_size=args.batch_size,
                epochs=args.epochs,
                criterion=criterion,
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

    parser.add_argument("-sera_sampling_rate", "--sera_sampling_rate", type=int, default=128)
    parser.add_argument("-sera_num_T", "--sera_num_T", type=int, default=32)
    parser.add_argument("-sera_patch_size", "--sera_patch_size", type=int, default=16)
    parser.add_argument("-sera_patch_stride", "--sera_patch_stride", type=int, default=8)
    parser.add_argument("-sera_dropout", "--sera_dropout", type=float, default=0.25)
    parser.add_argument("-sera_pool", "--sera_pool", type=int, default=2)
    parser.add_argument("-sera_dimz", "--sera_dimz", type=int, default=32)
    parser.add_argument("-sera_m", "--sera_m", type=int, default=3)
    parser.add_argument("-sera_transformer_depth", "--sera_transformer_depth", type=int, default=2)
    parser.add_argument("-sera_num_head", "--sera_num_head", type=int, default=16)

    parser.add_argument("-sera_lambda_domain", "--sera_lambda_domain", type=float, default=0.001)
    parser.add_argument("-sera_lambda_rec", "--sera_lambda_rec", type=float, default=0.0001)
    parser.add_argument("-sera_lambda_ort", "--sera_lambda_ort", type=float, default=0.01)
    parser.add_argument("-sera_lambda_ta", "--sera_lambda_ta", type=float, default=0.001)

    parser.add_argument("-sera_weight_decay", "--sera_weight_decay", type=float, default=1e-4)
    parser.add_argument("-sera_scheduler_step", "--sera_scheduler_step", type=int, default=100)
    parser.add_argument("-sera_scheduler_gamma", "--sera_scheduler_gamma", type=float, default=0.3)

    parser.add_argument(
        "-sera_target_strategy",
        "--sera_target_strategy",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help=(
            "Which split is used as unlabeled target data for Sera domain alignment. "
            "train is benchmark-safe. test is transductive."
        ),
    )

    parser.add_argument(
        "-sera_auto_patch",
        "--sera_auto_patch",
        action="store_true",
        help="Automatically adjust patch_size if current sample_length is too short.",
    )

    args = parser.parse_args()
    main(args)