import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from config.setting import preset_setting, set_setting_by_args
from data_utils.load_data import get_data
from data_utils.split import get_split_index, index_to_data, merge_to_part
from models.Models import Model
from Trainer.FATTraining import train
from utils.args import get_args_parser
from utils.store import make_output_dir
from utils.utils import result_log, setup_seed, sub_result_log


param_path = "config/model_param/FAT.yaml"


def _load_train_param():
    try:
        with open(param_path, "r", encoding="utf-8") as fd:
            cfg = yaml.load(fd, Loader=yaml.FullLoader)
        return cfg.get("train", {})
    except IOError:
        print("\n{} may not exist or not available".format(param_path))
        return {}


def main(args):
    train_cfg = _load_train_param()

    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)

    args.seed = 2024
    setup_seed(args.seed)
    data, label, channels, feature_dim, num_classes = get_data(setting)
    data, label = merge_to_part(data, label, setting)
    device = torch.device(args.device)

    best_metrics = []
    dependent_metrics = [[] for _ in range(len(data))]

    for rridx, (data_i, label_i) in enumerate(zip(data, label), 1):
        tts = get_split_index(data_i, label_i, setting)

        for ridx, (train_indexes, test_indexes, val_indexes) in enumerate(zip(tts["train"], tts["test"], tts["val"]), 1):
            setup_seed(args.seed)
            if val_indexes[0] == -1:
                print(f"train indexes:{train_indexes}, test indexes:{test_indexes}")
            else:
                print(f"train indexes:{train_indexes}, val indexes:{val_indexes}, test indexes:{test_indexes}")

            test_sub_label = None
            if setting.experiment_mode == "subject-independent":
                train_data, train_label, val_data, val_label, test_data, test_label = index_to_data(
                    data_i,
                    label_i,
                    train_indexes,
                    test_indexes,
                    val_indexes,
                    True,
                )
                test_sub_num = len(test_data)
                test_sub_label = []
                for i in range(test_sub_num):
                    test_sub_count = len(test_data[i])
                    test_sub_label.extend([i + 1 for _ in range(test_sub_count)])
                test_sub_label = np.array(test_sub_label)

            train_data, train_label, val_data, val_label, test_data, test_label = index_to_data(
                data_i,
                label_i,
                train_indexes,
                test_indexes,
                val_indexes,
                args.keep_dim,
            )

            if len(val_data) == 0:
                print("skip one split because val split is empty under strict no-test-leak protocol")
                continue

            model = Model["FAT"](channels, feature_dim, num_classes)

            dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_data), torch.Tensor(train_label))
            dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_data), torch.Tensor(val_label))
            dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_data), torch.Tensor(test_label))

            mode = setting.experiment_mode
            if mode == "subject-independent":
                default_epochs = int(train_cfg.get("epochs_independent", 50))
                default_lr = float(train_cfg.get("lr_independent", 0.001))
                default_weight_decay = float(train_cfg.get("weight_decay_independent", 1e-4))
                default_optimizer = str(train_cfg.get("optimizer_independent", "adam")).lower()
            else:
                default_epochs = int(train_cfg.get("epochs_dependent", 200))
                default_lr = float(train_cfg.get("lr_dependent", 0.0003))
                default_weight_decay = float(train_cfg.get("weight_decay_dependent", 1e-4))
                default_optimizer = str(train_cfg.get("optimizer_dependent", "adamw")).lower()

            # Preserve CLI override; fallback to original FAT training defaults.
            effective_epochs = args.epochs if args.epochs != 40 else default_epochs
            effective_lr = args.lr if args.lr != 0.001 else default_lr

            if default_optimizer == "adam":
                optimizer = optim.Adam(model.parameters(), lr=effective_lr, weight_decay=default_weight_decay)
            else:
                optimizer = optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=default_weight_decay, eps=1e-4)

            criterion = nn.CrossEntropyLoss()

            output_dir = make_output_dir(args, "FAT")

            round_metric = train(
                model=model,
                dataset_train=dataset_train,
                dataset_val=dataset_val,
                dataset_test=dataset_test,
                device=device,
                output_dir=output_dir,
                metrics=args.metrics,
                metric_choose=args.metric_choose,
                optimizer=optimizer,
                batch_size=args.batch_size,
                epochs=effective_epochs,
                criterion=criterion,
                num_workers=args.num_workers,
                experiment_mode=setting.experiment_mode,
            )

            best_metrics.append(round_metric)
            if setting.experiment_mode == "subject-dependent":
                dependent_metrics[rridx - 1].append(round_metric)

    if setting.experiment_mode == "subject-dependent":
        sub_result_log(args, dependent_metrics)
    else:
        result_log(args, best_metrics)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
