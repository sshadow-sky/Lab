import numpy as np
from data_utils.preprocess import normalize
from models.Models import Model
from config.setting import seed_sub_dependent_front_back_setting, preset_setting, set_setting_by_args
from data_utils.load_data import get_data
from data_utils.split import merge_to_part, index_to_data, get_split_index
from utils.args import get_args_parser
from utils.store import make_output_dir
from utils.utils import state_log, result_log, setup_seed, sub_result_log
from Trainer.FBSTCTraining import train
import torch
import torch.optim as optim
import torch.nn as nn


def main(args):
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)
    setup_seed(args.seed)
    setting.onehot = False
    data, label, channels, feature_dim, num_classes = get_data(setting)
    data, label = merge_to_part(data, label, setting)
    device = torch.device(args.device)
    best_metrics = []
    torch.set_default_tensor_type('torch.FloatTensor')
    subjects_metrics = [[]for _ in range(len(data))]
    for rridx, (data_i, label_i) in enumerate(zip(data, label), 1):
        tts = get_split_index(data_i, label_i, setting)
        for ridx, (train_indexes, test_indexes, val_indexes) in enumerate(zip(tts['train'], tts['test'], tts['val']), 1):
            setup_seed(args.seed)
            if val_indexes[0] == -1:
                print(f"train indexes:{train_indexes}, test indexes:{test_indexes}")
            else:
                print(f"train indexes:{train_indexes}, val indexes:{val_indexes}, test indexes:{test_indexes}")

            test_sub_label = None
            # record who each sample belong to
            if setting.experiment_mode == "subject-independent":
                # extract the subject label
                train_data, train_label, val_data, val_label, test_data, test_label = \
                    index_to_data(data_i, label_i, train_indexes, test_indexes, val_indexes, True)
                test_sub_num = len(test_data)
                test_sub_label = []
                for i in range(test_sub_num):
                    test_sub_count = len(test_data[i])
                    test_sub_label.extend([i + 1 for j in range(test_sub_count)])
                test_sub_label = np.array(test_sub_label)

            # split train and test data by specified experiment mode
            train_data, train_label, val_data, val_label, test_data, test_label = \
                index_to_data(data_i, label_i, train_indexes, test_indexes, val_indexes, args.keep_dim)
            # print(len(train_data))
            if len(val_data) == 0:
                val_data = test_data
                val_label = test_label
            train_data, val_data, test_data = normalize(train_data, val_data, test_data, dim='sample', method="z-score")
            # model to train
            filterRange = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 36), (36, 40),
                           (40, 44), (44, 48), (48, 52)]
            freq = 200
            if setting.dataset.startswith("seed") or setting.dataset.startswith("mped"):
                freq = 200
            elif setting.dataset.startswith("hci") or setting.dataset.startswith("deap"):
                freq = 128

            model = Model['FBSTCNet'](channels , num_classes, fs=freq, filterRange=filterRange,
                                      input_window_samples=feature_dim, same_filters_for_features = False).to(device)
            # Train one round using the train one round function defined in the model
            dataset_train = torch.utils.data.TensorDataset(torch.tensor(train_data).to(device), torch.tensor(train_label).to(device))
            dataset_val = torch.utils.data.TensorDataset(torch.tensor(val_data).to(device), torch.tensor(val_label).to(device))
            dataset_test = torch.utils.data.TensorDataset(torch.tensor(test_data).to(device), torch.tensor(test_label).to(device))
            lr = args.lr
            weight_decay = 1e-4
            output_dir = make_output_dir(args, "FBSTCNet")
            round_metric = train(model=model, dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test, device=device
                                 ,output_dir=output_dir, metrics=args.metrics, metric_choose=args.metric_choose, lr = lr, weight_decay=weight_decay,
                                 batch_size=args.batch_size, epochs=args.epochs, n_classes=num_classes, test_sub_label=test_sub_label)
            best_metrics.append(round_metric)
            if setting.experiment_mode == "subject-dependent":
                subjects_metrics[rridx-1].append(round_metric)
    # best metrics: every round metrics dict
    # subjects metrics: (subject, sub_round_metric)
    if setting.experiment_mode == "subject-dependent":
        sub_result_log(args, subjects_metrics)
    else:
        result_log(args, best_metrics)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    # log out train state
    main(args)
