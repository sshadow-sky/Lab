

from data_utils.preprocess import normalize
from models.Models import Model
from config.setting import seed_sub_dependent_front_back_setting, preset_setting, set_setting_by_args
from data_utils.load_data import get_data
from data_utils.split import merge_to_part, index_to_data, get_split_index
from utils.args import get_args_parser
from utils.store import make_output_dir
from utils.utils import state_log, result_log, setup_seed, sub_result_log
from Trainer.PRPLtraining import train_and_test_GAN
import torch
from models.PRRL import PRRL,discriminator, DomainAdversarialLoss
import numpy as np
from torch.optim import RMSprop


def feature_wrap(feature_3d):
    """将 shape 为 [channel, trial, freq] 的 3D 特征转换为 [trial, 310] 的 2D 特征"""
    n_trials = feature_3d.shape[1]
    feature_2d = np.zeros((n_trials, 310))
    for i in range(n_trials):
        feature_2d[i, :] = feature_3d[:, i, :].reshape(1, 310)
    return feature_2d

def main(args):
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)
    setup_seed(args.seed)
    data, label, channels, feature_dim, num_classes = get_data(setting)#
    data, label = merge_to_part(data, label, setting)
    device = torch.device(args.device)
    best_metrics = []
    subjects_metrics = [[] for _ in range(len(data))]
    for rridx, (data_i, label_i) in enumerate(zip(data, label), 1):
        tts = get_split_index(data_i, label_i, setting)
        for ridx, (train_indexes, test_indexes, val_indexes) in enumerate(zip(tts['train'], tts['test'], tts['val']),
                                                                          1):
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
                test_sub_label = torch.from_numpy(np.array(test_sub_label))

            # split train and test data by specified experiment mode
            train_data, train_label, val_data, val_label, test_data, test_label = \
                index_to_data(data_i, label_i, train_indexes, test_indexes, val_indexes, args.keep_dim)
            # print(len(train_data))
            # model to train
            if len(val_data) == 0:
                val_data = test_data
                val_label = test_label
            train_data, val_data, test_data = normalize(train_data, val_data, test_data, dim='sample', method="minmax")
            train_data = train_data.reshape(train_data.shape[0], -1)
            val_data = val_data.reshape(val_data.shape[0], -1)
            test_data = test_data.reshape(test_data.shape[0], -1)
            model = Model['PRRL'](channels, feature_dim, num_classes)
            target_set={'feature':test_data,'label':test_label}
            validation_set={'feature':val_data, 'label':val_label}
            source_set={'feature':train_data,'label':train_label}
            output_dir = make_output_dir(args, "PR_RL")

            round_metric = train_and_test_GAN(model=model, target_set=target_set, validation_set = validation_set,
                                 source_set=source_set, test_sub_label=test_sub_label, device=device,
                                 output_dir=output_dir, metrics=args.metrics, metric_choose=args.metric_choose,
                                 batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,threshold_update=True)
            best_metrics.append(round_metric)
            if setting.experiment_mode == "subject-dependent":
                subjects_metrics[rridx - 1].append(round_metric)
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