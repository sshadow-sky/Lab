from models.Models import Model
from config.setting import seed_sub_independent_leave_one_out_setting, preset_setting, set_setting_by_args
from data_utils.load_data import get_data
from data_utils.split import merge_to_part, index_to_data, get_split_index
from utils.args import get_args_parser
from utils.store import make_output_dir
from utils.utils import state_log, result_log, setup_seed, sub_result_log
from Trainer.NSAL_DGAT_Training import train
from models.NSAL_DGAT import Discriminator, DAANLoss
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from data_utils.preprocess import ele_normalize


def main(args):
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)
    setup_seed(args.seed)

    data, label, channels, feature_dim, num_classes = get_data(setting)
    data, label = merge_to_part(data, label, setting)
    device = torch.device(args.device)
    best_metrics = []
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
                index_to_data(data_i, label_i, train_indexes, test_indexes, val_indexes, keep_dim=False)

            model = Model['NSAL_DGAT'](channels=channels, feature_dim=feature_dim,num_of_class= num_classes, device=device)
            # Train one round using the train one round function defined in the model
            # model to train
            if len(val_data) == 0:
                val_data = test_data
                val_label = test_label
            dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_data),torch.arange(len(train_data)).long(), torch.Tensor(train_label))
            dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_data), torch.Tensor(val_label))
            dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_data), torch.Tensor(test_label))
            hidden_2=64
            output_dir = make_output_dir(args, "NSAL_DGAT")
            domain_discriminator = Discriminator(hidden_2)

            # loss criterion
            criterion = nn.CrossEntropyLoss()
            # Use GPU

            model = model.to(device)
            domain_discriminator = domain_discriminator.to(device)
            criterion = criterion.to(device)
            dann_loss = DAANLoss(domain_discriminator, num_class=num_classes).to(device)

            # Optimizer
            optimizer = torch.optim.AdamW(
                list(model.parameters()) + list(domain_discriminator.parameters()),
                lr=args.lr, weight_decay=0.001)
            lr_scheduler = StepwiseLR_GRL(optimizer, init_lr=args.lr, gamma=10, decay_rate=0.75, max_iter=args.epochs)
            round_metric = train(model=model, dann_loss=dann_loss,hidden_2=hidden_2, dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test, output_dir=output_dir, device=device, metrics=args.metrics, metric_choose=args.metric_choose, optimizer=optimizer,
                                 scheduler=lr_scheduler,batch_size=args.batch_size, epochs=args.epochs, criterion=criterion, test_sub_label=test_sub_label)
            best_metrics.append(round_metric)
    # best metrics: every round metrics dict
    # subjects metrics: (subject, sub_round_metric)
    result_log(args, best_metrics)

from torch.optim.optimizer import Optimizer
from typing import Optional
class StepwiseLR_GRL:
    def __init__(self, optimizer: Optimizer, init_lr: Optional[float] = 0.01,
                 gamma: Optional[float] = 0.001, decay_rate: Optional[float] = 0.75, max_iter: Optional[float] = 1000):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0
        self.max_iter = max_iter

    def get_lr(self) -> float:
        lr = self.init_lr / (1.0 + self.gamma * (self.iter_num / self.max_iter)) ** (self.decay_rate)
        return lr

    def step(self):
        """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if 'lr_mult' not in param_group:
                param_group['lr_mult'] = 1.
            param_group['lr'] = lr * param_group['lr_mult']

        self.iter_num += 1


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    # log out train state
    main(args)

