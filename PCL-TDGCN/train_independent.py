"""
Domain Adaptation Model Training Script
Author: YI Yang
Date: 2024
Purpose: Implementation of a domain adaptation framework for cross-subject EEG emotion recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import math
import os
import argparse
from typing import Tuple, List, Dict, Any
from sklearn.metrics import confusion_matrix

from model import DomainAdaptationModel, Discriminator
from Adversarial import DAANLoss
import utils
from utils import create_logger


def set_seed(seed: int = 20) -> None:
    """Set random seed for reproducibility across all random number generators."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(2)


class StepwiseLR_GRL:
    """Gradual learning rate scheduler with gradient reversal layer support."""

    def __init__(self, optimizer: torch.optim.Optimizer,
                 init_lr: float = 0.01, gamma: float = 0.001,
                 decay_rate: float = 0.75, max_iter: int = 1000):
        """
        Initialize the learning rate scheduler.

        Args:
            optimizer: Optimizer instance
            init_lr: Initial learning rate
            gamma: Decay coefficient
            decay_rate: Decay rate exponent
            max_iter: Maximum number of iterations
        """
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0
        self.max_iter = max_iter

    def get_lr(self) -> float:
        """Calculate current learning rate using polynomial decay."""
        lr = self.init_lr / (1.0 + self.gamma * (self.iter_num / self.max_iter)) ** (self.decay_rate)
        return lr

    def step(self) -> None:
        """Update learning rate for all parameter groups."""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group.setdefault('lr_mult', 1.)
            param_group['lr'] = lr * param_group['lr_mult']
        self.iter_num += 1


def test(test_loader: DataLoader, model: nn.Module,
         criterion: nn.Module, args: argparse.Namespace) -> Tuple[torch.Tensor, float, np.ndarray]:
    """
    Evaluate model performance on test dataset.

    Args:
        test_loader: Test data loader
        model: Model to evaluate
        criterion: Loss function
        args: Configuration parameters

    Returns:
        Average loss, accuracy, and confusion matrix
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for test_input, _, label in test_loader:
            test_input, label = test_input.to(args.device), label.to(args.device)
            output = model.target_predict(test_input)
            loss = criterion(output, label.view(-1))
            total_loss += loss.item()

            _, pred = torch.max(output, dim=1)
            correct += pred.eq(label.data.view_as(pred)).sum().item()

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / len(test_loader.dataset)

    # Compute confusion matrix
    all_classes = np.arange(args.cls)
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=all_classes)

    return avg_loss, accuracy, conf_matrix


def initialize_source_banks(train_loader: DataLoader, model: nn.Module, args: argparse.Namespace) -> None:
    """Initialize source domain feature memory banks."""
    model.eval()
    with torch.no_grad():
        for tran_input, tran_idx, _ in train_loader:
            tran_input, tran_idx = tran_input.to(args.device), tran_idx.to(args.device)
            model.get_init_banks(tran_input, tran_idx)


def initialize_target_banks(train_loader: DataLoader, model: nn.Module, args: argparse.Namespace) -> None:
    """Initialize target domain feature memory banks."""
    model.eval()
    with torch.no_grad():
        for tran_input, tran_idx, _ in train_loader:
            tran_input, tran_idx = tran_input.to(args.device), tran_idx.to(args.device)
            model.get_init_banks_tgt(tran_input, tran_idx)


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross-entropy loss for regularization."""

    def __init__(self, classes: int = 3, epsilon: float = 0.0005):
        """
        Initialize label smoothing loss.

        Args:
            classes: Number of classes
            epsilon: Smoothing parameter
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.classes = classes
        self.epsilon = epsilon

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for label smoothing loss.

        Args:
            input: Model predictions [batch_size, num_classes]
            target: Ground truth labels [batch_size]

        Returns:
            Smoothed cross-entropy loss
        """
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.epsilon / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.epsilon))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


def prepare_data(
    args: argparse.Namespace,
    test_id: int,
    validation_id: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Build a strict outer LOSO split: N-2 source, 1 validation, 1 test.

    The held-out test subject is still used as the *unlabeled target domain*
    during adaptation.  Its labels are kept only for the single final
    evaluation after the validation-selected checkpoint has been restored.
    The validation subject is never used for optimization or adaptation.
    """
    data, label = utils.load_data(
        args.dataset,
        dataset_root=args.dataset_path,
        rebuild_cache=args.rebuild_deap_cache,
    )
    if args.session < 0 or args.session >= len(data):
        raise ValueError(
            'Session index {} is invalid for {} ({} session(s)).'
            .format(args.session, args.dataset, len(data))
        )

    data_session = [
        np.ascontiguousarray(subject, dtype=np.float32)
        for subject in data[args.session]
    ]
    label_session = [
        np.ascontiguousarray(subject, dtype=np.int64).reshape(-1, 1)
        for subject in label[args.session]
    ]
    num_subjects = len(data_session)
    if not 0 <= test_id < num_subjects:
        raise ValueError(f'test_id must be in [0, {num_subjects - 1}].')
    if not 0 <= validation_id < num_subjects:
        raise ValueError(f'validation_id must be in [0, {num_subjects - 1}].')
    if validation_id == test_id:
        raise ValueError('validation_id and test_id must be different.')

    train_ids = [
        subject_id for subject_id in range(num_subjects)
        if subject_id not in {test_id, validation_id}
    ]

    source_set = {
        'feature': np.ascontiguousarray(
            np.vstack([data_session[i] for i in train_ids]), dtype=np.float32
        ),
        'label': np.ascontiguousarray(
            np.vstack([label_session[i] for i in train_ids]), dtype=np.int64
        ),
    }
    validation_set = {
        'feature': data_session[validation_id],
        'label': label_session[validation_id],
    }
    target_set = {
        'feature': data_session[test_id],
        'label': label_session[test_id],
    }
    return target_set, validation_set, source_set


def create_data_loaders(
    source_set: Dict[str, Any],
    target_set: Dict[str, Any],
    validation_set: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[Dict[str, DataLoader], int, int]:
    """Create loaders for source training, unlabeled target adaptation,
    validation-only selection, and final target testing.
    """
    for split in (source_set, target_set, validation_set):
        split['feature'] = np.ascontiguousarray(split['feature'], dtype=np.float32)
        split['label'] = np.ascontiguousarray(split['label'], dtype=np.int64)

    source_sample_num = source_set['feature'].shape[0]
    target_sample_num = target_set['feature'].shape[0]

    source_dataset = TensorDataset(
        torch.from_numpy(source_set['feature']).float(),
        torch.arange(source_sample_num).long(),
        torch.from_numpy(source_set['label']).long(),
    )
    target_dataset = TensorDataset(
        torch.from_numpy(target_set['feature']).float(),
        torch.arange(target_sample_num).long(),
        torch.from_numpy(target_set['label']).long(),
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(validation_set['feature']).float(),
        torch.arange(validation_set['feature'].shape[0]).long(),
        torch.from_numpy(validation_set['label']).long(),
    )

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return {
        'source_loader': DataLoader(source_dataset, shuffle=True, **loader_kwargs),
        # Test-subject features participate without labels in adaptation.
        'target_loader': DataLoader(target_dataset, shuffle=True, **loader_kwargs),
        # Validation is evaluation-only and never enters train_epoch().
        'validation_loader': DataLoader(validation_dataset, shuffle=False, **loader_kwargs),
        'test_loader': DataLoader(target_dataset, shuffle=False, **loader_kwargs),
    }, source_sample_num, target_sample_num


def train_epoch(model: nn.Module, domain_discriminator: nn.Module,
                dann_loss: nn.Module, criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                data_loaders: Dict[str, DataLoader],
                epoch: int, args: argparse.Namespace) -> Tuple[float, float, Dict[str, float]]:
    """
    Train the model for one epoch.

    Args:
        model: Main domain adaptation model
        domain_discriminator: Domain discriminator for adversarial training
        dann_loss: Domain adversarial adaptation loss
        criterion: Classification loss
        optimizer: Model optimizer
        data_loaders: Data loader dictionary
        epoch: Current epoch number
        args: Configuration parameters

    Returns:
        Average training loss, accuracy, and detailed loss dictionary
    """
    model.train()
    dann_loss.train()

    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    loss_dict = {}

    # Create data iterators
    src_iter = iter(data_loaders["source_loader"])
    tar_iter = iter(data_loaders["target_loader"])
    num_batches = min(len(data_loaders["source_loader"]), len(data_loaders["target_loader"]))
    for batch_idx in range(num_batches):
        # Get batch data
        src_data, src_idx, src_label = next(src_iter)
        tar_data, tar_idx, _ = next(tar_iter)

        # Move to device
        src_data, src_idx, src_label = (
            src_data.to(args.device),
            src_idx.to(args.device),
            src_label.to(args.device).view(-1)
        )
        tar_data, tar_idx = (
            tar_data.to(args.device),
            tar_idx.to(args.device)
        )

        # Forward pass
        (src_output_cls, src_feature, tar_output_cls, tar_feature,
         source_att, target_att, src_sim, tgt_sim, tgt_cluster_label,
         s2t_pro, t2s_pro, s2s_pro, t2t_pro) = model(
            src_data, tar_data, src_label, src_idx, tar_idx, epoch, args.epochs
        )

        # Classification loss
        cls_loss = criterion(src_output_cls, src_label)

        # Source domain loss with confidence filtering
        src_prob = F.softmax(src_output_cls, dim=1)
        max_prob, _ = src_prob.max(dim=1)
        mask = max_prob > 0.7

        if mask.any():
            filtered_prob = src_prob[mask]
            filtered_label = src_label[mask]
            source_loss = criterion(filtered_prob, filtered_label)
        else:
            source_loss = torch.tensor(0.0, device=src_prob.device)

        # Target domain classification loss
        target_loss = criterion(tgt_sim, tgt_cluster_label.long())

        # Domain adversarial loss
        global_transfer_loss = dann_loss(
            src_feature + 0.005 * torch.randn_like(src_feature).to(args.device),
            tar_feature + 0.005 * torch.randn_like(tar_feature).to(args.device),
            src_prob,tar_output_cls)


        # Cross-domain and within-domain consistency losses
        boost_factor = 2.0 * (2.0 / (1.0 + math.exp(-epoch / 1000)) - 1)

        s2t_entropy = -torch.sum(s2t_pro * torch.log(s2t_pro + 1e-10), dim=1).mean()
        t2s_entropy = -torch.sum(t2s_pro * torch.log(t2s_pro + 1e-10), dim=1).mean()
        cross_domain_loss = s2t_entropy + t2s_entropy

        s2s_entropy = -torch.sum(s2s_pro * torch.log(s2s_pro + 1e-10), dim=1).mean()
        t2t_entropy = -torch.sum(t2t_pro * torch.log(t2t_pro + 1e-10), dim=1).mean()
        in_domain_loss = s2s_entropy + t2t_entropy

        # Total loss
        loss = (cls_loss + global_transfer_loss + source_loss +
                boost_factor * target_loss + 0.2 * (cross_domain_loss + in_domain_loss))

        # Check for NaN values
        if torch.isnan(loss).any():
            print(f"Warning: NaN loss detected at epoch {epoch}, batch {batch_idx}")
            continue

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        _, pred = torch.max(src_prob, dim=1)
        total_correct += pred.eq(src_label).sum().item()
        total_samples += src_label.size(0)
        total_loss += loss.item()

        # Record detailed losses (first batch only)
        if batch_idx == 0:
            loss_dict = {
                'cls_loss': cls_loss.item(),
                'source_loss': source_loss.item(),
                'target_loss': target_loss.item(),
                'global_transfer_loss': global_transfer_loss.item(),
                'cross_domain_loss': cross_domain_loss.item(),
                'in_domain_loss': in_domain_loss.item(),
                'total_loss': loss.item()
            }

    avg_loss = total_loss / num_batches
    accuracy = total_correct / total_samples

    return avg_loss, accuracy, loss_dict


def main(
    test_id: int,
    validation_id: int,
    writer: SummaryWriter,
    args: argparse.Namespace,
) -> Tuple[float, float, int, np.ndarray]:
    """Train one strict LOSO fold and evaluate the test subject once.

    Model selection and early stopping use only ``validation_id``.  The labels
    of ``test_id`` are not inspected until the best validation checkpoint has
    been restored.
    """
    set_seed(args.seed)

    target_set, validation_set, source_set = prepare_data(
        args, test_id, validation_id
    )
    data_loaders, source_sample_num, target_sample_num = create_data_loaders(
        source_set, target_set, validation_set, args
    )

    model = DomainAdaptationModel(
        in_planes=args.in_planes,
        layers=args.layers,
        hidden_1=args.hidden_1,
        hidden_2=args.hidden_2,
        num_of_class=args.cls,
        device=args.device,
        source_num=source_sample_num,
        target_num=target_sample_num,
    ).to(args.device)
    domain_discriminator = Discriminator(args.hidden_2).to(args.device)
    criterion = LabelSmoothingCrossEntropy(classes=args.cls).to(args.device)
    dann_loss = DAANLoss(domain_discriminator, num_class=args.cls).to(args.device)

    optimizer = torch.optim.RMSprop(
        list(model.parameters()) + list(domain_discriminator.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = StepwiseLR_GRL(
        optimizer,
        init_lr=args.lr,
        gamma=10,
        decay_rate=0.75,
        max_iter=args.epochs,
    )

    model.eval()
    initialize_source_banks(data_loaders['source_loader'], model, args)
    initialize_target_banks(data_loaders['target_loader'], model, args)

    best_val_acc = -1.0
    best_val_loss = float('inf')
    best_epoch = -1
    patience_counter = 0
    eval_interval = args.eval_interval

    model_dir = os.path.join(
        args.output_model_dir,
        args.dataset,
        'strict_loso',
        f'session_{args.session}',
    )
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(
        model_dir,
        f'test_{test_id:02d}_val_{validation_id:02d}.pth',
    )

    logger.info(
        'Strict fold: train=%d subjects, validation=%d, test=%d; '
        'test features are used unlabeled for adaptation.',
        args.num_subjects - 2,
        validation_id,
        test_id,
    )

    for epoch in range(args.epochs):
        train_loss, train_acc, loss_dict = train_epoch(
            model,
            domain_discriminator,
            dann_loss,
            criterion,
            optimizer,
            data_loaders,
            epoch,
            args,
        )
        lr_scheduler.step()

        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/accuracy', train_acc, epoch)
        for loss_name, loss_value in loss_dict.items():
            writer.add_scalar(f'train/{loss_name}', loss_value, epoch)

        should_evaluate = (
            epoch % eval_interval == 0 or epoch == args.epochs - 1
        )
        if not should_evaluate:
            continue

        val_loss, val_acc, val_confusion = test(
            data_loaders['validation_loader'], model, criterion, args
        )
        writer.add_scalar('validation/loss', val_loss, epoch)
        writer.add_scalar('validation/accuracy', val_acc, epoch)

        improved = (
            val_acc > best_val_acc
            or (np.isclose(val_acc, best_val_acc) and val_loss < best_val_loss)
        )
        if improved:
            best_val_acc = float(val_acc)
            best_val_loss = float(val_loss)
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            logger.info(
                'Epoch %d: saved validation-best checkpoint '
                '(val_acc=%.4f, val_loss=%.6f).',
                epoch,
                val_acc,
                val_loss,
            )
        else:
            patience_counter += 1

        logger.info(
            'Epoch %d: train_acc=%.4f, val_acc=%.4f, best_val=%.4f@%d',
            epoch,
            train_acc,
            val_acc,
            best_val_acc,
            best_epoch,
        )
        if patience_counter >= args.patience:
            logger.info(
                'Early stopping after %d validation checks without improvement.',
                args.patience,
            )
            break

    if best_epoch < 0 or not os.path.isfile(model_path):
        raise RuntimeError('No validation checkpoint was saved for this fold.')

    state_dict = torch.load(model_path, map_location=args.device)
    model.load_state_dict(state_dict)
    test_loss, test_acc, test_confusion = test(
        data_loaders['test_loader'], model, criterion, args
    )
    logger.info(
        'Final test subject %d (selected by validation subject %d): '
        'accuracy=%.4f, loss=%.6f, best_epoch=%d.',
        test_id,
        validation_id,
        test_acc,
        test_loss,
        best_epoch,
    )
    writer.add_scalar('final_test/loss', test_loss, best_epoch)
    writer.add_scalar('final_test/accuracy', test_acc, best_epoch)
    return float(test_acc), best_val_acc, best_epoch, test_confusion



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PCL-TDGCN cross-subject training')

    # Dataset names: seed3 / seed_de_lds, seed4 / seediv_de_lds,
    # deap_a / deap-a, and deap_v / deap-v.
    parser.add_argument('--dataset', type=str, default='seed3', help='dataset name')
    parser.add_argument(
        '--dataset_path', type=str, default=None,
        help='dataset root; for DEAP point to the root or data_preprocessed_python',
    )
    parser.add_argument(
        '--session', type=int, default=0,
        help='zero-based session index; DEAP only supports 0',
    )
    parser.add_argument(
        '--cls', type=int, default=None,
        help='number of classes; inferred from dataset when omitted',
    )
    parser.add_argument(
        '--in_planes', type=int, nargs=2, default=None, metavar=('BANDS', 'CHANNELS'),
        help='input layout; inferred as 5x62 for SEED and 5x32 for DEAP',
    )
    parser.add_argument(
        '--rebuild_deap_cache', action='store_true',
        help='recompute DEAP DE-LDS features instead of using the local cache',
    )
    parser.add_argument(
        '--validation_offset', type=int, default=1,
        help='cyclic offset from test subject to validation subject',
    )
    parser.add_argument(
        '--eval_interval', type=int, default=10,
        help='evaluate the validation subject every N epochs',
    )
    parser.add_argument(
        '--patience', type=int, default=10,
        help='number of validation checks without improvement before stopping',
    )

    parser.add_argument('--layers', type=int, default=2, help='number of DGCN layers')
    parser.add_argument('--hidden_1', type=int, default=256)
    parser.add_argument('--hidden_2', type=int, default=64)
    parser.add_argument('--k', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--device', type=str,
        default='cuda:1' if torch.cuda.is_available() else 'cpu',
        help='PyTorch device',
    )
    parser.add_argument('--output_log_dir', default='./train_log', type=str)
    parser.add_argument('--output_model_dir', default='./model', type=str)
    args = parser.parse_args()

    metadata = utils.get_dataset_metadata(args.dataset)
    if args.cls is None:
        args.cls = metadata['num_classes']
    elif args.cls != metadata['num_classes']:
        raise ValueError(
            '--cls={} conflicts with {} classes for {}.'
            .format(args.cls, metadata['num_classes'], args.dataset)
        )

    inferred_planes = [metadata['num_bands'], metadata['num_channels']]
    if args.in_planes is None:
        args.in_planes = inferred_planes
    elif list(args.in_planes) != inferred_planes:
        raise ValueError(
            '--in_planes={} conflicts with required {} for {}.'
            .format(args.in_planes, inferred_planes, args.dataset)
        )

    if args.session < 0 or args.session >= metadata['num_sessions']:
        raise ValueError(
            '--session={} is invalid for {}; valid range is 0..{}.'
            .format(args.session, args.dataset, metadata['num_sessions'] - 1)
        )

    args.device = torch.device(args.device)
    args.num_subjects = metadata['num_subjects']

    logger = create_logger(args)
    logger.info(f"Training Configuration: {args}")
    logger.info(
        'Dataset geometry: sessions=%d, subjects=%d, bands=%d, channels=%d, classes=%d',
        metadata['num_sessions'], metadata['num_subjects'], metadata['num_bands'],
        metadata['num_channels'], metadata['num_classes'],
    )

    if args.num_subjects < 3:
        raise ValueError('Strict LOSO requires at least three subjects.')
    offset = args.validation_offset % args.num_subjects
    if offset == 0:
        raise ValueError('--validation_offset cannot be a multiple of subject count.')

    all_test_accuracies = []
    all_validation_accuracies = []
    all_best_epochs = []
    all_conf_matrices = []
    logger.info(
        'Starting strict N-2 + 1 + 1 LOSO: source training subjects, '
        'one validation subject, one held-out test subject.'
    )

    for test_id in range(args.num_subjects):
        validation_id = (test_id + offset) % args.num_subjects
        train_ids = [
            i for i in range(args.num_subjects)
            if i not in {test_id, validation_id}
        ]
        writer_dir = (
            f'data/tensorboard/experiment_{args.dataset}/'
            f'session_{args.session}_strict_loso/'
            f'test_{test_id:02d}_val_{validation_id:02d}'
        )
        writer = SummaryWriter(writer_dir)

        logger.info(
            'Fold %d/%d | train=%s | validation=%d | test=%d',
            test_id + 1,
            args.num_subjects,
            train_ids,
            validation_id,
            test_id,
        )
        test_acc, val_acc, best_epoch, conf_matrix = main(
            test_id, validation_id, writer, args
        )
        writer.close()
        all_test_accuracies.append(test_acc)
        all_validation_accuracies.append(val_acc)
        all_best_epochs.append(best_epoch)
        all_conf_matrices.append(conf_matrix)

    all_test_accuracies = np.asarray(all_test_accuracies, dtype=np.float64)
    total_conf_matrix = np.sum(all_conf_matrices, axis=0)
    logger.info('=' * 60)
    logger.info('Strict N-2 + 1 + 1 LOSO complete')
    logger.info(
        'Final test accuracy: %.4f ± %.4f',
        all_test_accuracies.mean(),
        all_test_accuracies.std(),
    )
    logger.info('Per-subject test accuracies: %s', all_test_accuracies)
    logger.info('Per-fold validation accuracies: %s', all_validation_accuracies)
    logger.info('Selected epochs: %s', all_best_epochs)
    logger.info('Aggregated final-test confusion matrix: %s', total_conf_matrix)
    logger.info('=' * 60)
