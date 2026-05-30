import copy
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, Dataset
from tqdm import tqdm

from utils.metric import Metric
from utils.store import save_state


param_path = "config/model_param/FAT.yaml"


def _load_train_param():
    try:
        with open(param_path, "r", encoding="utf-8") as fd:
            cfg = yaml.load(fd, Loader=yaml.FullLoader)
        return cfg.get("train", {})
    except IOError:
        print("\n{} may not exist or not available".format(param_path))
        return {}


def _to_label_index(targets):
    if len(targets.shape) > 1 and targets.shape[-1] > 1:
        return torch.argmax(targets, dim=1).long()
    return targets.long().view(-1)


def _scale_frequency_bands(features, scale_low=0.9, scale_high=1.1):
    scale_factors = np.random.uniform(scale_low, scale_high, size=features.shape[-1])
    return features * scale_factors


def _add_periodic_perturbation(features, frequency=2, amplitude=0.05):
    num_bands = features.shape[-1]
    timeline = np.linspace(0, 2 * np.pi, num_bands)
    sinusoid = amplitude * np.sin(frequency * timeline)
    return features + sinusoid


class FATDataset(Dataset):
    def __init__(self, features, labels, augment=False, scale_low=0.9, scale_high=1.1, frequency=2, amplitude=0.05):
        self.features = features
        self.labels = labels
        self.augment = augment
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.frequency = frequency
        self.amplitude = amplitude

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        if self.augment:
            feature_np = feature.detach().cpu().numpy()
            feature_np = _scale_frequency_bands(feature_np, self.scale_low, self.scale_high)
            feature_np = _add_periodic_perturbation(feature_np, self.frequency, self.amplitude)
            feature = torch.tensor(feature_np, dtype=torch.float32)

        return feature.float(), label


def _build_loader(dataset, sampler, batch_size, num_workers):
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers)


@torch.no_grad()
def evaluate(model, data_loader, device, metrics, criterion):
    model.eval()
    metric = Metric(metrics)
    for _, (samples, targets) in tqdm(enumerate(data_loader), total=len(data_loader), desc="Evaluating : "):
        samples = samples.to(device)
        targets = _to_label_index(targets.to(device))
        outputs = model(samples)
        loss = criterion(outputs, targets)
        metric.update(torch.argmax(outputs, dim=1), targets, loss.item())
    print("\033[34m eval state: " + metric.value())
    return metric.values


def train(
    model,
    dataset_train,
    dataset_val,
    dataset_test,
    device,
    output_dir="result/",
    metrics=None,
    metric_choose=None,
    optimizer=None,
    batch_size=16,
    epochs=40,
    criterion=None,
    num_workers=4,
    experiment_mode="subject-dependent",
):
    train_cfg = _load_train_param()

    if metrics is None:
        metrics = ["acc"]
    if metric_choose is None:
        metric_choose = metrics[0]

    augment_train = bool(train_cfg.get("augment_train", True))
    scale_low = float(train_cfg.get("scale_low", 0.9))
    scale_high = float(train_cfg.get("scale_high", 1.1))
    periodic_frequency = float(train_cfg.get("periodic_frequency", 2))
    periodic_amplitude = float(train_cfg.get("periodic_amplitude", 0.05))
    mixup_alpha = float(train_cfg.get("mixup_alpha", 0.2))
    mixup_dependent = bool(train_cfg.get("mixup_dependent", False))
    mixup_independent = bool(train_cfg.get("mixup_independent", True))
    rollback_margin = float(train_cfg.get("rollback_margin", 0.1))
    rollback_patience_divisor = int(train_cfg.get("rollback_patience_divisor", 10))
    evaluate_val = bool(train_cfg.get("evaluate_val", True))
    select_metric_on = str(train_cfg.get("select_metric_on", "val")).lower()

    use_mixup = mixup_independent if experiment_mode == "subject-independent" else mixup_dependent

    train_features, train_labels = dataset_train.tensors
    val_features, val_labels = dataset_val.tensors
    test_features, test_labels = dataset_test.tensors

    fat_train_dataset = FATDataset(
        train_features,
        train_labels,
        augment=augment_train,
        scale_low=scale_low,
        scale_high=scale_high,
        frequency=periodic_frequency,
        amplitude=periodic_amplitude,
    )
    fat_val_dataset = TensorDataset(val_features, val_labels)
    fat_test_dataset = TensorDataset(test_features, test_labels)

    sampler_train = RandomSampler(fat_train_dataset)
    sampler_val = SequentialSampler(fat_val_dataset)
    sampler_test = SequentialSampler(fat_test_dataset)

    data_loader_train = _build_loader(fat_train_dataset, sampler_train, batch_size, num_workers)
    data_loader_val = _build_loader(fat_val_dataset, sampler_val, batch_size, num_workers)
    data_loader_test = _build_loader(fat_test_dataset, sampler_test, batch_size, num_workers)

    model = model.to(device)

    best_metric = {m: 0.0 for m in metrics}
    best_state = copy.deepcopy(model.state_dict())
    no_improve_epoch = 0
    rollback_patience = max(1, epochs // max(1, rollback_patience_divisor))

    for epoch in range(epochs):
        model.train()
        metric = Metric(metrics)
        train_bar = tqdm(
            enumerate(data_loader_train),
            total=len(data_loader_train),
            desc=f"Train Epoch {epoch}/{epochs}: lr:{optimizer.param_groups[0]['lr']}",
        )

        for _, (samples, targets) in train_bar:
            samples = samples.to(device)
            targets = _to_label_index(targets.to(device))

            optimizer.zero_grad()

            if use_mixup:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                rand_index = torch.randperm(samples.size(0), device=device)
                samples_shuffled = samples[rand_index]
                targets_shuffled = targets[rand_index]
                samples_mix = lam * samples + (1 - lam) * samples_shuffled
                outputs = model(samples_mix)
                loss = lam * criterion(outputs, targets) + (1 - lam) * criterion(outputs, targets_shuffled)
            else:
                outputs = model(samples)
                loss = criterion(outputs, targets)

            metric.update(torch.argmax(outputs, dim=1), targets, loss.item())
            train_bar.set_postfix_str(f"loss: {loss.item():.2f}")

            loss.backward()
            optimizer.step()

        print("\033[32m train state: " + metric.value())

        metric_value_val = evaluate(model, data_loader_val, device, metrics, criterion)

        if select_metric_on != "val":
            print("[FAT] override select_metric_on to 'val' under strict no-test-leak protocol")
        metric_source = metric_value_val
        improved = False

        for m in metrics:
            if metric_source[m] > best_metric[m]:
                best_metric[m] = metric_source[m]
                improved = True
                save_state(output_dir, model, optimizer, epoch + 1, metric=m)

        if improved:
            no_improve_epoch = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_improve_epoch += 1

        if no_improve_epoch >= rollback_patience and metric_source[metric_choose] < best_metric[metric_choose] - rollback_margin:
            print("######################### Reloading best model state #########################")
            model.load_state_dict(best_state)
            no_improve_epoch = 0

    model.load_state_dict(torch.load(f"{output_dir}/checkpoint-best{metric_choose}")["model"])
    metric_value = evaluate(model, data_loader_test, device, metrics, criterion)

    for m in metrics:
        print(f"best_select_{m}: {best_metric[m]:.2f}")
        print(f"best_test_{m}: {metric_value[m]:.2f}")

    return metric_value
