from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import MinMaxScaler


SUPPORTED_SPLIT_TYPES = ("train-val-test", "leave-one-out")


@dataclass(frozen=True)
class DomainProtocol:
    split_type: str
    target_role: str
    selection_role: str
    final_role: str
    test_leak: bool


def resolve_domain_protocol(experiment_mode, split_type, val_indexes):
    if experiment_mode != "subject-independent":
        raise ValueError(
            "DS-AGC and PCL-TDGCN reproduction requires "
            "experiment_mode='subject-independent'."
        )
    if split_type not in SUPPORTED_SPLIT_TYPES:
        raise ValueError(
            "Supported reproduction split types are train-val-test and "
            f"leave-one-out, got {split_type!r}."
        )
    if split_type == "train-val-test":
        if not val_indexes or val_indexes[0] == -1:
            raise ValueError("train-val-test requires a non-empty validation split.")
        return DomainProtocol(split_type, "test", "val", "test", False)
    return DomainProtocol(split_type, "test", "test", "test", True)


def concatenate_parts(parts, dtype=None):
    arrays = [np.asarray(part, dtype=dtype) for part in parts]
    if not arrays or any(array.shape[0] == 0 for array in arrays):
        raise ValueError("Cannot concatenate an empty subject/part collection.")
    return np.concatenate(arrays, axis=0)


def minmax_scale_part(part):
    array = np.asarray(part, dtype=np.float32)
    if array.size == 0 or array.shape[0] == 0:
        raise ValueError("Cannot normalize an empty subject/part.")
    shape = array.shape
    flat = array.reshape(shape[0], -1)
    scaled = MinMaxScaler(feature_range=(-1, 1)).fit_transform(flat)
    return scaled.astype(np.float32).reshape(shape)


def normalize_and_concatenate_parts(parts):
    return concatenate_parts([minmax_scale_part(part) for part in parts])


def normalize_labeled_subject_parts(subject_indexes, data_parts, label_parts):
    indexes = list(subject_indexes)
    data_parts = list(data_parts)
    label_parts = list(label_parts)
    if not indexes or not (
        len(indexes) == len(data_parts) == len(label_parts)
    ):
        raise ValueError(
            "Subject indexes, data parts, and label parts must have the same "
            "non-zero length."
        )

    return [
        (
            int(subject_index),
            minmax_scale_part(data_part),
            np.asarray(label_part),
        )
        for subject_index, data_part, label_part in zip(
            indexes, data_parts, label_parts
        )
    ]
