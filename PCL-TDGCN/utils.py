from torch.utils.data import Dataset, DataLoader

import copy
import pickle
import logging
import os
import random
import re
import time

import numpy as np
import scipy.io as scio
from scipy import signal
import sklearn.preprocessing as preprocessing
import torch
import torch.nn as nn


# Set these paths to the dataset ROOT directories used by LibEER:
#   seed3 -> <SEED root>/ExtractedFeatures
#   seed4 -> <SEED-IV root>/eeg_feature_smooth/{1,2,3}
# The loader also accepts a path that already points directly to
# ExtractedFeatures or eeg_feature_smooth.
dataset_path = {
    'seed4': '/mnt/sdc/sdc1/yangli/yangli/EEG/EEG_Dataset/SEED_IV',
    'seed3': '/mnt/sdc/sdc1/yangli/yangli/EEG/EEG_Dataset/SEED/SEED_EEG',
    'deafseed3': '/home/user_yy/Dataset/deafseed',
    # May point either to the DEAP root or directly to data_preprocessed_python.
    'deap': '/path/to/DEAP/data_preprocessed_python',
}


# File order copied from LibEER's seed_de_lds reader.  The outer dimension is
# session and the inner dimension is subject.
SEED_DE_LDS_FILES = [
    [
        '1_20131027.mat', '2_20140404.mat', '3_20140603.mat',
        '4_20140621.mat', '5_20140411.mat', '6_20130712.mat',
        '7_20131027.mat', '8_20140511.mat', '9_20140620.mat',
        '10_20131130.mat', '11_20140618.mat', '12_20131127.mat',
        '13_20140527.mat', '14_20140601.mat', '15_20130709.mat',
    ],
    [
        '1_20131030.mat', '2_20140413.mat', '3_20140611.mat',
        '4_20140702.mat', '5_20140418.mat', '6_20131016.mat',
        '7_20131030.mat', '8_20140514.mat', '9_20140627.mat',
        '10_20131204.mat', '11_20140625.mat', '12_20131201.mat',
        '13_20140603.mat', '14_20140615.mat', '15_20131016.mat',
    ],
    [
        '1_20131107.mat', '2_20140419.mat', '3_20140629.mat',
        '4_20140705.mat', '5_20140506.mat', '6_20131113.mat',
        '7_20131106.mat', '8_20140521.mat', '9_20140704.mat',
        '10_20131211.mat', '11_20140630.mat', '12_20131207.mat',
        '13_20140610.mat', '14_20140627.mat', '15_20131105.mat',
    ],
]


# File order copied from LibEER's seediv_de_lds reader.
SEEDIV_DE_LDS_FILES = [
    [
        '1_20160518.mat', '2_20150915.mat', '3_20150919.mat',
        '4_20151111.mat', '5_20160406.mat', '6_20150507.mat',
        '7_20150715.mat', '8_20151103.mat', '9_20151028.mat',
        '10_20151014.mat', '11_20150916.mat', '12_20150725.mat',
        '13_20151115.mat', '14_20151205.mat', '15_20150508.mat',
    ],
    [
        '1_20161125.mat', '2_20150920.mat', '3_20151018.mat',
        '4_20151118.mat', '5_20160413.mat', '6_20150511.mat',
        '7_20150717.mat', '8_20151110.mat', '9_20151119.mat',
        '10_20151021.mat', '11_20150921.mat', '12_20150804.mat',
        '13_20151125.mat', '14_20151208.mat', '15_20150514.mat',
    ],
    [
        '1_20161126.mat', '2_20151012.mat', '3_20151101.mat',
        '4_20151123.mat', '5_20160420.mat', '6_20150512.mat',
        '7_20150721.mat', '8_20151117.mat', '9_20151209.mat',
        '10_20151023.mat', '11_20151011.mat', '12_20150807.mat',
        '13_20161130.mat', '14_20151215.mat', '15_20150527.mat',
    ],
]


_DATASET_ALIASES = {
    'seed_de_lds': 'seed3',
    'seediv_de_lds': 'seed4',
    'deap-a': 'deap_a',
    'deap_a': 'deap_a',
    'deap_arousal': 'deap_a',
    'deap-v': 'deap_v',
    'deap_v': 'deap_v',
    'deap_valence': 'deap_v',
}


def _canonical_dataset_name(dataset_name):
    """Allow LibEER-style names without changing the rest of PCL-TDGCN."""
    return _DATASET_ALIASES.get(dataset_name, dataset_name)




def get_dataset_metadata(dataset_name):
    """Return the structural parameters required by the training script."""
    dataset_name = _canonical_dataset_name(dataset_name)
    metadata = {
        'seed3': {
            'num_sessions': 3,
            'num_subjects': 15,
            'num_classes': 3,
            'num_bands': 5,
            'num_channels': 62,
        },
        'seed4': {
            'num_sessions': 3,
            'num_subjects': 15,
            'num_classes': 4,
            'num_bands': 5,
            'num_channels': 62,
        },
        'deap_a': {
            'num_sessions': 1,
            'num_subjects': 32,
            'num_classes': 2,
            'num_bands': 5,
            'num_channels': 32,
        },
        'deap_v': {
            'num_sessions': 1,
            'num_subjects': 32,
            'num_classes': 2,
            'num_bands': 5,
            'num_channels': 32,
        },
    }
    if dataset_name not in metadata:
        raise ValueError('Unexpected dataset name: {}'.format(dataset_name))
    return metadata[dataset_name].copy()


def norminx(data):
    """Normalize each row independently."""
    for i in range(data.shape[0]):
        data[i] = normalization(data[i])
    return data


def norminy(data):
    data_t = data.T
    for i in range(data_t.shape[0]):
        data_t[i] = normalization(data_t[i])
    return data_t.T


def norminy_2d(data, band_num=5):
    """Normalize every band/electrode feature without assuming 62 channels."""
    if data.ndim != 2 or data.shape[1] % band_num != 0:
        raise ValueError('Expected [samples, bands * channels], got {}'.format(data.shape))
    channel_num = data.shape[1] // band_num
    data = data.reshape([-1, band_num, channel_num])
    for j in range(band_num):
        for i in range(channel_num):
            data[:, j, i] = normalization(data[:, j, i])
    return data.reshape([-1, band_num * channel_num])


def normalization(data):
    data_range = np.max(data) - np.min(data)
    return (data - np.min(data)) / data_range


class CustomDataset(Dataset):
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.LongTensor(self.Label[index])
        return data, label.squeeze()


# MMD loss and Gaussian kernel.
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1))
    )
    total1 = total.unsqueeze(1).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1))
    )
    l2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(l2_distance.data) / (n_samples ** 2 - n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [
        torch.exp(-l2_distance / bandwidth_temp)
        for bandwidth_temp in bandwidth_list
    ]
    return sum(kernel_val)


def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(
        source,
        target,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma,
    )
    xx = kernels[:batch_size, :batch_size]
    yy = kernels[batch_size:, batch_size:]
    xy = kernels[:batch_size, batch_size:]
    yx = kernels[batch_size:, :batch_size]
    return torch.mean(xx + yy - xy - yx)


def mmd_rbf_accelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(
        source,
        target,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma,
    )
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i + 1) % batch_size
        t1, t2 = s1 + batch_size, s2 + batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)


def mmd_linear(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    return torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))


def CORAL(source, target):
    d = source.data.shape[1]
    xm = torch.mean(source, 1, keepdim=True) - source
    xc = torch.matmul(torch.transpose(xm, 0, 1), xm)
    xmt = torch.mean(target, 1, keepdim=True) - target
    xct = torch.matmul(torch.transpose(xmt, 0, 1), xmt)
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    return loss / (4 * d * 4)


def EntropyLoss(input_):
    mask = input_.ge(0.000001)
    mask_out = torch.masked_select(input_, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))


def PADA(features, ad_net, grl_layer, weight_ad, use_gpu=True):
    ad_out = ad_net(grl_layer(features))
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(
        np.array([[1]] * batch_size + [[0]] * batch_size)
    ).float()
    if use_gpu:
        dc_target = dc_target.cuda()
        weight_ad = weight_ad.cuda()
    return nn.BCELoss(weight=weight_ad.view(-1))(
        ad_out.view(-1), dc_target.view(-1)
    )


def get_number_of_label_n_trial(dataset_name):
    """Return trial count, class count, and trial labels for each session."""
    dataset_name = _canonical_dataset_name(dataset_name)

    label_seed4 = [
        [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
        [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
        [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
    ]
    label_seed3 = [
        [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
        [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
        [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
    ]

    if dataset_name == 'seed3':
        return 15, 3, label_seed3
    if dataset_name == 'seed4':
        return 24, 4, label_seed4
    raise ValueError('Unexpected dataset name: {}'.format(dataset_name))


def reshape_data(data, label):
    """
    Convert every trial from [channel, sample, frequency] to [sample, 310]
    and concatenate trials in trial order.
    """
    reshape_data_array = None
    reshape_label = None

    for i in range(len(data)):
        trial = np.asarray(data[i])
        if trial.ndim != 3:
            raise ValueError(
                'Each de_LDS trial must be a 3-D array, but trial {} has shape {}.'
                .format(i, trial.shape)
            )

        # Official SEED/SEED-IV feature files use [62, samples, 5].
        # Keep PCL-TDGCN's original flattening order: frequency first, channel second.
        if trial.shape[0] == 62 and trial.shape[2] == 5:
            one_data = np.reshape(np.transpose(trial, (1, 2, 0)), (-1, 310))
        elif trial.shape[1] == 62 and trial.shape[2] == 5:
            # Defensive support for data already transposed to [samples, 62, 5].
            one_data = np.reshape(np.transpose(trial, (0, 2, 1)), (-1, 310))
        else:
            raise ValueError(
                'Unsupported de_LDS trial shape {}. Expected [62, samples, 5] '
                'or [samples, 62, 5].'.format(trial.shape)
            )

        one_label = np.full((one_data.shape[0], 1), label[i])
        if reshape_data_array is None:
            reshape_data_array = one_data
            reshape_label = one_label
        else:
            reshape_data_array = np.vstack((reshape_data_array, one_data))
            reshape_label = np.vstack((reshape_label, one_label))

    return reshape_data_array, reshape_label


def _natural_key(name):
    """Sort de_LDS1, de_LDS2, ..., de_LDS10 in numeric trial order."""
    match = re.search(r'(\d+)$', name)
    return int(match.group(1)) if match else name


def _extract_de_lds_trials(mat_data, dataset_name):
    """
    Extract de_LDS trials from one official feature MAT file.

    Primary path: select variables whose names start with ``de_LDS`` and sort
    them by trial number.  Fallback path: use the same feature-index logic as
    LibEER (SEED: 12 features/trial, de_lds index 1; SEED-IV: 4 features/trial,
    de_lds index 1).
    """
    dataset_name = _canonical_dataset_name(dataset_name)
    expected_trials = 15 if dataset_name == 'seed3' else 24

    de_lds_keys = [key for key in mat_data.keys() if key.startswith('de_LDS')]
    de_lds_keys.sort(key=_natural_key)

    if len(de_lds_keys) == expected_trials:
        return [np.asarray(mat_data[key]) for key in de_lds_keys]

    # Exact LibEER fallback based on variable insertion order.
    data_keys = [key for key in mat_data.keys() if not key.startswith('__')]
    feature_stride = 12 if dataset_name == 'seed3' else 4
    feature_index = 1
    required_count = expected_trials * feature_stride

    if len(data_keys) < required_count:
        raise ValueError(
            '{} contains {} de_LDS variables and {} data variables; expected '
            '{} de_LDS trials.'.format(
                'MAT file', len(de_lds_keys), len(data_keys), expected_trials
            )
        )

    selected_keys = [
        data_keys[trial_id * feature_stride + feature_index]
        for trial_id in range(expected_trials)
    ]
    return [np.asarray(mat_data[key]) for key in selected_keys]


def get_data_label_frommat(mat_path, dataset_name, session_id):
    """Load one subject MAT file and return normalized [N, 310] data/labels."""
    dataset_name = _canonical_dataset_name(dataset_name)
    _, _, labels = get_number_of_label_n_trial(dataset_name)

    try:
        mat_data = scio.loadmat(mat_path, simplify_cells=True)
    except TypeError:
        # Compatibility with older SciPy releases that do not support
        # simplify_cells.
        mat_data = scio.loadmat(mat_path)

    mat_de_data = _extract_de_lds_trials(mat_data, dataset_name)
    one_sub_data, one_sub_label = reshape_data(
        mat_de_data, labels[session_id]
    )

    one_sub_data = one_sub_data.astype(np.float32)
    one_sub_label = one_sub_label.astype(np.int64)

    # Preserve the original PCL-TDGCN subject/session-wise normalization.
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    one_sub_data = min_max_scaler.fit_transform(one_sub_data).astype(np.float32)
    return one_sub_data, one_sub_label


def sample_by_value(values, value, number):
    result_index = []
    index_for_value = [i for i, v in enumerate(values) if v == value]
    result_index.extend(random.sample(index_for_value, number))
    return result_index


def _resolve_feature_root(dataset_name):
    """Resolve LibEER-style dataset roots and legacy direct feature paths."""
    dataset_name = _canonical_dataset_name(dataset_name)
    if dataset_name not in dataset_path:
        raise KeyError(
            'No path configured for {}. Available keys: {}'.format(
                dataset_name, sorted(dataset_path.keys())
            )
        )

    configured_path = os.path.abspath(os.path.expanduser(dataset_path[dataset_name]))
    if not os.path.isdir(configured_path):
        raise FileNotFoundError(
            'Dataset path does not exist: {}. Update dataset_path[\'{}\'].'
            .format(configured_path, dataset_name)
        )

    if dataset_name == 'seed3':
        child = os.path.join(configured_path, 'ExtractedFeatures')
        if os.path.isdir(child):
            return child
        # The path may already be ExtractedFeatures, or may use the old
        # PCL-TDGCN session-folder layout.
        return configured_path

    if dataset_name == 'seed4':
        child = os.path.join(configured_path, 'eeg_feature_smooth')
        if os.path.isdir(child):
            return child
        return configured_path

    return configured_path


def get_allmats_name(dataset_name):
    """
    Return the feature root and the explicit LibEER session/subject file table.

    Unlike the original implementation, this does not depend on ``os.listdir``
    sorting or manual list rotations, so subject ordering is deterministic.
    """
    dataset_name = _canonical_dataset_name(dataset_name)
    path = _resolve_feature_root(dataset_name)

    if dataset_name == 'seed3':
        allmats = copy.deepcopy(SEED_DE_LDS_FILES)
    elif dataset_name == 'seed4':
        allmats = copy.deepcopy(SEEDIV_DE_LDS_FILES)
    else:
        raise ValueError('Unexpected dataset name: {}'.format(dataset_name))

    missing = []
    for session_id, session_files in enumerate(allmats):
        for filename in session_files:
            try:
                _get_mat_path(path, dataset_name, session_id, filename)
            except FileNotFoundError:
                missing.append((session_id + 1, filename))

    if missing:
        preview = ', '.join(
            'session {}: {}'.format(session_id, filename)
            for session_id, filename in missing[:8]
        )
        if len(missing) > 8:
            preview += ', ... ({} missing in total)'.format(len(missing))
        raise FileNotFoundError(
            'Required feature MAT files were not found under {}. Missing: {}'
            .format(path, preview)
        )

    return path, allmats


def _get_mat_path(feature_root, dataset_name, session_id, filename):
    """Resolve both LibEER's official layout and the old PCL layout."""
    dataset_name = _canonical_dataset_name(dataset_name)

    if dataset_name == 'seed3':
        candidates = [
            # LibEER/official SEED: all 45 files in ExtractedFeatures.
            os.path.join(feature_root, filename),
            # Backward compatibility with manually split 1/2/3 folders.
            os.path.join(feature_root, str(session_id + 1), filename),
        ]
    elif dataset_name == 'seed4':
        candidates = [
            # LibEER/official SEED-IV: eeg_feature_smooth/1~3/file.mat.
            os.path.join(feature_root, str(session_id + 1), filename),
            # Defensive support for a flattened feature directory.
            os.path.join(feature_root, filename),
        ]
    else:
        candidates = [os.path.join(feature_root, filename)]

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        'Cannot find {}. Checked: {}'.format(filename, candidates)
    )



DEAP_SAMPLE_RATE = 128
DEAP_NUM_SUBJECTS = 32
DEAP_NUM_TRIALS = 40
DEAP_NUM_CHANNELS = 32
DEAP_BASELINE_SECONDS = 3
DEAP_STIMULUS_SECONDS = 60
DEAP_TIME_WINDOW_SECONDS = 1
DEAP_EXTRACT_BANDS = ((0.5, 4), (4, 8), (8, 14), (14, 30), (30, 50))
DEAP_CACHE_FILENAME = 'pcl_tdgc_deap_de_lds_cache.npz'


def _resolve_deap_cache_path(dataset_root):
    """Return a writable cache path without requiring dataset-folder write access."""
    preferred = os.path.join(dataset_root, DEAP_CACHE_FILENAME)
    if os.access(dataset_root, os.W_OK):
        return preferred

    cache_dir = os.environ.get(
        'PCL_TDGCN_CACHE_DIR',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache'),
    )
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, DEAP_CACHE_FILENAME)
_DEAP_FEATURE_RUNTIME_CACHE = {}
_DEAP_TASK_RUNTIME_CACHE = {}


def _resolve_deap_root(dataset_root=None):
    """Resolve the LibEER DEAP ``data_preprocessed_python`` directory."""
    configured = dataset_root if dataset_root is not None else dataset_path.get('deap')
    if not configured:
        raise ValueError(
            "DEAP path is not configured. Set dataset_path['deap'] or pass "
            "--dataset_path /path/to/DEAP/data_preprocessed_python."
        )
    configured = os.path.abspath(os.path.expanduser(configured))
    candidates = [
        configured,
        os.path.join(configured, 'data_preprocessed_python'),
        os.path.join(configured, 'DEAP', 'data_preprocessed_python'),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, 's01.dat')):
            return candidate
    raise FileNotFoundError(
        'Cannot find DEAP data_preprocessed_python. Checked: {}'.format(candidates)
    )


def _libeer_lds(data):
    """LibEER LDS smoothing for an array shaped [time, channel, feature]."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError('LDS expects [time, channel, feature], got {}'.format(data.shape))
    num_t, num_channel, num_feature = data.shape
    flattened = data.reshape(num_t, -1)

    prior_correlation = 0.01
    transition_matrix = 1.0
    noise_correlation = 0.0001
    observation_matrix = 1.0
    observation_correlation = 1.0

    mean = np.mean(flattened, axis=0)
    observations = flattened.T
    num_features, num_samples = observations.shape
    p_mat = np.zeros_like(observations)
    u_mat = np.zeros_like(observations)
    k_mat = np.zeros_like(observations)
    v_mat = np.zeros_like(observations)

    k_mat[:, 0] = (
        prior_correlation * observation_matrix
        / (observation_matrix * prior_correlation * observation_matrix + observation_correlation)
    )
    u_mat[:, 0] = mean + k_mat[:, 0] * (
        observations[:, 0] - observation_matrix * prior_correlation
    )
    v_mat[:, 0] = (1.0 - k_mat[:, 0] * observation_matrix) * prior_correlation

    for idx in range(1, num_samples):
        p_mat[:, idx - 1] = (
            transition_matrix * v_mat[:, idx - 1] * transition_matrix
            + noise_correlation
        )
        k_mat[:, idx] = (
            p_mat[:, idx - 1] * observation_matrix
            / (
                observation_matrix * p_mat[:, idx - 1] * observation_matrix
                + observation_correlation
            )
        )
        u_mat[:, idx] = (
            transition_matrix * u_mat[:, idx - 1]
            + k_mat[:, idx]
            * (
                observations[:, idx]
                - observation_matrix * transition_matrix * u_mat[:, idx - 1]
            )
        )
        v_mat[:, idx] = (1.0 - k_mat[:, idx] * observation_matrix) * p_mat[:, idx - 1]

    return u_mat.T.reshape(num_t, num_channel, num_feature)


def _extract_de_lds_deap_batch(raw_batch):
    """
    Reproduce LibEER's DEAP preprocessing for a batch of trials.

    Input shape: [trials, 32, 8064] from data_preprocessed_python.
    Output shape: [trials, 60, 32, 5].
    """
    raw_batch = np.asarray(raw_batch, dtype=np.float64)
    expected_points = (DEAP_BASELINE_SECONDS + DEAP_STIMULUS_SECONDS) * DEAP_SAMPLE_RATE
    if raw_batch.ndim != 3 or raw_batch.shape[1] < DEAP_NUM_CHANNELS:
        raise ValueError('Unexpected DEAP data shape: {}'.format(raw_batch.shape))
    if raw_batch.shape[2] < expected_points:
        raise ValueError(
            'DEAP trial is too short: {} points, expected at least {}.'
            .format(raw_batch.shape[2], expected_points)
        )

    eeg = raw_batch[:, :DEAP_NUM_CHANNELS, :expected_points]
    baseline_points = DEAP_BASELINE_SECONDS * DEAP_SAMPLE_RATE
    baseline = eeg[:, :, :baseline_points].reshape(
        eeg.shape[0], DEAP_NUM_CHANNELS, DEAP_BASELINE_SECONDS, DEAP_SAMPLE_RATE
    ).mean(axis=2)

    stimulus = eeg[:, :, baseline_points:].copy()
    stimulus = stimulus.reshape(
        eeg.shape[0], DEAP_NUM_CHANNELS, DEAP_STIMULUS_SECONDS, DEAP_SAMPLE_RATE
    )
    stimulus -= baseline[:, :, None, :]
    stimulus = stimulus.reshape(eeg.shape[0], DEAP_NUM_CHANNELS, -1)

    # LibEER: fifth-order 0.3-50 Hz band-pass before feature extraction.
    nyquist = 0.5 * DEAP_SAMPLE_RATE
    b_filter, a_filter = signal.butter(
        N=5, Wn=[0.3 / nyquist, 50.0 / nyquist], btype='bandpass'
    )
    stimulus = signal.filtfilt(b_filter, a_filter, stimulus, axis=-1)

    num_windows = DEAP_STIMULUS_SECONDS // DEAP_TIME_WINDOW_SECONDS
    window_size = DEAP_TIME_WINDOW_SECONDS * DEAP_SAMPLE_RATE
    de_data = np.zeros(
        (eeg.shape[0], num_windows, DEAP_NUM_CHANNELS, len(DEAP_EXTRACT_BANDS)),
        dtype=np.float64,
    )

    for band_idx, (low, high) in enumerate(DEAP_EXTRACT_BANDS):
        b_band, a_band = signal.butter(
            3, [low / nyquist, high / nyquist], btype='bandpass'
        )
        band_signal = signal.filtfilt(b_band, a_band, stimulus, axis=-1)
        band_windows = band_signal.reshape(
            eeg.shape[0], DEAP_NUM_CHANNELS, num_windows, window_size
        )
        variance = np.var(band_windows, axis=-1, ddof=1)
        # [trial, channel, time] -> [trial, time, channel]
        de_data[:, :, :, band_idx] = np.transpose(
            0.5 * np.log2(2.0 * np.pi * np.e * variance), (0, 2, 1)
        )

    for trial_idx in range(de_data.shape[0]):
        de_data[trial_idx] = _libeer_lds(de_data[trial_idx])

    return de_data


def _load_or_build_deap_features(dataset_root=None, rebuild_cache=False, batch_trials=5):
    """Load cached DE-LDS features, or build them from LibEER's DEAP .dat files."""
    root = _resolve_deap_root(dataset_root)
    cache_path = _resolve_deap_cache_path(root)
    runtime_key = os.path.realpath(root)

    # Even when rebuild_cache=True, rebuild only once per Python process.
    # This function is called once per LOSO fold.
    if runtime_key in _DEAP_FEATURE_RUNTIME_CACHE:
        return _DEAP_FEATURE_RUNTIME_CACHE[runtime_key]

    if os.path.isfile(cache_path) and not rebuild_cache:
        cached = np.load(cache_path, allow_pickle=False)
        features = cached['features'].astype(np.float32, copy=False)
        ratings = cached['ratings'].astype(np.float32, copy=False)
        _DEAP_FEATURE_RUNTIME_CACHE[runtime_key] = (features, ratings)
        return features, ratings

    subject_features = []
    subject_ratings = []
    for subject_id in range(1, DEAP_NUM_SUBJECTS + 1):
        file_path = os.path.join(root, 's{:02d}.dat'.format(subject_id))
        if not os.path.isfile(file_path):
            raise FileNotFoundError('Missing DEAP subject file: {}'.format(file_path))
        with open(file_path, 'rb') as handle:
            subject = pickle.load(handle, encoding='latin1')

        raw = np.asarray(subject['data'])
        ratings = np.asarray(subject['labels'], dtype=np.float32)
        if raw.shape[0] != DEAP_NUM_TRIALS or ratings.shape[0] != DEAP_NUM_TRIALS:
            raise ValueError(
                '{} has data/label shapes {} and {}; expected 40 trials.'
                .format(file_path, raw.shape, ratings.shape)
            )

        trial_features = []
        for begin in range(0, DEAP_NUM_TRIALS, batch_trials):
            end = min(begin + batch_trials, DEAP_NUM_TRIALS)
            de_lds = _extract_de_lds_deap_batch(raw[begin:end])
            # PCL-TDGCN flattening order is band first, electrode second.
            flat = np.transpose(de_lds, (0, 1, 3, 2)).reshape(
                -1, len(DEAP_EXTRACT_BANDS) * DEAP_NUM_CHANNELS
            )
            trial_features.append(flat)

        subject_feature = np.vstack(trial_features).astype(np.float32)
        if not np.isfinite(subject_feature).all():
            subject_feature = np.nan_to_num(
                subject_feature, nan=0.0, posinf=0.0, neginf=0.0
            )
        subject_features.append(subject_feature)
        subject_ratings.append(ratings)
        print('Prepared DEAP subject {:02d}/32: {}'.format(subject_id, subject_feature.shape))

    features = np.stack(subject_features, axis=0)
    ratings = np.stack(subject_ratings, axis=0)
    try:
        np.savez_compressed(cache_path, features=features, ratings=ratings)
        print('Saved DEAP DE-LDS cache to {}'.format(cache_path))
    except OSError as error:
        print('Warning: could not save DEAP cache {}: {}'.format(cache_path, error))
    _DEAP_FEATURE_RUNTIME_CACHE[runtime_key] = (features, ratings)
    return features, ratings


def _load_deap_task(dataset_name, dataset_root=None, rebuild_cache=False):
    """Return PCL-TDGCN data for DEAP-A or DEAP-V as one session × 32 subjects."""
    dataset_name = _canonical_dataset_name(dataset_name)
    if dataset_name not in ('deap_a', 'deap_v'):
        raise ValueError('Expected deap_a or deap_v, got {}'.format(dataset_name))

    root = _resolve_deap_root(dataset_root)
    task_key = (dataset_name, os.path.realpath(root))
    # Reuse the already prepared task on later LOSO folds in this process.
    if task_key in _DEAP_TASK_RUNTIME_CACHE:
        return _DEAP_TASK_RUNTIME_CACHE[task_key]

    features, ratings = _load_or_build_deap_features(
        dataset_root=root, rebuild_cache=rebuild_cache
    )
    rating_index = 1 if dataset_name == 'deap_a' else 0
    data = [[None for _ in range(DEAP_NUM_SUBJECTS)]]
    label = [[None for _ in range(DEAP_NUM_SUBJECTS)]]

    for subject_id in range(DEAP_NUM_SUBJECTS):
        subject_feature = features[subject_id]
        windows_per_trial = subject_feature.shape[0] // DEAP_NUM_TRIALS
        if windows_per_trial * DEAP_NUM_TRIALS != subject_feature.shape[0]:
            raise ValueError(
                'Subject {} has {} samples, not divisible by 40 trials.'
                .format(subject_id, subject_feature.shape[0])
            )

        # LibEER label_process(bounds=[5, 5]) maps rating <= 5 to 0 and > 5 to 1.
        trial_label = (ratings[subject_id, :, rating_index] > 5.0).astype(np.int64)
        sample_label = np.repeat(trial_label, windows_per_trial).reshape(-1, 1)

        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        subject_feature = scaler.fit_transform(subject_feature).astype(np.float32)
        data[0][subject_id] = subject_feature
        label[0][subject_id] = sample_label

    _DEAP_TASK_RUNTIME_CACHE[task_key] = (data, label)
    return data, label


def load_data(dataset_name, dataset_root=None, rebuild_cache=False):
    """
    Load PCL-TDGCN data while following LibEER's SEED/SEED-IV file layout.

    Returns
    -------
    data : list
        ``data[session][subject]``; each item is ``[N, 310]`` for SEED/SEED-IV
        and ``[N, 160]`` for DEAP.
    label : list
        ``label[session][subject]``; each item has shape ``[N, 1]``.
    """
    dataset_name = _canonical_dataset_name(dataset_name)

    if dataset_name in ('deap_a', 'deap_v'):
        return _load_deap_task(
            dataset_name, dataset_root=dataset_root, rebuild_cache=rebuild_cache
        )

    if dataset_name in ['seed3', 'seed4']:
        if dataset_root is not None:
            previous_path = dataset_path.get(dataset_name)
            dataset_path[dataset_name] = dataset_root
        else:
            previous_path = None
        try:
            path, allmats = get_allmats_name(dataset_name)
            data = [[None for _ in range(15)] for _ in range(3)]
            label = [[None for _ in range(15)] for _ in range(3)]

            for session_id, session_files in enumerate(allmats):
                for subject_id, filename in enumerate(session_files):
                    mat_path = _get_mat_path(
                        path, dataset_name, session_id, filename
                    )
                    one_data, one_label = get_data_label_frommat(
                        mat_path, dataset_name, session_id
                    )
                    data[session_id][subject_id] = one_data.copy()
                    label[session_id][subject_id] = one_label.copy()

            return data, label
        finally:
            if dataset_root is not None:
                if previous_path is None:
                    dataset_path.pop(dataset_name, None)
                else:
                    dataset_path[dataset_name] = previous_path

    if dataset_name == 'deafseed3':
        allmats = sorted(
            file for file in os.listdir(dataset_path[dataset_name])
            if file.lower().endswith('.mat')
        )
        data = [[None] * 15]
        label = [[None] * 15]
        for i, mat_name in enumerate(allmats):
            mat_path = os.path.join(dataset_path['deafseed3'], mat_name)
            mat_data = scio.loadmat(mat_path)
            one_sub_data = mat_data['XData']
            one_sub_label = mat_data['YLabel']
            one_sub_data = np.reshape(one_sub_data, [-1, 310]).astype(np.float32)
            one_sub_label = np.asarray(one_sub_label).astype(np.int64)
            min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
            one_sub_data = min_max_scaler.fit_transform(one_sub_data).astype(np.float32)
            data[0][i] = one_sub_data.copy()
            label[0][i] = one_sub_label.copy()
        return np.array(data, dtype=object), np.array(label, dtype=object)

    raise ValueError('Unexpected dataset name: {}'.format(dataset_name))


def pick_one_data(dataset_name, session_id=1, cd_count=4, sub_id=0):
    """Pick calibration and uncalibrated trials for one subject."""
    dataset_name = _canonical_dataset_name(dataset_name)
    path, allmats = get_allmats_name(dataset_name)
    mat_path = _get_mat_path(
        path, dataset_name, session_id, allmats[session_id][sub_id]
    )

    try:
        mat_data = scio.loadmat(mat_path, simplify_cells=True)
    except TypeError:
        mat_data = scio.loadmat(mat_path)
    mat_de_data = _extract_de_lds_trials(mat_data, dataset_name)

    cd_list = []
    ud_list = []
    number_trial, number_label, labels = get_number_of_label_n_trial(dataset_name)
    session_label_one_data = labels[session_id]

    for label_id in range(number_label):
        cd_list.extend(
            sample_by_value(
                session_label_one_data,
                label_id,
                int(cd_count / number_label),
            )
        )
    ud_list.extend([i for i in range(number_trial) if i not in cd_list])

    cd_label_list = copy.deepcopy(cd_list)
    ud_label_list = copy.deepcopy(ud_list)

    for i in range(len(cd_list)):
        trial_index = cd_list[i]
        cd_list[i] = mat_de_data[trial_index]
        cd_label_list[i] = labels[session_id][trial_index]

    for i in range(len(ud_list)):
        trial_index = ud_list[i]
        ud_list[i] = mat_de_data[trial_index]
        ud_label_list[i] = labels[session_id][trial_index]

    cd_data, cd_label = reshape_data(cd_list, cd_label_list)
    ud_data, ud_label = reshape_data(ud_list, ud_label_list)
    return cd_data, cd_label, ud_data, ud_label


class CustomDatasetWithIdx(Dataset):
    def __init__(self, Data1, Label, idx):
        self.Data1 = Data1
        self.Label = Label
        self.idx = idx

    def __len__(self):
        return len(self.Data1)

    def __getitem__(self, index):
        data1 = torch.Tensor(self.Data1[index])
        label = torch.LongTensor(self.Label[index])
        idx = torch.LongTensor(self.idx[index])
        return data1, label, idx


def create_logger(args):
    os.makedirs(args.output_log_dir, exist_ok=True)
    time_str = time.strftime('%m-%d-%H-%M')
    log_file = (
        'CrossSubject_C3DA_' + args.dataset + '_lr_' + str(args.lr)
        + '_seed_' + str(args.seed) + '_{}.log'.format(time_str)
    )
    final_log_file = os.path.join(args.output_log_dir, log_file)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = '[%(asctime)s] %(message)s'

    file_handler = logging.FileHandler(filename=final_log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S')
    )
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(
        logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S')
    )
    logger.addHandler(console)
    return logger


if __name__ == '__main__':
    data, label = load_data('seed3')
    data_tmp = copy.deepcopy(data)
    label_tmp = copy.deepcopy(label)
    for i in range(len(data_tmp)):
        for j in range(len(data_tmp[0])):
            data_tmp[i][j] = norminy(data_tmp[i][j])
