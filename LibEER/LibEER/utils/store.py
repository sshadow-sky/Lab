import argparse
import hashlib
import json
import os.path
import time
from datetime import datetime
from pathlib import Path

import torch


_RESULT_PARAM_SUFFIX = {}

# Keep path-friendly parameter display keys; full args are still serialized and hashed.
_DISPLAY_PARAM_KEYS = [
    'batch_size',
    'lr',
    'epochs',
    'seed',
    'feature_type',
    'time_window',
    'sample_length',
    'stride',
    'split_type',
    'experiment_mode',
    'label_used',
    'bounds',
]

# Exclude noisy or self-referential keys from full-path display to avoid path explosion.
_EXCLUDED_PARAM_KEYS = {
    'output_dir',
    'log_dir',
    'dataset_path',
    'data_dir',
    'time',
    'checkpoint',
}


def _sanitize_name(value):
    """Make text safe for filesystem names."""
    text = str(value)
    for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']:
        text = text.replace(ch, '-')
    return text


def _format_param_value(value):
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_format_param_value(v) for v in value) + "]"
    if value is None:
        return "None"
    return str(value)


def _collect_all_param_items(args):
    # Collect all public args for fully traceable path naming.
    items = []
    for key, value in sorted(vars(args).items(), key=lambda kv: kv[0]):
        if key.startswith('_'):
            continue
        if key in _EXCLUDED_PARAM_KEYS:
            continue
        items.append((key, _format_param_value(value)))
    return items


def _build_param_path(args):
    all_items = _collect_all_param_items(args)
    if not all_items:
        return Path("paramsNA")

    # Full serialization for deterministic identity.
    full_text = json.dumps({k: v for k, v in all_items}, ensure_ascii=False, sort_keys=True)
    digest = hashlib.md5(full_text.encode('utf-8')).hexdigest()[:12]

    # Human-readable short tag from key params.
    display_parts = []
    for key in _DISPLAY_PARAM_KEYS:
        if hasattr(args, key):
            value = _format_param_value(getattr(args, key))
            display_parts.append(f"{key}={value}")
    if not display_parts:
        display_parts = ["key-paramsNA"]

    text = _sanitize_name("__".join(display_parts))
    if len(text) > 160:
        text = text[:160]

    return Path(f"run-{text}__{digest}")


def _build_setting_name(args):
    if hasattr(args, 'setting') and args.setting:
        return _sanitize_name(args.setting)
    return _sanitize_name(f"{getattr(args, 'experiment_mode', 'mode')}_{getattr(args, 'split_type', 'split')}")


def _build_log_filename(args, timestamp):
    # Keep filename short/readable; full params are encoded in parent path.
    model_name = _sanitize_name(getattr(args, 'model', 'model'))
    return f"log_{model_name}_{timestamp}.txt"


def _build_result_filename(args, timestamp):
    model_name = _sanitize_name(getattr(args, 'model', 'model'))
    return f"result_{model_name}_{timestamp}.txt"


def _resolve_log_file(args):
    # Cache file path so one run appends into one file.
    if hasattr(args, '_log_file'):
        return Path(args._log_file)

    log_base = Path(args.log_dir)
    setting = _build_setting_name(args)
    param_path = _build_param_path(args)
    log_dir = log_base / _sanitize_name(getattr(args, 'dataset', 'dataset')) / _sanitize_name(getattr(args, 'model', 'model')) / setting / param_path
    add_dir(log_dir)

    # Use microseconds to avoid collisions when multiple runs start in one second.
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log_file = log_dir / _build_log_filename(args, timestamp)
    args._log_file = str(log_file)
    return log_file


def _resolve_result_file(args):
    # Cache result path so one run appends into one file.
    if hasattr(args, '_result_file'):
        return Path(args._result_file)

    output_base = Path(args.output_dir)
    setting = _build_setting_name(args)
    param_path = _build_param_path(args)
    result_dir = output_base / _sanitize_name(getattr(args, 'dataset', 'dataset')) / _sanitize_name(getattr(args, 'model', 'model')) / setting / param_path
    add_dir(result_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    result_file = result_dir / _build_result_filename(args, timestamp)
    args._result_file = str(result_file)
    return result_file


def _build_result_param_suffix(args):
    items = _collect_all_param_items(args)
    if not items:
        return "paramsNA"
    full_text = json.dumps({k: v for k, v in items}, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(full_text.encode('utf-8')).hexdigest()[:12]


def _resolve_result_param_suffix(output_dir):
    path = Path(output_dir)
    for candidate in [path] + list(path.parents):
        cache_key = str(candidate)
        if cache_key in _RESULT_PARAM_SUFFIX:
            return _RESULT_PARAM_SUFFIX[cache_key]
    return None

def make_output_dir(args, model):
    output_dir = Path(args.output_dir)
    # Save model checkpoints under dataset/model/setting/full-params.
    output_dir = output_dir / _sanitize_name(args.dataset) / _sanitize_name(model)
    if args.setting is not None:
        output_dir = output_dir / _sanitize_name(args.setting)
    else:
        output_dir = output_dir / _sanitize_name(args.experiment_mode)
        output_dir = output_dir / _sanitize_name(args.split_type)

    output_dir = output_dir / _build_param_path(args)

    _RESULT_PARAM_SUFFIX[str(output_dir)] = _build_result_param_suffix(args)
    return output_dir

def save_state(output_dir, model, optimizer, epoch, r_idx='last', rr_idx='last', metric=None, state='best'):
    # compatibility
    if type(output_dir) is argparse.Namespace:
        output_dir = make_output_dir(output_dir, output_dir.model)
    else:
        output_dir = Path(output_dir)
    if not ( r_idx == 'last' and rr_idx == 'last'):
        output_dir = output_dir / str(r_idx)
        output_dir = output_dir / str(rr_idx)

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"An error occurred: {e.strerror}")

    param_suffix = _resolve_result_param_suffix(output_dir)
    checkpoint_path = output_dir / f'checkpoint-{str(epoch)}' if metric is None \
        else output_dir / f'checkpoint-{state}{metric}'
    tagged_checkpoint_path = None
    if param_suffix is not None:
        tagged_checkpoint_path = Path(str(checkpoint_path) + f"-{param_suffix}")

    save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    # Keep original checkpoint names for existing loading logic.
    torch.save(save, checkpoint_path)
    print(f"save model to {checkpoint_path}")
    # Add a parameter-suffixed archive file as requested.
    if tagged_checkpoint_path is not None:
        torch.save(save, tagged_checkpoint_path)
        print(f"archive model to {tagged_checkpoint_path}")


def save_data(args, data, label):
    save_dir = Path(args.data_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    mode = {
        'subject-dependent': 'sub-dep',
        'subject-independent': 'sub-In',
        'cross-session': 'cro-sess'
    }
    save_path = save_dir / f'{args.dataset}'
    save_path = save_path / f'{args.feature_type}-tw-{args.time_window}ol-{args.overlap}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(f"Saving Processed Data To {save_path}")



def save_res(args, metric):
    log_file = _resolve_log_file(args)
    result_file = _resolve_result_file(args)

    # Persist full argument snapshot once per run for complete traceability.
    items = _collect_all_param_items(args)
    arg_snapshot = {k: v for k, v in items}
    arg_file_log = log_file.parent / "args_full.json"
    arg_file_result = result_file.parent / "args_full.json"
    if not os.path.exists(arg_file_log):
        with open(arg_file_log, 'w', encoding='utf-8') as f:
            json.dump(arg_snapshot, f, indent=2, ensure_ascii=False)
    if not os.path.exists(arg_file_result):
        with open(arg_file_result, 'w', encoding='utf-8') as f:
            json.dump(arg_snapshot, f, indent=2, ensure_ascii=False)

    # Write args once, then append metrics/results.
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write(str(args) + '\n')
    with open(log_file, 'a') as f:
        f.write('\n' + str(metric))

    if not os.path.exists(result_file):
        with open(result_file, 'w') as f:
            f.write(str(args) + '\n')
    with open(result_file, 'a') as f:
        f.write('\n' + str(metric))


def add_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
