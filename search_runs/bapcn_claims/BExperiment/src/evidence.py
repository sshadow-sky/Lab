"""Export checkpoint evidence using full-source prototypes and frozen eval batches."""

from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from experiment_model import BAPCN


BANDS = ("delta", "theta", "alpha", "beta", "gamma")


@contextmanager
def preserve_rng_state():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path, content):
    Path(path).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _validate_split(checkpoint, subjects):
    split = checkpoint["split"]
    protocol = checkpoint["profile"]["protocol"]
    if protocol not in ("loso", "tvt"):
        raise ValueError(f"Unsupported protocol: {protocol}")
    for role in ("source", "target", "test"):
        ids = split[role]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"{role} subject IDs must be nonempty and unique")
        if any(not isinstance(sid, (int, np.integer)) or sid < 1 or sid not in subjects for sid in ids):
            raise ValueError(f"Missing or invalid one-based {role} subject IDs")
    if set(split["source"]) & (set(split["target"]) | set(split["test"])):
        raise ValueError("Source subjects overlap target or test subjects")
    if protocol == "tvt" and set(split["target"]) & set(split["test"]):
        raise ValueError("TVT target and test subjects overlap")
    if protocol == "loso" and (split["target"] != split["test"] or len(split["test"]) != 1):
        raise ValueError("LOSO requires one held-out subject shared by target and test")
    if checkpoint["profile"].get("sessions", [1]) != [1]:
        raise ValueError("This frozen exporter requires session 1; its model session index is zero")
    return protocol, split


@torch.no_grad()
def _encode_subjects(model, subjects, ids, device, batch_size):
    feature_parts, weight_parts, logits_parts = None, [], []
    label_parts, id_parts, index_parts, sample_ids, batches = [], [], [], [], []
    for sid in ids:
        x, labels = (np.asarray(value) for value in subjects[sid])
        if x.ndim != 3 or len(x) == 0 or labels.shape != (len(x),):
            raise ValueError(f"Invalid subject {sid} feature/label shape: {x.shape}, {labels.shape}")
        if not np.all(np.isfinite(x)) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"Subject {sid} requires finite features and integer class labels")
        if np.any(labels < 0) or np.any(labels >= model.num_classes):
            raise ValueError(f"Subject {sid} labels must be zero-based class indices")
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))
            inputs = torch.as_tensor(np.ascontiguousarray(x[start:stop]), dtype=torch.float32, device=device)
            subject_index = torch.full((stop - start,), int(sid) - 1, dtype=torch.long, device=device)
            encoded = model._encode_all(inputs, subject_index, torch.zeros_like(subject_index))
            features, weights = model.prototype_inputs(encoded)
            if feature_parts is None:
                feature_parts = [[] for _ in features]
            if len(features) != len(feature_parts):
                raise ValueError("Inconsistent prototype set count across encoding batches")
            for parts, feature in zip(feature_parts, features):
                parts.append(feature.detach().cpu())
            weight_parts.append(weights.detach().cpu())
            logits_parts.append(encoded["logits"].detach().cpu())
            batches.append({"subject_id": int(sid), "start_inclusive": start, "stop_exclusive": stop})
        label_parts.append(labels.astype(np.int64, copy=False))
        id_parts.append(np.full(len(x), int(sid), dtype=np.int64))
        index_parts.append(np.arange(len(x), dtype=np.int64))
        sample_ids.extend(f"subject_{sid:02d}:sample_{index:07d}" for index in range(len(x)))
    return {"features": [torch.cat(parts) for parts in feature_parts],
            "weights": torch.cat(weight_parts), "logits": torch.cat(logits_parts),
            "labels": np.concatenate(label_parts), "subject_ids": np.concatenate(id_parts),
            "sample_indices": np.concatenate(index_parts), "sample_ids": np.asarray(sample_ids),
            "batches": batches}


def _sample_csv(path, arrays, set_names):
    labels = arrays["labels"]
    band_predictions = arrays["band_posteriors"].argmax(axis=-1)
    consensus_predictions = arrays["consensus_posteriors"].argmax(axis=-1)
    classifier_predictions = arrays["classifier_posteriors"].argmax(axis=-1)
    fields = ["sample_id", "subject_id", "sample_index", "label", "consensus_prediction",
              "consensus_correct", "classifier_prediction", "classifier_correct"]
    for name in set_names:
        fields.extend((f"{name}_prediction", f"{name}_correct", f"{name}_weight"))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, label in enumerate(labels):
            row = {"sample_id": arrays["sample_ids"][index], "subject_id": int(arrays["subject_ids"][index]),
                   "sample_index": int(arrays["sample_indices"][index]), "label": int(label),
                   "consensus_prediction": int(consensus_predictions[index]),
                   "consensus_correct": int(consensus_predictions[index] == label),
                   "classifier_prediction": int(classifier_predictions[index]),
                   "classifier_correct": int(classifier_predictions[index] == label)}
            for band_index, name in enumerate(set_names):
                row.update({f"{name}_prediction": int(band_predictions[index, band_index]),
                            f"{name}_correct": int(band_predictions[index, band_index] == label),
                            f"{name}_weight": float(arrays["band_weights"][index, band_index])})
            writer.writerow(row)


def export_checkpoint(checkpoint_path, subjects, output_dir, device="cpu", batch_size=512):
    """Export a trusted locally generated checkpoint without mutating caller RNG/model state.

    LOSO writes predictions.npz in output_dir. TVT writes target/ and test/ exports.
    The returned mapping contains each cohort's predictions, metadata, and CSV paths.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    checkpoint_path, output_dir = Path(checkpoint_path).resolve(), Path(output_dir).resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Refusing to overwrite existing evidence: {output_dir}")
    device = torch.device(device)
    with preserve_rng_state():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        required = {"model_state", "net_params", "prototype_space", "profile", "split", "epoch", "checkpoint_tag"}
        missing = required - set(checkpoint)
        if missing:
            raise ValueError(f"Checkpoint is missing required fields: {sorted(missing)}")
        protocol, split = _validate_split(checkpoint, subjects)
        parameters = dict(checkpoint["net_params"])
        parameters["DEVICE"] = device
        model = BAPCN(parameters, prototype_space=checkpoint["prototype_space"]).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.subject_grl.iter_num = int(checkpoint.get("grl_iter", 0))
        model.eval()
        source = _encode_subjects(model, subjects, split["source"], device, batch_size)
        source_labels = torch.as_tensor(source["labels"], dtype=torch.long)
        source_class_counts = np.bincount(source["labels"], minlength=model.num_classes)
        set_names = list(BANDS) if checkpoint["prototype_space"] == "band" else ["fused"]
        if len(source["features"]) != len(set_names):
            raise ValueError("Export requires five canonical band sets or one fused set")
        local_root = Path(__file__).resolve().parent
        common = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": _sha256(checkpoint_path),
                  "checkpoint_tag": str(checkpoint["checkpoint_tag"]), "epoch": int(checkpoint["epoch"]),
                  "grl_iter": int(checkpoint.get("grl_iter", 0)), "model_digest": checkpoint.get("model_digest"),
                  "local_model_sha256": {name: _sha256(local_root / name)
                                          for name in ("baseline_model.py", "experiment_model.py", "evidence.py")},
                  "protocol": protocol, "dataset": checkpoint["profile"].get("key"),
                  "prototype_space": checkpoint["prototype_space"], "prototype_set_names": set_names,
                  "split": split, "subject_ids_one_based": True, "class_labels_zero_based": True,
                  "session": 1, "model_session_index": 0, "device": str(device), "batch_size": int(batch_size),
                  "drop_last": False, "shuffle": False, "source_n_samples": int(len(source_labels)),
                  "source_class_counts": source_class_counts.tolist(), "source_batches": source["batches"],
                  "prototype_construction": "All source samples: normalize each feature, mean within class, normalize each class mean.",
                  "normalization": "Original eval normalization within each fixed subject batch; no stored global normalization statistics.",
                  "posterior_helper": "BAPCN.source_prototype_posteriors", "epsilon": float(model.ugfcda_eps),
                  "temperature": float(model.temperature),
                  "consensus_definition": "Sum of prototype-set posteriors weighted by helper-returned weights; training epsilon retained.",
                  "target_label_use": "Diagnostics only; target/test labels never enter encoding, prototypes, or posterior computation.",
                  "sample_selection": "All target/test samples, without a reliability threshold filter.",
                  "rng_policy": "Restore caller Python, NumPy, Torch CPU and CUDA RNG states after fresh-model export."}
        cohorts = ("test",) if protocol == "loso" else ("target", "test")
        output_dir.mkdir(parents=True, exist_ok=True)
        result = {"source_prototypes": str(output_dir / "source_prototypes.npz"),
                  "metadata": str(output_dir / "export_metadata.json")}
        for cohort in cohorts:
            target = _encode_subjects(model, subjects, split[cohort], device, batch_size)
            evidence = model.source_prototype_posteriors(source["features"], source_labels,
                                                        target["features"], target["weights"])
            arrays = {"band_posteriors": evidence["band_posteriors"].cpu().numpy(),
                      "band_weights": evidence["weights"].cpu().numpy(),
                      "consensus_posteriors": evidence["consensus_posterior"].cpu().numpy(),
                      "classifier_posteriors": torch.softmax(target["logits"], dim=-1).numpy(),
                      "labels": target["labels"], "subject_ids": target["subject_ids"],
                      "sample_indices": target["sample_indices"], "sample_ids": target["sample_ids"],
                      "checkpoint_tag": np.asarray(str(checkpoint["checkpoint_tag"])),
                      "source_subject_ids": np.asarray(split["source"], dtype=np.int64),
                      "source_class_counts": source_class_counts,
                      "source_prototypes": evidence["prototypes"].cpu().numpy(),
                      "valid_classes": evidence["valid_classes"].cpu().numpy()}
            cohort_dir = output_dir if protocol == "loso" else output_dir / cohort
            cohort_dir.mkdir(parents=True, exist_ok=True)
            npz_path, csv_path = cohort_dir / "predictions.npz", cohort_dir / "sample_predictions.csv"
            metadata_path = cohort_dir / "metadata.json"
            np.savez_compressed(npz_path, **arrays)
            _sample_csv(csv_path, arrays, set_names)
            metadata = {**common, "cohort": cohort, "n_samples": int(len(target["labels"])),
                        "subject_ids": [int(sid) for sid in split[cohort]], "batches": target["batches"],
                        "predictions_sha256": _sha256(npz_path), "valid_classes": arrays["valid_classes"].tolist()}
            _json_write(metadata_path, metadata)
            result[cohort] = {"predictions": str(npz_path), "metadata": str(metadata_path), "csv": str(csv_path)}
            if cohort == cohorts[0]:
                np.savez_compressed(result["source_prototypes"],
                                    source_subject_ids=arrays["source_subject_ids"],
                                    source_sample_subject_ids=source["subject_ids"],
                                    source_sample_indices=source["sample_indices"],
                                    source_class_counts=source_class_counts,
                                    source_prototypes=arrays["source_prototypes"], valid_classes=arrays["valid_classes"])
        _json_write(result["metadata"], {**common, "exports": result})
        return result


if __name__ == "__main__":
    raise SystemExit("Use run.py --dataset <dataset> --run-dir <existing-run> --export-only")
