#!/usr/bin/env python3
"""
Run the Rust `run_transformer` helper binary on a JSONL test file and compute F1.

This script calls the Rust helper binary `run_transformer` (built from
`examples/serve_ml/src/bin/run_transformer.rs`) to produce predictions (one JSON value per line).
It reads the test JSONL to obtain the true `s_mac` labels and computes classification metrics
using scikit-learn.

Usage:
  python3 scripts/run_transformer.py --weights ./model_weights.ot --test test.jsonl

If you built the helper binary in debug mode, the default Rust binary path is `./target/debug/run_transformer`.
If you built release, pass `--rust-bin /path/to/run_transformer`.
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Optional

from sklearn.metrics import f1_score, classification_report


def read_true_labels(test_path: Path) -> List[str]:
    trues = []
    with test_path.open("r") as fh:
        for ln, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            if "s_mac" not in j:
                raise ValueError(f"missing s_mac on line {ln} in {test_path}")
            trues.append(j["s_mac"])
    return trues


def load_meta(weights: Path) -> Optional[dict]:
    meta_path = weights.with_suffix(weights.suffix + ".meta.json")
    # If weights is like model.ot, weights.suffix = '.ot', so meta file becomes 'model.ot.meta.json'
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return None


def build_cmd(rust_bin: Path, weights: Path, labels: Path, test_file: Path, meta: Optional[dict]) -> List[str]:
    cmd = [str(rust_bin), "--weights", str(weights), "--labels-file", str(labels), "--test-file", str(test_file)]

    # If meta is present, map known hyperparameters to CLI flags
    if meta:
        # mapping: meta keys -> CLI flags (clap converts underscores to hyphens)
        mapping = {
            "num_variates": "--num-variates",
            "lookback_len": "--lookback-len",
            "depth": "--depth",
            "dim": "--dim",
            "num_tokens_per_variate": "--num-tokens-per-variate",
            "pred_length": "--pred-length",
            "dim_head": "--dim-head",
            "heads": "--heads",
        }
        for key, flag in mapping.items():
            if key in meta and meta[key] is not None:
                val = meta[key]
                # pred_length may be an array in meta; convert to comma-separated
                if key == "pred_length" and isinstance(val, list):
                    val = ",".join(str(int(x)) for x in val)
                cmd += [flag, str(val)]

        # boolean flags
        if meta.get("use_reversible_instance_norm", False):
            cmd.append("--use-reversible-instance-norm")
        if meta.get("flash_attn", False):
            cmd.append("--flash-attn")

    return cmd


def run_rust_predict(rust_bin: Path, weights: Path, labels: Path, test_file: Path) -> List[str]:
    meta = load_meta(weights)
    cmd = build_cmd(rust_bin, weights, labels, test_file, meta)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"rust binary failed: {proc.returncode}\nstderr:\n{proc.stderr}\ncmd: {cmd}")
    preds = []
    for ln, out_line in enumerate(proc.stdout.splitlines(), start=1):
        out_line = out_line.strip()
        if not out_line:
            continue
        try:
            val = json.loads(out_line)
        except Exception as e:
            # If output is raw (not JSON), treat the entire line as the prediction string
            val = out_line
        preds.append(val)
    return preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--labels", required=False, type=Path, help="Optional labels sidecar JSON file (defaults to <weights>.labels.json)")
    p.add_argument("--test", required=True, type=Path, help="JSONL file with test samples (must include s_mac)")
    p.add_argument("--rust-bin", required=False, type=Path, default=Path("./target/debug/run_transformer"), help="Path to the compiled Rust helper binary")
    args = p.parse_args()

    weights = args.weights
    labels = args.labels if args.labels else Path(str(weights) + ".labels.json")

    if not labels.exists():
        raise FileNotFoundError(f"labels sidecar not found at {labels}; train_transformer writes <weights>.labels.json")

    trues = read_true_labels(args.test)
    preds = run_rust_predict(args.rust_bin, weights, labels, args.test)

    if len(trues) != len(preds):
        raise RuntimeError(f"number of true labels ({len(trues)}) != number of predictions ({len(preds)})")

    print(f"Samples: {len(trues)}")
    print("\nClassification report:")
    print(classification_report(trues, preds, digits=4))

    f1_macro = f1_score(trues, preds, average="macro")
    f1_weighted = f1_score(trues, preds, average="weighted")
    print(f"F1 (macro): {f1_macro:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}")


if __name__ == "__main__":
    main()
