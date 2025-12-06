#!/usr/bin/env python3
"""
Run a Rust-serialized SmartCore model on a JSONL test file and compute F1.

This script calls the Rust helper binary `run_model` (build with `cargo build --release -p serve_ml`) to
produce predictions (one JSON string per line). It then reads the test JSONL to obtain the
true `s_mac` labels and computes classification metrics using scikit-learn.

Usage:
  python3 scripts/run_model.py --model target/release/model.bin --labels model.bin.labels.json --test test.jsonl

If you built the helper binary, the default Rust binary path is `./target/release/run_model`.
If your binary is elsewhere, pass `--rust-bin /path/to/run_model`.
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import List

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


def run_rust_predict(rust_bin: Path, model_file: Path, labels_file: Path, test_file: Path) -> List[str]:
    cmd = [str(rust_bin), "--model-file", str(model_file), "--labels-file", str(labels_file), "--test-file", str(test_file)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"rust binary failed: {proc.returncode}\nstderr:\n{proc.stderr}")
    preds = []
    for ln, out_line in enumerate(proc.stdout.splitlines(), start=1):
        out_line = out_line.strip()
        if not out_line:
            continue
        # The Rust helper prints a JSON string per line (e.g. "aa:bb:cc:.."), so parse it
        try:
            val = json.loads(out_line)
        except Exception as e:
            raise RuntimeError(f"failed to parse prediction JSON on line {ln}: {e}\nline: {out_line}")
        preds.append(val)
    return preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--labels", required=True, type=Path)
    p.add_argument("--test", required=True, type=Path, help="JSONL file with test samples (must include s_mac)")
    p.add_argument("--rust-bin", required=False, type=Path, default=Path("./target/release/run_model"), help="Path to the compiled Rust helper binary")
    args = p.parse_args()

    trues = read_true_labels(args.test)
    preds = run_rust_predict(args.rust_bin, args.model, args.labels, args.test)

    if len(trues) != len(preds):
        raise RuntimeError(f"number of true labels ({len(trues)}) != number of predictions ({len(preds)})")

    # Compute macro and weighted F1 and print a report
    print(f"Samples: {len(trues)}")
    print("\nClassification report:")
    print(classification_report(trues, preds, digits=4))

    f1_macro = f1_score(trues, preds, average="macro")
    f1_weighted = f1_score(trues, preds, average="weighted")
    print(f"F1 (macro): {f1_macro:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}")


if __name__ == "__main__":
    main()
