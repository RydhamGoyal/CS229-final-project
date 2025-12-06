#!/usr/bin/env python3
"""Split a JSON Lines (jsonl) file into two jsonl files by percentage.

Usage examples are in the README.md.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split a JSONL file into two JSONL files by percentage")
    p.add_argument("input", help="Path to input .jsonl file")
    p.add_argument("percentage", type=float, help="Percentage (0-100) of lines to put into the first output file")
    p.add_argument("--output1", help="Path for first output file (defaults to <input>_part1_<pct>.jsonl)")
    p.add_argument("--output2", help="Path for second output file (defaults to <input>_part2_<pct>.jsonl)")
    p.add_argument("--shuffle", dest="shuffle", action="store_true", help="Shuffle lines before splitting")
    p.add_argument("--seed", type=int, default=None, help="Optional random seed used when shuffling (only used if --shuffle is set)")
    return p.parse_args()


def read_jsonl(path: str) -> List[str]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            line = raw.rstrip("\n")
            if not line:
                # skip empty lines
                continue
            # basic validation: ensure this is valid JSON
            try:
                json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSON on line {i} of {path}: {e}")
            lines.append(line)
    return lines


def write_jsonl(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
            f.write("\n")


def make_default_paths(input_path: str, pct: float) -> Tuple[str, str]:
    base = os.path.splitext(os.path.basename(input_path))[0]
    out1 = f"{base}_part1_{int(round(pct))}.jsonl"
    out2 = f"{base}_part2_{int(round(pct))}.jsonl"
    return out1, out2


def main() -> int:
    args = parse_args()

    # Validate percentage
    pct = args.percentage
    if not (0.0 <= pct <= 100.0):
        print("Error: percentage must be between 0 and 100", file=sys.stderr)
        return 2

    try:
        lines = read_jsonl(args.input)
    except Exception as e:
        print(f"Error reading input: {e}", file=sys.stderr)
        return 3

    n = len(lines)
    if n == 0:
        print("Warning: input file contains no JSON objects (0 lines). Producing two empty files.")

    # Optionally shuffle (default: keep original order)
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(lines)

    # Compute split index
    idx = int(round(n * (pct / 100.0)))

    part1 = lines[:idx]
    part2 = lines[idx:]

    out1 = args.output1
    out2 = args.output2
    if not out1 or not out2:
        default1, default2 = make_default_paths(args.input, pct)
        if not out1:
            out1 = default1
        if not out2:
            out2 = default2

    try:
        write_jsonl(out1, part1)
        write_jsonl(out2, part2)
    except Exception as e:
        print(f"Error writing outputs: {e}", file=sys.stderr)
        return 4

    print(f"Input: {args.input}")
    print(f"Total JSON lines: {n}")
    print(f"Percentage for first file: {pct}% -> {len(part1)} lines")
    print(f"Wrote: {out1}")
    print(f"Wrote: {out2}")
    if args.shuffle:
        seed_msg = f" (seed={args.seed})" if args.seed is not None else ""
        print(f"Shuffled before splitting{seed_msg}.")
    else:
        print("Kept original order (no shuffle).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
