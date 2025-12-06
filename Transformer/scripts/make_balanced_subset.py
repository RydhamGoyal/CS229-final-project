#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

# Create a balanced subset by selecting classes with at least `min_count`
# and sampling up to `per_class` examples per selected class.

p = argparse.ArgumentParser()
p.add_argument("--in", dest="infile", required=True, type=Path)
p.add_argument("--out", dest="outfile", required=True, type=Path)
p.add_argument("--test-out", dest="test_outfile", required=False, type=Path)
p.add_argument("--per-class", dest="per_class", type=int, default=2000)
p.add_argument("--min-count", dest="min_count", type=int, default=2000)
p.add_argument("--seed", dest="seed", type=int, default=42)
args = p.parse_args()

random.seed(args.seed)

# First pass: collect indices (file offsets) per class and counts
class_lines = {}
lines = []
with args.infile.open('r') as fh:
    for i, line in enumerate(fh):
        line = line.rstrip('\n')
        if not line:
            continue
        try:
            j = json.loads(line)
        except Exception:
            continue
        s_mac = j.get('s_mac')
        if s_mac is None:
            continue
        class_lines.setdefault(s_mac, []).append(line)

# Select classes with at least min_count
selected = [c for c, lst in class_lines.items() if len(lst) >= args.min_count]
selected.sort()
print(f"Found {len(class_lines)} classes; selecting {len(selected)} classes with >= {args.min_count} samples")

# For each selected class, sample up to per_class lines
out_lines = []
for c in selected:
    lst = class_lines[c]
    if len(lst) <= args.per_class:
        out = lst.copy()
    else:
        out = random.sample(lst, args.per_class)
    out_lines.extend(out)

# Shuffle final output
random.shuffle(out_lines)

args.outfile.parent.mkdir(parents=True, exist_ok=True)
with args.outfile.open('w') as fh:
    for l in out_lines:
        fh.write(l + '\n')

# Optionally write a filtered test file containing only selected classes
if args.test_outfile:
    test_in_path = Path(str(args.infile).replace('part1_80', 'part2_20'))
    if test_in_path.exists():
        with test_in_path.open('r') as inf, args.test_outfile.open('w') as outf:
            for line in inf:
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                if j.get('s_mac') in selected:
                    outf.write(line)

# Also dump the selected class list for reference
sel_path = args.outfile.with_suffix('.classes.json')
with sel_path.open('w') as fh:
    json.dump(selected, fh, indent=2)

print(f"Wrote {len(out_lines)} samples to {args.outfile}")
print(f"Wrote selected classes to {sel_path}")
if args.test_outfile:
    print(f"Wrote filtered test file to {args.test_outfile}")
