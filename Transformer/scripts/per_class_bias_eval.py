#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.metrics import f1_score, classification_report

# Paths (edit if needed)
TRAIN = Path('/home/adamsas/iot_pcaps_features/all_features_with_mac_part1_80.jsonl')
PROM_LOGITS = Path('/tmp/logits_promoted_full.jsonl')
LABELS_SIDE = Path('analysis/final_models/model_gentle_unfreeze.ot.labels.json')
TEST = Path('/home/adamsas/iot_pcaps_features/all_features_with_mac_part2_20.jsonl')
OUT_DIR = Path('analysis/finetune_full_cw/ensemble_results')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load labels (canonical order)
labels = json.loads(LABELS_SIDE.read_text()) if LABELS_SIDE.exists() else None
if labels is None:
    raise SystemExit('labels sidecar not found at ' + str(LABELS_SIDE))
num_labels = len(labels)
label_to_idx = {l:i for i,l in enumerate(labels)}

# Compute training priors
train_counts = Counter()
if TRAIN.exists():
    with TRAIN.open() as fh:
        for ln, line in enumerate(fh, start=1):
            line=line.strip()
            if not line:
                continue
            j = json.loads(line)
            sm = j.get('s_mac')
            if sm is not None:
                train_counts[sm]+=1
else:
    print('Warning: train file not found at', TRAIN)

# Build prior vector aligned to labels
total = sum(train_counts.values())
priors = np.ones(num_labels, dtype=float) * 1e-12
if total>0:
    for l,idx in label_to_idx.items():
        priors[idx] = train_counts.get(l, 0) / total
else:
    # fallback: uniform
    priors[:] = 1.0/num_labels

# Read promoted logits (topk=full list expected)
prom_vecs = []
true_labels = []
with PROM_LOGITS.open() as fh:
    for line in fh:
        if not line.strip():
            continue
        j = json.loads(line)
        topk = j.get('topk', [])
        true = j.get('true')
        vec = np.zeros(num_labels, dtype=float)
        for lbl,prob in topk:
            if lbl in label_to_idx:
                vec[label_to_idx[lbl]] = prob
        # if probs don't sum to ~1 (maybe truncated), renormalize
        s = vec.sum()
        if s>0:
            vec = vec / s
        else:
            # fallback uniform
            vec[:] = 1.0/num_labels
        prom_vecs.append(vec)
        true_labels.append(true)

N = len(prom_vecs)
print(f'Loaded {N} promoted logits vectors')

# Beta grid to try
betas = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
results = {}
for beta in betas:
    preds = []
    for v in prom_vecs:
        # multiply by prior^beta: equivalent to add beta*log(prior) in logit space
        adj = v * (priors ** beta)
        s = adj.sum()
        if s>0:
            adj = adj / s
        else:
            adj = v
        preds.append(labels[int(adj.argmax())])
    # Evaluate against true_labels (some trues may be None)
    trues = true_labels
    # filter None entries
    paired = [(t,p) for t,p in zip(trues,preds) if t is not None]
    if not paired:
        print('No true labels available to evaluate')
        break
    tvec, pvec = zip(*paired)
    w = f1_score(tvec, pvec, average='weighted')
    m = f1_score(tvec, pvec, average='macro')
    results[beta] = {'weighted': float(w), 'macro': float(m)}
    print(f'beta={beta}: weighted={w:.6f}, macro={m:.6f}')

# Pick best beta by weighted F1
best_beta = max(results.keys(), key=lambda b: results[b]['weighted'])
best = results[best_beta]
print('\nBest beta by weighted-F1:', best_beta, best)

# Save results and best preds
OUT_DIR.joinpath('bias_results.json').write_text(json.dumps({'results':results, 'best_beta':best_beta, 'best':best}, indent=2))
# also save preds for best beta
preds_best = []
beta = best_beta
for v in prom_vecs:
    adj = v * (priors ** beta)
    s = adj.sum()
    if s>0:
        adj = adj / s
    else:
        adj = v
    preds_best.append(labels[int(adj.argmax())])
with (OUT_DIR / 'preds_promoted_bias_beta_{:.3f}.jsonl'.format(beta)).open('w') as fh:
    for p in preds_best:
        fh.write(json.dumps({'pred':p}) + '\n')

print('Saved bias results to', OUT_DIR)
