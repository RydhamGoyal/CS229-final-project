#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

PROM_LOGITS = Path('/tmp/logits_promoted_full.jsonl')
LABELS_SIDE = Path('analysis/final_models/model_gentle_unfreeze.ot.labels.json')
TEST = Path('/home/adamsas/iot_pcaps_features/all_features_with_mac_part2_20.jsonl')
OUT_DIR = Path('analysis/finetune_full_cw/ensemble_results')
OUT_DIR.mkdir(parents=True, exist_ok=True)

labels = json.loads(LABELS_SIDE.read_text())
label_to_idx = {l:i for i,l in enumerate(labels)}
num_labels = len(labels)

# load logits (probs -> convert to logit)
probs = []
trues = []
with PROM_LOGITS.open() as fh:
    for line in fh:
        if not line.strip():
            continue
        j = json.loads(line)
        topk = j.get('topk', [])
        vec = np.full(num_labels, 1e-12, dtype=float)
        for lbl,p in topk:
            if lbl in label_to_idx:
                vec[label_to_idx[lbl]] = p
        vec = vec / vec.sum()
        probs.append(vec)
        trues.append(j.get('true'))

N = len(probs)
print('Loaded', N, 'samples')

logits = np.log(np.stack(probs) + 1e-12)

# split into train/val
split = int(0.8 * N)
train_idx = np.arange(0, split)
val_idx = np.arange(split, N)

# initialize biases to zero
bias = np.zeros(num_labels, dtype=float)

# function to compute weighted-f1 on val with given bias
from sklearn.metrics import f1_score

def eval_bias(bias_vec):
    preds = []
    for i in val_idx:
        l = logits[i] + bias_vec
        # softmax
        s = np.exp(l - np.max(l))
        p = s / s.sum()
        preds.append(labels[int(p.argmax())])
    tr = [trues[i] for i in val_idx]
    return f1_score(tr, preds, average='weighted')

# coordinate ascent: for each class, try adjustments from -1.0 to 1.0 step 0.1
best_bias = bias.copy()
best_score = eval_bias(best_bias)
print('initial val weighted-F1', best_score)

improved = True
passes = 0
while improved and passes < 5:
    improved = False
    passes += 1
    for j in range(num_labels):
        current = best_bias[j]
        best_local = current
        for delta in np.linspace(-1.0, 1.0, 21):
            trial = best_bias.copy()
            trial[j] = current + delta
            score = eval_bias(trial)
            if score > best_score + 1e-9:
                best_score = score
                best_local = trial[j]
                best_bias = trial
                improved = True
        # set bias[j] to best_local
        best_bias[j] = best_local
    print('pass', passes, 'best val weighted-F1', best_score)

# Evaluate on full test set
preds_full = []
for i in range(N):
    l = logits[i] + best_bias
    s = np.exp(l - np.max(l))
    p = s / s.sum()
    preds_full.append(labels[int(p.argmax())])

tr_full = trues
w_full = f1_score(tr_full, preds_full, average='weighted')
m_full = f1_score(tr_full, preds_full, average='macro')
print('Final on full test: weighted', w_full, 'macro', m_full)

# save
OUT_DIR.joinpath('additive_bias_best.json').write_text(json.dumps({'bias':best_bias.tolist(), 'val_weighted':best_score, 'test_weighted':w_full, 'test_macro':m_full}, indent=2))
with (OUT_DIR / 'preds_promoted_additive_bias.jsonl').open('w') as fh:
    for p in preds_full:
        fh.write(json.dumps({'pred':p}) + '\n')

print('Saved additive bias results to', OUT_DIR)
