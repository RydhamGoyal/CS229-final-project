#!/usr/bin/env python3
import json
import math
import random
from pathlib import Path
from typing import List, Dict
from sklearn.metrics import f1_score, classification_report
import numpy as np

# Paths
BASE = Path('/home/adamsas/retina_cato')
LOGIT_BASE = Path('/tmp/logits_conservative_full.jsonl')
LOGIT_PROM = Path('/tmp/logits_promoted_full.jsonl')
TEST = Path('/home/adamsas/iot_pcaps_features/all_features_with_mac_part2_20.jsonl')
OUT = Path('analysis/finetune_full_cw/ensemble_results')
OUT.mkdir(parents=True, exist_ok=True)

# Read trues
trues = [json.loads(l)['s_mac'] for l in open(TEST) if l.strip()]

# read topk as dicts of probs
def read_topk_dict(path: Path) -> List[Dict[str,float]]:
    res = []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            j = json.loads(line)
            topk = j.get('topk', [])
            res.append({lbl: prob for lbl, prob in topk})
    return res

base = read_topk_dict(LOGIT_BASE)
prom = read_topk_dict(LOGIT_PROM)
N = min(len(base), len(prom), len(trues))
base = base[:N]
prom = prom[:N]
trues = trues[:N]

# Build label set union (should be identical)
labels = sorted({lbl for d in base for lbl in d.keys()} | {lbl for d in prom for lbl in d.keys()})

# Helper to densify dict to vector
label_to_idx = {l:i for i,l in enumerate(labels)}

def dict_to_vec(d):
    v = np.zeros(len(labels), dtype=float)
    for k,p in d.items():
        v[label_to_idx[k]] = p
    return v

base_vecs = [dict_to_vec(d) for d in base]
prom_vecs = [dict_to_vec(d) for d in prom]

# Baseline predictions (argmax of base_vecs)
base_preds = [labels[int(v.argmax())] for v in base_vecs]
prom_preds = [labels[int(v.argmax())] for v in prom_vecs]

# Ensemble average (simple mean)
ens_preds = []
for b,p in zip(base_vecs, prom_vecs):
    avg = (b + p) / 2.0
    ens_preds.append(labels[int(avg.argmax())])

# Evaluate
results = {}
for name, preds in [('baseline', base_preds), ('promoted', prom_preds), ('ensemble', ens_preds)]:
    f1w = f1_score(trues, preds, average='weighted')
    f1m = f1_score(trues, preds, average='macro')
    results[name] = {'weighted': float(f1w), 'macro': float(f1m)}

# Temperature scaling: fit on a small validation split (use last 10% as val)
split = int(0.9 * N)
val_idx = range(split, N)
train_idx = range(0, split)

# We'll temperature-scale the ensemble logits (avg of probs turned to logits via log)
# For ensemble, compute logits = log(avg_probs + eps)
EPS = 1e-12
ens_logits = [np.log((b + p) / 2.0 + EPS) for b,p in zip(base_vecs, prom_vecs)]

# Search temperature via grid on validation set
def apply_temp(logits, T):
    scaled = [l / T for l in logits]
    probs = [np.exp(l - np.logaddexp.reduce(l)) for l in scaled]
    preds = [labels[int(p.argmax())] for p in probs]
    return preds

bestT = 1.0
bestW = results['ensemble']['weighted']
for T in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
    # compute on val
    preds = []
    for i in val_idx:
        l = ens_logits[i]
        s = l / T
        p = np.exp(s - np.logaddexp.reduce(s))
        preds.append(labels[int(p.argmax())])
    w = f1_score([trues[i] for i in val_idx], preds, average='weighted')
    if w > bestW:
        bestW = w
        bestT = T

# Apply best temperature to whole test set
ens_temp_preds = []
for l in ens_logits:
    s = l / bestT
    p = np.exp(s - np.logaddexp.reduce(s))
    ens_temp_preds.append(labels[int(p.argmax())])

results['ensemble_temp'] = {'weighted': float(f1_score(trues, ens_temp_preds, average='weighted')), 'macro': float(f1_score(trues, ens_temp_preds, average='macro')), 'T': float(bestT)}

# Save results
import json
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'results.json').write_text(json.dumps(results, indent=2))
print('Results saved to', OUT)
print(results)

# Save preds for inspection
with open(OUT / 'preds_baseline.jsonl', 'w') as fh:
    for p in base_preds:
        fh.write(json.dumps({'pred':p}) + '\n')
with open(OUT / 'preds_promoted.jsonl', 'w') as fh:
    for p in prom_preds:
        fh.write(json.dumps({'pred':p}) + '\n')
with open(OUT / 'preds_ensemble.jsonl', 'w') as fh:
    for p in ens_preds:
        fh.write(json.dumps({'pred':p}) + '\n')
with open(OUT / 'preds_ensemble_temp.jsonl', 'w') as fh:
    for p in ens_temp_preds:
        fh.write(json.dumps({'pred':p}) + '\n')
