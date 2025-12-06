#!/usr/bin/env python3
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
import numpy as np

TRAIN_LOGITS = Path('/tmp/features_train_promoted_logits.jsonl')
TEST_LOGITS = Path('/tmp/logits_promoted_full.jsonl')
LABELS_SIDE = Path('analysis/final_models/model_gentle_unfreeze.ot.labels.json')
OUT_DIR = Path('analysis/finetune_full_cw/ensemble_results')
OUT_DIR.mkdir(parents=True, exist_ok=True)

labels = json.loads(LABELS_SIDE.read_text())
label_to_idx = {l:i for i,l in enumerate(labels)}

def load_logits(path):
    X = []
    y = []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            j = json.loads(line)
            topk = j.get('topk', [])
            vec = np.zeros(len(labels), dtype=float)
            for lbl,p in topk:
                if lbl in label_to_idx:
                    vec[label_to_idx[lbl]] = p
            s = vec.sum()
            if s>0:
                vec = vec / s
            else:
                vec[:] = 1.0/len(labels)
            X.append(vec)
            y.append(j.get('true'))
    return np.array(X), np.array(y)

X_train, y_train = load_logits(TRAIN_LOGITS)
X_test, y_test = load_logits(TEST_LOGITS)
print('Shapes:', X_train.shape, X_test.shape)

# encode y to indices, filtering unknown labels
train_mask = [y in label_to_idx for y in y_train]
test_mask = [y in label_to_idx for y in y_test]
X_train = X_train[train_mask]
y_train = y_train[train_mask]
X_test = X_test[test_mask]
y_test = y_test[test_mask]

y_train_idx = np.array([label_to_idx[y] for y in y_train])
y_test_idx = np.array([label_to_idx[y] for y in y_test])

# Train multinomial logistic regression on probs features
clf = LogisticRegression(multi_class='multinomial', max_iter=1000, C=1.0, solver='lbfgs')
clf.fit(X_train, y_train_idx)

pred_idx = clf.predict(X_test)
preds = [labels[i] for i in pred_idx]

w = f1_score(y_test, preds, average='weighted')
m = f1_score(y_test, preds, average='macro')
print('Sklearn head distill weighted-F1', w, 'macro-F1', m)
print(classification_report(y_test, preds, digits=4))

OUT_DIR.joinpath('sklearn_distill_results.json').write_text(json.dumps({'weighted':float(w),'macro':float(m)}))
with open(OUT_DIR / 'preds_sklearn_distill.jsonl','w') as fh:
    for p in preds:
        fh.write(json.dumps({'pred':p}) + '\n')
