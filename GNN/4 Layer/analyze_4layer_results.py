import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    f1_score,
    accuracy_score,
)

# Import the 4-layer training module in this folder
import train_4layer as train4

# ----------------------------
# Reuse device and constants
# ----------------------------
device = train4.device
num_classes = train4.num_classes
num_features = train4.num_features
HIDDEN_DIM = train4.HIDDEN_DIM
DROPOUT = train4.DROPOUT

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------
# Recreate the 4-layer model and load best weights
# ----------------------------
model = train4.SAGEModelDeep4Layer(
    in_dim=num_features,
    hidden_dim=HIDDEN_DIM,
    num_classes=num_classes,
    dropout=DROPOUT,
).to(device)

state_path = os.path.join(BASE_DIR, "best_model_4layer.pt")
state_dict = torch.load(state_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# ----------------------------
# Reuse test data & graph from train_4layer.py
# ----------------------------
X_test = train4.X_test
y_test = train4.y_test
edge_index_test = train4.edge_index_test

with torch.no_grad():
    out_test = model(X_test, edge_index_test)
    pred_test = out_test.argmax(dim=1)

y_test_np = y_test.cpu().numpy()
pred_test_np = pred_test.cpu().numpy()

# Sanity-check metrics
acc = accuracy_score(y_test_np, pred_test_np)
macro_f1 = f1_score(y_test_np, pred_test_np, average="macro", zero_division=0)
weighted_f1 = f1_score(y_test_np, pred_test_np, average="weighted", zero_division=0)

print(f"[ANALYZE-4LAYER] Accuracy:   {acc:.3f}")
print(f"[ANALYZE-4LAYER] Macro-F1:   {macro_f1:.3f}")
print(f"[ANALYZE-4LAYER] Weighted-F1:{weighted_f1:.3f}")

# ============================================================
# 1. Normalized Confusion Matrix (Recall)
# ============================================================

cm = confusion_matrix(y_test_np, pred_test_np, labels=list(range(num_classes)))
cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

plt.figure(figsize=(12, 10))
plt.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
plt.title("Normalized Confusion Matrix (Per-Class Recall) - 4-layer")
plt.colorbar(label="Recall (row-normalized)")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.tight_layout()
cm_norm_path = os.path.join(BASE_DIR, "confusion_matrix_normalized_4layer.png")
plt.savefig(cm_norm_path)
plt.close()
print(f"Saved {cm_norm_path}")

# ============================================================
# 2. Precision Matrix (Column-normalized) WITH TEXT ANNOTATION
# ============================================================

cm_prec = cm.astype(float) / (cm.sum(axis=0, keepdims=True) + 1e-9)

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(cm_prec, interpolation="nearest", cmap="Greens", vmin=0, vmax=1)

ax.set_title("Precision Matrix (Column-normalized) - 4-layer")
fig.colorbar(im, ax=ax, label="Precision")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")

# Add the numeric value inside each cell
num_classes_local = cm_prec.shape[0]
for i in range(num_classes_local):
    for j in range(num_classes_local):
        value = cm_prec[i, j]
        text_color = "black" if value < 0.5 else "white"
        ax.text(
            j, i,
            f"{value:.2f}",
            ha="center",
            va="center",
            color=text_color,
            fontsize=8,
        )

plt.tight_layout()
prec_path = os.path.join(BASE_DIR, "precision_matrix_4layer.png")
plt.savefig(prec_path)
plt.close()
print(f"Saved {prec_path}")

# ============================================================
# 3. Per-Class Precision, Recall, F1
# ============================================================

prec, rec, f1, support = precision_recall_fscore_support(
    y_test_np, pred_test_np, labels=list(range(num_classes)), zero_division=0
)

x = np.arange(num_classes)

plt.figure(figsize=(18, 6))
plt.bar(x - 0.25, prec, width=0.25, label="Precision")
plt.bar(x, rec, width=0.25, label="Recall")
plt.bar(x + 0.25, f1, width=0.25, label="F1-score")
plt.xlabel("Class label")
plt.ylabel("Score")
plt.title("Per-Class Precision / Recall / F1 - 4-layer")
plt.legend()
plt.tight_layout()
per_class_path = os.path.join(BASE_DIR, "per_class_metrics_4layer.png")
plt.savefig(per_class_path)
plt.close()
print(f"Saved {per_class_path}")

# ============================================================
# 4. Test Class Support
# ============================================================

plt.figure(figsize=(16, 5))
plt.bar(x, support)
plt.xlabel("Class label")
plt.ylabel("Number of test samples")
plt.title("Class Support in Test Set - 4-layer")
plt.tight_layout()
support_path = os.path.join(BASE_DIR, "class_support_4layer.png")
plt.savefig(support_path)
plt.close()
print(f"Saved {support_path}")

print("All 4-layer analysis figures generated successfully.")