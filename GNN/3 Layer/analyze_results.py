import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import train  # our training module

# ----------------------------
# Reuse device and constants
# ----------------------------
device = train.device
num_classes = train.num_classes

# ----------------------------
# Recreate the model and load best weights
# ----------------------------
model = train.SAGEModelDeep(
    in_dim=train.num_features,
    hidden_dim=train.GONE_HIDDEN_DIM if hasattr(train, "GONE_HIDDEN_DIM") else train.HIDDEN_DIM,
    num_classes=num_classes,
    dropout=train.DROPOUT,
).to(device)

state_dict = torch.load("best_model.pt", map_location=device)
model.load_state_dict(state_dict)
model.eval()

# ----------------------------
# Reuse test data & graph from train.py
# ----------------------------
X_test = train.X_test          # [N_test, num_features], on device
y_test = train.y_test          # [N_test], on device
edge_index_test = train.edge_index_test  # [2, E_test], on device

with torch.no_grad():
    out_test = model(X_test, edge_index_test)
    pred_test = out_test.argmax(dim=1)

y_test_np = y_test.cpu().numpy()
pred_test_np = pred_test.cpu().numpy()

from sklearn.metrics import f1_score, accuracy_score

# Sanity-check metrics
acc = accuracy_score(y_test_np, pred_test_np)
macro_f1 = f1_score(y_test_np, pred_test_np, average="macro", zero_division=0)
weighted_f1 = f1_score(y_test_np, pred_test_np, average="weighted", zero_division=0)

print(f"[ANALYZE] Accuracy:   {acc:.3f}")
print(f"[ANALYZE] Macro-F1:   {macro_f1:.3f}")
print(f"[ANALYZE] Weighted-F1:{weighted_f1:.3f}")

# ============================================================
# 1. Normalized Confusion Matrix (Recall)
# ============================================================

cm = confusion_matrix(y_test_np, pred_test_np, labels=list(range(num_classes)))
cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

plt.figure(figsize=(12, 10))
plt.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
plt.title("Normalized Confusion Matrix (Per-Class Recall)")
plt.colorbar(label="Recall (row-normalized)")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.tight_layout()
plt.savefig("confusion_matrix_normalized.png")
plt.close()
print("Saved confusion_matrix_normalized.png")

# ============================================================
# 2. Precision Matrix (Column-normalized) WITH TEXT ANNOTATION
# ============================================================

cm_prec = cm.astype(float) / (cm.sum(axis=0, keepdims=True) + 1e-9)

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(cm_prec, interpolation="nearest", cmap="Greens", vmin=0, vmax=1)

# Title, labels
ax.set_title("Precision Matrix (Column-normalized)")
fig.colorbar(im, ax=ax, label="Precision")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")

# === ADD TEXT TO EACH CELL ===
num_classes_local = cm_prec.shape[0]
for i in range(num_classes_local):
    for j in range(num_classes_local):
        value = cm_prec[i, j]
        text_color = "black" if value < 0.5 else "white"
        ax.text(
            j, i,
            f"{value:.2f}",          # 2 decimals
            ha="center",
            va="center",
            color=text_color,
            fontsize=8
        )

plt.tight_layout()
plt.savefig("precision_matrix.png")
plt.close()
print("Saved precision_matrix.png with annotations")


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
plt.title("Per-Class Precision / Recall / F1")
plt.legend()
plt.tight_layout()
plt.savefig("per_class_metrics.png")
plt.close()
print("Saved per_class_metrics.png")

# ============================================================
# 4. Test Class Support
# ============================================================

plt.figure(figsize=(16, 5))
plt.bar(x, support)
plt.xlabel("Class label")
plt.ylabel("Number of test samples")
plt.title("Class Support in Test Set")
plt.tight_layout()
plt.savefig("class_support.png")
plt.close()
print("Saved class_support.png")

print("All analysis figures generated successfully.")