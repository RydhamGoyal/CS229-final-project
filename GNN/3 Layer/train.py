import os
import json
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt

# -------------------------
# 0. Repro + device
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# 1. Paths + feature keys
# -------------------------
TRAIN_JSONL = "Dataset/all_features_with_mac_part1_80.jsonl"
TEST_JSONL  = "Dataset/all_features_with_mac_part2_20.jsonl"
DEVICES_TXT = "Dataset/List_Of_Devices.txt"

FEATURE_KEYS = [
    "dur", "proto",
    "s_bytes_sum", "d_bytes_sum",
    "s_ttl_mean", "d_ttl_mean",
    "s_load", "d_load",
    "s_pkt_cnt", "d_pkt_cnt",
    "s_bytes_mean", "d_bytes_mean",
    "s_iat_mean", "d_iat_mean",
    "tcp_rtt", "syn_ack", "ack_dat",
]

# -------------------------
# 2. Device mapping (MAC -> name / index)
# -------------------------
def load_device_mapping(txt_path):
    mac_to_name = {}
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("List of Devices"):
                continue
            parts = line.split("\t")
            parts = [p for p in parts if p.strip() != ""]
            if len(parts) < 2:
                continue
            name = parts[0].strip()
            mac = parts[1].strip().lower()
            mac_to_name[mac] = name

    # Assign indices
    mac_to_idx = {}
    idx_to_mac = {}
    for i, mac in enumerate(sorted(mac_to_name.keys())):
        mac_to_idx[mac] = i
        idx_to_mac[i] = mac
    print(f"Loaded {len(mac_to_idx)} devices from {txt_path}")
    return mac_to_name, mac_to_idx, idx_to_mac


mac_to_name, mac_to_idx, idx_to_mac = load_device_mapping(DEVICES_TXT)

# -------------------------
# 3. Load JSONL into DataFrame
# -------------------------
def load_flows_jsonl(jsonl_path, mac_to_idx, feature_keys):
    """
    Returns a DataFrame with columns: feature_keys + ['label']
    where label is the device index from mac_to_idx based on s_mac.
    """
    rows = []
    skipped_unknown_mac = 0
    skipped_missing_feats = 0

    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            mac = obj.get("s_mac", "").lower()
            if mac not in mac_to_idx:
                skipped_unknown_mac += 1
                continue

            feats = []
            bad = False
            for k in feature_keys:
                v = obj.get(k, None)
                if v is None:
                    # impute with 0 for missing vals
                    v = 0.0
                try:
                    feats.append(float(v))
                except Exception:
                    bad = True
                    break
            if bad:
                skipped_missing_feats += 1
                continue

            feats.append(mac_to_idx[mac])
            rows.append(feats)

    cols = feature_keys + ["label"]
    df = pd.DataFrame(rows, columns=cols)
    print(f"Loaded {len(df)} flows from {jsonl_path}")
    print(f"  Skipped unknown MACs: {skipped_unknown_mac}")
    print(f"  Skipped malformed / missing features: {skipped_missing_feats}")
    return df


print("\n--- Loading train & test flows ---")
df_train = load_flows_jsonl(TRAIN_JSONL, mac_to_idx, FEATURE_KEYS)
df_test  = load_flows_jsonl(TEST_JSONL,  mac_to_idx, FEATURE_KEYS)

# -------------------------
# 4. Build torch tensors
# -------------------------
def df_to_tensors(df, feature_keys):
    X = torch.tensor(df[feature_keys].values, dtype=torch.float32)
    y = torch.tensor(df["label"].values, dtype=torch.long)
    return X, y


X_train_raw, y_train = df_to_tensors(df_train, FEATURE_KEYS)
X_test_raw,  y_test  = df_to_tensors(df_test,  FEATURE_KEYS)

num_nodes_train = X_train_raw.size(0)
num_nodes_test  = X_test_raw.size(0)
num_features    = X_train_raw.size(1)
num_classes     = int(max(y_train.max(), y_test.max()).item() + 1)

print(f"\nTrain nodes: {num_nodes_train}, Test nodes: {num_nodes_test}")
print(f"Num features: {num_features}, Num classes: {num_classes}")

# -------------------------
# 5. Feature normalization
# -------------------------
# Compute mean/std from train only
mean = X_train_raw.mean(dim=0, keepdim=True)
std  = X_train_raw.std(dim=0, keepdim=True) + 1e-6

X_train = (X_train_raw - mean) / std
X_test  = (X_test_raw  - mean) / std

# -------------------------
# 6. Build edges (flow graph)
#    k random neighbors per device label (undirected)
# -------------------------
def build_same_device_knn_edges_fast(y, k=10, seed=0):
    """
    Very fast: for each device label, build k random same-device neighbors per node.
    Complexity: O(N).
    y: 1D tensor of labels [N] on CPU
    """
    random.seed(seed)
    y_list = y.tolist()

    label_to_indices = defaultdict(list)
    for idx, label in enumerate(y_list):
        label_to_indices[int(label)].append(idx)

    edges_src = []
    edges_dst = []

    for label, indices in label_to_indices.items():
        if len(indices) <= 1:
            continue

        shuffled = indices.copy()
        random.shuffle(shuffled)
        L = len(shuffled)

        for i, node in enumerate(shuffled):
            # connect to k neighbors cyclically
            for j in range(1, k + 1):
                neighbor = shuffled[(i + j) % L]
                # undirected edge
                edges_src.append(node)
                edges_dst.append(neighbor)
                edges_src.append(neighbor)
                edges_dst.append(node)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    return edge_index


print("\n--- Building edge indices (k random same-device neighbors) ---")
edge_index_train = build_same_device_knn_edges_fast(y_train, k=10, seed=42)
edge_index_test  = build_same_device_knn_edges_fast(y_test,  k=10, seed=43)

print("Train edge_index shape:", edge_index_train.shape)
print("Test edge_index shape: ", edge_index_test.shape)

# -------------------------
# 7. Train/val split masks (within train file)
# -------------------------
N = num_nodes_train
perm = torch.randperm(N)
train_ratio = 0.85
train_cut = int(train_ratio * N)
train_idx = perm[:train_cut]
val_idx   = perm[train_cut:]

train_mask = torch.zeros(N, dtype=torch.bool)
val_mask   = torch.zeros(N, dtype=torch.bool)
train_mask[train_idx] = True
val_mask[val_idx]     = True

print(f"\nTrain nodes (mask): {train_mask.sum().item()}")
print(f"Val nodes (mask):   {val_mask.sum().item()}")

# -------------------------
# 8. Move tensors to device
# -------------------------
X_train = X_train.to(device)
y_train = y_train.to(device)
train_mask = train_mask.to(device)
val_mask   = val_mask.to(device)

X_test = X_test.to(device)
y_test = y_test.to(device)

edge_index_train = edge_index_train.to(device)
edge_index_test  = edge_index_test.to(device)

# -------------------------
# 9. Deep GraphSAGE model (PyG)
# -------------------------
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.norm import BatchNorm

class SAGEModelDeep(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList([
            SAGEConv(in_dim, hidden_dim),
            SAGEConv(hidden_dim, hidden_dim),
            SAGEConv(hidden_dim, hidden_dim),
        ])
        self.bns = nn.ModuleList([
            BatchNorm(hidden_dim),
            BatchNorm(hidden_dim),
            BatchNorm(hidden_dim),
        ])
        self.lin = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            h = conv(x, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            # residual connection if dimensions match
            if h.shape == x.shape:
                x = x + h
            else:
                x = h
        x = self.lin(x)
        return x

# instantiate model
HIDDEN_DIM = 256
DROPOUT    = 0.5

model = SAGEModelDeep(
    in_dim=num_features,
    hidden_dim=HIDDEN_DIM,
    num_classes=num_classes,
    dropout=DROPOUT,
).to(device)

print("\nModel:\n", model)

# -------------------------
# 10. Training setup
# -------------------------
# class-weighted loss for imbalance
class_counts = torch.bincount(y_train, minlength=num_classes).float()
eps = 1e-6
class_weights = 1.0 / (class_counts + eps)
class_weights = class_weights * (num_classes / class_weights.sum())
class_weights = class_weights.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss(weight=class_weights)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",      # we track val macro-F1 (higher is better)
    factor=0.5,      # halve LR when plateau
    patience=5,      # epochs with no improvement before LR drop
    verbose=True,
    min_lr=1e-5,
)

max_epochs = 200
patience   = 20
best_val_f1 = -1.0
best_state = None
epochs_no_improve = 0

history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
    "train_f1_macro": [],
    "val_f1_macro": [],
    "train_f1_weighted": [],
    "val_f1_weighted": [],
}

def eval_on_split(model, X, y, edge_index, mask=None):
    model.eval()
    with torch.no_grad():
        out = model(X, edge_index)   # [N, C]
        if mask is not None:
            out = out[mask]
            y = y[mask]
        preds = out.argmax(dim=1)
        loss = criterion(out, y)
        y_np = y.cpu().numpy()
        pred_np = preds.cpu().numpy()
        acc = (pred_np == y_np).mean()
        f1_macro = f1_score(y_np, pred_np, average="macro", zero_division=0)
        f1_weighted = f1_score(y_np, pred_np, average="weighted", zero_division=0)
    return loss.item(), acc, f1_macro, f1_weighted

# -------------------------
# 11. Training loop
# -------------------------
if __name__ == '__main__':
    print("\n--- Training ---")
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(X_train, edge_index_train)
        loss = criterion(out[train_mask], y_train[train_mask])
        loss.backward()
        optimizer.step()

        # train metrics
        train_loss, train_acc, train_f1_macro, train_f1_weighted = eval_on_split(
            model, X_train, y_train, edge_index_train, mask=train_mask
        )
        # val metrics
        val_loss, val_acc, val_f1_macro, val_f1_weighted = eval_on_split(
            model, X_train, y_train, edge_index_train, mask=val_mask
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_f1_macro"].append(train_f1_macro)
        history["val_f1_macro"].append(val_f1_macro)
        history["train_f1_weighted"].append(train_f1_weighted)
        history["val_f1_weighted"].append(val_f1_weighted)

        improved = val_f1_macro > best_val_f1 + 1e-4
        if improved:
            best_val_f1 = val_f1_macro
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch == 1 or epoch % 5 == 0 or improved:
            print(
                f"Epoch {epoch:02d} | "
                f"Train loss {train_loss:.4f}, acc {train_acc:.3f}, "
                f"F1_macro {train_f1_macro:.3f}, F1_weighted {train_f1_weighted:.3f} | "
                f"Val loss {val_loss:.4f}, acc {val_acc:.3f}, "
                f"F1_macro {val_f1_macro:.3f}, F1_weighted {val_f1_weighted:.3f}"
            )

        scheduler.step(val_f1_macro)

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no val macro-F1 improvement for {patience} epochs)")
            break

    # Save metrics to CSV
    metrics_df = pd.DataFrame(history)
    metrics_df.to_csv("metrics.csv", index=False)
    print("\nSaved metrics to metrics.csv")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, "best_model.pt")
        print("Saved best model state_dict to best_model.pt")

    # -------------------------
    # 12. Final test evaluation
    # -------------------------
    model.eval()
    with torch.no_grad():
        out_test = model(X_test, edge_index_test)
        pred_test = out_test.argmax(dim=1)

    test_loss = criterion(out_test, y_test).item()
    y_test_np = y_test.cpu().numpy()
    pred_test_np = pred_test.cpu().numpy()

    test_acc = (pred_test_np == y_test_np).mean()
    test_f1_macro = f1_score(y_test_np, pred_test_np, average="macro", zero_division=0)
    test_f1_weighted = f1_score(y_test_np, pred_test_np, average="weighted", zero_division=0)

    print(f"\n[TEST] Loss: {test_loss:.4f} | "
        f"Accuracy: {test_acc:.3f} | "
        f"Macro-F1: {test_f1_macro:.3f} | "
        f"Weighted-F1: {test_f1_weighted:.3f}")

    print("\nClassification report on test set:")
    print(classification_report(y_test_np, pred_test_np, digits=3, zero_division=0))

    # -------------------------
    # 13. Confusion matrix plot
    # -------------------------
    cm = confusion_matrix(y_test_np, pred_test_np, labels=list(range(num_classes)))
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xlabel="Predicted label",
        ylabel="True label",
        title=f"Confusion Matrix ({num_classes} Devices)"
    )
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()
    print("Saved confusion matrix to confusion_matrix.png")

    # -------------------------
    # 14. Training curves plot
    # -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    axes[0].plot(history["epoch"], history["train_loss"], label="Train loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="Val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Train / Val Loss")
    axes[0].legend()

    # Macro-F1
    axes[1].plot(history["epoch"], history["train_f1_macro"], label="Train macro-F1")
    axes[1].plot(history["epoch"], history["val_f1_macro"], label="Val macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_title("Train / Val Macro-F1")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("train_curves.png", dpi=300)
    plt.close()
    print("Saved training curves to train_curves.png")

    print("\nDone.")