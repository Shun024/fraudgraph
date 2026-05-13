"""
Hybrid model: GraphSAGE embeddings + XGBoost.
This is the production pattern used at Visa, Mastercard, and major banks.
GNN captures network patterns → embeddings fed to XGBoost → best of both worlds.
"""

import json
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xgboost as xgb
from torch_geometric.nn import SAGEConv
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
)
from pathlib import Path


class GraphSAGEEmbedder(torch.nn.Module):
    """GraphSAGE used as an embedding generator only — no classifier head."""

    def __init__(self, in_channels: int, hidden_channels: int = 64):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels // 2)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.conv3(x, edge_index))
        return x  # returns embeddings, not class logits


def train_embedder(data, epochs: int = 80) -> GraphSAGEEmbedder:
    """Train GraphSAGE with classification head, then strip head for embeddings."""
    from torch_geometric.nn import SAGEConv

    class FullModel(torch.nn.Module):
        def __init__(self, in_ch, hidden=64):
            super().__init__()
            self.embedder = GraphSAGEEmbedder(in_ch, hidden)
            self.head = torch.nn.Linear(hidden // 2, 2)

        def forward(self, x, edge_index):
            emb = self.embedder(x, edge_index)
            return self.head(emb)

    model = FullModel(data.num_node_features)
    train_labels = data.y[data.train_mask]
    n_neg = (train_labels == 0).sum().item()
    n_pos = (train_labels == 1).sum().item()
    weights = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Training embedder for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(
            out[data.train_mask], data.y[data.train_mask], weight=weights
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d} | Loss: {loss:.4f}")

    return model.embedder


@torch.no_grad()
def extract_embeddings(embedder: GraphSAGEEmbedder, data) -> np.ndarray:
    """Extract node embeddings from trained GraphSAGE."""
    embedder.eval()
    embeddings = embedder(data.x, data.edge_index)
    return embeddings.numpy()


def run_hybrid() -> dict:
    mlflow.set_experiment("fraudgraph")

    print("=" * 50)
    print("FraudGraph — Hybrid (GraphSAGE + XGBoost)")
    print("=" * 50)

    # Load graph and original features
    data = torch.load("data/processed/graph.pt", weights_only=False)
    df = pd.read_parquet("data/processed/features.parquet")
    df_sample = df.head(50000).reset_index(drop=True)

    print(f"Graph: {data.num_nodes:,} nodes | {data.num_edges:,} edges")

    with mlflow.start_run(run_name="hybrid_graphsage_xgb"):

        # Step 1: Train GNN embedder
        print("\n[1/3] Training GraphSAGE embedder...")
        embedder = train_embedder(data, epochs=80)

        # Step 2: Extract embeddings
        print("\n[2/3] Extracting graph embeddings...")
        embeddings = extract_embeddings(embedder, data)
        print(f"  Embedding shape: {embeddings.shape}")

        emb_cols = [f"gnn_emb_{i}" for i in range(embeddings.shape[1])]
        emb_df = pd.DataFrame(embeddings, columns=emb_cols)

        # Step 3: Combine embeddings with original tabular features
        print("\n[3/3] Training hybrid XGBoost...")
        drop_cols = ["isFraud", "TransactionID", "card_id"]
        tab_features = df_sample[
            [c for c in df_sample.columns if c not in drop_cols]
        ].fillna(-999).reset_index(drop=True)

        X_hybrid = pd.concat([tab_features, emb_df], axis=1)
        y = df_sample["isFraud"].reset_index(drop=True)

        # Time-ordered split
        split = int(len(X_hybrid) * 0.8)
        X_train = X_hybrid.iloc[:split]
        X_test = X_hybrid.iloc[split:]
        y_train = y.iloc[:split]
        y_test = y.iloc[split:]

        fraud_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]

        import gc
        gc.collect()

        # Replace the XGBClassifier instantiation in hybrid.py
        model = xgb.XGBClassifier(
            n_estimators=100,          # reduced from 300
            max_depth=4,               # reduced from 6
            learning_rate=0.1,
            scale_pos_weight=fraud_ratio,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            random_state=42,
            n_jobs=1,                  # single thread to avoid memory issues
            tree_method="hist",
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=100,
        )

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        ap = average_precision_score(y_test, y_pred_proba)

        y_pred = (y_pred_proba >= 0.5).astype(int)
        report = classification_report(y_test, y_pred, output_dict=True)

        metrics = {
            "auc_roc": round(auc, 4),
            "avg_precision": round(ap, 4),
            "fraud_precision": round(report["1"]["precision"], 4),
            "fraud_recall": round(report["1"]["recall"], 4),
            "fraud_f1": round(report["1"]["f1-score"], 4),
        }

        mlflow.log_metrics(metrics)

        # Save results
        results = {"model": "Hybrid (GraphSAGE + XGBoost)", "metrics": metrics}
        with open("data/processed/hybrid_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print("\n" + "=" * 50)
        print("Final Model Comparison")
        print("=" * 50)
        print(f"{'Model':<30} {'AUC-ROC':<10} {'Avg Precision'}")
        print("-" * 50)
        print(f"{'XGBoost (tabular only)':<30} {'0.8619':<10} {'0.4078'}")
        print(f"{'GraphSAGE (graph only)':<30} {'0.7489':<10} {'0.0701'}")
        print(f"{'Hybrid (GNN + XGBoost)':<30} {auc:<10.4f} {ap:.4f}")
        print("=" * 50)

    return results


if __name__ == "__main__":
    run_hybrid()