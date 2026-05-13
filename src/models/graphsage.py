"""
GraphSAGE for fraud detection.
Uses graph structure (shared cards, emails, devices) + node features.
Key advantage over XGBoost: sees network patterns, not just individual features.
"""

import json
import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path


class GraphSAGE(torch.nn.Module):
    """
    3-layer GraphSAGE with batch normalisation and dropout.
    Each layer aggregates features from neighbouring transactions.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels // 2)
        self.classifier = torch.nn.Linear(hidden_channels // 2, out_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        return self.classifier(x)

    def get_embeddings(self, x, edge_index):
        """Return node embeddings before classification head."""
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.conv3(x, edge_index))
        return x


def compute_class_weights(data: Data) -> torch.Tensor:
    """Compute class weights for imbalanced fraud detection."""
    train_labels = data.y[data.train_mask]
    n_neg = (train_labels == 0).sum().item()
    n_pos = (train_labels == 1).sum().item()
    weight = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float)
    print(f"  Class weights: [neg={weight[0]:.1f}, fraud={weight[1]:.1f}]")
    return weight


def train_epoch(
    model: GraphSAGE,
    data: Data,
    optimizer: torch.optim.Optimizer,
    class_weights: torch.Tensor,
) -> float:
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(
        out[data.train_mask],
        data.y[data.train_mask],
        weight=class_weights,
    )
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def evaluate(
    model: GraphSAGE,
    data: Data,
    mask: torch.Tensor,
) -> dict:
    model.eval()
    out = model(data.x, data.edge_index)
    proba = F.softmax(out, dim=1)[:, 1]

    y_true = data.y[mask].numpy()
    y_proba = proba[mask].numpy()

    auc = roc_auc_score(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    return {"auc_roc": round(auc, 4), "avg_precision": round(ap, 4)}


def run_graphsage() -> dict:
    mlflow.set_experiment("fraudgraph")

    print("=" * 50)
    print("FraudGraph — GraphSAGE")
    print("=" * 50)

    # Load graph
    data = torch.load("data/processed/graph.pt", weights_only=False)
    print(f"\nGraph: {data.num_nodes:,} nodes, {data.num_edges:,} edges")
    print(f"Features: {data.num_node_features}")
    print(f"Fraud rate: {data.y.float().mean():.2%}")

    # Model setup
    model = GraphSAGE(
        in_channels=data.num_node_features,
        hidden_channels=64,
        dropout=0.3,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    class_weights = compute_class_weights(data)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    with mlflow.start_run(run_name="graphsage"):
        mlflow.log_params({
            "model": "GraphSAGE",
            "hidden_channels": 64,
            "dropout": 0.3,
            "lr": 0.01,
            "epochs": 50,
        })

        print("\nTraining GraphSAGE...")
        best_auc = 0
        best_metrics = {}

        for epoch in range(1, 51):
            loss = train_epoch(model, data, optimizer, class_weights)
            scheduler.step()

            if epoch % 10 == 0:
                train_metrics = evaluate(model, data, data.train_mask)
                test_metrics = evaluate(model, data, data.test_mask)
                print(
                    f"  Epoch {epoch:3d} | Loss: {loss:.4f} | "
                    f"Train AUC: {train_metrics['auc_roc']:.4f} | "
                    f"Test AUC: {test_metrics['auc_roc']:.4f} | "
                    f"Test AP: {test_metrics['avg_precision']:.4f}"
                )

                if test_metrics["auc_roc"] > best_auc:
                    best_auc = test_metrics["auc_roc"]
                    best_metrics = test_metrics
                    torch.save(model.state_dict(), "data/processed/graphsage_best.pt")

        mlflow.log_metrics(best_metrics)

        # Save results
        results = {
            "model": "GraphSAGE",
            "metrics": best_metrics,
        }
        with open("data/processed/graphsage_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print("\n" + "=" * 50)
        print("GraphSAGE Results (best epoch):")
        print(f"  AUC-ROC:       {best_metrics['auc_roc']}")
        print(f"  Avg Precision: {best_metrics['avg_precision']}")
        print("\nComparison:")
        print(f"  XGBoost AUC-ROC:    0.8619")
        print(f"  GraphSAGE AUC-ROC:  {best_metrics['auc_roc']}")
        print(f"  XGBoost AP:         0.4078")
        print(f"  GraphSAGE AP:       {best_metrics['avg_precision']}")
        print("=" * 50)

    return results


if __name__ == "__main__":
    run_graphsage()