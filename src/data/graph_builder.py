"""
Graph construction for fraud detection.
Nodes: transactions
Edges: shared card, shared email, shared IP, shared device ID

Key insight: fraudulent transactions cluster together in the graph
because fraudsters reuse cards, emails, devices across multiple victims.
"""

import pandas as pd
import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from pathlib import Path
from sklearn.preprocessing import StandardScaler


def build_transaction_graph(df: pd.DataFrame) -> tuple:
    """
    Build a transaction graph where edges connect transactions
    that share identifying attributes (card, email, device).

    Returns: (networkx graph, PyG Data object)
    """
    print("Building transaction graph...")

    # Use a subset for speed — first 50k transactions
    sample = df.head(50000).copy().reset_index(drop=True)
    n = len(sample)
    print(f"  Using {n:,} transactions")

    # Build edges based on shared attributes
    edges = []
    edge_types = []

    shared_attrs = {
        "card1": "shared_card",
        "addr1": "shared_address",
        "P_emaildomain": "shared_email",
        "DeviceInfo": "shared_device",
    }

    for attr, edge_type in shared_attrs.items():
        if attr not in sample.columns:
            continue

        # Group transactions by shared attribute value
        valid = sample[sample[attr] != -999][[attr]].copy()
        valid["idx"] = valid.index

        grouped = valid.groupby(attr)["idx"].apply(list)

        for group in grouped:
            if len(group) < 2:
                continue
            # Connect all pairs within group (limit to avoid explosion)
            group = group[:20]  # max 20 per group
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    edges.append((group[i], group[j]))
                    edge_types.append(edge_type)

        print(f"  {attr}: {len(edges):,} edges so far")

    print(f"  Total edges: {len(edges):,}")

    # Build NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)

    # Node features
    feature_cols = [
        "TransactionAmt", "log_amount", "hour", "day",
        "card1", "card2", "card3", "card4", "card5", "card6",
        "P_emaildomain", "R_emaildomain", "same_email_domain",
        "addr1", "addr2", "dist1",
        "C1", "C2", "C3", "C4", "C5", "C6",
        "card_mean_amt", "card_std_amt", "card_count", "amt_vs_card_mean",
    ]
    feature_cols = [c for c in feature_cols if c in sample.columns]
    node_features = sample[feature_cols].fillna(-999).values

    # Normalise features
    scaler = StandardScaler()
    node_features = scaler.fit_transform(node_features)

    # Labels
    labels = sample["isFraud"].values

    # Build PyG Data object
    if edges:
        edge_index = torch.tensor(
            [[e[0] for e in edges], [e[1] for e in edges]],
            dtype=torch.long,
        )
        # Make undirected
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(labels, dtype=torch.long),
    )

    # Train/test mask — time-ordered split
    split = int(n * 0.8)
    data.train_mask = torch.zeros(n, dtype=torch.bool)
    data.test_mask = torch.zeros(n, dtype=torch.bool)
    data.train_mask[:split] = True
    data.test_mask[split:] = True

    print(f"\nGraph summary:")
    print(f"  Nodes: {data.num_nodes:,}")
    print(f"  Edges: {data.num_edges:,}")
    print(f"  Node features: {data.num_node_features}")
    print(f"  Fraud nodes: {labels.sum():,} ({labels.mean():.2%})")
    print(f"  Train nodes: {data.train_mask.sum():,}")
    print(f"  Test nodes: {data.test_mask.sum():,}")

    return G, data, sample


def save_graph(data: Data, output_dir: str = "data/processed") -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(data, f"{output_dir}/graph.pt")
    print(f"Graph saved to {output_dir}/graph.pt")


def load_graph(path: str = "data/processed/graph.pt") -> Data:
    return torch.load(path)


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_parquet("data/processed/features.parquet")
    G, data, sample = build_transaction_graph(df)
    save_graph(data)