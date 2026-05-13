"""
SHAP explainability for FraudGraph.
Answers: which features most drive fraud predictions?
Also visualises suspicious transaction subgraphs.
"""

import json
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score


def load_data():
    df = pd.read_parquet("data/processed/features.parquet")
    drop_cols = ["isFraud", "TransactionID", "card_id"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].fillna(-999)
    y = df["isFraud"]
    split = int(len(df) * 0.8)
    return (
        X.iloc[:split], X.iloc[split:],
        y.iloc[:split], y.iloc[split:],
        df,
    )


def run_shap_analysis(output_dir: str = "data/processed/shap") -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("FraudGraph — SHAP Explainability")
    print("=" * 50)

    X_train, X_test, y_train, y_test, df = load_data()

    # Retrain XGBoost
    fraud_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=fraud_ratio,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1,
        tree_method="hist",
    )
    print("Training XGBoost for SHAP analysis...")
    model.fit(X_train, y_train, verbose=False)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"AUC-ROC: {auc:.4f}")

    # SHAP values — use TreeExplainer (fast for XGBoost)
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.head(2000))

    # --- Plot 1: Global feature importance ---
    print("Generating SHAP importance plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_test.head(2000),
        plot_type="bar",
        max_display=20,
        show=False,
    )
    plt.title("Top 20 Features Driving Fraud Predictions", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/shap_importance.png")

    # --- Plot 2: SHAP beeswarm ---
    print("Generating SHAP beeswarm plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_test.head(2000),
        max_display=20,
        show=False,
    )
    plt.title("SHAP Summary — Fraud Feature Impact", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/shap_beeswarm.png")

    # --- Plot 3: Fraud network visualisation ---
    print("Generating fraud network visualisation...")
    import torch
    data = torch.load("data/processed/graph.pt", weights_only=False)

    edge_index = data.edge_index.numpy()
    fraud_indices = set((data.y == 1).nonzero(as_tuple=True)[0].tolist()[:15])

    # Build small subgraph — only direct neighbours of first 15 fraud nodes
    G = nx.Graph()
    subgraph_nodes = set(fraud_indices)

    for i in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, i]), int(edge_index[1, i])
        if src in fraud_indices or dst in fraud_indices:
            subgraph_nodes.add(src)
            subgraph_nodes.add(dst)
        if len(subgraph_nodes) > 60:
            break

    node_list = list(subgraph_nodes)
    for i in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, i]), int(edge_index[1, i])
        if src in subgraph_nodes and dst in subgraph_nodes:
            G.add_edge(src, dst)
    G.add_nodes_from(node_list)

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=3)
    colors = ["#FF4444" if n in fraud_indices else "#4488FF" for n in G.nodes()]
    sizes = [250 if n in fraud_indices else 80 for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, ax=ax, alpha=0.85)
    nx.draw_networkx_edges(G, pos, edge_color="#888888", alpha=0.4, width=1.0, ax=ax)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#FF4444", label=f"Fraudulent ({len(fraud_indices)} nodes)"),
        Patch(color="#4488FF", label="Legitimate (connected)"),
    ], loc="upper left", fontsize=11)
    ax.set_title(
        "Fraud Transaction Network\nRed nodes share cards/emails/devices with known fraud",
        fontsize=12,
    )
    ax.set_facecolor("#0a0a0a")
    fig.patch.set_facecolor("#0a0a0a")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fraud_network.png", dpi=120,
                bbox_inches="tight", facecolor="#0a0a0a")
    plt.close()
    print(f"  Saved: {output_dir}/fraud_network.png")

    # Print top SHAP features
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_features = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": mean_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    print("\n" + "=" * 50)
    print("Top 10 Fraud Drivers (SHAP)")
    print("=" * 50)
    for _, row in top_features.head(10).iterrows():
        bar = "█" * int(row["mean_abs_shap"] / top_features["mean_abs_shap"].max() * 30)
        print(f"  {row['feature']:<25} {bar} {row['mean_abs_shap']:.4f}")

    # Save results
    results = {
        "top_features": top_features.head(20).to_dict(orient="records"),
        "auc_roc": round(auc, 4),
    }
    with open(f"{output_dir}/shap_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    run_shap_analysis()