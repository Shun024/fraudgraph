# 🔍 FraudGraph

Transaction fraud detection using Graph Neural Networks. Compares XGBoost (tabular), GraphSAGE (graph structure), and a hybrid GNN+XGBoost model on 590,540 real transactions from the IEEE-CIS Fraud Detection dataset.

---

## Results

| Model | AUC-ROC | Avg Precision | Approach |
|---|---|---|---|
| **XGBoost** | **0.8619** | **0.4078** | Tabular features only |
| GraphSAGE | 0.7489 | 0.0701 | Graph structure only |
| Hybrid (GNN+XGB) | 0.7833 | 0.2167 | GNN embeddings + XGBoost |

**Key finding:** XGBoost outperforms GNN on this dataset because the IEEE-CIS features (C1–C14) already encode rich network signals. GNNs add most value when raw transaction logs lack pre-engineered features — the production case at most banks.

---

## Architecture

```
IEEE-CIS Fraud Dataset (590,540 transactions · 3.5% fraud)
        │
        ▼
Graph Construction
├── Nodes: transactions (50k sample)
├── Edges: shared card, address, email, device (290k edges)
└── Node features: amount, time, card, email, device (26 features)
        │
        ├── XGBoost (tabular baseline)     AUC 0.8619
        ├── GraphSAGE (3-layer GNN)        AUC 0.7489
        └── Hybrid (GNN embeddings + XGB)  AUC 0.7833
                │
                ▼
        SHAP Explainability
        + Fraud Network Visualisation
        + Plotly Dash Dashboard
```

---

## SHAP — Top Fraud Drivers

| Feature | SHAP | Interpretation |
|---|---|---|
| C5 | 0.543 | Transaction count feature — frequency signal |
| C1 | 0.199 | Card usage count — velocity check |
| amount_cents | 0.192 | Round amounts signal card testing |
| card_mean_amt | 0.173 | Deviation from card's historical average |
| same_email_domain | 0.143 | Mismatched billing/shipping email = account takeover |

---

## Stack

PyTorch Geometric · XGBoost · NetworkX · SHAP · Plotly Dash · MLflow · scikit-learn · pandas

---

## Quickstart

```bash
git clone https://github.com/Shun024/fraudgraph.git
cd fraudgraph
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download IEEE-CIS data from Kaggle → data/raw/
# kaggle.com/competitions/ieee-fraud-detection/data

PYTHONPATH=. python -m src.data.processor
PYTHONPATH=. python -m src.data.graph_builder
PYTHONPATH=. python -m src.models.baseline_xgb
PYTHONPATH=. python -m src.models.graphsage
PYTHONPATH=. python -m src.models.hybrid
PYTHONPATH=. python -m src.evaluation.explainability

# Launch dashboard
PYTHONPATH=. python src/dashboard/app.py
# Open http://localhost:8052
```

---

## Why XGBoost Beats GNN Here

This is an honest result worth explaining. The IEEE-CIS dataset contains Vesta Corporation's pre-engineered V-features and C-features (count of transactions per card, email, IP) which already encode the network information that GNNs would otherwise discover from the graph. In production fraud systems with only raw transaction logs — no pre-engineered features — GNNs consistently outperform tabular models by 5–15% AUC.

The hybrid model (GNN embeddings + XGBoost) represents the production architecture used at major payment networks.

---

## Author

**Shun Le Yi Mon (Sheryl)** · Data Scientist · NLP & GenAI  
[LinkedIn](https://www.linkedin.com/in/shunleyimon724) · [GitHub](https://github.com/Shun024)
