"""
FraudGraph — Interactive Plotly Dash Dashboard
"""

import json
from pathlib import Path
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from dash import dcc, html, Input, Output


# Load results
def load_results():
    base = Path("data/processed")
    with open(base / "xgb_results.json") as f:
        xgb = json.load(f)
    with open(base / "graphsage_results.json") as f:
        gnn = json.load(f)
    with open(base / "hybrid_results.json") as f:
        hybrid = json.load(f)
    with open(base / "shap/shap_results.json") as f:
        shap = json.load(f)
    return xgb, gnn, hybrid, shap


xgb_r, gnn_r, hybrid_r, shap_r = load_results()

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="FraudGraph",
    assets_folder=str(Path("data/processed/shap").absolute()),
)


def metric_card(title, value, subtitle, color="primary"):
    return dbc.Card([
        dbc.CardBody([
            html.H6(title, className="text-muted mb-1", style={"fontSize": "0.75rem"}),
            html.H3(value, className=f"text-{color} mb-0"),
            html.Small(subtitle, className="text-muted"),
        ])
    ], className="mb-3")


# Model comparison data
models = ["XGBoost\n(tabular)", "GraphSAGE\n(graph)", "Hybrid\n(GNN+XGB)"]
auc_scores = [
    xgb_r["metrics"]["auc_roc"],
    gnn_r["metrics"]["auc_roc"],
    hybrid_r["metrics"]["auc_roc"],
]
ap_scores = [
    xgb_r["metrics"]["avg_precision"],
    gnn_r["metrics"]["avg_precision"],
    hybrid_r["metrics"]["avg_precision"],
]

app.layout = dbc.Container([

    # Header
    dbc.Row([
        dbc.Col([
            html.H2("🔍 FraudGraph", className="mt-4 mb-0"),
            html.P(
                "Transaction Fraud Detection · XGBoost vs GraphSAGE vs Hybrid GNN",
                className="text-muted mb-4",
            ),
        ])
    ]),

    # Metric cards
    dbc.Row([
        dbc.Col(metric_card(
            "XGBoost AUC-ROC",
            f"{xgb_r['metrics']['auc_roc']}",
            "Tabular features only",
            "secondary"
        ), md=3),
        dbc.Col(metric_card(
            "GraphSAGE AUC-ROC",
            f"{gnn_r['metrics']['auc_roc']}",
            "Graph structure only",
            "secondary"
        ), md=3),
        dbc.Col(metric_card(
            "Hybrid AUC-ROC",
            f"{hybrid_r['metrics']['auc_roc']}",
            "GNN embeddings + XGBoost",
            "warning"
        ), md=3),
        dbc.Col(metric_card(
            "Dataset",
            "590,540",
            "Transactions · 3.5% fraud rate",
            "info"
        ), md=3),
    ]),

    # Model comparison chart
    dbc.Row([
        dbc.Col([
            html.H5("Model Comparison", className="mt-2 mb-3"),
            dcc.Graph(
                id="comparison-chart",
                figure=go.Figure(data=[
                    go.Bar(
                        name="AUC-ROC",
                        x=["XGBoost", "GraphSAGE", "Hybrid"],
                        y=auc_scores,
                        marker_color=["#7B68EE", "#FFA500", "#00FF7F"],
                        text=[f"{s:.4f}" for s in auc_scores],
                        textposition="outside",
                    ),
                    go.Bar(
                        name="Avg Precision",
                        x=["XGBoost", "GraphSAGE", "Hybrid"],
                        y=ap_scores,
                        marker_color=["#4B4899", "#B37000", "#009955"],
                        text=[f"{s:.4f}" for s in ap_scores],
                        textposition="outside",
                    ),
                ]).update_layout(
                    template="plotly_dark",
                    barmode="group",
                    title="AUC-ROC and Average Precision by Model",
                    yaxis=dict(range=[0, 1]),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", y=1.1),
                ),
                style={"height": "400px"},
            ),
        ], md=7),

        dbc.Col([
            html.H5("Model Comparison Table", className="mt-2 mb-3"),
            dbc.Table([
                html.Thead(html.Tr([
                    html.Th("Model"),
                    html.Th("AUC-ROC"),
                    html.Th("Avg Precision"),
                    html.Th("vs XGBoost"),
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td("XGBoost"),
                        html.Td(f"{xgb_r['metrics']['auc_roc']}"),
                        html.Td(f"{xgb_r['metrics']['avg_precision']}"),
                        html.Td("baseline"),
                    ]),
                    html.Tr([
                        html.Td("GraphSAGE"),
                        html.Td(f"{gnn_r['metrics']['auc_roc']}"),
                        html.Td(f"{gnn_r['metrics']['avg_precision']}"),
                        html.Td(
                            f"{(gnn_r['metrics']['auc_roc']-xgb_r['metrics']['auc_roc'])*100:+.1f}%",
                            style={"color": "red"},
                        ),
                    ]),
                    html.Tr([
                        html.Td("Hybrid (GNN+XGB)"),
                        html.Td(f"{hybrid_r['metrics']['auc_roc']}"),
                        html.Td(f"{hybrid_r['metrics']['avg_precision']}"),
                        html.Td(
                            f"{(hybrid_r['metrics']['auc_roc']-xgb_r['metrics']['auc_roc'])*100:+.1f}%",
                            style={"color": "orange"},
                        ),
                    ]),
                ])
            ], bordered=True, hover=True, striped=True),

            html.Hr(),
            html.H6("Key Finding", className="text-muted"),
            html.P(
                "XGBoost outperforms GNN on this dataset because the IEEE-CIS "
                "features already encode rich network signals (C1-C14 count features). "
                "GNN adds most value when raw transaction logs lack engineered features.",
                className="text-muted",
                style={"fontSize": "0.8rem"},
            ),
        ], md=5),
    ]),

    # SHAP section
    dbc.Row([
        dbc.Col([
            html.H5("SHAP Explainability — Fraud Drivers", className="mt-4 mb-3"),
            dbc.Tabs([
                dbc.Tab(
                    html.Img(
                        src="assets/shap_importance.png",
                        style={"width": "100%", "marginTop": "15px"},
                    ),
                    label="Feature Importance",
                ),
                dbc.Tab(
                    html.Img(
                        src="assets/shap_beeswarm.png",
                        style={"width": "100%", "marginTop": "15px"},
                    ),
                    label="SHAP Beeswarm",
                ),
                dbc.Tab(
                    html.Img(
                        src="assets/fraud_network.png",
                        style={"width": "100%", "marginTop": "15px"},
                    ),
                    label="Fraud Network Graph",
                ),
            ]),
        ])
    ]),

    # SHAP top features bar
    dbc.Row([
        dbc.Col([
            html.H5("Top Fraud Signal Features", className="mt-4 mb-3"),
            dcc.Graph(
                figure=go.Figure(data=[
                    go.Bar(
                        x=[f["mean_abs_shap"] for f in shap_r["top_features"][:10]][::-1],
                        y=[f["feature"] for f in shap_r["top_features"][:10]][::-1],
                        orientation="h",
                        marker_color="#FF4444",
                        text=[f"{f['mean_abs_shap']:.3f}" for f in shap_r["top_features"][:10]][::-1],
                        textposition="outside",
                    )
                ]).update_layout(
                    template="plotly_dark",
                    title="Mean |SHAP Value| — Top 10 Fraud Predictors",
                    xaxis_title="Mean |SHAP value|",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=400,
                ),
                style={"height": "400px"},
            ),
        ])
    ]),

    html.Hr(className="mt-4"),
    html.P(
        "Data: IEEE-CIS Fraud Detection · Models: XGBoost, GraphSAGE (PyTorch Geometric), Hybrid · Explainability: SHAP TreeExplainer",
        className="text-muted text-center mb-4",
        style={"fontSize": "0.75rem"},
    ),

], fluid=True)


if __name__ == "__main__":
    app.run(debug=True, port=8052)