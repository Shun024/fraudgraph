"""
XGBoost baseline for fraud detection.
Uses only tabular features — no graph structure.
This is what most DS teams have today.
"""

import json
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)


def load_data(path: str = "data/processed/features.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)


def prepare_features(df: pd.DataFrame) -> tuple:
    drop_cols = ["isFraud", "TransactionID", "card_id"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].fillna(-999)
    y = df["isFraud"]
    return X, y


def evaluate(y_true, y_pred_proba, threshold: float = 0.5) -> dict:
    y_pred = (y_pred_proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_pred_proba)
    ap = average_precision_score(y_true, y_pred_proba)
    report = classification_report(y_true, y_pred, output_dict=True)

    print(f"\n  AUC-ROC:           {auc:.4f}")
    print(f"  Avg Precision:     {ap:.4f}")
    print(f"  Fraud Precision:   {report['1']['precision']:.4f}")
    print(f"  Fraud Recall:      {report['1']['recall']:.4f}")
    print(f"  Fraud F1:          {report['1']['f1-score']:.4f}")

    return {
        "auc_roc": round(auc, 4),
        "avg_precision": round(ap, 4),
        "fraud_precision": round(report["1"]["precision"], 4),
        "fraud_recall": round(report["1"]["recall"], 4),
        "fraud_f1": round(report["1"]["f1-score"], 4),
    }


def run_xgb_baseline() -> dict:
    mlflow.set_experiment("fraudgraph")

    print("=" * 50)
    print("FraudGraph — XGBoost Baseline")
    print("=" * 50)

    df = load_data()
    X, y = prepare_features(df)

    # Time-ordered split
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Class weight for imbalance
    fraud_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
    print(f"\nClass weight (scale_pos_weight): {fraud_ratio:.1f}")

    with mlflow.start_run(run_name="xgboost_baseline"):
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=fraud_ratio,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )

        print("\nTraining XGBoost...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50,
        )

        print("\nEvaluating...")
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, y_pred_proba)

        mlflow.log_params({
            "model": "XGBoost",
            "n_estimators": 300,
            "max_depth": 6,
            "scale_pos_weight": round(fraud_ratio, 2),
        })
        mlflow.log_metrics(metrics)

        # Feature importance
        importance = pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        print("\nTop 10 features:")
        print(importance.head(10).to_string(index=False))

        # Save results
        results = {
            "model": "XGBoost",
            "metrics": metrics,
            "top_features": importance.head(20).to_dict(orient="records"),
        }
        Path("data/processed").mkdir(exist_ok=True)
        with open("data/processed/xgb_results.json", "w") as f:
            json.dump(results, f, indent=2)

    print("\n" + "=" * 50)
    print(f"XGBoost Baseline AUC-ROC: {metrics['auc_roc']}")
    print(f"XGBoost Baseline Avg Precision: {metrics['avg_precision']}")
    print("=" * 50)

    return results, model, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_xgb_baseline()