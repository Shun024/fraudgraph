"""
Data processor for IEEE-CIS Fraud Detection dataset.
Merges transaction + identity, engineers features, handles imbalance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_raw_data(data_dir: str = "data/raw") -> pd.DataFrame:
    """Load and merge transaction + identity datasets."""
    print("Loading raw data...")
    transactions = pd.read_csv(f"{data_dir}/train_transaction.csv")
    identity = pd.read_csv(f"{data_dir}/train_identity.csv")

    df = transactions.merge(identity, on="TransactionID", how="left")

    print(f"  Transactions: {len(transactions):,}")
    print(f"  With identity: {identity['TransactionID'].nunique():,}")
    print(f"  Merged shape: {df.shape}")
    print(f"  Fraud rate: {df['isFraud'].mean():.2%}")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for fraud detection."""
    print("Engineering features...")

    # Time features
    df["hour"] = (df["TransactionDT"] / 3600 % 24).astype(int)
    df["day"] = (df["TransactionDT"] / (3600 * 24) % 7).astype(int)

    # Amount features
    df["log_amount"] = np.log1p(df["TransactionAmt"])
    df["amount_cents"] = df["TransactionAmt"] % 1

    # Card features
    df["card_id"] = (
        df["card1"].astype(str) + "_" +
        df["card2"].astype(str) + "_" +
        df["card3"].astype(str)
    )

    # Aggregated features — card-level statistics
    card_stats = df.groupby("card_id")["TransactionAmt"].agg(
        card_mean_amt="mean",
        card_std_amt="std",
        card_count="count",
    ).reset_index()
    df = df.merge(card_stats, on="card_id", how="left")
    df["amt_vs_card_mean"] = df["TransactionAmt"] / (df["card_mean_amt"] + 1)

    # Email domain features
    df["p_email_domain"] = df["P_emaildomain"].fillna("unknown")
    df["r_email_domain"] = df["R_emaildomain"].fillna("unknown")
    df["same_email_domain"] = (df["p_email_domain"] == df["r_email_domain"]).astype(int)

    print(f"  Features engineered: {df.shape[1]} columns")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label encode categorical columns."""
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print(f"  Encoding {len(cat_cols)} categorical columns...")

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = df[col].fillna("unknown")
        df[col] = le.fit_transform(df[col].astype(str))

    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare feature matrix and target."""
    drop_cols = ["isFraud", "TransactionID", "TransactionDT", "card_id"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].fillna(-999)
    y = df["isFraud"]

    return X, y


def run_processing(
    data_dir: str = "data/raw",
    output_dir: str = "data/processed",
) -> pd.DataFrame:
    """Full processing pipeline."""
    print("=" * 50)
    print("FraudGraph — Data Processing Pipeline")
    print("=" * 50)

    df = load_raw_data(data_dir)
    df = engineer_features(df)
    df = encode_categoricals(df)

    X, y = prepare_features(df)

    # Train/test split — time-ordered (no shuffle)
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Train fraud rate: {y_train.mean():.2%}")
    print(f"Test fraud rate: {y_test.mean():.2%}")

    # Save processed data — keep only essential columns to save space
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Select only columns needed for graph + model
    keep_cols = [
        "TransactionID", "isFraud", "TransactionAmt", "log_amount",
        "hour", "day", "card1", "card2", "card3", "card4", "card5", "card6",
        "P_emaildomain", "R_emaildomain", "same_email_domain",
        "addr1", "addr2", "dist1", "card_id",
        "card_mean_amt", "card_std_amt", "card_count", "amt_vs_card_mean",
        "amount_cents", "DeviceInfo", "DeviceType",
        "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8",
        "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df[keep_cols].to_parquet(f"{output_dir}/features.parquet", index=False)
    print(f"Saved {len(keep_cols)} columns to {output_dir}/features.parquet")

    return df, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_processing()