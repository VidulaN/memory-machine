"""
train_model.py
Train a RandomForest drought classifier on the pre-built dataset and save to model/.
Usage:
    python train_model.py [--data data/dataset.csv] [--out model/drought_model.pkl]
"""

import argparse
import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

os.makedirs("model", exist_ok=True)

FEATURES = [
    "temp_max",
    "precipitation",
    "precip_7day",
    "precip_30day",
    "temp_7day",
    "evapotranspiration",
]
TARGET = "drought_risk"


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, parse_dates=["date"])
    missing = [f for f in FEATURES + [TARGET] if f not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    X = df[FEATURES]
    y = df[TARGET]
    return X, y


def train(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Balance classes via sample weights
    classes    = np.unique(y_train)
    weights    = compute_class_weight("balanced", classes=classes, y=y_train)
    class_dict = dict(zip(classes, weights))

    clf = RandomForestClassifier(
        n_estimators     = 300,
        max_depth        = 12,
        min_samples_leaf = 4,
        class_weight     = class_dict,
        random_state     = 42,
        n_jobs           = -1,
    )
    clf.fit(X_train, y_train)

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print("\n── Classification report ─────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["No drought", "Drought"]))

    print("── Confusion matrix ──────────────────────────────────")
    print(confusion_matrix(y_test, y_pred))

    auc = roc_auc_score(y_test, y_prob)
    print(f"\nROC-AUC : {auc:.4f}")

    # Cross-validation
    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"5-fold CV AUC : {scores.mean():.4f} ± {scores.std():.4f}")

    # Feature importance
    print("\n── Feature importances ───────────────────────────────")
    for feat, imp in sorted(
        zip(FEATURES, clf.feature_importances_), key=lambda x: -x[1]
    ):
        bar = "█" * int(imp * 40)
        print(f"  {feat:<22} {bar}  {imp:.4f}")

    return clf


def main():
    parser = argparse.ArgumentParser(description="Train drought classifier")
    parser.add_argument("--data", default="data/dataset.csv")
    parser.add_argument("--out",  default="model/drought_model.pkl")
    args = parser.parse_args()

    print(f"Loading dataset from {args.data} ...")
    X, y = load_data(args.data)
    print(f"Samples : {len(X)}  |  Drought rate : {y.mean() * 100:.1f}%")

    clf = train(X, y)

    with open(args.out, "wb") as f:
        pickle.dump(clf, f)
    print(f"\nModel saved → {args.out}")


if __name__ == "__main__":
    main()
