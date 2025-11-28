#!/usr/bin/env python3
"""
src/train_models.py

Trains ML models for fake job detection using:
 - meta_features.csv (dense engineered features)
 - desc_tfidf.npz (sparse TF-IDF matrix)

Models:
 - Logistic Regression (best for sparse + SVD)
 - Random Forest
 - XGBoost

Outputs:
 - results/metrics.csv
 - models/*.joblib
 - preprocessors/svd.joblib (if SVD is used)

Usage:
 python src/train_models.py --meta data/meta_features.csv --tfidf data/desc_tfidf.npz
"""

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
LOG = logging.getLogger("train_models")


# ------------------ ARGPARSE ------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--meta", default="data/meta_features.csv")
    p.add_argument("--tfidf", default="data/desc_tfidf.npz")
    p.add_argument("--use-svd", action="store_true", help="Apply TruncatedSVD to TF-IDF")
    p.add_argument("--svd-dim", type=int, default=200, help="Number of SVD components")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--results", default="results/metrics.csv")
    p.add_argument("--models-dir", default="models")
    return p.parse_args()


# ------------------ METRICS UTILS ------------------
def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    }


# ------------------ MAIN PIPELINE ------------------
def main():
    args = parse_args()
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("preprocessors").mkdir(parents=True, exist_ok=True)

    # ---------------- LOAD DATA ----------------
    LOG.info("Loading meta features from %s", args.meta)
    meta = pd.read_csv(args.meta)

    if "fake_label" not in meta.columns:
        raise ValueError("meta_features.csv does not contain fake_label column.")

    y = meta["fake_label"].values

    # drop Job ID and label
    drop_cols = ["Job ID", "fake_label"]
    X_meta = meta.drop(columns=[c for c in drop_cols if c in meta.columns]).values

    LOG.info("Meta feature matrix shape: %s", X_meta.shape)

    # TF-IDF
    LOG.info("Loading TF-IDF sparse matrix from %s", args.tfidf)
    X_tfidf = sparse.load_npz(args.tfidf)
    LOG.info("TF-IDF shape before SVD: %s", X_tfidf.shape)

    # ---------------- OPTIONAL SVD ----------------
    if args.use_svd:
        LOG.info("Applying Truncated SVD (n_components=%d)...", args.svd_dim)
        svd = TruncatedSVD(n_components=args.svd_dim, random_state=args.random_state)
        X_tfidf_reduced = svd.fit_transform(X_tfidf)
        LOG.info("TF-IDF shape after SVD: %s", X_tfidf_reduced.shape)

        # save SVD transformer
        joblib.dump(svd, "preprocessors/svd.joblib")
    else:
        LOG.info("Skipping SVD; using full TF-IDF matrix")
        X_tfidf_reduced = X_tfidf  # use sparse directly

    # ---------------- CONCAT META + TF-IDF ----------------
    LOG.info("Concatenating meta features + text features")
    if sparse.issparse(X_tfidf_reduced):
        X_final = sparse.hstack([X_meta, X_tfidf_reduced]).tocsr()
    else:
        X_final = np.hstack([X_meta, X_tfidf_reduced])

    LOG.info("Final feature shape: %s", X_final.shape)

    # ---------------- SPLIT ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=args.test_size,
        stratify=y, random_state=args.random_state
    )

    LOG.info("Train shape: %s, Test shape: %s", X_train.shape, X_test.shape)

    # ---------------- TRAIN MODELS ----------------
    metrics_out = []

    # ---- 1. Logistic Regression ----
    LOG.info("Training Logistic Regression")
    lr = LogisticRegression(max_iter=5000, class_weight="balanced")
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)[:, 1]
    m_lr = compute_metrics(y_test, y_pred, y_prob)
    m_lr["model"] = "LogisticRegression"
    metrics_out.append(m_lr)

    joblib.dump(lr, f"{args.models_dir}/logistic_regression.joblib")
    LOG.info("Saved Logistic Regression model")

    # ---- 2. Random Forest ----
    LOG.info("Training Random Forest")
    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=args.random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    m_rf = compute_metrics(y_test, y_pred, y_prob)
    m_rf["model"] = "RandomForest"
    metrics_out.append(m_rf)

    joblib.dump(rf, f"{args.models_dir}/random_forest.joblib")
    LOG.info("Saved Random Forest model")

    # ---- 3. XGBoost ----
    LOG.info("Training XGBoost")
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=1.0,   # can adjust if dataset is unbalanced
        random_state=args.random_state,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)[:, 1]
    m_xgb = compute_metrics(y_test, y_pred, y_prob)
    m_xgb["model"] = "XGBoost"
    metrics_out.append(m_xgb)

    joblib.dump(xgb, f"{args.models_dir}/xgboost.joblib")
    LOG.info("Saved XGBoost model")

    # ---------------- SAVE METRICS ----------------
    df_metrics = pd.DataFrame(metrics_out)
    df_metrics.to_csv(args.results, index=False)

    LOG.info("Training completed. Metrics saved to %s", args.results)
    LOG.info("\n%s", df_metrics)


if __name__ == "__main__":
    main()