import argparse
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             precision_recall_curve, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

LOG = logging.getLogger("train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--meta", default="data/meta_features.csv")
    p.add_argument("--tfidf", default="data/desc_tfidf.npz")
    p.add_argument("--use-svd", action="store_true")
    p.add_argument("--svd-dim", type=int, default=300)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--results", default="results/metrics.csv")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--preprocessors-dir", default="preprocessors")
    p.add_argument("--calibrate", action="store_true", help="Calibrate probabilities (slower)")
    # calibration options
    p.add_argument("--calib-subfrac", type=float, default=0.3, help="Fraction of train used for calibration (0-1)")
    return p.parse_args()


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    }


def find_best_threshold(y_true, y_prob, metric="f1"):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1s = 2 * (prec * rec) / (prec + rec + 1e-12)
    idx = np.nanargmax(f1s)
    if idx >= len(thr):
        return 0.5
    return float(thr[idx])


def save_feature_importances(model, meta_col_names, tfidf_dim, out_path):
    """
    Save top feature importances for tree models (XGBoost/RandomForest).
    meta_col_names: list of meta column names (order must match X_meta used)
    tfidf_dim: number of TF-IDF columns used (if reduced, that's the reduced dimension)
    """
    try:
        # Attempt XGBoost booster score (gain)
        if hasattr(model, "get_booster"):
            fi = model.get_booster().get_score(importance_type="gain")
            items = []
            for k, v in fi.items():
                idx = int(k[1:])  # 'f123' -> 123
                if idx < len(meta_col_names):
                    name = meta_col_names[idx]
                else:
                    tf_idx = idx - len(meta_col_names)
                    name = f"tfidf_{tf_idx}"
                items.append((name, v))
            df_fi = pd.DataFrame(items, columns=["feature", "gain"]).sort_values("gain", ascending=False)
            df_fi.to_csv(out_path, index=False)
        # RandomForest path (feature_importances_)
        elif hasattr(model, "feature_importances_"):
            fi_arr = model.feature_importances_
            items = []
            for idx, v in enumerate(fi_arr):
                if idx < len(meta_col_names):
                    name = meta_col_names[idx]
                else:
                    tf_idx = idx - len(meta_col_names)
                    name = f"tfidf_{tf_idx}"
                items.append((name, float(v)))
            df_fi = pd.DataFrame(items, columns=["feature", "importance"]).sort_values("importance", ascending=False)
            df_fi.to_csv(out_path, index=False)
        else:
            LOG.info("Model has no known feature importance attributes.")
    except Exception as e:
        LOG.exception("Could not save feature importances: %s", e)


def main():
    args = parse_args()
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)
    Path(args.preprocessors_dir).mkdir(parents=True, exist_ok=True)

    LOG.info("Loading meta features from %s", args.meta)
    meta = pd.read_csv(args.meta)
    if "fake_label" not in meta.columns:
        raise ValueError("meta file must contain fake_label column")

    # Log meta columns & check for new features requested by referee
    LOG.info("Meta columns sample: %s", meta.columns.tolist()[:80])
    if "fraud_trigger_index" in meta.columns:
        try:
            LOG.info("Fraud Trigger Index stats: min=%.4f median=%.4f max=%.4f",
                     meta['fraud_trigger_index'].min(),
                     meta['fraud_trigger_index'].median(),
                     meta['fraud_trigger_index'].max())
        except Exception:
            LOG.info("fraud_trigger_index present but could not compute stats")
    else:
        LOG.info("fraud_trigger_index column not found in meta features")

    if "india_scam_keyword_count" in meta.columns:
        LOG.info("india_scam_keyword_count unique value counts (top): %s", meta['india_scam_keyword_count'].value_counts().head(5).to_dict())

    # load feature_groups if available
    fg_path = os.path.join(args.preprocessors_dir, "feature_groups.joblib")
    if os.path.exists(fg_path):
        try:
            feature_groups = joblib.load(fg_path)
            LOG.info("Loaded feature_groups keys: %s", list(feature_groups.keys()))
        except Exception:
            LOG.info("feature_groups exists but could not be loaded")
    else:
        LOG.info("feature_groups.joblib not found at %s", fg_path)
        feature_groups = None

    y = meta["fake_label"].values
    # Drop Job ID and label; keep numeric columns only and preserve column order for mapping later
    X_meta_df = meta.drop(columns=[c for c in ["Job ID", "fake_label"] if c in meta.columns])
    X_meta_numeric = X_meta_df.select_dtypes(include=[np.number]).fillna(0)
    meta_col_names = list(X_meta_numeric.columns)
    X_meta = X_meta_numeric.values
    LOG.info("Meta matrix shape: %s (meta cols: %d)", X_meta.shape, len(meta_col_names))

    LOG.info("Loading TF-IDF from %s", args.tfidf)
    X_tfidf = sparse.load_npz(args.tfidf)
    LOG.info("TF-IDF shape: %s", X_tfidf.shape)

    # --- SVD guard + apply ---
    if args.use_svd:
        n_features = X_tfidf.shape[1]
        desired = args.svd_dim
        if desired >= n_features:
            new_dim = max(1, n_features - 1)
            LOG.warning("svd-dim %d >= n_features %d -> reducing to %d", desired, n_features, new_dim)
            desired = new_dim
        LOG.info("Applying TruncatedSVD n_components=%d", desired)
        svd = TruncatedSVD(n_components=desired, random_state=args.random_state)
        X_tfidf_reduced = svd.fit_transform(X_tfidf)
        joblib.dump(svd, os.path.join(args.preprocessors_dir, "svd.joblib"))
        LOG.info("Saved SVD to %s", os.path.join(args.preprocessors_dir, "svd.joblib"))
        tfidf_dim = X_tfidf_reduced.shape[1]
    else:
        X_tfidf_reduced = X_tfidf
        tfidf_dim = X_tfidf.shape[1]

    # --- concatenate meta + text ---
    LOG.info("Concatenating meta + text")
    if sparse.issparse(X_tfidf_reduced):
        X = sparse.hstack([X_meta, X_tfidf_reduced]).tocsr()
    else:
        X = np.hstack([X_meta, X_tfidf_reduced])
    LOG.info("Final X shape: %s", X.shape)

    # --- train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )
    LOG.info("Train: %s Test: %s", X_train.shape, X_test.shape)

    results = []

    # Logistic Regression (sparse-friendly) - use single-core to avoid nested parallelism
    LOG.info("Training Logistic Regression (single-core solver)")
    lr = LogisticRegression(max_iter=5000, class_weight="balanced", C=2.0, solver="liblinear")
    lr_model = None
    if args.calibrate:
        LOG.info("Calibrating Logistic Regression with CV=3 (safe mode: single-core, subsample for calibration)")
        # avoid nested parallelism by ensuring single-core
        try:
            lr.n_jobs = 1
        except Exception:
            pass
        # subsample train for calibration to save time
        calib_frac = float(args.calib_subfrac) if args.calib_subfrac > 0 and args.calib_subfrac <= 1.0 else 0.3
        n_train = X_train.shape[0]
        calib_size = max(100, int(n_train * calib_frac))
        rng = np.random.RandomState(args.random_state)
        idx = rng.choice(n_train, size=calib_size, replace=False)
        X_cal = X_train[idx] if not sparse.issparse(X_train) else X_train[idx, :]
        y_cal = y_train[idx]
        lr_clf = CalibratedClassifierCV(lr, cv=3, n_jobs=1)
        lr_clf.fit(X_cal, y_cal)
        lr_model = lr_clf
    else:
        lr.fit(X_train, y_train)
        lr_model = lr

    y_prob = lr_model.predict_proba(X_test)[:, 1]
    y_pred_default = (y_prob >= 0.5).astype(int)
    best_thr = find_best_threshold(y_test, y_prob)
    y_pred_tuned = (y_prob >= best_thr).astype(int)

    m_def = compute_metrics(y_test, y_pred_default, y_prob); m_def["model"] = "LogisticRegression"; m_def["threshold"] = 0.5
    m_tun = compute_metrics(y_test, y_pred_tuned, y_prob); m_tun["model"] = "LogisticRegression_Tuned"; m_tun["threshold"] = best_thr
    results += [m_def, m_tun]
    joblib.dump(lr_model, os.path.join(args.models_dir, "logistic_regression.joblib"))
    LOG.info("Saved LogisticRegression -> %s", os.path.join(args.models_dir, "logistic_regression.joblib"))

    # Random Forest
    LOG.info("Training RandomForest")
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", random_state=args.random_state, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_prob = rf.predict_proba(X_test)[:, 1]
    y_pred_default = (y_prob >= 0.5).astype(int)
    best_thr = find_best_threshold(y_test, y_prob)
    y_pred_tuned = (y_prob >= best_thr).astype(int)
    m_def = compute_metrics(y_test, y_pred_default, y_prob); m_def["model"] = "RandomForest"; m_def["threshold"] = 0.5
    m_tun = compute_metrics(y_test, y_pred_tuned, y_prob); m_tun["model"] = "RandomForest_Tuned"; m_tun["threshold"] = best_thr
    results += [m_def, m_tun]
    joblib.dump(rf, os.path.join(args.models_dir, "random_forest.joblib"))
    LOG.info("Saved RandomForest -> %s", os.path.join(args.models_dir, "random_forest.joblib"))

    # Save RandomForest importances (mapped to feature names)
    try:
        feature_names = meta_col_names + [f"tfidf_{i}" for i in range(tfidf_dim)]
        save_feature_importances(rf, meta_col_names, tfidf_dim, out_path="results/top_features_randomforest.csv")
        LOG.info("Saved RandomForest top features -> results/top_features_randomforest.csv")
    except Exception as e:
        LOG.exception("Could not save RF importances: %s", e)

    # XGBoost
    LOG.info("Training XGBoost")
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    scale_pos = neg / pos if pos > 0 else 1.0
    LOG.info("XGBoost scale_pos_weight ~= %.4f (neg/pos)", scale_pos)
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        scale_pos_weight=scale_pos,
        random_state=args.random_state,
        n_jobs=-1,
        use_label_encoder=False
    )
    xgb.fit(X_train, y_train)
    y_prob = xgb.predict_proba(X_test)[:, 1]
    y_pred_default = (y_prob >= 0.5).astype(int)
    best_thr = find_best_threshold(y_test, y_prob)
    y_pred_tuned = (y_prob >= best_thr).astype(int)
    m_def = compute_metrics(y_test, y_pred_default, y_prob); m_def["model"] = "XGBoost"; m_def["threshold"] = 0.5
    m_tun = compute_metrics(y_test, y_pred_tuned, y_prob); m_tun["model"] = "XGBoost_Tuned"; m_tun["threshold"] = best_thr
    results += [m_def, m_tun]
    joblib.dump(xgb, os.path.join(args.models_dir, "xgboost.joblib"))
    LOG.info("Saved XGBoost -> %s", os.path.join(args.models_dir, "xgboost.joblib"))

    # Save XGBoost importances (mapped to feature names)
    try:
        feature_names = meta_col_names + [f"tfidf_{i}" for i in range(tfidf_dim)]
        save_feature_importances(xgb, meta_col_names, tfidf_dim, out_path="results/top_features_xgboost.csv")
        LOG.info("Saved XGBoost top features -> results/top_features_xgboost.csv")
    except Exception as e:
        LOG.exception("Could not save XGBoost importances: %s", e)

    # --- save metrics ---
    df_res = pd.DataFrame(results)
    df_res.to_csv(args.results, index=False)
    LOG.info("Saved metrics -> %s", args.results)
    LOG.info("\n%s", df_res.to_string(index=False))


if __name__ == "__main__":
    main()