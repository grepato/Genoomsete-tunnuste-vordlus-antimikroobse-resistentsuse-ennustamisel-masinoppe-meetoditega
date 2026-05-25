#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    make_scorer,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.naive_bayes import BernoulliNB
from imblearn.ensemble import BalancedRandomForestClassifier


# =========================================================
# ENV / WARNINGS
# =========================================================
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
warnings.simplefilter("ignore", FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="imblearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# =========================================================
# SETTINGS
# =========================================================
POS_LABEL = 1
OUTER_SPLITS = 5
INNER_SPLITS = 3
RANDOM_STATE = 88

def get_n_jobs(default: int = 1) -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        return max(1, int(slurm_cpus))

    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        cpu_count = os.cpu_count()
        return max(1, cpu_count if cpu_count is not None else default)

N_JOBS = get_n_jobs()

ID_COLS = ["Isolate", "Phenotype"]

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# =========================================================
# HELPERS
# =========================================================
def sanitize_name(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace("+", "plus")
    )


def get_continuous_scores(model, X_input, pos_label=1):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_input)
        model_classes = model.classes_
        pos_idx = int(np.where(model_classes == pos_label)[0][0])
        return proba[:, pos_idx]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_input)
        model_classes = model.classes_

        if np.ndim(scores) == 1:
            if model_classes[1] != pos_label:
                scores = -scores
            return scores

        raise ValueError("Multiclass decision_function detected, but this code expects binary classification.")

    raise ValueError(f"Model {type(model).__name__} has neither predict_proba nor decision_function.")


def threshold_predict(scores, threshold, pos_label, neg_label):
    return np.where(scores >= threshold, pos_label, neg_label)


def specificity_score_from_labels(y_true, y_pred, pos_label=1):
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    if len(labels) != 2:
        return np.nan

    neg_label = [x for x in labels if x != pos_label][0]
    cm = confusion_matrix(y_true, y_pred, labels=[neg_label, pos_label])
    tn, fp, fn, tp = cm.ravel()

    if (tn + fp) == 0:
        return np.nan
    return tn / (tn + fp)


def metrics_from_predictions(y_true, y_pred, scores, pos_label=1):
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    if len(labels) != 2:
        raise ValueError(f"Expected binary classification, got labels: {labels}")

    neg_label = [c for c in labels if c != pos_label][0]
    cm = confusion_matrix(y_true, y_pred, labels=[neg_label, pos_label])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    specificity = specificity_score_from_labels(y_true, y_pred, pos_label=pos_label)

    try:
        auc_val = roc_auc_score(y_true, scores)
    except ValueError:
        auc_val = np.nan

    return {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "auc": float(auc_val) if not pd.isna(auc_val) else np.nan,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "recall": float(sensitivity),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity) if not pd.isna(specificity) else np.nan,
        "f1": float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def choose_best_threshold_balacc(y_true, scores, pos_label=1):
    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError(f"Expected binary y_true, got classes: {classes}")

    neg_label = [c for c in classes if c != pos_label][0]

    unique_scores = np.unique(scores)
    candidate_thresholds = np.concatenate([
        [unique_scores.min() - 1e-12],
        unique_scores,
        [unique_scores.max() + 1e-12],
    ])

    rows = []
    for thr in candidate_thresholds:
        y_pred = threshold_predict(scores, thr, pos_label=pos_label, neg_label=neg_label)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        rows.append((thr, bal_acc, mcc, f1))

    thresh_df = pd.DataFrame(rows, columns=["threshold", "balanced_accuracy", "mcc", "f1"])
    best = thresh_df.sort_values(
        ["balanced_accuracy", "mcc", "f1", "threshold"],
        ascending=[False, False, False, False]
    ).iloc[0]

    return float(best["threshold"])


def tune_threshold_with_inner_cv(best_estimator, X_train, y_train, inner_cv, pos_label=1):
    chosen_thresholds = []

    for inner_train_idx, inner_valid_idx in inner_cv.split(X_train, y_train):
        X_inner_train = X_train.iloc[inner_train_idx]
        y_inner_train = y_train.iloc[inner_train_idx]
        X_inner_valid = X_train.iloc[inner_valid_idx]
        y_inner_valid = y_train.iloc[inner_valid_idx]

        model = clone(best_estimator)
        model.fit(X_inner_train, y_inner_train)

        scores_valid = get_continuous_scores(model, X_inner_valid, pos_label=pos_label)
        best_thr = choose_best_threshold_balacc(y_inner_valid, scores_valid, pos_label=pos_label)
        chosen_thresholds.append(best_thr)

    return float(np.median(chosen_thresholds)), chosen_thresholds


def summarize_metric_list(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def load_dataset(parquet_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(parquet_path)

    for col in ID_COLS:
        if col not in df.columns:
            raise ValueError(f"{parquet_path} is missing required column: {col}")

    df = df.copy()
    df["Isolate"] = df["Isolate"].astype(str)
    df["Phenotype"] = pd.to_numeric(df["Phenotype"], errors="raise").astype(int)

    X = df.drop(columns=ID_COLS)
    y = df["Phenotype"]

    if X.shape[1] == 0:
        raise ValueError(f"{parquet_path} has no feature columns after removing {ID_COLS}")

    if y.nunique() != 2:
        raise ValueError(f"{parquet_path} is not binary after loading. Classes: {sorted(y.unique())}")

    return X, y


# =========================================================
# SCORER
# =========================================================
balacc_scorer = make_scorer(balanced_accuracy_score)

# =========================================================
# MODELS + PARAM GRIDS
# =========================================================
model_configs: dict[str, dict[str, Any]] = {
    "Linear SVM": {
        "pipeline": Pipeline([
            ("clf", LinearSVC(class_weight="balanced", max_iter=30000))
        ]),
        "param_grid": {
            "clf__C": [0.001, 0.01, 0.1, 1, 10, 100]
        }
    },

    "RBF SVM": {
        "pipeline": Pipeline([
            ("clf", SVC(class_weight="balanced", probability=True))
        ]),
        "param_grid": {
            "clf__C": [0.1, 1, 10, 100],
            "clf__gamma": ["scale", "auto", 0.01, 0.1, 1]
        }
    },

    "LogReg (L1)": {
        "pipeline": Pipeline([
            ("clf", LogisticRegression(
                penalty="l1",
                solver="liblinear",
                class_weight="balanced",
                max_iter=30000
            ))
        ]),
        "param_grid": {
            "clf__C": [0.001, 0.01, 0.1, 1, 10, 100]
        }
    },

    "LogReg (L2)": {
        "pipeline": Pipeline([
            ("clf", LogisticRegression(
                penalty="l2",
                solver="liblinear",
                class_weight="balanced",
                max_iter=30000
            ))
        ]),
        "param_grid": {
            "clf__C": [0.001, 0.01, 0.1, 1, 10, 100]
        }
    },

    "SGD Logistic": {
        "pipeline": Pipeline([
            ("clf", SGDClassifier(
                loss="log_loss",
                class_weight="balanced",
                max_iter=10000,
                tol=1e-3,
                random_state=RANDOM_STATE
            ))
        ]),
        "param_grid": {
            "clf__alpha": [1e-5, 1e-4, 1e-3, 1e-2]
        }
    },

    "KNN": {
        "pipeline": Pipeline([
            ("clf", KNeighborsClassifier())
        ]),
        "param_grid": {
            "clf__n_neighbors": [1, 3, 5, 7, 9, 11, 15],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1]
        }
    },
    
    "Decision Tree": {
        "pipeline": Pipeline([
            ("clf", DecisionTreeClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),
        "param_grid": {
            "clf__max_depth": [None, 2, 3, 5, 10, 20],
            "clf__min_samples_split": [2, 5, 10, 20],
            "clf__min_samples_leaf": [1, 2, 5, 10]
        }
    },

    "Random Forest": {
        "pipeline": Pipeline([
            ("clf", RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1
            ))
        ]),
        "param_grid": {
            "clf__n_estimators": [200, 500],
            "clf__max_depth": [None, 3, 5, 10, 20],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 5]
        }
    },

    "Balanced Random Forest": {
        "pipeline": Pipeline([
            ("clf", BalancedRandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=1
            ))
        ]),
        "param_grid": {
            "clf__n_estimators": [200, 500],
            "clf__max_depth": [None, 3, 5, 10, 20],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 5]
        }
    },

    "Extra Trees": {
        "pipeline": Pipeline([
            ("clf", ExtraTreesClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1
            ))
        ]),
        "param_grid": {
            "clf__n_estimators": [200, 500],
            "clf__max_depth": [None, 3, 5, 10, 20],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 5]
        }
    },

    "Gradient Boosting": {
        "pipeline": Pipeline([
            ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE))
        ]),
        "param_grid": {
            "clf__n_estimators": [100, 200, 500],
            "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "clf__max_depth": [1, 2, 3]
        }
    },

    "HistGB": {
        "pipeline": Pipeline([
            ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE))
        ]),
        "param_grid": {
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [3, 6],
            "clf__max_leaf_nodes": [31, 63],
            "clf__min_samples_leaf": [10, 50],
            "clf__l2_regularization": [0.0, 1.0]
        }
    },

    "AdaBoost": {
        "pipeline": Pipeline([
            ("clf", AdaBoostClassifier(random_state=RANDOM_STATE))
        ]),
        "param_grid": {
            "clf__n_estimators": [50, 100, 200, 500],
            "clf__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0]
        }
    },

    "BernoulliNB": {
        "pipeline": Pipeline([
            ("clf", BernoulliNB())
        ]),
        "param_grid": {
            "clf__alpha": [0.01, 0.1, 1.0, 10.0],
            "clf__binarize": [None]
        }
    }
}

# =========================================================
# EVALUATION
# =========================================================
def evaluate_dataset_with_model(
    dataset_name: str,
    model_name: str,
    config: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    pos_label: int = 1,
) -> dict[str, Any]:
    outer_cv = StratifiedKFold(
        n_splits=OUTER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_metrics = []
    fold_best_params = []
    fold_thresholds = []

    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"{dataset_name}: expected 2 classes, got {classes}")
    neg_label = [c for c in classes if c != pos_label][0]

    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        print(f"  Outer fold {fold_i}/{OUTER_SPLITS}", flush=True)

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        inner_cv = StratifiedKFold(
            n_splits=INNER_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        grid = GridSearchCV(
            estimator=config["pipeline"],
            param_grid=config["param_grid"],
            scoring=balacc_scorer,
            cv=inner_cv,
            n_jobs=N_JOBS,
            refit=True,
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        fold_best_params.append(grid.best_params_)
        
        best_threshold, _ = tune_threshold_with_inner_cv(
            best_model,
            X_train,
            y_train,
            inner_cv=inner_cv,
            pos_label=pos_label,
        )
        fold_thresholds.append(best_threshold)
        
        scores_test = get_continuous_scores(best_model, X_test, pos_label=pos_label)
        y_pred = threshold_predict(scores_test, best_threshold, pos_label=pos_label, neg_label=neg_label)

        fold_result = metrics_from_predictions(y_test, y_pred, scores_test, pos_label=pos_label)
        fold_result["threshold"] = float(best_threshold)
        fold_metrics.append(fold_result)

        print(
            f"    best_params={grid.best_params_} | "
            f"threshold={best_threshold:.6f} | "
            f"bal_acc={fold_result['balanced_accuracy']:.3f} | "
            f"auc={fold_result['auc']:.3f} | "
            f"mcc={fold_result['mcc']:.3f}",
            flush=True
        )

    metric_names = [
        "tp", "fp", "tn", "fn",
        "auc", "accuracy", "balanced_accuracy",
        "recall", "sensitivity", "specificity",
        "f1", "mcc", "threshold"
    ]

    summary = {
        "dataset": dataset_name,
        "model": model_name,
        "n_features": X.shape[1],
        "fold_best_params": fold_best_params,
        "fold_thresholds": fold_thresholds,
        "median_threshold": float(np.median(fold_thresholds)),
    }

    for metric in metric_names:
        vals = [fm[metric] for fm in fold_metrics]
        mean_val, std_val = summarize_metric_list(vals)
        summary[f"{metric}_mean"] = mean_val
        summary[f"{metric}_std"] = std_val

    return summary


def train_final_model(
    dataset_name: str,
    model_name: str,
    config: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    pos_label: int = 1,
) -> dict[str, Any]:
    full_inner_cv = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    grid = GridSearchCV(
        estimator=config["pipeline"],
        param_grid=config["param_grid"],
        scoring=balacc_scorer,
        cv=full_inner_cv,
        n_jobs=N_JOBS,
        refit=True,
    )
    grid.fit(X, y)

    final_model = grid.best_estimator_
    final_threshold, per_fold_thresholds = tune_threshold_with_inner_cv(
        final_model,
        X,
        y,
        inner_cv=full_inner_cv,
        pos_label=pos_label,
    )

    payload = {
        "dataset": dataset_name,
        "model_name": model_name,
        "model": final_model,
        "threshold": final_threshold,
        "thresholds_per_fold": per_fold_thresholds,
        "best_params": grid.best_params_,
        "feature_columns": list(X.columns),
        "pos_label": pos_label,
    }

    filename = f"{sanitize_name(dataset_name)}__{sanitize_name(model_name)}.pkl"
    filepath = MODELS_DIR / filename
    joblib.dump(payload, filepath)

    return {
        "dataset": dataset_name,
        "model": model_name,
        "filepath": str(filepath),
        "best_params": grid.best_params_,
        "threshold": final_threshold,
        "n_features": X.shape[1],
    }


# =========================================================
# MAIN
# =========================================================
def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python train_one_dataset_balacc.py /path/to/dataset.parquet"
        )

    parquet_path = Path(sys.argv[1])
    if not parquet_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {parquet_path}")

    dataset_name = parquet_path.stem

    print("==================================================", flush=True)
    print(f"DATASET: {dataset_name}", flush=True)
    print("==================================================", flush=True)

    X, y = load_dataset(parquet_path)
    print(f"Rows: {len(X)} | Features: {X.shape[1]} | Classes: {sorted(y.unique())}", flush=True)

    dataset_results = []
    final_model_info = []

    for model_name, config in model_configs.items():
        print(f"\nModel: {model_name}", flush=True)

        result = evaluate_dataset_with_model(
            dataset_name=dataset_name,
            model_name=model_name,
            config=config,
            X=X,
            y=y,
            pos_label=POS_LABEL,
        )
        dataset_results.append(result)

        final_info = train_final_model(
            dataset_name=dataset_name,
            model_name=model_name,
            config=config,
            X=X,
            y=y,
            pos_label=POS_LABEL,
        )
        final_model_info.append(final_info)

    dataset_results_df = pd.DataFrame(dataset_results).sort_values(
        ["balanced_accuracy_mean", "mcc_mean", "auc_mean", "f1_mean"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    best_result_df = dataset_results_df.head(1).copy()
    final_model_info_df = pd.DataFrame(final_model_info).sort_values(
        ["model"]
    ).reset_index(drop=True)

    dataset_results_path = RESULTS_DIR / f"{sanitize_name(dataset_name)}__all_models_summary.csv"
    best_result_path = RESULTS_DIR / f"{sanitize_name(dataset_name)}__best_model_summary.csv"
    saved_models_path = RESULTS_DIR / f"{sanitize_name(dataset_name)}__saved_models.csv"

    dataset_results_df.to_csv(dataset_results_path, index=False)
    best_result_df.to_csv(best_result_path, index=False)
    final_model_info_df.to_csv(saved_models_path, index=False)

    print("\n==================================================", flush=True)
    print("FINISHED", flush=True)
    print("==================================================", flush=True)
    print(f"Saved all-model summary: {dataset_results_path}", flush=True)
    print(f"Saved best-model summary: {best_result_path}", flush=True)
    print(f"Saved model registry: {saved_models_path}", flush=True)


if __name__ == "__main__":
    main()