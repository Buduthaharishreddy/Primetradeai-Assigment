"""
modeling.py
-----------
Machine Learning pipeline for predicting trade win/loss outcome.

Models trained:
  1. Logistic Regression (baseline)
  2. Random Forest       (ensemble)
  3. XGBoost             (gradient boosting)

Features used:
  - position_value      : trade notional size in USD
  - sentiment_encoded   : ordinal sentiment (0=Extreme Fear … 4=Extreme Greed)
  - direction_encoded   : 1=Buy/Long, 0=Sell/Short
  - fee                 : trade fee (proxy for market impact)
  - log_position_value  : log-transformed position size (handles skew)

Outputs:
  - Classification reports printed to stdout
  - Feature importance bar chart saved to outputs/figures/
  - ROC curve chart saved to outputs/figures/
  - Trained models saved as .pkl files to outputs/models/
  - Summary metrics dict returned for the notebook
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.utils import (
    apply_dark_style, save_fig,
    DARK_BG, PANEL_BG, TEXT_COLOR, GRID_COLOR, ACCENT
)

warnings.filterwarnings("ignore")

FEATURES = [
    "position_value",
    "sentiment_encoded",
    "direction_encoded",
    "fee",
    "log_position_value",
]
TARGET = "win"


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_ml_data(df: pd.DataFrame):
    """
    Extract features and target, drop rows with missing values,
    and split into 80/20 train/test sets (stratified by target).

    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    ml_df = df[FEATURES + [TARGET]].dropna()

    if len(ml_df) < 100:
        raise ValueError(f"Too few clean rows for ML: {len(ml_df)}")

    X = ml_df[FEATURES].values
    y = ml_df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[modeling] Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")
    print(f"[modeling] Class balance — Win: {y.mean():.1%}  Loss: {1-y.mean():.1%}\n")
    return X_train, X_test, y_train, y_test, FEATURES


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

def build_models() -> dict:
    """
    Return a dict of named sklearn-compatible model pipelines.
    Each pipeline includes a StandardScaler (required for Logistic Regression;
    harmless for tree models — keeps the API uniform).
    """
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",   # handles class imbalance
                random_state=42,
                solver="lbfgs",
            )),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=20,       # prevent overfitting on noisy trade data
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )),
        ])
    else:
        print("[modeling] XGBoost not installed — skipping XGBoost model.")

    return models


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(name: str, model, X_train, X_test, y_train, y_test) -> dict:
    """
    Fit a model, print a full classification report, and return key metrics.
    """
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(classification_report(y_test, y_pred, target_names=["Loss (0)", "Win (1)"]))

    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    if roc_auc:
        print(f"  ROC-AUC: {roc_auc:.4f}")

    # 5-fold CV accuracy on training set
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"  5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return {
        "name":    name,
        "model":   model,
        "y_pred":  y_pred,
        "y_proba": y_proba,
        "roc_auc": roc_auc,
        "cv_mean": cv_scores.mean(),
        "cv_std":  cv_scores.std(),
    }


def train_all_models(
    df: pd.DataFrame,
    output_dir_figs: str = "outputs/figures",
    output_dir_models: str = "outputs/models",
) -> dict:
    """
    Full ML pipeline: prepare data → train all models → evaluate → plot → save.
    Returns a dict of results keyed by model name.
    """
    os.makedirs(output_dir_figs, exist_ok=True)
    os.makedirs(output_dir_models, exist_ok=True)

    X_train, X_test, y_train, y_test, feature_names = prepare_ml_data(df)
    models  = build_models()
    results = {}

    print("\n═══ MODEL TRAINING & EVALUATION ═══\n")
    for name, model in models.items():
        res = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results[name] = res

    # ── Feature importance (Random Forest) ───────────────────────────────────
    if "Random Forest" in results:
        plot_feature_importance(
            results["Random Forest"]["model"],
            feature_names, output_dir_figs
        )

    # ── ROC curves for all models ─────────────────────────────────────────────
    plot_roc_curves(results, y_test, output_dir_figs)

    # ── Confusion matrix for best model ───────────────────────────────────────
    best_name = max(
        [n for n, r in results.items() if r["roc_auc"] is not None],
        key=lambda n: results[n]["roc_auc"],
        default=list(results.keys())[0]
    )
    plot_confusion_matrix(results[best_name], y_test, output_dir_figs)

    # ── Save models ───────────────────────────────────────────────────────────
    for name, res in results.items():
        safe_name = name.lower().replace(" ", "_")
        path = os.path.join(output_dir_models, f"{safe_name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(res["model"], f)
        print(f"  ✓ Model saved → {path}")

    print_model_comparison(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(model, feature_names: list[str], output_dir: str) -> None:
    """Horizontal bar chart of Random Forest feature importances."""
    apply_dark_style()

    clf = model.named_steps["clf"]
    importances = pd.Series(clf.feature_importances_, index=feature_names).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [ACCENT if v == importances.max() else "#58a6ff80" for v in importances.values]
    ax.barh(importances.index, importances.values, color=colors, edgecolor=PANEL_BG)

    for i, val in enumerate(importances.values):
        ax.text(val + 0.002, i, f"{val:.3f}", va="center", fontsize=10, color=TEXT_COLOR)

    ax.set_title("Random Forest — Feature Importances")
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x")
    fig.tight_layout()
    save_fig(fig, "11_feature_importance.png", output_dir)


def plot_roc_curves(results: dict, y_test, output_dir: str) -> None:
    """ROC curves for all models on the same axes."""
    apply_dark_style()
    COLORS = [ACCENT, "#2dc9a7", "#ff764a", "#ffa600"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color=GRID_COLOR, label="Random Baseline")

    for (name, res), color in zip(results.items(), COLORS):
        if res["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
            ax.plot(fpr, tpr, color=color, linewidth=1.8,
                    label=f"{name} (AUC={res['roc_auc']:.3f})")

    ax.set_title("ROC Curves — Win/Loss Prediction")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=9)
    ax.grid()
    fig.tight_layout()
    save_fig(fig, "12_roc_curves.png", output_dir)


def plot_confusion_matrix(res: dict, y_test, output_dir: str) -> None:
    """Styled confusion matrix for the best model."""
    apply_dark_style()

    cm = confusion_matrix(y_test, res["y_pred"])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted Loss", "Predicted Win"],
        yticklabels=["Actual Loss", "Actual Win"],
        linewidths=0.5, linecolor=DARK_BG, ax=ax,
        annot_kws={"size": 13}
    )
    ax.set_title(f"Confusion Matrix — {res['name']}")
    fig.tight_layout()
    save_fig(fig, "13_confusion_matrix.png", output_dir)


# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def print_model_comparison(results: dict) -> None:
    """Print a compact model comparison table."""
    print("\n" + "═"*60)
    print("  MODEL COMPARISON SUMMARY")
    print("═"*60)
    print(f"  {'Model':<25} {'ROC-AUC':>8}  {'CV Acc':>8}  {'CV Std':>7}")
    print("  " + "─"*50)
    for name, res in results.items():
        roc = f"{res['roc_auc']:.4f}" if res['roc_auc'] else "  N/A "
        print(f"  {name:<25} {roc:>8}  {res['cv_mean']:>7.4f}  {res['cv_std']:>6.4f}")
    print("═"*60 + "\n")
