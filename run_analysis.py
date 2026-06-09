"""
run_analysis.py
---------------
Entry-point script. Run this from the project root:

    python run_analysis.py

It executes the full pipeline:
  1. Load & preprocess data
  2. EDA — 10 dark-theme charts + statistical tests + trader segmentation
  3. ML  — Logistic Regression, Random Forest, XGBoost

All outputs (figures, models) are written to the outputs/ directory.
"""

import sys
import os

# Allow running from the project root without installing the package
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing import load_and_prepare
from src.eda import run_full_eda
from src.modeling import train_all_models

# ── Paths ──────────────────────────────────────────────────────────────────────
FEAR_GREED_PATH  = "data/fear_greed.csv"
HISTORICAL_PATH  = "data/historical_data.csv"
FIGURES_DIR      = "outputs/figures"
MODELS_DIR       = "outputs/models"


def main():
    print("=" * 60)
    print("  Bitcoin Sentiment × Trader Performance Analysis")
    print("  Primetrade.ai — Hyperliquid Historical Data")
    print("=" * 60)

    # ── Step 1: Load & preprocess ──────────────────────────────────────────────
    print("\n[STEP 1] Loading and preprocessing data …")
    df = load_and_prepare(FEAR_GREED_PATH, HISTORICAL_PATH)

    # ── Step 2: EDA ────────────────────────────────────────────────────────────
    print("\n[STEP 2] Running EDA …")
    stats_results = run_full_eda(df, output_dir=FIGURES_DIR)

    # ── Step 3: Machine Learning ───────────────────────────────────────────────
    print("\n[STEP 3] Training ML models …")
    ml_results = train_all_models(df, output_dir_figs=FIGURES_DIR, output_dir_models=MODELS_DIR)

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Figures  → {FIGURES_DIR}/")
    print(f"  Models   → {MODELS_DIR}/")
    print("\n  Key Findings:")

    if "ttest" in stats_results:
        t = stats_results["ttest"]
        better = "Greed" if t["mean_greed"] > t["mean_fear"] else "Fear"
        print(f"  • Higher avg PnL during {better} periods "
              f"(Fear={t['mean_fear']:.2f}, Greed={t['mean_greed']:.2f})")
        sig = "statistically significant" if t["significant"] else "NOT statistically significant"
        print(f"  • T-Test p={t['p_value']:.4f} — {sig}")

    best_model = max(
        [(n, r) for n, r in ml_results.items() if r["roc_auc"]],
        key=lambda x: x[1]["roc_auc"],
        default=(None, {"roc_auc": None})
    )
    if best_model[0]:
        print(f"  • Best ML model: {best_model[0]}  (ROC-AUC={best_model[1]['roc_auc']:.4f})")

    print("\n  Open outputs/figures/ to view all charts.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
