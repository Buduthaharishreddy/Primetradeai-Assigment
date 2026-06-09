<div align="center">

# 📊 Bitcoin Market Sentiment × Trader Performance Analysis

### *Hyperliquid Perpetual Futures — Powered by Fear & Greed Index*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-006600?style=for-the-badge)](https://xgboost.readthedocs.io)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **An end-to-end data science project** analyzing how Bitcoin market sentiment (Fear & Greed Index) drives trader profitability, behavior, and risk on Hyperliquid — covering **211,218 trades** from 32 unique traders across **2 years (May 2023 – May 2025)**.

[📁 Project Structure](#-project-structure) · [🚀 Quick Start](#-quick-start) · [📈 Results](#-key-results--findings) · [🖼️ Charts](#%EF%B8%8F-visual-outputs) · [🤖 ML Models](#-machine-learning) · [🔮 Future Work](#-future-enhancements)

</div>

---

## 🧠 Project Overview

This project investigates a core question in quantitative trading:

> **Does the Bitcoin market's emotional state (fear vs greed) meaningfully affect how well traders perform?**

Using the **Alternative.me Fear & Greed Index** merged with **Hyperliquid perpetual futures execution data**, we run a complete data science pipeline covering:

- 🧹 **Data Engineering** — merging two independent datasets on daily date keys
- 🔬 **Exploratory Data Analysis** — 10 dark-theme publication-quality charts
- 📐 **Statistical Testing** — T-Test, ANOVA, Chi-Square, Pearson Correlation
- 👥 **Trader Segmentation** — identifying High / Medium / Low performers
- 🤖 **Machine Learning** — predicting win/loss with 3 models (LR, RF, XGBoost)
- 💡 **Actionable Insights** — strategy recommendations grounded in the data

---

## 📁 Project Structure

```
Primetradeai-Assignment/
│
├── 📂 data/
│   ├── fear_greed.csv              # Bitcoin Fear & Greed Index (2018–2025)
│   └── historical_data.csv         # Hyperliquid trade executions (211K rows)
│
├── 📂 notebooks/
│   └── analysis.ipynb              # Full interactive walkthrough (Jupyter)
│
├── 📂 src/
│   ├── __init__.py
│   ├── preprocessing.py            # Load → clean → merge → feature engineering
│   ├── eda.py                      # 10 charts + 4 statistical tests + segmentation
│   ├── modeling.py                 # ML pipeline (LR, RF, XGBoost)
│   └── utils.py                    # Dark-theme style helpers
│
├── 📂 outputs/
│   ├── 📂 figures/                 # 13 auto-generated PNG charts
│   ├── 📂 models/                  # 3 trained .pkl model files
│   └── 📂 reports/                 # Reserved for PDF/HTML reports
│
├── run_analysis.py                 # ▶️  One-command entry point
├── requirements.txt                # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Buduthaharishreddy/Primetradeai-Assigment.git
cd Primetradeai-Assigment

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your data files in the data/ folder
#    data/fear_greed.csv
#    data/historical_data.csv
```

### Run

**Option A — Single script (fastest)**
```bash
python run_analysis.py
```

**Option B — Jupyter Notebook (interactive, charts inline)**
```bash
jupyter notebook notebooks/analysis.ipynb
```
> Open in browser → run cells with **Shift + Enter**

All 13 charts are saved to `outputs/figures/` and all 3 trained models to `outputs/models/`.

---

## 📊 Dataset Overview

| Dataset | Source | Rows | Date Range | Key Columns |
|---|---|---|---|---|
| Fear & Greed Index | Alternative.me | 2,644 days | 2018 – 2025 | `date`, `value`, `classification` |
| Hyperliquid Trades | Primetrade.ai | 211,218 | May 2023 – May 2025 | `Account`, `Closed PnL`, `Size USD`, `Direction`, `Timestamp IST` |

**After merging on date key:** `211,218 trades × 28 engineered features`

### Sentiment Distribution in Trade Data

| Sentiment | Trades | % of Total |
|---|---|---|
| 🔴 Fear | 61,837 | 29.3% |
| 🟡 Greed | 50,303 | 23.8% |
| 🟢 Extreme Greed | 39,992 | 18.9% |
| 🟠 Neutral | 37,686 | 17.8% |
| 🔴 Extreme Fear | 21,400 | 10.1% |

---

## 🔧 Feature Engineering

The preprocessing pipeline derives the following features from raw data:

| Feature | Description | Type |
|---|---|---|
| `win` | 1 if Closed PnL > 0, else 0 | Binary |
| `loss` | 1 if Closed PnL < 0, else 0 | Binary |
| `abs_pnl` | Absolute value of Closed PnL | Float |
| `position_value` | Trade notional = Size USD | Float |
| `log_position_value` | log1p(position_value) — reduces skew | Float |
| `sentiment_encoded` | Ordinal: Extreme Fear=0 … Extreme Greed=4 | Integer |
| `sentiment_binary` | Fear-family=0, Greed-family=1 | Binary |
| `direction_encoded` | Buy/Long=1, Sell/Short=0 | Binary |

---

## 📈 Key Results & Findings

### 🏆 Sentiment vs. Profitability

| Sentiment | Avg PnL / Trade | Win Rate | Trade Count |
|---|---|---|---|
| 🟢 **Extreme Greed** | **$67.89** ← highest | **46.5%** ← highest | 39,992 |
| 🟡 Fear | $54.29 | 42.1% | 61,837 |
| 🟡 Greed | $42.74 | 38.5% | 50,303 |
| 🟠 Neutral | $34.31 | 39.7% | 37,686 |
| 🔴 Extreme Fear | $34.54 | 37.1% | 21,400 |

> **📌 Key Insight:** Extreme Greed periods deliver the highest per-trade profitability AND win rate. Fear periods show relatively high average PnL — suggesting skilled traders exploit panic-driven market dislocations.

### 📐 Statistical Test Results

| Test | Statistic | p-value | Conclusion |
|---|---|---|---|
| Welch's T-Test (Fear vs Greed) | t = 1.851 | 0.064 | Not significant at α=0.05 alone |
| **One-Way ANOVA** (all 5 groups) | F = 9.06 | **< 0.0001** | ✅ Significant differences across all sentiments |
| **Chi-Square** (Sentiment × Win/Loss) | χ² = 821.99 | **< 0.0001** | ✅ Sentiment significantly affects win/loss outcome |
| Pearson Correlation (Sentiment ↔ PnL) | r = 0.006 | — | Weak linear relationship |

### 👥 Trader Segmentation

| Segment | # Traders | Avg Total PnL | Avg Win Rate | Avg Position Size | Avg Trades |
|---|---|---|---|---|---|
| 🥇 High Performer | 11 | **$812,049** | **43%** | $9,085 | 10,750 |
| 🥈 Medium Performer | 10 | $130,680 | 39% | $2,787 | 4,871 |
| 🥉 Low Performer | 11 | $1,378 | 38% | $5,855 | 4,024 |

> **High performers** trade more frequently AND use larger position sizes, suggesting conviction-driven sizing, not random gambling.

---

## 🖼️ Visual Outputs

All 13 charts use a consistent **dark theme** optimized for professional presentations.

---

### Chart 1 — Sentiment Distribution
**`outputs/figures/01_sentiment_distribution.png`**

> Shows how many trades (out of 211K) fall within each sentiment bucket. Fear dominates with 61,837 trades (29.3%), revealing that the dataset captures significant market stress periods. This provides context for all downstream sentiment-stratified analyses.

---

### Chart 2 — Daily Profit Trend
**`outputs/figures/02_daily_profit_trend.png`**

> Time-series of total daily Closed PnL from May 2023 to May 2025, with sentiment-coloured background shading. Highlights volatility spikes, profitable streaks, and the correlation between sentiment regime shifts and aggregate trader returns over the 2-year window.

---

### Chart 3 — PnL Distribution by Sentiment (Violin)
**`outputs/figures/03_pnl_violin_by_sentiment.png`**

> Violin plots reveal the *shape* of the PnL distribution for each sentiment, not just the average. Wider violins indicate more variance. The quartile lines inside each violin show median and IQR. Clipped to 1st–99th percentile to remove extreme outliers and preserve visual clarity.

---

### Chart 4 — Average PnL per Trade by Sentiment
**`outputs/figures/04_avg_pnl_by_sentiment.png`**

> Bar chart showing mean Closed PnL per trade by sentiment category. **Extreme Greed ($67.89)** clearly leads, followed by Fear ($54.29). Green bars = positive PnL, red = negative. The clear ordering across sentiments motivates the ANOVA and statistical tests.

---

### Chart 5 — Win Rate by Sentiment
**`outputs/figures/05_win_rate_by_sentiment.png`**

> Win rate (% of trades with positive PnL) per sentiment. A 50% dashed reference line is included. **Extreme Greed achieves the highest win rate (46.5%)** while Extreme Fear has the lowest (37.1%). The consistent below-50% win rates tell us these traders rely on large wins to overcome loss frequency.

---

### Chart 6 — Position Size (USD) by Sentiment
**`outputs/figures/06_position_size_by_sentiment.png`**

> Box plots of trade notional size (USD) across sentiments. Reveals whether traders increase or decrease position size during fear vs. greed. Larger position sizes during Fear suggests **panic-driven over-sizing** or contrarian conviction bets.

---

### Chart 7 — Trade Direction Mix by Sentiment
**`outputs/figures/07_direction_by_sentiment.png`**

> 100% stacked bar showing the proportion of Buy vs. Sell trades per sentiment. A relatively balanced mix across all categories suggests these are sophisticated traders, not momentum chasers exclusively going long during greed and short during fear.

---

### Chart 8 — Long vs. Short PnL by Sentiment
**`outputs/figures/08_long_vs_short_pnl.png`**

> Side-by-side comparison of average PnL for Long vs. Short trades within each sentiment. Reveals which direction is more profitable under each emotional regime — critical for strategy design (e.g., "go long during Extreme Greed" vs. "go short during Extreme Fear").

---

### Chart 9 — Feature Correlation Heatmap
**`outputs/figures/09_correlation_heatmap.png`**

> Pearson correlation matrix across the key numeric features: PnL, position size, sentiment encoding, win flag, and fee. Lower-triangle only (removes redundancy). Notable finding: `position_value` and `closedPnL` have the strongest correlation (r=0.12), confirming that larger bets produce larger absolute outcomes.

---

### Chart 10 — Cumulative PnL Over Time
**`outputs/figures/10_cumulative_pnl.png`**

> Running total of all trade PnL from day 1 to day last. Green fill = cumulative profit, red fill = drawdown periods. Gives an at-a-glance view of the overall profitability trajectory and the depth/duration of losing streaks, which is key for risk assessment.

---

### Chart 11 — Random Forest Feature Importances
**`outputs/figures/11_feature_importance.png`**

> Horizontal bar chart of feature importance scores from the Random Forest model. Reveals which features the model finds most predictive of win/loss. `position_value` and `log_position_value` typically dominate, confirming that trade sizing is the strongest signal — larger trades tend to be better-researched.

---

### Chart 12 — ROC Curves (All 3 Models)
**`outputs/figures/12_roc_curves.png`**

> Receiver Operating Characteristic curves for Logistic Regression, Random Forest, and XGBoost plotted on the same axes. Area Under the Curve (AUC) is annotated per model. **XGBoost achieves the best AUC (0.68)**, significantly above the random baseline of 0.50. Demonstrates that sentiment and sizing features have genuine predictive power.

---

### Chart 13 — Confusion Matrix (Best Model)
**`outputs/figures/13_confusion_matrix.png`**

> Confusion matrix for XGBoost (best model) on the held-out test set (42,244 trades). Shows true positives, false positives, true negatives, false negatives. Useful for evaluating class-specific performance, especially given the 59%/41% class imbalance between losses and wins.

---

## 🤖 Machine Learning

### Models & Performance

| Model | ROC-AUC | 5-Fold CV Accuracy | Notes |
|---|---|---|---|
| Logistic Regression | 0.6025 | 51.7% | Linear baseline, fast |
| Random Forest | 0.6751 | 57.5% | Best interpretability |
| **XGBoost** | **0.6808** | **63.4%** | **Best overall performance** |

### Features Used for Prediction

```python
features = [
    "position_value",       # Trade size in USD
    "sentiment_encoded",    # Ordinal sentiment (0–4)
    "direction_encoded",    # Buy=1 / Sell=0
    "fee",                  # Trading fee paid
    "log_position_value",   # Log-transformed size (handles skew)
]
target = "win"              # 1 = profitable trade, 0 = loss
```

### Why ROC-AUC of 0.68 is meaningful
Trade outcome prediction is inherently noisy — markets are semi-efficient and short-term outcomes have high randomness. An AUC of 0.68 (vs. random baseline of 0.50) on 42,000+ real trades is a **strong result** for this domain, indicating sentiment and sizing do carry real predictive signal.

---

## 💡 Strategic Insights

Based on the full analysis, here are data-driven trading strategy recommendations:

1. **Trade more aggressively during Extreme Greed** — highest win rate (46.5%) and avg PnL ($67.89) align to maximize expected value
2. **Don't avoid Fear** — Fear periods show surprisingly high avg PnL ($54.29), suggesting expert traders profit from exploiting panic
3. **Manage position sizing in Extreme Fear** — variance is highest, risk of large losses elevated
4. **Win rate < 50% across all sentiment regimes** — winning strategies here rely on asymmetric reward (large wins outweigh frequent small losses)
5. **Larger trades ≠ reckless** — High Performers use bigger position sizes AND more trades, showing conviction-based sizing

---

## 🔮 Future Enhancements

| Enhancement | Description | Impact |
|---|---|---|
| 🕐 **Intraday sentiment** | Use hourly Fear & Greed instead of daily for finer-grained signal | High |
| 📊 **Sharpe / Sortino Ratio** | Risk-adjusted performance metrics per sentiment | High |
| 🔁 **Backtesting Engine** | Simulate a sentiment-aware trading strategy on historical data | Very High |
| 🌐 **On-chain features** | Add BTC funding rates, open interest, liquidation volumes | High |
| 🧠 **LSTM / Transformer** | Time-series deep learning models for sequential trade pattern detection | Medium |
| 📰 **NLP Sentiment** | Incorporate Twitter/Reddit sentiment to complement Fear & Greed | Medium |
| 🏦 **Cross-exchange** | Extend analysis to Binance, Bybit, dYdX for generalizability | Medium |
| 📱 **Dashboard** | Interactive Streamlit or Dash app with live Fear & Greed API | Medium |
| ⚡ **Real-time alerts** | Signal system that triggers trade recommendations on sentiment shifts | High |
| 🔍 **Coin-level analysis** | Break down BTC vs. ETH vs. altcoin performance by sentiment | Low |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Data Processing | `pandas` · `numpy` |
| Visualization | `matplotlib` · `seaborn` |
| Statistics | `scipy` (T-Test, ANOVA, Chi-Square) |
| Machine Learning | `scikit-learn` · `xgboost` |
| Notebook | `jupyter` |
| Model Persistence | `pickle` |

---

## 📝 Resume-Worthy Talking Points

When presenting this project in interviews:

1. **"Merged two independent real-world datasets"** — aligned a time-series sentiment index with 211K trade executions via date-key merging
2. **"Validated hypotheses statistically"** — applied ANOVA (p<0.0001) and Chi-Square (χ²=821.99) rather than relying on visuals alone
3. **"Handled class imbalance"** — used `class_weight='balanced'` and stratified splits for the 59/41 loss/win split
4. **"Built modular, production-style code"** — separated preprocessing, EDA, and modeling into independent importable modules
5. **"Discovered a counterintuitive finding"** — Fear periods show higher avg PnL than Greed, challenging the naive assumption that greed = better performance
6. **"Achieved 0.68 ROC-AUC on 42K test trades"** — meaningful predictive signal in a genuinely noisy domain

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">

**Built for Primetrade.ai Data Science Internship Assignment**

*By Budutha Harish Reddy*

⭐ Star this repo if you found it useful!

</div>
