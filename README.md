# 🔮 StockOracle AI — Stock Prediction Engine

> **Predict whether a stock will Rise, Fall, or Stay Flat using 15 Supervised Machine Learning Models — powered by live Yahoo Finance data.**

## 📌 What Is This?

**StockOracle AI** is a fully interactive stock prediction dashboard built with Streamlit. It fetches live historical price data from Yahoo Finance, engineers 40+ technical features, trains up to 15 different supervised ML classification models, and predicts whether a selected stock will **BUY (rise)**, **SELL (fall)**, or **HOLD (stay flat)** on the next trading day.

It is designed for:
- 📊 **Traders & investors** who want data-driven signals
- 🎓 **Students & researchers** learning applied ML on financial data
- 💻 **Developers** building ML pipelines on time-series data

> ⚠️ This tool is for **educational and informational purposes only**. It does not constitute financial advice.

---

## 🚀 Quick Start

### 1. Clone or download
```bash
git clone https://github.com/YOUR_USERNAME/stock-prediction.git
cd stock-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run predict_app.py
```

Your browser opens at `http://localhost:8501` automatically.

---

## 📁 File Structure

```
stock-prediction/
├── predict_app.py       ← Entire application (single file)
├── requirements.txt     ← All dependencies
└── README.md            ← This file
```

---

## 🌐 Deploy Free (Public Access)

### Streamlit Community Cloud (Recommended)
1. Push this repo to GitHub
2. Go to **https://share.streamlit.io**
3. Connect your GitHub → select `predict_app.py` → click **Deploy**
4. Get a permanent public URL instantly — no credit card needed

### Other free options
| Platform | Notes |
|---|---|
| **Render** | Free tier, add `streamlit run predict_app.py --server.port=10000 --server.address=0.0.0.0` as start command |
| **Railway** | $5 free credit/month, very fast deploys |
| **Hugging Face Spaces** | Select Streamlit as SDK, free forever |

---

## 🗂️ App Structure — What You'll Find

### 🔧 Sidebar Controls

| Control | What It Does |
|---|---|
| **Search box** | Filter stocks by name or ticker symbol |
| **Stock selector** | Choose from 100+ global stocks and assets |
| **Data Period** | 6 Months / 1 Year / 2 Years / 5 Years of historical training data |
| **Hold Band (±%)** | Threshold to define HOLD — returns within ±X% are classified as HOLD, not BUY/SELL |
| **Select All 15 Models** | Toggle all ML models on or off at once |
| **Individual model checkboxes** | Pick exactly which models to train |
| **Test Split %** | Percentage of data reserved for testing (10–40%) |
| **CV Folds** | Number of cross-validation folds (3–10) |
| **Use Cross-Validation toggle** | Enable/disable CV scoring alongside test accuracy |
| **TRAIN & PREDICT button** | Triggers full pipeline: fetch → feature engineering → train → predict |

---

## 📊 The 5 Analysis Tabs

### 🏆 Tab 1 — Model Leaderboard

The central ranking hub. After training, every model is ranked by test accuracy.

**What you'll find:**
- Horizontal bar chart comparing **Test Accuracy** and **CV Accuracy** side by side for all trained models
- Medal rankings (🥇🥈🥉) with full metric cards for every model showing:
  - Test Accuracy %
  - F1 Score %
  - CV Score % (if enabled)
  - Next-day BUY / HOLD / SELL signal prediction
  - Model type (Ensemble, Linear, Neural Net, etc.)
  - Speed indicator (Fast / Medium / Slow)
- Visual accuracy progress bar per model
- "BEST" badge on the top-performing model
- Error log for any models that failed to train

---

### 📊 Tab 2 — Deep Analysis

Drill into any single model in detail.

**What you'll find:**
- **Model selector** — pick any trained model to inspect
- **5 metric pills** — Test Accuracy, F1 Score, Precision, Recall, CV Score
- **Confusion Matrix heatmap** — shows where the model is correct and where it confuses BUY/SELL/HOLD
- **Signal Distribution chart** — Predicted vs Actual class counts stacked bar chart
- **ROC Curves (One-vs-Rest)** — one curve per class (BUY, HOLD, SELL) with AUC scores
- **Top 20 Feature Importances** — horizontal bar chart showing which of the 40+ features the model relies on most (available for tree-based models and linear models)
- **Full Classification Report** — expandable table with per-class precision, recall, F1, and support counts

---

### 🔄 Tab 3 — Compare Models

Side-by-side comparison of multiple models at once.

**What you'll find:**
- **Model multi-selector** — choose 2 to 8 models to compare simultaneously
- **Radar Chart** — multi-axis spider chart plotting Accuracy, F1, Precision, and Recall for each selected model overlaid on the same chart
- **Grouped Bar Chart** — all 4 metrics side by side per model
- **Next-Day Signal Agreement Table** — shows what each model predicts (BUY/HOLD/SELL) for the next day, with probability breakdown and accuracy
- **Model Consensus Bar Chart** — shows what % of selected models voted BUY vs HOLD vs SELL

---

### 💡 Tab 4 — Buy / Sell / Hold

The final recommendation engine. This is the actionable output.

**What you'll find:**
- **Model selector** — choose which model's signal to base the recommendation on
- **Big Signal Display** — large BUY / HOLD / SELL card with color (green/yellow/red), description, and the model + accuracy it's based on
- **3 Probability Gauges** — circular gauge charts showing the model's confidence in SELL %, HOLD %, and BUY % for the next day
- **Ensemble Vote Panel** — combines signals from ALL trained models:
  - BUY vote count + accuracy-weighted percentage
  - HOLD vote count + accuracy-weighted percentage
  - SELL vote count + accuracy-weighted percentage
  - Final **Ensemble Signal** (accuracy-weighted majority vote)
- **Per-model signal breakdown** — lists every model under its signal group (BUY / HOLD / SELL) with its accuracy score
- **Risk Disclaimer** — clearly states this is not financial advice

---

### 📈 Tab 5 — Price + Signals

Visualizes the model's predictions overlaid directly on the price chart.

**What you'll find:**
- **Model selector** — choose which model's signals to visualize
- **Price chart** with buy/sell signal markers:
  - 🟢 Green upward triangles = BUY signal days
  - 🔴 Red downward triangles = SELL signal days
  - Grey = HOLD (no marker, price line only)
- **Volume bar chart** below the price chart (green/red colored by up/down days)
- **Signal counts** — total BUY, HOLD, SELL signals generated on the test set
- **Test Accuracy pill** for the selected model
- **BUY/SELL Probability Timeline** — line chart showing the model's BUY probability and SELL probability evolving over the test period

---

## 🤖 The 15 ML Models

| # | Model | Type | Speed | Notes |
|---|-------|------|-------|-------|
| 1 | **Logistic Regression** | Linear | ⚡ Fast | Baseline linear classifier |
| 2 | **Random Forest** | Ensemble | ⚡ Fast | 200 trees, robust to noise |
| 3 | **Gradient Boosting** | Ensemble | 🐢 Slow | Sequential boosting, high accuracy |
| 4 | **Extra Trees** | Ensemble | ⚡ Fast | Randomized splits, fast & strong |
| 5 | **AdaBoost** | Ensemble | 🔶 Medium | Adaptive boosting on weak learners |
| 6 | **Support Vector Machine** | Kernel | 🐢 Slow | RBF kernel, strong on smaller datasets |
| 7 | **K-Nearest Neighbors** | Instance | 🔶 Medium | K=7 neighbors, non-parametric |
| 8 | **Decision Tree** | Tree | ⚡ Fast | Depth-8 tree, interpretable |
| 9 | **Naive Bayes** | Probabilistic | ⚡ Fast | Gaussian, fast baseline |
| 10 | **LDA** | Linear | ⚡ Fast | Linear Discriminant Analysis |
| 11 | **QDA** | Quadratic | ⚡ Fast | Quadratic DA with reg_param=0.3 |
| 12 | **Ridge Classifier** | Linear | ⚡ Fast | L2-regularized linear classifier |
| 13 | **SGD Classifier** | Linear | ⚡ Fast | Stochastic gradient descent |
| 14 | **MLP Neural Network** | Neural Net | 🐢 Slow | 3-layer: 128→64→32 neurons |
| 15 | **Bagging Classifier** | Ensemble | 🔶 Medium | Bootstrap aggregation, 100 estimators |

---

## 🧠 Feature Engineering — 40+ Features

The app automatically engineers the following features from raw OHLCV data:

### Moving Averages
- Simple Moving Averages: SMA 5, 10, 20, 50, 200
- Exponential Moving Averages: EMA 5, 10, 20, 50, 200
- Price vs SMA ratio for each window

### Returns
- Daily returns: 1d, 2d, 3d, 5d, 10d, 20d lookback

### Volatility
- Rolling standard deviation of returns: 5d, 10d, 20d windows

### Momentum Indicators
- **RSI** (Relative Strength Index, 14-period)
- **MACD** (12, 26, 9) — line, signal, histogram
- **Stochastic Oscillator** — %K and %D
- **Williams %R** (14-period)
- **Rate of Change** — ROC 5, 10, 20

### Volatility Indicators
- **Bollinger Bands** — upper, lower, width, position within bands
- **ATR** (Average True Range, 14-period) — absolute and % of price

### Volume Indicators
- **OBV** (On-Balance Volume) — raw and signal
- Volume ratio vs 20-day average
- Volume % change

### Candlestick Pattern Features
- High-low range as % of close
- Open-to-close % change
- Upper shadow size
- Lower shadow size

### Target Variable
- **BUY** — next day return > +Hold Band %
- **SELL** — next day return < −Hold Band %
- **HOLD** — next day return within ±Hold Band %

---

## 📈 Supported Stocks & Assets

100+ global stocks and assets including:

**US Technology:** AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, AMD, INTC, ORCL, CRM, ADBE, NFLX, and more

**US Finance:** JPM, V, MA, BAC, GS, MS, BLK, AXP, WFC, SCHW, and more

**US Healthcare:** LLY, UNH, JNJ, ABBV, MRK, PFE, AMGN, TMO, and more

**US Consumer:** WMT, COST, HD, MCD, SBUX, NKE, TGT, LOW, and more

**US Energy:** XOM, CVX, COP, SLB, and more

**US Industrials:** GE, CAT, RTX, HON, UPS, BA, LMT, DE, and more

**Global:** TSM, ASML, NVO, TM, SAP, SHOP, SPOT, and more

**Crypto & Commodities:** BTC-USD, ETH-USD, GC=F (Gold), CL=F (Oil)

**Indices:** ^GSPC (S&P 500), ^DJI (Dow Jones), ^IXIC (NASDAQ)

---

## ⚙️ How the Pipeline Works

```
Yahoo Finance (live data)
        ↓
Raw OHLCV History (up to 5 years)
        ↓
Feature Engineering (40+ technical indicators)
        ↓
Label Creation (BUY / HOLD / SELL based on next-day return)
        ↓
Train/Test Split (time-ordered, no data leakage)
        ↓
StandardScaler (fit on train, transform both)
        ↓
Train 15 ML Models (+ optional cross-validation)
        ↓
Evaluate: Accuracy, F1, Precision, Recall, ROC-AUC
        ↓
Predict next day → BUY / HOLD / SELL + probabilities
        ↓
Ensemble vote (accuracy-weighted across all models)
```

---

## 📦 Dependencies

```
streamlit>=1.32.0
yfinance>=0.2.37
pandas>=2.0.0
numpy>=1.26.0
plotly>=5.20.0
scikit-learn>=1.4.0
joblib>=1.3.0
requests>=2.31.0
```

---

## 🔑 Key Design Decisions

**No data leakage** — The train/test split is always time-ordered (oldest data trains, newest data tests). No shuffling that would let the model "see the future."

**Hold Band** — A configurable dead zone (default ±0.5%) prevents labeling tiny noise movements as BUY or SELL. This makes the 3-class problem more meaningful.

**StandardScaler** — Fitted only on training data and applied to test data, preventing information from the test set bleeding into preprocessing.

**1-hour cache** — `@st.cache_data(ttl=3600)` prevents hammering Yahoo Finance on every UI interaction while keeping data fresh.

**Session state for models** — Trained models are stored in `st.session_state` so switching tabs doesn't retrain everything from scratch.

---

## ⚠️ Limitations & Important Notes

- Predictions are based purely on **historical price patterns** — they do not account for news, earnings, macroeconomics, or sentiment
- Stock markets are **inherently unpredictable** — no model achieves consistent accuracy above ~60% on daily direction
- The **ensemble signal** is generally more reliable than any single model
- Higher accuracy on the leaderboard does not guarantee future performance
- **Not financial advice** — always do your own research before making investment decisions

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built with ❤️ using Streamlit, scikit-learn, yfinance, and Plotly*
