import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── ML Imports ──────────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier,
                               VotingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                              roc_auc_score, f1_score, precision_score, recall_score,
                              roc_curve, auc)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import joblib
import io
import base64

# ── Page Config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockOracle AI | Prediction Engine",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&family=Outfit:wght@300;400;600;700;800;900&display=swap');

:root {
    --bg: #04080f;
    --surface: #080f1e;
    --surface2: #0c1527;
    --surface3: #111d35;
    --border: #162440;
    --border2: #1e3060;
    --cyan: #00f5ff;
    --green: #00ff88;
    --red: #ff3366;
    --yellow: #ffcc00;
    --orange: #ff6b35;
    --purple: #8b5cf6;
    --blue: #3b82f6;
    --text: #e2eaf5;
    --muted: #4a6080;
    --muted2: #2a3f5f;
}

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border2) !important;
}

/* Header glow */
.oracle-header {
    background: linear-gradient(135deg, #04080f 0%, #0c1527 50%, #04080f 100%);
    border: 1px solid var(--border2);
    border-radius: 16px;
    padding: 28px 32px;
    position: relative;
    overflow: hidden;
    margin-bottom: 24px;
}
.oracle-header::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(0,245,255,0.04) 0%, transparent 60%),
                radial-gradient(circle at 70% 50%, rgba(139,92,246,0.04) 0%, transparent 60%);
    pointer-events: none;
}

/* Model cards */
.model-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 8px;
    transition: border-color 0.2s;
    position: relative;
    overflow: hidden;
}
.model-card:hover { border-color: var(--border2); }
.model-card.selected { border-color: var(--cyan) !important; }
.model-card.best { border-color: var(--green) !important; }
.model-card::after {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 3px; height: 100%;
    background: transparent;
}
.model-card.best::after { background: var(--green); }
.model-card.selected::after { background: var(--cyan); }

/* Metric pill */
.metric-pill {
    display: inline-block;
    background: var(--surface3);
    border: 1px solid var(--border2);
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
}
.metric-pill .label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 1.5px; }
.metric-pill .value { font-family: 'JetBrains Mono', monospace; font-size: 20px; font-weight: 700; }

/* Accuracy bar */
.acc-bar-wrap { background: var(--surface3); border-radius: 4px; height: 8px; margin-top: 6px; }
.acc-bar { height: 8px; border-radius: 4px; }

/* Signal badge */
.signal-buy   { background: rgba(0,255,136,0.15); border: 1px solid var(--green); color: var(--green); border-radius: 8px; padding: 6px 18px; font-weight: 700; }
.signal-sell  { background: rgba(255,51,102,0.15); border: 1px solid var(--red);   color: var(--red);   border-radius: 8px; padding: 6px 18px; font-weight: 700; }
.signal-hold  { background: rgba(255,204,0,0.15);  border: 1px solid var(--yellow); color: var(--yellow); border-radius: 8px; padding: 6px 18px; font-weight: 700; }

/* Section header */
.sec-hdr {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 2.5px; color: var(--cyan); padding: 6px 0;
    border-bottom: 1px solid var(--border2); margin-bottom: 16px;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface2); border-radius: 10px;
    padding: 4px; gap: 4px; border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent; color: var(--muted);
    border-radius: 8px; font-size: 13px; font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #162440, #1e3060) !important;
    color: var(--cyan) !important; border: 1px solid var(--border2) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #162440, #1e3060) !important;
    color: var(--cyan) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 8px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1e3060, #2a4080) !important;
    border-color: var(--cyan) !important;
}

/* Progress bar */
.stProgress > div > div { background: var(--cyan) !important; }

/* DataFrame */
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ── TOP 200 STOCKS ───────────────────────────────────────────────────────────────
TOP_200 = {
    "AAPL":"Apple Inc.","MSFT":"Microsoft","NVDA":"NVIDIA","GOOGL":"Alphabet A",
    "AMZN":"Amazon","META":"Meta Platforms","TSLA":"Tesla","AVGO":"Broadcom",
    "ORCL":"Oracle","AMD":"AMD","INTC":"Intel","QCOM":"Qualcomm",
    "CRM":"Salesforce","ADBE":"Adobe","NFLX":"Netflix","NOW":"ServiceNow",
    "INTU":"Intuit","TXN":"Texas Instruments","AMAT":"Applied Materials",
    "MU":"Micron","IBM":"IBM","DELL":"Dell","HPQ":"HP Inc.",
    "JPM":"JPMorgan","V":"Visa","MA":"Mastercard","BAC":"Bank of America",
    "WFC":"Wells Fargo","GS":"Goldman Sachs","MS":"Morgan Stanley",
    "BLK":"BlackRock","AXP":"Amex","C":"Citigroup","SCHW":"Schwab",
    "LLY":"Eli Lilly","UNH":"UnitedHealth","JNJ":"J&J","ABBV":"AbbVie",
    "MRK":"Merck","TMO":"Thermo Fisher","DHR":"Danaher","ABT":"Abbott",
    "PFE":"Pfizer","AMGN":"Amgen","GILD":"Gilead","ISRG":"Intuitive Surgical",
    "WMT":"Walmart","COST":"Costco","HD":"Home Depot","MCD":"McDonald's",
    "SBUX":"Starbucks","NKE":"Nike","TGT":"Target","LOW":"Lowe's",
    "XOM":"Exxon Mobil","CVX":"Chevron","COP":"ConocoPhillips","SLB":"SLB",
    "GE":"GE Aerospace","CAT":"Caterpillar","RTX":"RTX","HON":"Honeywell",
    "UPS":"UPS","BA":"Boeing","LMT":"Lockheed","DE":"Deere",
    "T":"AT&T","VZ":"Verizon","TMUS":"T-Mobile","NEE":"NextEra",
    "TSM":"TSMC","ASML":"ASML","NVO":"Novo Nordisk","TM":"Toyota",
    "PYPL":"PayPal","UBER":"Uber","ABNB":"Airbnb","COIN":"Coinbase",
    "CRWD":"CrowdStrike","PANW":"Palo Alto","SNOW":"Snowflake",
    "PLTR":"Palantir","DDOG":"Datadog","NET":"Cloudflare","MDB":"MongoDB",
    "SHOP":"Shopify","SPOT":"Spotify","DIS":"Disney","CMCSA":"Comcast",
    "PG":"P&G","KO":"Coca-Cola","PEP":"PepsiCo","BRK-B":"Berkshire B",
    "SPGI":"S&P Global","MCO":"Moody's","CME":"CME Group",
    "AMT":"American Tower","PLD":"Prologis","EQIX":"Equinix",
    "LIN":"Linde","SHW":"Sherwin-Williams","ECL":"Ecolab",
    "BKNG":"Booking Holdings","MAR":"Marriott","ZM":"Zoom",
    "OKTA":"Okta","ZS":"Zscaler","S":"SentinelOne","PATH":"UiPath",
    "^GSPC":"S&P 500 Index","^DJI":"Dow Jones","^IXIC":"NASDAQ",
    "BTC-USD":"Bitcoin","ETH-USD":"Ethereum","GC=F":"Gold","CL=F":"Oil",
}

# ── 15 SUPERVISED MODELS ─────────────────────────────────────────────────────────
ALL_MODELS = {
    "Logistic Regression":      LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":            RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting":        GradientBoostingClassifier(n_estimators=150, random_state=42),
    "XGBoost (Extra Trees)":    ExtraTreesClassifier(n_estimators=200, random_state=42),
    "AdaBoost":                 AdaBoostClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine":   SVC(kernel='rbf', probability=True, random_state=42),
    "K-Nearest Neighbors":      KNeighborsClassifier(n_neighbors=7),
    "Decision Tree":            DecisionTreeClassifier(max_depth=8, random_state=42),
    "Naive Bayes":              GaussianNB(),
    "LDA":                      LinearDiscriminantAnalysis(),
    "QDA":                      QuadraticDiscriminantAnalysis(reg_param=0.3),
    "Ridge Classifier":         RidgeClassifier(),
    "SGD Classifier":           SGDClassifier(loss='modified_huber', random_state=42, max_iter=1000),
    "MLP Neural Network":       MLPClassifier(hidden_layer_sizes=(128,64,32), max_iter=500, random_state=42),
    "Bagging Classifier":       BaggingClassifier(n_estimators=100, random_state=42),
}

MODEL_INFO = {
    "Logistic Regression":     {"type": "Linear",      "speed": "⚡ Fast",    "icon": "📐"},
    "Random Forest":           {"type": "Ensemble",    "speed": "⚡ Fast",    "icon": "🌲"},
    "Gradient Boosting":       {"type": "Ensemble",    "speed": "🐢 Slow",   "icon": "🚀"},
    "XGBoost (Extra Trees)":   {"type": "Ensemble",    "speed": "⚡ Fast",    "icon": "⚡"},
    "AdaBoost":                {"type": "Ensemble",    "speed": "🔶 Medium", "icon": "🎯"},
    "Support Vector Machine":  {"type": "Kernel",      "speed": "🐢 Slow",   "icon": "🔵"},
    "K-Nearest Neighbors":     {"type": "Instance",    "speed": "🔶 Medium", "icon": "🔗"},
    "Decision Tree":           {"type": "Tree",        "speed": "⚡ Fast",    "icon": "🌿"},
    "Naive Bayes":             {"type": "Probabilistic","speed": "⚡ Fast",   "icon": "🎲"},
    "LDA":                     {"type": "Linear",      "speed": "⚡ Fast",    "icon": "📊"},
    "QDA":                     {"type": "Quadratic",   "speed": "⚡ Fast",    "icon": "📈"},
    "Ridge Classifier":        {"type": "Linear",      "speed": "⚡ Fast",    "icon": "📏"},
    "SGD Classifier":          {"type": "Linear",      "speed": "⚡ Fast",    "icon": "🏃"},
    "MLP Neural Network":      {"type": "Neural Net",  "speed": "🐢 Slow",   "icon": "🧠"},
    "Bagging Classifier":      {"type": "Ensemble",    "speed": "🔶 Medium", "icon": "🛍️"},
}


# ── Feature Engineering ──────────────────────────────────────────────────────────
def build_features(hist):
    df = hist.copy()
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # Price-based features
    for w in [5, 10, 20, 50, 200]:
        df[f"SMA_{w}"] = close.rolling(w).mean()
        df[f"EMA_{w}"] = close.ewm(span=w, adjust=False).mean()
        df[f"close_vs_sma{w}"] = (close / df[f"SMA_{w}"]) - 1

    # Returns
    for d in [1, 2, 3, 5, 10, 20]:
        df[f"ret_{d}d"] = close.pct_change(d)

    # Volatility
    df["vol_5"]  = df["ret_1d"].rolling(5).std()
    df["vol_10"] = df["ret_1d"].rolling(10).std()
    df["vol_20"] = df["ret_1d"].rolling(20).std()

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]

    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BB_upper"] = sma20 + 2 * std20
    df["BB_lower"] = sma20 - 2 * std20
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / sma20
    df["BB_pos"]   = (close - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"]).replace(0, np.nan)

    # Stochastic
    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df["%K"] = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
    df["%D"] = df["%K"].rolling(3).mean()

    # ATR
    hl  = high - low
    hc  = (high - close.shift()).abs()
    lc  = (low  - close.shift()).abs()
    df["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df["ATR_pct"] = df["ATR"] / close

    # Williams %R
    df["Williams_R"] = -100 * (high14 - close) / (high14 - low14).replace(0, np.nan)

    # OBV
    direction = np.where(close > close.shift(1), 1, -1)
    df["OBV"] = (direction * vol).cumsum()
    df["OBV_sma"] = df["OBV"].rolling(20).mean()
    df["OBV_signal"] = (df["OBV"] - df["OBV_sma"]) / df["OBV_sma"].abs().replace(0, np.nan)

    # Volume features
    df["vol_ratio"]  = vol / vol.rolling(20).mean()
    df["vol_change"] = vol.pct_change()

    # Price patterns
    df["high_low_pct"] = (high - low) / close
    df["open_close_pct"] = (close - df["Open"]) / df["Open"]
    df["upper_shadow"] = (high - close.clip(lower=df["Open"])) / close
    df["lower_shadow"] = (close.clip(upper=df["Open"]) - low) / close

    # Momentum
    df["ROC_5"]  = close.pct_change(5) * 100
    df["ROC_10"] = close.pct_change(10) * 100
    df["ROC_20"] = close.pct_change(20) * 100

    # ── TARGET: predict next day direction ──────────────────────────────────────
    future_ret = close.shift(-1) / close - 1
    threshold  = 0.005  # 0.5% band for "HOLD"

    df["target_raw"] = future_ret
    df["target"] = np.where(future_ret > threshold, 2,       # BUY (rise)
                   np.where(future_ret < -threshold, 0, 1))  # SELL / HOLD

    df = df.dropna()
    return df


def get_feature_cols(df):
    exclude = {"Open","High","Low","Close","Volume","Dividends","Stock Splits",
               "target","target_raw"}
    return [c for c in df.columns if c not in exclude]


# ── Data Fetching ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(ticker, period="2y"):
    tk = yf.Ticker(ticker)
    hist = tk.history(period=period, auto_adjust=True)
    info = tk.info
    return hist, info

@st.cache_data(ttl=3600, show_spinner=False)
def get_current_price(ticker):
    tk = yf.Ticker(ticker)
    h = tk.history(period="5d")
    if h.empty:
        return None, None, None
    cur = h["Close"].iloc[-1]
    prev = h["Close"].iloc[-2] if len(h) > 1 else cur
    chg = (cur - prev) / prev * 100
    return cur, prev, chg


# ── Chart Theme ──────────────────────────────────────────────────────────────────
CT = {"bg":"#04080f","surface":"#080f1e","surface2":"#0c1527","border":"#162440",
      "border2":"#1e3060","cyan":"#00f5ff","green":"#00ff88","red":"#ff3366",
      "yellow":"#ffcc00","orange":"#ff6b35","purple":"#8b5cf6","blue":"#3b82f6","text":"#e2eaf5","muted":"#4a6080"}

def theme(fig, height=400, title=None):
    fig.update_layout(
        height=height,
        title=dict(text=title, font=dict(family="Outfit", size=13, color=CT["text"])) if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,15,30,0.7)",
        font=dict(family="Outfit", size=11, color=CT["text"]),
        legend=dict(bgcolor="rgba(8,15,30,0.9)", bordercolor=CT["border2"], borderwidth=1),
        xaxis=dict(gridcolor=CT["border"], linecolor=CT["border"], zeroline=False, tickfont=dict(size=10)),
        yaxis=dict(gridcolor=CT["border"], linecolor=CT["border"], zeroline=False, tickfont=dict(size=10)),
        margin=dict(l=50, r=20, t=40 if title else 20, b=40),
        hovermode="x unified",
    )
    return fig


# ── SIDEBAR ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:12px 0 20px;">
        <div style="font-size:22px;font-weight:900;letter-spacing:-1px;font-family:'Outfit',sans-serif;">
            🔮 StockOracle <span style="color:#00f5ff;">AI</span>
        </div>
        <div style="font-size:10px;color:#4a6080;letter-spacing:2.5px;text-transform:uppercase;margin-top:4px;">
            Prediction Engine v2.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">🎯 Stock Selection</div>', unsafe_allow_html=True)
    search = st.text_input("Search", placeholder="Apple, NVDA, BTC...")
    filtered = {k: v for k, v in TOP_200.items()
                if search.lower() in v.lower() or search.upper() in k} if search else TOP_200
    ticker = st.selectbox("Select Stock", list(filtered.keys()),
                          format_func=lambda x: f"{x} — {filtered.get(x, '')}")

    st.markdown('<div class="sec-hdr">📅 Training Data</div>', unsafe_allow_html=True)
    period_map = {"6 Months":"6mo","1 Year":"1y","2 Years":"2y","5 Years":"5y"}
    period_lbl = st.select_slider("Data Period", list(period_map.keys()), value="2 Years")
    period = period_map[period_lbl]

    threshold_pct = st.slider("Hold Band (±%)", min_value=0.1, max_value=2.0, value=0.5, step=0.1,
                               help="Returns within ±X% = HOLD signal")

    st.markdown('<div class="sec-hdr">🤖 Model Selection</div>', unsafe_allow_html=True)
    st.caption("Pick models to train & compare")

    # Initialize defaults on first load
    for name in MODEL_INFO:
        if f"chk_{name}" not in st.session_state:
            st.session_state[f"chk_{name}"] = name in ["Random Forest", "Gradient Boosting", "Logistic Regression"]

    # Toggle all on/off
    def toggle_all():
        new_val = not all(st.session_state.get(f"chk_{n}", False) for n in MODEL_INFO)
        for n in MODEL_INFO:
            st.session_state[f"chk_{n}"] = new_val

    all_checked = all(st.session_state.get(f"chk_{n}", False) for n in MODEL_INFO)
    st.checkbox(
        "Select All 15 Models",
        value=all_checked,
        on_change=toggle_all,
        key="select_all_toggle"
    )

    selected_models = []
    for name, info_m in MODEL_INFO.items():
        if st.checkbox(f"{info_m['icon']} {name}", key=f"chk_{name}"):
            selected_models.append(name)

    st.markdown('<div class="sec-hdr">⚙️ Training Config</div>', unsafe_allow_html=True)
    test_size = st.slider("Test Split %", 10, 40, 20) / 100
    cv_folds  = st.slider("CV Folds", 3, 10, 5)
    use_cv    = st.toggle("Use Cross-Validation", value=True)

    run_btn = st.button("🚀 TRAIN & PREDICT", use_container_width=True)


# ── HEADER ───────────────────────────────────────────────────────────────────────
cur_price, prev_price, price_chg = get_current_price(ticker)
price_color = CT["green"] if (price_chg or 0) >= 0 else CT["red"]
arrow = "▲" if (price_chg or 0) >= 0 else "▼"

st.markdown(f"""
<div class="oracle-header">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px;">
        <div>
            <div style="font-size:13px;color:#4a6080;letter-spacing:2px;text-transform:uppercase;margin-bottom:4px;">
                ML Prediction Engine
            </div>
            <div style="font-size:32px;font-weight:900;letter-spacing:-1px;">
                {TOP_200.get(ticker, ticker)}
                <span style="font-size:16px;color:#4a6080;font-weight:400;margin-left:8px;">{ticker}</span>
            </div>
        </div>
        <div style="text-align:right;">
            <div style="font-family:'JetBrains Mono';font-size:36px;font-weight:700;color:{price_color};">
                ${f"{cur_price:.2f}" if cur_price else "N/A"}
            </div>
            <div style="font-family:'JetBrains Mono';font-size:14px;color:{price_color};">
                {arrow} {abs(price_chg):.2f}% today
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if not selected_models:
    st.warning("👈 Please select at least one model from the sidebar.")
    st.stop()


# ── MAIN LOGIC ───────────────────────────────────────────────────────────────────
if run_btn or "results" in st.session_state:

    if run_btn:
        # Reset results
        st.session_state.pop("results", None)
        st.session_state.pop("df_feat", None)
        st.session_state.pop("trained_models", None)

        with st.spinner(f"📡 Fetching {ticker} data from Yahoo Finance..."):
            hist, info = fetch_data(ticker, period)

        if hist.empty or len(hist) < 100:
            st.error("Not enough data to train models. Try a longer period or different stock.")
            st.stop()

        with st.spinner("🔧 Engineering 40+ features..."):
            df_feat = build_features(hist)
            df_feat["target"] = np.where(df_feat["target_raw"] > threshold_pct/100, 2,
                                 np.where(df_feat["target_raw"] < -threshold_pct/100, 0, 1))
            feature_cols = get_feature_cols(df_feat)
            X = df_feat[feature_cols].values
            y = df_feat["target"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        results = {}
        trained_models = {}
        total = len(selected_models)
        prog_bar = st.progress(0, text="Training models...")

        for i, model_name in enumerate(selected_models):
            prog_bar.progress((i) / total, text=f"⚙️ Training {model_name}...")
            model = ALL_MODELS[model_name]

            try:
                # Cross-val score
                cv_score = None
                if use_cv:
                    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                    cv_scores = cross_val_score(model, X_train_sc, y_train, cv=skf, scoring="accuracy")
                    cv_score = cv_scores.mean()
                    cv_std   = cv_scores.std()
                else:
                    cv_std = 0.0

                # Train on full train set
                model.fit(X_train_sc, y_train)
                y_pred  = model.predict(X_test_sc)
                y_proba = None
                try:
                    y_proba = model.predict_proba(X_test_sc)
                except:
                    pass

                acc  = accuracy_score(y_test, y_pred)
                f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)

                # Predict NEXT day
                latest_features = df_feat[feature_cols].iloc[-1].values.reshape(1, -1)
                latest_sc = scaler.transform(latest_features)
                next_pred = model.predict(latest_sc)[0]
                next_proba = None
                try:
                    next_proba = model.predict_proba(latest_sc)[0]
                except:
                    pass

                results[model_name] = {
                    "accuracy": acc, "cv_score": cv_score, "cv_std": cv_std,
                    "f1": f1, "precision": prec, "recall": rec,
                    "y_pred": y_pred, "y_test": y_test,
                    "y_proba": y_proba,
                    "next_pred": next_pred, "next_proba": next_proba,
                    "cm": confusion_matrix(y_test, y_pred),
                }
                trained_models[model_name] = (model, scaler)

            except Exception as e:
                results[model_name] = {"error": str(e), "accuracy": 0}

        prog_bar.progress(1.0, text="✅ All models trained!")

        st.session_state["results"]        = results
        st.session_state["df_feat"]        = df_feat
        st.session_state["trained_models"] = trained_models
        st.session_state["feature_cols"]   = feature_cols
        st.session_state["X_test_sc"]      = X_test_sc
        st.session_state["y_test"]         = y_test
        st.session_state["ticker"]         = ticker
        st.session_state["hist"]           = hist

    # ── Pull from session state ───────────────────────────────────────────────────
    results        = st.session_state["results"]
    df_feat        = st.session_state["df_feat"]
    feature_cols   = st.session_state["feature_cols"]
    X_test_sc      = st.session_state["X_test_sc"]
    y_test         = st.session_state["y_test"]
    hist           = st.session_state["hist"]

    valid_results = {k: v for k, v in results.items() if "error" not in v}
    if not valid_results:
        st.error("All models failed. Try different settings or stocks.")
        st.stop()

    # Sort by accuracy
    sorted_models = sorted(valid_results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    best_model_name = sorted_models[0][0]
    best_result     = sorted_models[0][1]

    label_map = {0: "🔴 SELL", 1: "🟡 HOLD", 2: "🟢 BUY"}
    label_short = {0: "SELL", 1: "HOLD", 2: "BUY"}
    label_color = {0: CT["red"], 1: CT["yellow"], 2: CT["green"]}


    # ═══════════════════════════════════════════════════════════════════════════
    # TABS
    # ═══════════════════════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 Model Leaderboard", "📊 Deep Analysis", "🔄 Compare Models",
        "💡 Buy / Sell / Hold", "📈 Price + Signals"
    ])


    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — LEADERBOARD
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="sec-hdr">🏆 Model Performance Leaderboard</div>', unsafe_allow_html=True)

        # Summary bar chart
        names   = [n for n, _ in sorted_models]
        accs    = [v["accuracy"]*100 for _, v in sorted_models]
        f1s     = [v["f1"]*100 for _, v in sorted_models]
        cv_accs = [v["cv_score"]*100 if v["cv_score"] else v["accuracy"]*100 for _, v in sorted_models]

        fig_lb = go.Figure()
        fig_lb.add_trace(go.Bar(
            y=names, x=accs, name="Test Accuracy", orientation="h",
            marker=dict(
                color=[CT["green"] if n == best_model_name else CT["cyan"] for n in names],
                opacity=0.85
            ),
            text=[f"{a:.1f}%" for a in accs], textposition="outside"
        ))
        if use_cv:
            fig_lb.add_trace(go.Bar(
                y=names, x=cv_accs, name="CV Accuracy", orientation="h",
                marker=dict(color=CT["purple"], opacity=0.5),
                text=[f"{a:.1f}%" for a in cv_accs], textposition="inside"
            ))
        fig_lb.update_layout(barmode="overlay", xaxis_title="Accuracy (%)")
        theme(fig_lb, height=max(350, len(sorted_models)*38+60), title="Model Accuracy Comparison")
        st.plotly_chart(fig_lb, use_container_width=True)

        # Leaderboard cards
        st.markdown("### 📋 Detailed Rankings")
        for rank, (mname, mres) in enumerate(sorted_models, 1):
            is_best = (mname == best_model_name)
            next_signal = label_short.get(mres["next_pred"], "HOLD")
            sig_color   = label_color.get(mres["next_pred"], CT["yellow"])
            acc_pct     = mres["accuracy"] * 100
            bar_color   = CT["green"] if is_best else CT["cyan"]
            medal = ["🥇","🥈","🥉"][rank-1] if rank <= 3 else f"#{rank}"

            st.markdown(f"""
            <div class="model-card {'best' if is_best else ''}">
              <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
                <div>
                  <span style="font-size:18px;">{medal}</span>
                  <span style="font-size:15px;font-weight:700;margin-left:8px;">{mname}</span>
                  <span style="font-size:11px;color:{CT['muted']};margin-left:8px;">
                    {MODEL_INFO[mname]['type']} · {MODEL_INFO[mname]['speed']}
                  </span>
                  {'<span style="margin-left:8px;font-size:11px;color:' + CT['green'] + ';border:1px solid ' + CT['green'] + ';border-radius:4px;padding:1px 6px;">BEST</span>' if is_best else ''}
                </div>
                <div style="display:flex;gap:20px;align-items:center;">
                  <div style="text-align:center;">
                    <div style="font-family:JetBrains Mono;font-size:20px;font-weight:700;color:{bar_color};">{acc_pct:.1f}%</div>
                    <div style="font-size:10px;color:{CT['muted']};">Test Acc</div>
                  </div>
                  <div style="text-align:center;">
                    <div style="font-family:JetBrains Mono;font-size:20px;font-weight:700;color:{CT['purple']};">{mres['f1']*100:.1f}%</div>
                    <div style="font-size:10px;color:{CT['muted']};">F1 Score</div>
                  </div>
                  {f'<div style="text-align:center;"><div style="font-family:JetBrains Mono;font-size:20px;font-weight:700;color:{CT["yellow"]};">{mres["cv_score"]*100:.1f}%</div><div style="font-size:10px;color:{CT["muted"]};">CV Score</div></div>' if mres.get("cv_score") else ""}
                  <div>
                    <span style="font-size:13px;font-weight:700;color:{sig_color};
                          border:1px solid {sig_color};border-radius:6px;padding:4px 12px;">
                      {next_signal}
                    </span>
                  </div>
                </div>
              </div>
              <div class="acc-bar-wrap" style="margin-top:10px;">
                <div class="acc-bar" style="width:{acc_pct}%;background:{bar_color};"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Errors
        errors = {k: v for k, v in results.items() if "error" in v}
        if errors:
            with st.expander("⚠️ Failed Models"):
                for k, v in errors.items():
                    st.error(f"**{k}**: {v['error']}")


    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — DEEP ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        # Select model to inspect
        inspect_model = st.selectbox(
            "🔍 Select model to inspect",
            [n for n, _ in sorted_models],
            index=0,
            format_func=lambda x: f"{'🥇 ' if x==best_model_name else ''}{x} — {valid_results[x]['accuracy']*100:.1f}%"
        )
        res = valid_results[inspect_model]

        # Metrics row
        mc = st.columns(5)
        metrics_list = [
            ("Test Accuracy", f"{res['accuracy']*100:.2f}%", CT["cyan"]),
            ("F1 Score",      f"{res['f1']*100:.2f}%",       CT["purple"]),
            ("Precision",     f"{res['precision']*100:.2f}%", CT["green"]),
            ("Recall",        f"{res['recall']*100:.2f}%",    CT["yellow"]),
            ("CV Score",      f"{res['cv_score']*100:.2f}%" if res['cv_score'] else "N/A", CT["blue"]),
        ]
        for i, (lbl, val, clr) in enumerate(metrics_list):
            with mc[i]:
                st.markdown(f"""<div class="metric-pill" style="border-color:{clr}33;">
                    <div class="label">{lbl}</div>
                    <div class="value" style="color:{clr};">{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_cm, col_dist = st.columns(2)

        with col_cm:
            cm = res["cm"]
            labels = ["SELL", "HOLD", "BUY"]
            # Keep only present classes
            present = sorted(np.unique(np.concatenate([res["y_test"], res["y_pred"]])))
            labels_present = [labels[i] for i in present]
            cm_data = cm.tolist()

            fig_cm = go.Figure(go.Heatmap(
                z=cm_data,
                x=labels_present, y=labels_present,
                colorscale=[[0,"#080f1e"],[0.5,"#162440"],[1,"#00f5ff"]],
                text=cm_data, texttemplate="%{text}",
                hovertemplate="Actual %{y} → Predicted %{x}: %{z}<extra></extra>"
            ))
            theme(fig_cm, height=320, title=f"Confusion Matrix — {inspect_model}")
            fig_cm.update_layout(
                xaxis_title="Predicted", yaxis_title="Actual"
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_dist:
            # Prediction distribution
            pred_counts = pd.Series(res["y_pred"]).map({0:"SELL",1:"HOLD",2:"BUY"}).value_counts()
            true_counts = pd.Series(res["y_test"]).map({0:"SELL",1:"HOLD",2:"BUY"}).value_counts()
            fig_dist = go.Figure()
            colors_dist = {"BUY": CT["green"], "HOLD": CT["yellow"], "SELL": CT["red"]}
            for lbl in ["BUY","HOLD","SELL"]:
                fig_dist.add_trace(go.Bar(
                    name=f"Predicted {lbl}",
                    x=["Predicted"],
                    y=[pred_counts.get(lbl, 0)],
                    marker_color=colors_dist[lbl], opacity=0.9,
                    text=[pred_counts.get(lbl, 0)], textposition="auto"
                ))
                fig_dist.add_trace(go.Bar(
                    name=f"Actual {lbl}",
                    x=["Actual"],
                    y=[true_counts.get(lbl, 0)],
                    marker_color=colors_dist[lbl], opacity=0.5,
                    text=[true_counts.get(lbl, 0)], textposition="auto",
                    showlegend=False
                ))
            fig_dist.update_layout(barmode="stack")
            theme(fig_dist, height=320, title="Predicted vs Actual Signal Distribution")
            st.plotly_chart(fig_dist, use_container_width=True)

        # ROC Curves (multi-class OvR)
        if res["y_proba"] is not None:
            present_classes = sorted(np.unique(res["y_test"]))
            fig_roc = go.Figure()
            roc_colors = {0: CT["red"], 1: CT["yellow"], 2: CT["green"]}
            roc_labels = {0: "SELL", 1: "HOLD", 2: "BUY"}
            for cls in present_classes:
                if cls < res["y_proba"].shape[1]:
                    y_bin = (res["y_test"] == cls).astype(int)
                    prob  = res["y_proba"][:, cls]
                    fpr, tpr, _ = roc_curve(y_bin, prob)
                    auc_val = auc(fpr, tpr)
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        name=f"{roc_labels[cls]} (AUC={auc_val:.3f})",
                        line=dict(color=roc_colors[cls], width=2)
                    ))
            fig_roc.add_trace(go.Scatter(
                x=[0,1], y=[0,1], line=dict(dash="dash", color=CT["muted"]),
                name="Random", showlegend=True
            ))
            theme(fig_roc, height=350, title="ROC Curves (One-vs-Rest)")
            fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig_roc, use_container_width=True)

        # Feature Importance
        model_obj, scaler_obj = st.session_state["trained_models"].get(inspect_model, (None, None))
        if model_obj is not None:
            importances = None
            try:
                if hasattr(model_obj, "feature_importances_"):
                    importances = model_obj.feature_importances_
                elif hasattr(model_obj, "coef_"):
                    coef = model_obj.coef_
                    if coef.ndim > 1:
                        importances = np.abs(coef).mean(axis=0)
                    else:
                        importances = np.abs(coef)
            except:
                pass

            if importances is not None and len(importances) == len(feature_cols):
                feat_df = pd.DataFrame({
                    "Feature": feature_cols,
                    "Importance": importances
                }).sort_values("Importance", ascending=False).head(20)

                fig_fi = go.Figure(go.Bar(
                    y=feat_df["Feature"], x=feat_df["Importance"],
                    orientation="h",
                    marker=dict(
                        color=feat_df["Importance"],
                        colorscale=[[0, CT["border2"]], [0.5, CT["cyan"]], [1, CT["green"]]],
                        showscale=False
                    )
                ))
                theme(fig_fi, height=420, title=f"Top 20 Feature Importances — {inspect_model}")
                fig_fi.update_layout(yaxis_autorange="reversed")
                st.plotly_chart(fig_fi, use_container_width=True)

        # Classification Report
        with st.expander("📋 Full Classification Report"):
            cr = classification_report(res["y_test"], res["y_pred"],
                                        target_names=["SELL","HOLD","BUY"],
                                        zero_division=0, output_dict=True)
            cr_df = pd.DataFrame(cr).T.round(3)
            st.dataframe(cr_df, use_container_width=True)


    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — COMPARE MODELS
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="sec-hdr">🔄 Side-by-Side Model Comparison</div>', unsafe_allow_html=True)

        compare_selection = st.multiselect(
            "Select models to compare (2–8)",
            options=[n for n, _ in sorted_models],
            default=[n for n, _ in sorted_models[:min(5, len(sorted_models))]],
            max_selections=8
        )

        if len(compare_selection) < 2:
            st.info("Select at least 2 models to compare.")
        else:
            comp_data = {m: valid_results[m] for m in compare_selection if m in valid_results}

            # Radar chart
            metrics_radar = ["accuracy","f1","precision","recall"]
            metric_labels = ["Accuracy","F1 Score","Precision","Recall"]
            fig_rad = go.Figure()
            palette = [CT["cyan"],CT["green"],CT["purple"],CT["yellow"],
                       CT["orange"],CT["red"],CT["blue"],"#ff69b4"]
            for i, (mname, mres) in enumerate(comp_data.items()):
                vals = [mres[m]*100 for m in metrics_radar]
                vals_closed = vals + [vals[0]]
                labs_closed = metric_labels + [metric_labels[0]]
                fig_rad.add_trace(go.Scatterpolar(
                    r=vals_closed, theta=labs_closed,
                    fill="toself", opacity=0.3,
                    line=dict(color=palette[i % len(palette)], width=2),
                    name=mname
                ))
            fig_rad.update_layout(
                polar=dict(
                    bgcolor="rgba(8,15,30,0.8)",
                    radialaxis=dict(visible=True, range=[0,100],
                                    gridcolor=CT["border2"], tickfont=dict(size=9)),
                    angularaxis=dict(gridcolor=CT["border2"])
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Outfit", color=CT["text"]),
                height=420,
                title=dict(text="Multi-Metric Radar Comparison",
                           font=dict(family="Outfit", size=13, color=CT["text"])),
                legend=dict(bgcolor="rgba(8,15,30,0.9)", bordercolor=CT["border2"], borderwidth=1),
            )
            st.plotly_chart(fig_rad, use_container_width=True)

            # Grouped bar chart
            fig_grp = go.Figure()
            metric_keys  = ["accuracy","f1","precision","recall"]
            metric_names = ["Accuracy","F1","Precision","Recall"]
            for i, mk in enumerate(metric_keys):
                fig_grp.add_trace(go.Bar(
                    name=metric_names[i],
                    x=list(comp_data.keys()),
                    y=[comp_data[m][mk]*100 for m in comp_data],
                    marker_color=[CT["cyan"],CT["green"],CT["purple"],CT["yellow"]][i],
                    opacity=0.85
                ))
            fig_grp.update_layout(barmode="group")
            theme(fig_grp, height=380, title="All Metrics — Grouped Comparison")
            st.plotly_chart(fig_grp, use_container_width=True)

            # Next-day signal table
            st.markdown("### 🎯 Next-Day Signal Agreement")
            sig_rows = []
            for mname in compare_selection:
                if mname in valid_results:
                    mr = valid_results[mname]
                    prob_str = ""
                    if mr["next_proba"] is not None:
                        p = mr["next_proba"]
                        prob_str = f"SELL:{p[0]*100:.0f}% | HOLD:{p[1]*100:.0f}% | BUY:{p[2]*100:.0f}%"
                    sig_rows.append({
                        "Model": mname,
                        "Signal": label_map.get(mr["next_pred"], "N/A"),
                        "Confidence": prob_str,
                        "Accuracy": f"{mr['accuracy']*100:.1f}%",
                        "F1": f"{mr['f1']*100:.1f}%",
                    })
            sig_df = pd.DataFrame(sig_rows)
            st.dataframe(sig_df, use_container_width=True, hide_index=True)

            # Agreement meter
            signals = [valid_results[m]["next_pred"] for m in compare_selection if m in valid_results]
            buy_pct  = signals.count(2) / len(signals) * 100
            sell_pct = signals.count(0) / len(signals) * 100
            hold_pct = signals.count(1) / len(signals) * 100

            fig_agree = go.Figure(go.Bar(
                x=["🟢 BUY","🟡 HOLD","🔴 SELL"],
                y=[buy_pct, hold_pct, sell_pct],
                marker_color=[CT["green"], CT["yellow"], CT["red"]],
                text=[f"{v:.0f}%" for v in [buy_pct, hold_pct, sell_pct]],
                textposition="outside"
            ))
            theme(fig_agree, height=280, title="Model Consensus — Next-Day Signal Distribution")
            st.plotly_chart(fig_agree, use_container_width=True)


    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — BUY / SELL / HOLD
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown('<div class="sec-hdr">💡 Intelligent Buy · Sell · Hold Recommendation</div>', unsafe_allow_html=True)

        # Choose which model's signal to use
        chosen_model = st.selectbox(
            "Base recommendation on which model?",
            options=[n for n, _ in sorted_models],
            index=0,
            format_func=lambda x: f"{'🥇 ' if x==best_model_name else ''}{x} — Acc: {valid_results[x]['accuracy']*100:.1f}%"
        )
        chosen_res = valid_results[chosen_model]
        signal     = chosen_res["next_pred"]
        signal_lbl = label_short[signal]
        sig_color  = label_color[signal]
        proba      = chosen_res.get("next_proba")

        # Big signal display
        emoji_map = {"BUY":"🟢","SELL":"🔴","HOLD":"🟡"}
        desc_map  = {
            "BUY":  "Model predicts price will rise >0.5% by next close. Consider entering a long position.",
            "SELL": "Model predicts price will fall >0.5% by next close. Consider reducing or exiting.",
            "HOLD": "Model predicts price will stay flat (±0.5%). No strong directional signal.",
        }
        st.markdown(f"""
        <div style="background:rgba({','.join(['0,255,136' if signal==2 else '255,51,102' if signal==0 else '255,204,0'][0].split(','))},0.08);
                    border:2px solid {sig_color};border-radius:16px;padding:32px;text-align:center;margin:16px 0;">
            <div style="font-size:64px;margin-bottom:8px;">{emoji_map[signal_lbl]}</div>
            <div style="font-size:48px;font-weight:900;color:{sig_color};letter-spacing:4px;font-family:'JetBrains Mono';">
                {signal_lbl}
            </div>
            <div style="font-size:14px;color:#a0b4c8;margin-top:12px;max-width:500px;margin-left:auto;margin-right:auto;">
                {desc_map[signal_lbl]}
            </div>
            <div style="font-size:12px;color:#4a6080;margin-top:8px;">
                Based on: <strong style="color:{sig_color};">{chosen_model}</strong> · 
                Accuracy: <strong style="color:{CT['cyan']};">{chosen_res['accuracy']*100:.1f}%</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability gauge
        if proba is not None and len(proba) == 3:
            col_g1, col_g2, col_g3 = st.columns(3)
            gauges = [
                (col_g1, "SELL Probability", proba[0]*100, CT["red"]),
                (col_g2, "HOLD Probability", proba[1]*100, CT["yellow"]),
                (col_g3, "BUY Probability",  proba[2]*100, CT["green"]),
            ]
            for col, lbl, pval, clr in gauges:
                with col:
                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=pval,
                        number={"suffix":"%","font":{"family":"JetBrains Mono","size":26,"color":clr}},
                        gauge={
                            "axis":{"range":[0,100],"tickfont":{"size":9},"tickcolor":CT["muted"]},
                            "bar":{"color":clr,"thickness":0.25},
                            "bgcolor":"rgba(8,15,30,0.8)",
                            "borderwidth":1,"bordercolor":CT["border2"],
                            "steps":[
                                {"range":[0,33],"color":"rgba(22,36,64,0.5)"},
                                {"range":[33,66],"color":"rgba(30,48,96,0.5)"},
                                {"range":[66,100],"color":"rgba(22,36,64,0.3)"},
                            ],
                            "threshold":{"line":{"color":clr,"width":3},"thickness":0.8,"value":pval}
                        },
                        title={"text":lbl,"font":{"size":12,"color":CT["muted"]}}
                    ))
                    fig_g.update_layout(
                        height=230, paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Outfit"), margin=dict(l=20,r=20,t=30,b=10)
                    )
                    st.plotly_chart(fig_g, use_container_width=True)

        # Ensemble vote
        st.markdown("---")
        st.markdown("### 🗳️ Ensemble Vote — All Models")
        all_signals = [(n, v["next_pred"], v["accuracy"]) for n, v in valid_results.items()]
        buy_votes  = [(n, a) for n, s, a in all_signals if s == 2]
        hold_votes = [(n, a) for n, s, a in all_signals if s == 1]
        sell_votes = [(n, a) for n, s, a in all_signals if s == 0]

        # Weighted vote
        w_buy  = sum(a for _, a in buy_votes)
        w_hold = sum(a for _, a in hold_votes)
        w_sell = sum(a for _, a in sell_votes)
        total_w = w_buy + w_hold + w_sell or 1

        ensemble_signal = "BUY" if w_buy >= w_hold and w_buy >= w_sell else \
                          "SELL" if w_sell >= w_hold else "HOLD"
        ens_color = {"BUY":CT["green"],"SELL":CT["red"],"HOLD":CT["yellow"]}[ensemble_signal]

        vc1, vc2, vc3, vc4 = st.columns(4)
        with vc1:
            st.markdown(f"""<div class="metric-pill" style="border-color:{CT['green']}44;">
                <div class="label">BUY Votes</div>
                <div class="value" style="color:{CT['green']};">{len(buy_votes)}</div>
                <div style="font-size:11px;color:{CT['muted']};">{w_buy/total_w*100:.0f}% weight</div>
            </div>""", unsafe_allow_html=True)
        with vc2:
            st.markdown(f"""<div class="metric-pill" style="border-color:{CT['yellow']}44;">
                <div class="label">HOLD Votes</div>
                <div class="value" style="color:{CT['yellow']};">{len(hold_votes)}</div>
                <div style="font-size:11px;color:{CT['muted']};">{w_hold/total_w*100:.0f}% weight</div>
            </div>""", unsafe_allow_html=True)
        with vc3:
            st.markdown(f"""<div class="metric-pill" style="border-color:{CT['red']}44;">
                <div class="label">SELL Votes</div>
                <div class="value" style="color:{CT['red']};">{len(sell_votes)}</div>
                <div style="font-size:11px;color:{CT['muted']};">{w_sell/total_w*100:.0f}% weight</div>
            </div>""", unsafe_allow_html=True)
        with vc4:
            st.markdown(f"""<div class="metric-pill" style="border-color:{ens_color};">
                <div class="label">Ensemble Signal</div>
                <div class="value" style="color:{ens_color};">{ensemble_signal}</div>
                <div style="font-size:11px;color:{CT['muted']};">Accuracy-weighted</div>
            </div>""", unsafe_allow_html=True)

        # Per-model signal breakdown
        st.markdown("<br>", unsafe_allow_html=True)
        breakdown_cols = st.columns(2)
        for half_idx, group_name, group_data in [
            (0, "🟢 BUY Signal Models", buy_votes),
            (0, "🟡 HOLD Signal Models", hold_votes),
            (1, "🔴 SELL Signal Models", sell_votes),
        ]:
            col = breakdown_cols[half_idx]
            with col:
                if group_data:
                    clr = CT["green"] if "BUY" in group_name else CT["yellow"] if "HOLD" in group_name else CT["red"]
                    st.markdown(f"**{group_name}**")
                    for mname, macc in sorted(group_data, key=lambda x: -x[1]):
                        st.markdown(f"""
                        <div style="display:flex;justify-content:space-between;padding:5px 10px;
                                    background:var(--surface2);border-radius:6px;margin-bottom:4px;
                                    border-left:3px solid {clr};">
                            <span style="font-size:12px;">{mname}</span>
                            <span style="font-family:'JetBrains Mono';font-size:12px;color:{clr};">{macc*100:.1f}%</span>
                        </div>
                        """, unsafe_allow_html=True)

        # Risk disclaimer
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:rgba(255,51,102,0.05);border:1px solid rgba(255,51,102,0.2);
                    border-radius:10px;padding:16px;font-size:12px;color:#64748b;">
            ⚠️ <strong style="color:#ff3366;">Disclaimer:</strong>
            These predictions are generated by ML models trained on historical price data and technical indicators.
            They do <strong>NOT</strong> constitute financial advice. Markets are inherently unpredictable.
            Always do your own research and consult a licensed financial advisor before making investment decisions.
            Past model performance does not guarantee future accuracy.
        </div>
        """, unsafe_allow_html=True)


    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 — PRICE + SIGNALS
    # ══════════════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown('<div class="sec-hdr">📈 Historical Price with Predicted Signals</div>', unsafe_allow_html=True)

        signal_model = st.selectbox(
            "Show signals from model:",
            [n for n, _ in sorted_models], index=0,
            format_func=lambda x: f"{x} — {valid_results[x]['accuracy']*100:.1f}%",
            key="sig_model_sel"
        )
        sr = valid_results[signal_model]

        # Get test portion of hist
        test_len  = len(sr["y_pred"])
        hist_test = hist.iloc[-test_len:]

        buy_dates  = hist_test.index[sr["y_pred"] == 2]
        sell_dates = hist_test.index[sr["y_pred"] == 0]
        hold_dates = hist_test.index[sr["y_pred"] == 1]

        fig_sig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                 row_heights=[0.75,0.25], vertical_spacing=0.03)

        # Full price line
        fig_sig.add_trace(go.Scatter(
            x=hist.index, y=hist["Close"],
            line=dict(color=CT["cyan"], width=1.5),
            name="Close Price", opacity=0.8
        ), row=1, col=1)

        # Signal markers
        close_test = hist_test["Close"]
        if len(buy_dates):
            fig_sig.add_trace(go.Scatter(
                x=buy_dates, y=close_test.reindex(buy_dates),
                mode="markers", name="BUY signal",
                marker=dict(color=CT["green"], size=8, symbol="triangle-up",
                            line=dict(width=1, color="white"))
            ), row=1, col=1)
        if len(sell_dates):
            fig_sig.add_trace(go.Scatter(
                x=sell_dates, y=close_test.reindex(sell_dates),
                mode="markers", name="SELL signal",
                marker=dict(color=CT["red"], size=8, symbol="triangle-down",
                            line=dict(width=1, color="white"))
            ), row=1, col=1)

        # Volume
        vol_colors = [CT["green"] if hist["Close"].iloc[i] >= hist["Open"].iloc[i]
                      else CT["red"] for i in range(len(hist))]
        fig_sig.add_trace(go.Bar(
            x=hist.index, y=hist["Volume"],
            marker_color=vol_colors, marker_opacity=0.5, name="Volume"
        ), row=2, col=1)

        fig_sig.update_xaxes(rangeslider_visible=False)
        theme(fig_sig, height=580, title=f"{ticker} — {signal_model} Buy/Sell Signals")
        st.plotly_chart(fig_sig, use_container_width=True)

        # Signal counts
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.markdown(f"""<div class="metric-pill" style="border-color:{CT['green']}44;">
                <div class="label">BUY signals</div>
                <div class="value" style="color:{CT['green']};">{len(buy_dates)}</div>
            </div>""", unsafe_allow_html=True)
        with sc2:
            st.markdown(f"""<div class="metric-pill" style="border-color:{CT['yellow']}44;">
                <div class="label">HOLD signals</div>
                <div class="value" style="color:{CT['yellow']};">{len(hold_dates)}</div>
            </div>""", unsafe_allow_html=True)
        with sc3:
            st.markdown(f"""<div class="metric-pill" style="border-color:{CT['red']}44;">
                <div class="label">SELL signals</div>
                <div class="value" style="color:{CT['red']};">{len(sell_dates)}</div>
            </div>""", unsafe_allow_html=True)
        with sc4:
            total_sig = len(buy_dates) + len(sell_dates) + len(hold_dates)
            directional_acc = (len(buy_dates[close_test.reindex(buy_dates).notna()]) /
                                max(total_sig, 1)) * 100
            st.markdown(f"""<div class="metric-pill" style="border-color:{CT['cyan']}44;">
                <div class="label">Test Accuracy</div>
                <div class="value" style="color:{CT['cyan']};">{sr['accuracy']*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)

        # Probability timeline (if available)
        if sr["y_proba"] is not None:
            fig_prob = go.Figure()
            fig_prob.add_trace(go.Scatter(
                x=hist_test.index, y=sr["y_proba"][:,2]*100,
                fill="tozeroy", fillcolor="rgba(0,255,136,0.08)",
                line=dict(color=CT["green"], width=1.5), name="BUY prob %"
            ))
            fig_prob.add_trace(go.Scatter(
                x=hist_test.index, y=sr["y_proba"][:,0]*100,
                fill="tozeroy", fillcolor="rgba(255,51,102,0.08)",
                line=dict(color=CT["red"], width=1.5), name="SELL prob %"
            ))
            fig_prob.add_hline(y=50, line_dash="dash", line_color=CT["muted"], line_width=1)
            theme(fig_prob, height=280, title="BUY / SELL Probability Over Time (%)")
            st.plotly_chart(fig_prob, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;color:#1e3060;font-size:11px;padding:24px 0 8px;
            font-family:'JetBrains Mono';">
    StockOracle AI · 15 Supervised ML Models · Data: Yahoo Finance via yfinance ·
    {datetime.now().strftime('%Y-%m-%d %H:%M UTC')} ·
    Not financial advice.
</div>
""", unsafe_allow_html=True)
