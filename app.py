import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Smart Live Trading Bot",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# API
# =========================================================

API_KEY = st.secrets.get("TWELVE_DATA_API_KEY", "")

PAIRS = [
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    "USD/CHF",
    "AUD/USD",
    "USD/CAD",
    "NZD/USD"
]

TIMEFRAMES = {
    "1 Minute": "1min",
    "5 Minutes": "5min",
    "15 Minutes": "15min"
}

# =========================================================
# SIMPLE UI
# =========================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #050816, #0b1728);
}

.title {
    text-align: center;
    font-size: 38px;
    font-weight: 800;
    color: white;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #9ca9bd;
    margin-bottom: 25px;
}

.card {
    background: rgba(20, 32, 52, 0.85);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 15px;
}

.label {
    color: #8d9bb2;
    font-size: 13px;
    text-transform: uppercase;
}

.value {
    color: white;
    font-size: 28px;
    font-weight: 800;
}

.up {
    background: rgba(0, 220, 130, 0.12);
    border: 1px solid rgba(0,255,150,0.35);
    border-radius: 20px;
    padding: 35px;
    text-align: center;
}

.down {
    background: rgba(255, 60, 80, 0.12);
    border: 1px solid rgba(255,70,90,0.35);
    border-radius: 20px;
    padding: 35px;
    text-align: center;
}

.up-text {
    color: #52ffc0;
    font-size: 42px;
    font-weight: 900;
}

.down-text {
    color: #ff7185;
    font-size: 42px;
    font-weight: 900;
}

.reason {
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
    padding: 10px;
    margin: 6px 0;
    color: #dce5f5;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">📊 SMART LIVE TRADING BOT</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Real Twelve Data market analysis</div>',
    unsafe_allow_html=True
)

# =========================================================
# API CHECK
# =========================================================

if not API_KEY:
    st.error("Twelve Data API key nahi mili.")
    st.info(
        "Streamlit Secrets mein TWELVE_DATA_API_KEY add karo."
    )
    st.stop()

# =========================================================
# DATA
# =========================================================

@st.cache_data(ttl=10)
def get_data(symbol, interval):

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": 200,
        "apikey": API_KEY
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        if response.status_code != 200:
            return None, f"HTTP error: {response.status_code}"

        data = response.json()

    except requests.exceptions.Timeout:
        return None, "Twelve Data response timeout."

    except Exception as e:
        return None, f"Connection error: {e}"

    if "values" not in data:
        return None, data.get(
            "message",
            "Twelve Data ne market data return nahi ki."
        )

    df = pd.DataFrame(data["values"])

    if df.empty:
        return None, "Market data empty hai."

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    for column in ["open", "high", "low", "close"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "datetime",
            "open",
            "high",
            "low",
            "close"
        ]
    )

    df = df.sort_values(
        "datetime"
    ).reset_index(drop=True)

    return df, None

# =========================================================
# INDICATORS
# =========================================================

def calculate_indicators(df):

    df = df.copy()

    # EMA
    df["ema9"] = df["close"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["ema21"] = df["close"].ewm(
        span=21,
        adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    # RSI
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi"] = 100 - (
        100 / (1 + rs)
    )

    # MACD
    ema12 = df["close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["macd"] = ema12 - ema26

    df["macd_signal"] = df["macd"].ewm(
        span=9,
        adjust=False
    ).mean()

    # Momentum
    df["momentum"] = df["close"].pct_change(3)

    # Candle strength
    df["body"] = (
        df["close"] - df["open"]
    ).abs()

    df["range"] = (
        df["high"] - df["low"]
    )

    df["body_ratio"] = (
        df["body"] /
        df["range"].replace(0, np.nan)
    )

    return df

# =========================================================
# SIGNAL
# =========================================================

def generate_signal(df):

    last = df.iloc[-1]

    up = 0
    down = 0

    reasons = []

    # EMA trend
    if (
        last["ema9"] >
        last["ema21"] >
        last["ema50"]
    ):
        up += 3
        reasons.append("Bullish EMA alignment")

    elif (
        last["ema9"] <
        last["ema21"] <
        last["ema50"]
    ):
        down += 3
        reasons.append("Bearish EMA alignment")

    # RSI
    if 52 <= last["rsi"] <= 68:
        up += 2
        reasons.append("RSI bullish zone")

    elif 32 <= last["rsi"] <= 48:
        down += 2
        reasons.append("RSI bearish zone")

    # MACD
    if last["macd"] > last["macd_signal"]:
        up += 2
        reasons.append("MACD bullish")

    elif last["macd"] < last["macd_signal"]:
        down += 2
        reasons.append("MACD bearish")

    # Momentum
    if last["momentum"] > 0:
        up += 2
        reasons.append("Positive momentum")

    elif last["momentum"] < 0:
        down += 2
        reasons.append("Negative momentum")

    # Candle
    if pd.notna(last["body_ratio"]):
        if last["body_ratio"] >= 0.55:

            if last["close"] > last["open"]:
                up += 2
                reasons.append("Strong bullish candle")

            elif last["close"] < last["open"]:
                down += 2
                reasons.append("Strong bearish candle")

    # Recent structure
    if len(df) >= 12:

        recent_high = df["high"].iloc[-11:-1].max()
        recent_low = df["low"].iloc[-11:-1].min()

        if last["close"] > recent_high:
            up += 2
            reasons.append("Recent high breakout")

        elif last["close"] < recent_low:
            down += 2
            reasons.append("Recent low breakdown")

    total = up + down

    # Always UP or DOWN
    if total == 0:
        direction = "UP"
        strength = 50.0
        reasons.append("Neutral market; direction forced to UP")
    elif up >= down:
        direction = "UP"
        strength = round(
            (up / total) * 100,
            1
        )
    else:
        direction = "DOWN"
        strength = round(
            (down / total) * 100,
            1
        )

    return (
        direction,
        strength,
        up,
        down,
        reasons
    )

# =========================================================
# BACKTEST
# =========================================================

def run_backtest(df):

    if len(df) < 70:
        return 0, 0, 0

    wins = 0
    losses = 0

    for i in range(60, len(df) - 1):

        historical = df.iloc[:i + 1].copy()

        signal, _, _, _, _ = generate_signal(
            historical
        )

        current_price = df.iloc[i]["close"]
        next_price = df.iloc[i + 1]["close"]

        if signal == "UP":
            correct = next_price > current_price
        else:
            correct = next_price < current_price

        if correct:
            wins += 1
        else:
            losses += 1

    total = wins + losses

    accuracy = (
        (wins / total) * 100
        if total > 0
        else 0
    )

    return accuracy, wins, losses

# =========================================================
# CONTROLS
# =========================================================

st.subheader("⚙️ Market Settings")

col1, col2 = st.columns(2)

with col1:
    pair = st.selectbox(
        "Trading Pair",
        PAIRS
    )

with col2:
    timeframe = st.selectbox(
        "Timeframe",
        list(TIMEFRAMES.keys())
    )

# =========================================================
# ANALYZE
# =========================================================

if st.button(
    "🚀 START ANALYZE",
    use_container_width=True
):

    with st.spinner("Live market data analyze ho raha hai..."):

        df, error = get_data(
            pair,
            TIMEFRAMES[timeframe]
        )

        if error:
            st.error(error)
            st.stop()

        if df is None or len(df) < 60:
            st.error(
                "Analysis ke liye enough historical candles nahi mili."
            )
            st.stop()

        df = calculate_indicators(df)

        (
            signal,
            strength,
            up_score,
            down_score,
            reasons
        ) = generate_signal(df)

        accuracy, wins, losses = run_backtest(df)

        last = df.iloc[-1]

        price = float(last["close"])
        rsi = last["rsi"]
        macd = last["macd"]
        momentum = last["momentum"]

    # =====================================================
    # MARKET INFO
    # =====================================================

    st.subheader("📡 Live Market")

    a, b, c, d = st.columns(4)

    with a:
        st.markdown(
            f"""
            <div class="card">
                <div class="label">Pair</div>
                <div class="value">{pair}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with b:
        st.markdown(
            f"""
            <div class="card">
                <div class="label">Timeframe</div>
                <div class="value">{timeframe}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c:
        st.markdown(
            f"""
            <div class="card">
                <div class="label">Live Price</div>
                <div class="value">{price:.6f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with d:
        st.markdown(
            f"""
            <div class="card">
                <div class="label">Updated</div>
                <div class="value">
                    {datetime.now().strftime("%H:%M:%S")}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # =====================================================
    # SIGNAL
    # =====================================================

    if signal == "UP":

        st.markdown(
            f"""
            <div class="up">
                <div style="font-size:45px;">🟢</div>
                <div class="up-text">UP</div>
                <div style="color:#b7c4d8;">
                    Signal Strength: {strength}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="down">
                <div style="font-size:45px;">🔴</div>
                <div class="down-text">DOWN</div>
                <div style="color:#b7c4d8;">
                    Signal Strength: {strength}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # =====================================================
    # SCORES
    # =====================================================

    st.subheader("⚖️ Direction Scores")

    s1, s2 = st.columns(2)

    with s1:
        st.metric(
            "🟢 UP Score",
            up_score
        )

    with s2:
        st.metric(
            "🔴 DOWN Score",
            down_score
        )

    # =====================================================
    # TECHNICAL
    # =====================================================

    st.subheader("📈 Technical Analysis")

    t1, t2, t3 = st.columns(3)

    with t1:
        st.metric(
            "RSI",
            f"{rsi:.2f}"
            if pd.notna(rsi)
            else "N/A"
        )

    with t2:
        st.metric(
            "MACD",
            f"{macd:.6f}"
            if pd.notna(macd)
            else "N/A"
        )

    with t3:
        st.metric(
            "Momentum",
            f"{momentum * 100:.4f}%"
            if pd.notna(momentum)
            else "N/A"
        )

    # =====================================================
    # REASONS
    # =====================================================

    st.subheader("🧠 Signal Reasons")

    for reason in reasons:
        st.markdown(
            f"""
            <div class="reason">
                ✓ {reason}
            </div>
            """,
            unsafe_allow_html=True
        )

    # =====================================================
    # BACKTEST
    # =====================================================

    st.subheader("📚 Historical Backtest")

    b1, b2, b3 = st.columns(3)

    with b1:
        st.metric(
            "Accuracy",
            f"{accuracy:.1f}%"
        )

    with b2:
        st.metric(
            "Wins",
            wins
        )

    with b3:
        st.metric(
            "Losses",
            losses
        )

    # =====================================================
    # NOTE
    # =====================================================

    st.info(
        "Signal strength indicator agreement score hai, "
        "guaranteed profit ya guaranteed accuracy nahi."
    )

else:

    st.info(
        "Pair aur timeframe select karke "
        "START ANALYZE press karein."
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#718096;
        padding:30px 0 10px 0;
    ">
        Smart Live Trading Bot
    </div>
    """,
    unsafe_allow_html=True
)
