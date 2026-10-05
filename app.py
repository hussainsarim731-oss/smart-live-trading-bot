import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Live Trading Signal Bot",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# TWELVE DATA API
# =========================================================

API_KEY = st.secrets.get("TWELVE_DATA_API_KEY", "").strip()

# =========================================================
# PAIRS
# =========================================================

PAIRS = [
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    "USD/CHF",
    "AUD/USD",
    "USD/CAD",
    "NZD/USD"
]

# =========================================================
# TIMEFRAMES
# =========================================================

TIMEFRAMES = {
    "1 Minute": "1min",
    "5 Minutes": "5min",
    "15 Minutes": "15min"
}

# =========================================================
# GET MARKET DATA
# =========================================================

def get_data(symbol, interval):

    if not API_KEY:
        return None, "Twelve Data API key is missing."

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
            timeout=15
        )

        if response.status_code != 200:
            return None, (
                f"Twelve Data HTTP error: "
                f"{response.status_code}"
            )

        data = response.json()

    except requests.exceptions.Timeout:
        return None, "Twelve Data response timeout. Dobara START ANALYZE karein."

    except requests.exceptions.RequestException as e:
        return None, f"Internet/API connection error: {e}"

    except ValueError:
        return None, "Twelve Data ne invalid response diya."

    if data.get("status") == "error":
        return None, data.get(
            "message",
            "Twelve Data market data error."
        )

    if "values" not in data:
        return None, data.get(
            "message",
            "Market data available nahi hai."
        )

    df = pd.DataFrame(data["values"])

    if df.empty:
        return None, "Market data empty hai."

    required_columns = [
        "datetime",
        "open",
        "high",
        "low",
        "close"
    ]

    for column in required_columns:
        if column not in df.columns:
            return None, f"Required market field missing: {column}"

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    for column in [
        "open",
        "high",
        "low",
        "close"
    ]:
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

    if len(df) < 60:
        return None, (
            f"Enough historical candles nahi hain. "
            f"Sirf {len(df)} candles mili hain."
        )

    return df, None


# =========================================================
# TECHNICAL INDICATORS
# =========================================================

def indicators(df):

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

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

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

    # ATR
    tr1 = df["high"] - df["low"]

    tr2 = (
        df["high"]
        - df["close"].shift()
    ).abs()

    tr3 = (
        df["low"]
        - df["close"].shift()
    ).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["atr"] = tr.rolling(14).mean()

    df["atr_avg"] = df["atr"].rolling(
        30
    ).mean()

    # Candle strength
    df["body"] = (
        df["close"]
        - df["open"]
    ).abs()

    df["range"] = (
        df["high"]
        - df["low"]
    )

    df["body_ratio"] = (
        df["body"]
        / df["range"].replace(
            0,
            np.nan
        )
    )

    # Momentum
    df["momentum"] = (
        df["close"].pct_change(3)
    )

    return df


# =========================================================
# SIGNAL ENGINE
# =========================================================

def signal_engine(df):

    last = df.iloc[-1]

    up = 0
    down = 0

    reasons = []

    # -----------------------------------------------------
    # EMA TREND
    # -----------------------------------------------------

    if (
        last["ema9"]
        > last["ema21"]
        > last["ema50"]
    ):
        up += 3
        reasons.append(
            "Strong bullish EMA trend"
        )

    elif (
        last["ema9"]
        < last["ema21"]
        < last["ema50"]
    ):
        down += 3
        reasons.append(
            "Strong bearish EMA trend"
        )

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    rsi = last["rsi"]

    if pd.notna(rsi):

        if 52 <= rsi <= 68:
            up += 2
            reasons.append(
                "Bullish RSI zone"
            )

        elif 32 <= rsi <= 48:
            down += 2
            reasons.append(
                "Bearish RSI zone"
            )

        elif rsi > 68:
            down += 1
            reasons.append(
                "RSI high / possible pullback"
            )

        elif rsi < 32:
            up += 1
            reasons.append(
                "RSI low / possible recovery"
            )

    # -----------------------------------------------------
    # MACD
    # -----------------------------------------------------

    if (
        pd.notna(last["macd"])
        and pd.notna(last["macd_signal"])
    ):

        if last["macd"] > last["macd_signal"]:
            up += 2
            reasons.append(
                "MACD bullish"
            )

        elif last["macd"] < last["macd_signal"]:
            down += 2
            reasons.append(
                "MACD bearish"
            )

    # -----------------------------------------------------
    # MOMENTUM
    # -----------------------------------------------------

    momentum = last["momentum"]

    if pd.notna(momentum):

        if momentum > 0:
            up += 2
            reasons.append(
                "Positive momentum"
            )

        elif momentum < 0:
            down += 2
            reasons.append(
                "Negative momentum"
            )

    # -----------------------------------------------------
    # CANDLE STRENGTH
    # -----------------------------------------------------

    body_ratio = last["body_ratio"]

    if pd.notna(body_ratio):

        if body_ratio >= 0.55:

            if last["close"] > last["open"]:
                up += 2
                reasons.append(
                    "Strong bullish candle"
                )

            elif last["close"] < last["open"]:
                down += 2
                reasons.append(
                    "Strong bearish candle"
                )

    # -----------------------------------------------------
    # RECENT BREAKOUT
    # -----------------------------------------------------

    if len(df) >= 12:

        recent_high = (
            df["high"]
            .iloc[-11:-1]
            .max()
        )

        recent_low = (
            df["low"]
            .iloc[-11:-1]
            .min()
        )

        if last["close"] > recent_high:
            up += 2
            reasons.append(
                "Recent high breakout"
            )

        elif last["close"] < recent_low:
            down += 2
            reasons.append(
                "Recent low breakdown"
            )

    # -----------------------------------------------------
    # VOLATILITY
    # -----------------------------------------------------

    if (
        pd.notna(last["atr"])
        and pd.notna(last["atr_avg"])
    ):

        if last["atr"] >= (
            last["atr_avg"] * 0.75
        ):

            if up > down:
                up += 1

            elif down > up:
                down += 1

    # -----------------------------------------------------
    # FORCE UP/DOWN
    # -----------------------------------------------------

    total = up + down

    if total == 0:

        # No "No Trade"
        # Always return direction

        if (
            last["close"]
            >= last["ema21"]
        ):
            return (
                "UP",
                50.0,
                up,
                down,
                ["Price above EMA21"]
            )

        else:
            return (
                "DOWN",
                50.0,
                up,
                down,
                ["Price below EMA21"]
            )

    if up >= down:

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

    if not reasons:
        reasons.append(
            "Direction selected from current market structure"
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

def backtest(df):

    wins = 0
    losses = 0

    rows = []

    if len(df) < 65:
        return (
            0,
            0,
            0,
            pd.DataFrame()
        )

    for i in range(
        60,
        len(df) - 1
    ):

        historical = df.iloc[
            :i + 1
        ].copy()

        (
            signal,
            strength,
            up,
            down,
            reasons
        ) = signal_engine(
            historical
        )

        current = df.iloc[i]["close"]

        following = df.iloc[
            i + 1
        ]["close"]

        if signal == "UP":
            result = following > current
        else:
            result = following < current

        if result:
            wins += 1
            outcome = "WIN"
        else:
            losses += 1
            outcome = "LOSS"

        rows.append({
            "time": df.iloc[i]["datetime"],
            "signal": signal,
            "strength": strength,
            "result": outcome
        })

    total = wins + losses

    if total > 0:
        accuracy = (
            wins / total
        ) * 100
    else:
        accuracy = 0

    return (
        accuracy,
        wins,
        losses,
        pd.DataFrame(rows)
    )


# =========================================================
# APP UI
# =========================================================

st.title(
    "📊 Live Trading Signal Bot"
)

st.caption(
    "Real-time market analysis + historical backtesting"
)

# =========================================================
# API KEY CHECK
# =========================================================

if not API_KEY:

    st.error(
        "❌ Twelve Data API key missing."
    )

    st.info(
        "Streamlit Secrets mein "
        "TWELVE_DATA_API_KEY add karein."
    )

    st.stop()

# =========================================================
# SETTINGS
# =========================================================

col1, col2 = st.columns(2)

with col1:

    pair = st.selectbox(
        "Pair",
        PAIRS
    )

with col2:

    timeframe = st.selectbox(
        "Timeframe",
        list(TIMEFRAMES.keys())
    )

# =========================================================
# ANALYZE BUTTON
# =========================================================

start = st.button(
    "🚀 START ANALYZE",
    use_container_width=True
)

# =========================================================
# ANALYSIS
# =========================================================

if start:

    with st.spinner(
        f"Analyzing {pair} — {timeframe}..."
    ):

        df, error = get_data(
            pair,
            TIMEFRAMES[timeframe]
        )

        if error:

            st.error(
                f"❌ {error}"
            )

            st.stop()

        try:

            df = indicators(df)

            (
                signal,
                strength,
                up,
                down,
                reasons
            ) = signal_engine(df)

            (
                accuracy,
                wins,
                losses,
                history
            ) = backtest(df)

        except Exception as e:

            st.error(
                f"Analysis error: {e}"
            )

            st.stop()

        price = float(
            df.iloc[-1]["close"]
        )

        rsi = df.iloc[-1]["rsi"]
        macd = df.iloc[-1]["macd"]
        momentum = df.iloc[-1]["momentum"]

    # =====================================================
    # RESULT
    # =====================================================

    st.success(
        "✅ Analysis completed"
    )

    st.divider()

    st.subheader(
        f"{pair} — {timeframe}"
    )

    # -----------------------------------------------------
    # MAIN METRICS
    # -----------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Live Price",
            f"{price:.6f}"
        )

    with c2:

        st.metric(
            "Signal Strength",
            f"{strength}%"
        )

    with c3:

        st.metric(
            "Signal",
            signal
        )

    # -----------------------------------------------------
    # DIRECTION
    # -----------------------------------------------------

    if signal == "UP":

        st.success(
            "🟢 CURRENT TRADING DIRECTION: UP"
        )

    else:

        st.error(
            "🔴 CURRENT TRADING DIRECTION: DOWN"
        )

    # -----------------------------------------------------
    # SCORES
    # -----------------------------------------------------

    st.divider()

    st.write(
        "### 📊 Signal Scores"
    )

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "UP Score",
            up
        )

    with c2:

        st.metric(
            "DOWN Score",
            down
        )

    # -----------------------------------------------------
    # TECHNICAL ANALYSIS
    # -----------------------------------------------------

    st.write(
        "### 🧠 Technical Analysis"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        if pd.notna(rsi):
            st.metric(
                "RSI",
                f"{rsi:.2f}"
            )
        else:
            st.metric(
                "RSI",
                "N/A"
            )

    with c2:

        if pd.notna(macd):
            st.metric(
                "MACD",
                f"{macd:.8f}"
            )
        else:
            st.metric(
                "MACD",
                "N/A"
            )

    with c3:

        if pd.notna(momentum):

            st.metric(
                "Momentum",
                f"{momentum * 100:.4f}%"
            )

        else:

            st.metric(
                "Momentum",
                "N/A"
            )

    # -----------------------------------------------------
    # REASONS
    # -----------------------------------------------------

    st.write(
        "### 🔎 Signal Reasons"
    )

    for reason in reasons:

        st.write(
            f"• {reason}"
        )

    # -----------------------------------------------------
    # BACKTEST
    # -----------------------------------------------------

    st.divider()

    st.write(
        "### 🧪 Historical Backtest"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Accuracy",
            f"{accuracy:.2f}%"
        )

    with c2:

        st.metric(
            "WIN",
            wins
        )

    with c3:

        st.metric(
            "LOSS",
            losses
        )

    st.caption(
        f"Tested on {len(history)} historical signals."
    )

    # -----------------------------------------------------
    # TIME
    # -----------------------------------------------------

    st.write(
        "### 🕒 Signal Time"
    )

    st.write(
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    st.warning(
        "⚠️ Historical accuracy future results "
        "ki guarantee nahi hai."
    )

else:

    # =====================================================
    # INITIAL SCREEN
    # =====================================================

    st.info(
        "Pair aur timeframe select karein, "
        "phir START ANALYZE dabayein."
    )

    st.caption(
        "Market data START ANALYZE dabane ke baad hi "
        "Twelve Data se fetch hoga."
        )
