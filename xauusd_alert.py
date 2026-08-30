"""
XAUUSD combined signal alert -> Telegram (FIXED: per-candle backfill)
-----------------------------------------------------------------------
Runs FOUR independent checks and sends a separate Telegram message for
each one that fires:
  1) ALIGNMENT + SWEEP (1-minute chart):
     Breakout + EMA cross + RSI overbought/oversold ALL fire and agree
     on the 1-min chart, AND at least one of 30min/1h/4h shows a fresh
     liquidity sweep in the same direction (evaluated AS OF that 1-min
     candle's time, not "now" -- see find_htf_sweep_asof below).
  2) BULLISH DIVERGENCE + STRUCTURE (1-minute chart)
  3) BEARISH DIVERGENCE + STRUCTURE (mirror of #2)
  4) RSI EXTREME -> 50 RETEST -> CONTINUATION
WHAT CHANGED FROM THE PREVIOUS VERSION
----------------------------------------
The old version called every check function against only `df.iloc[-1]`
-- the single newest closed candle at fetch time. Since this runs every
~15 min on 1-min data, any signal that occurred on one of the other
~14 candles between runs was never evaluated at all. This was
especially damaging for Alert #4, which is intentionally designed to
fire on exactly ONE bar (the exact bar RSI turns) -- if that bar wasn't
the one candle happening to be "last" at cron time, the alert was gone
permanently, not just late.
This version is now STATEFUL (previously it was intentionally
stateless). It persists a `last_processed_time` cursor to
alert_state.json and, each run, walks forward through every new closed
1-min candle since the last successful run -- in chronological order --
running all four checks against each one. Your GitHub Actions workflow
must commit alert_state.json back to the repo after each run (same
pattern as your liquidity/MSS script) or the cursor resets every run
and you're back to only checking the latest candle.
HTF SWEEP LOOKUP (Alert #1) is now time-aware: for a given 1-min
candle being backfilled, it finds "the most recently closed 30min/1h/4h
candle as of THAT candle's time" and checks whether THAT specific HTF
candle swept the swing window immediately before it -- reproducing the
original real-time logic point-by-point instead of only checking
against the current moment.
Required environment variables:
  TWELVE_DATA_API_KEY
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID
"""
import os
import json
import requests
import pandas as pd
SYMBOL = "XAU/USD"
STATE_FILE = "alert_state.json"
# --- Shared 1-minute chart settings ---
SIGNAL_INTERVAL = "1min"
SIGNAL_OUTPUT_SIZE = 300  # widened: needs enough history behind the FIRST
                          # new candle to satisfy every lookback below,
                          # even after a delayed run backfills several bars
EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
# --- Alert #1: breakout + MA cross + RSI alignment ---
BREAKOUT_LOOKBACK = 20
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
REQUIRED_SIGNAL_TYPES = {"breakout", "ma_cross", "rsi"}
# --- Alert #1: higher-timeframe liquidity sweep filter ---
HTF_CONFIGS = [  # (interval, outputsize, swing lookback bars)
    ("30min", 150, 20),
    ("1h", 150, 20),
    ("4h", 150, 20),
]
# --- Alerts #1/#2/#3: SL/TP structure reference ---
STRUCTURE_LOOKBACK = 30
SL_BUFFER = 0.30
MIN_RR = 2
# --- Alerts #2/#3: divergence + structure ---
FRACTAL_N = 3
RSI_RETEST_LOW = 45
RSI_RETEST_HIGH = 55
# --- Alert #4: RSI extreme -> 50 retest ---
RSI_OVERBOUGHT_ZONE = 70
RSI_OVERSOLD_ZONE = 30
RSI_DIP_CONFIRM = 45
RSI_RISE_CONFIRM = 55
RETEST_SCAN_LOOKBACK = 80
TWELVE_DATA_API_KEY = os.environ["TWELVE_DATA_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
# ---------------------------------------------------------------------
# State
# ---------------------------------------------------------------------
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_processed_time": None}
def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
# ---------------------------------------------------------------------
# Data fetch / indicators
# ---------------------------------------------------------------------
def fetch_candles(interval: str, outputsize: int) -> pd.DataFrame:
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": SYMBOL,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_API_KEY,
        "order": "ASC",
    }
    resp = requests.get(url, params=params, timeout=20)
    data = resp.json()
    if "values" not in data:
        raise RuntimeError(f"Twelve Data error ({interval}): {data}")
    df = pd.DataFrame(data["values"])
    df = df.rename(columns={"datetime": "time"})
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df
def compute_structure_reference(df: pd.DataFrame, direction: str) -> dict | None:
    entry = df.iloc[-1]["close"]
    lookback = df.iloc[-(STRUCTURE_LOOKBACK + 1):-1]
    if len(lookback) < STRUCTURE_LOOKBACK:
        return None
    if direction == "bullish":
        swing_low = lookback["low"].min()
sl = swing_low - SL_BUFFER
        risk = entry - sl
        if risk <= 0:
            return None
        tp1 = entry + risk
        tp2 = entry + risk * MIN_RR
    else:
        swing_high = lookback["high"].max()
        sl = swing_high + SL_BUFFER
        risk = sl - entry
        if risk <= 0:
            return None
        tp1 = entry - risk
        tp2 = entry - risk * MIN_RR
    return {"entry": entry, "sl": sl, "risk": risk, "tp1_breakeven": tp1, "tp2": tp2}
# ---------------------------------------------------------------------
# Alert #1: breakout + MA cross + RSI alignment, filtered by HTF sweep
# ---------------------------------------------------------------------
def check_signals(df: pd.DataFrame) -> list[dict]:
    signals = []
    last = df.iloc[-1]
    prev = df.iloc[-2]
    lookback = df.iloc[-(BREAKOUT_LOOKBACK + 1):-1]
    recent_high = lookback["high"].max()
    recent_low = lookback["low"].min()
    if last["close"] > recent_high:
        signals.append({
            "type": "breakout", "direction": "bullish",
            "text": f"🟢 BREAKOUT UP: closed at {last['close']:.2f}, "
                    f"above {BREAKOUT_LOOKBACK}-bar high ({recent_high:.2f})",
        })
    elif last["close"] < recent_low:
        signals.append({
            "type": "breakout", "direction": "bearish",
            "text": f"🔴 BREAKOUT DOWN: closed at {last['close']:.2f}, "
                    f"below {BREAKOUT_LOOKBACK}-bar low ({recent_low:.2f})",
        })
    if prev["ema_fast"] <= prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]:
        signals.append({
            "type": "ma_cross", "direction": "bullish",
            "text": f"🟢 MA CROSS UP: EMA{EMA_FAST} > EMA{EMA_SLOW} at {last['close']:.2f}",
        })
    elif prev["ema_fast"] >= prev["ema_slow"] and last["ema_fast"] < last["ema_slow"]:
        signals.append({
            "type": "ma_cross", "direction": "bearish",
            "text": f"🔴 MA CROSS DOWN: EMA{EMA_FAST} < EMA{EMA_SLOW} at {last['close']:.2f}",
        })
    if last["rsi"] >= RSI_OVERBOUGHT:
        signals.append({
            "type": "rsi", "direction": "bearish",
            "text": f"⚠️ RSI OVERBOUGHT: RSI({RSI_PERIOD}) = {last['rsi']:.1f}",
        })
    elif last["rsi"] <= RSI_OVERSOLD:
        signals.append({
            "type": "rsi", "direction": "bullish",
            "text": f"⚠️ RSI OVERSOLD: RSI({RSI_PERIOD}) = {last['rsi']:.1f}",
        })
    return signals
def all_signals_agree(signals: list[dict]) -> str | None:
    types_present = {s["type"] for s in signals}
    if types_present != REQUIRED_SIGNAL_TYPES:
        return None
    directions = {s["direction"] for s in signals}
    if len(directions) != 1:
        return None
    return directions.pop()
def build_htf_sweep_table(interval: str, outputsize: int, lookback: int) -> pd.DataFrame:
    """Precomputes, for every candle in the fetched window, whether THAT
    candle (using the `lookback` bars before it) swept a swing high/low.
    This lets us look up 'was there a sweep as of time T' for any T,
    instead of only ever checking the single latest candle."""
    df = fetch_candles(interval, outputsize)
    df["swing_high"] = df["high"].shift(1).rolling(lookback).max()
    df["swing_low"] = df["low"].shift(1).rolling(lookback).min()
    df["sweep_bearish"] = (df["high"] > df["swing_high"]) & (df["close"] < df["swing_high"])
    df["sweep_bullish"] = (df["low"] < df["swing_low"]) & (df["close"] > df["swing_low"])
    return df
def find_htf_sweep_asof(htf_tables: dict, target_time: pd.Timestamp, direction: str) -> dict | None:
    """For each configured HTF interval, finds the most recently closed
    candle as of `target_time` and checks whether THAT candle swept --
    reproducing what the original script did in real time, but usable
    for any historical 1-min candle being backfilled."""
    col = "sweep_bearish" if direction == "bearish" else "sweep_bullish"
    level_col = "swing_high" if direction == "bearish" else "swing_low"
    wick_col = "high" if direction == "bearish" else "low"
    for interval, _, _ in HTF_CONFIGS:
        df = htf_tables[interval]
        eligible = df[df["time"] <= target_time]
        if eligible.empty:
            continue
        row = eligible.iloc[-1]
        if bool(row[col]):
            return {"interval": interval, "level": row[level_col], "wick": row[wick_col]}
    return None
def build_alignment_message(df: pd.DataFrame, direction: str, signals: list[dict], sweep: dict, candle_time) -> str:
    lines = [f"XAUUSD signal ({SIGNAL_INTERVAL} chart, {direction}) -- ALIGNMENT + SWEEP  [{candle_time}]"]
    for sig in signals:
        lines.append(sig["text"])
    lines.append("")
    lines.append(
        f"💧 {sweep['interval']} LIQUIDITY SWEEP: wicked to {sweep['wick']:.2f}, "
        f"swept level {sweep['level']:.2f}, closed back inside"
    )
    ref = compute_structure_reference(df, direction)
    if ref:
        lines.append("")
        lines.append(f"1M structure reference ({direction}):")
        lines.append(f"Entry ~ {ref['entry']:.2f}")
        lines.append(f"SL (1M swing): {ref['sl']:.2f}")
        lines.append(f"Move to breakeven at: {ref['tp1_breakeven']:.2f}  (1R)")
        lines.append(f"TP2 (1:{MIN_RR}): {ref['tp2']:.2f}")
    lines.append("")
    lines.append("Manual step: confirm structure/rejection on chart before")
    lines.append("entering. This alert does not place trades or move your SL.")
    return "\n".join(lines)
# ---------------------------------------------------------------------
# Alerts #2/#3: divergence + structure shift + retest
# ---------------------------------------------------------------------
def find_fractal_swings(df: pd.DataFrame, n: int = FRACTAL_N):
    highs, lows = [], []
    for i in range(n, len(df) - n):
        window_low = df["low"].iloc[i - n: i + n + 1]
        window_high = df["high"].iloc[i - n: i + n + 1]
        if df["low"].iloc[i] == window_low.min():
            lows.append(i)
        if df["high"].iloc[i] == window_high.max():
            highs.append(i)
    return highs, lows
def check_bullish_divergence(df: pd.DataFrame, highs_idx, lows_idx):
    if len(lows_idx) < 2:
        return None
    i1, i2 = lows_idx[-2], lows_idx[-1]
    price1, price2 = df["low"].iloc[i1], df["low"].iloc[i2]
    rsi1, rsi2 = df["rsi"].iloc[i1], df["rsi"].iloc[i2]
    if not (price2 < price1 and rsi2 > rsi1):
        return None
    candidate_highs = [h for h in highs_idx if h > i1]
    if not candidate_highs:
        return None
    structure_high_idx = candidate_highs[0]
    structure_high_price = df["high"].iloc[structure_high_idx]
