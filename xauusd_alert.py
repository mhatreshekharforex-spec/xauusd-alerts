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
