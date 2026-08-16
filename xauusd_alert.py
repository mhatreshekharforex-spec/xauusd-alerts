"""
XAUUSD signal alert -> Telegram
15M scan for breakout / EMA cross / RSI extreme.
When a signal fires, pulls 3M data to compute a TIGHT stop loss from
recent 3M swing structure (instead of using the wider 15M range),
plus a suggested breakeven-move reference point.

IMPORTANT: This script only sends alerts. It does NOT place trades,
does NOT move your stop loss automatically, and does NOT connect to
any broker. Every number it sends is a reference for you to act on
manually in your own trading platform.

Required environment variables:
  TWELVE_DATA_API_KEY
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID
"""

import os
import sys
import requests
import pandas as pd

SYMBOL = "XAU/USD"

# 15M signal-scan settings
SIGNAL_INTERVAL = "15min"
SIGNAL_OUTPUT_SIZE = 60
BREAKOUT_LOOKBACK = 20
EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# 3M structure settings (for the tight SL reference)
STRUCTURE_INTERVAL = "3min"
STRUCTURE_OUTPUT_SIZE = 30
STRUCTURE_LOOKBACK = 15   # bars used to find the recent 3M swing high/low
SL_BUFFER = 0.30          # extra cushion beyond the swing point, in price units
MIN_RR = 2                # your checklist's minimum R:R -> sets TP2

TWELVE_DATA_API_KEY = os.environ["TWELVE_DATA_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]


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


def check_signals(df: pd.DataFrame) -> list[dict]:
    """Returns a list of signal dicts: {text, direction} where
    direction is 'bullish' or 'bearish' (used to pick the right
    3M structure side for the SL)."""
    signals = []
    last = df.iloc[-1]
    prev = df.iloc[-2]

    lookback = df.iloc[-(BREAKOUT_LOOKBACK + 1):-1]
    recent_high = lookback["high"].max()
    recent_low = lookback["low"].min()
    if last["close"] > recent_high:
        signals.append({
            "direction": "bullish",
            "text": f"🟢 BREAKOUT UP: closed at {last['close']:.2f}, "
                    f"above {BREAKOUT_LOOKBACK}-bar high ({recent_high:.2f})",
        })
    elif last["close"] < recent_low:
        signals.append({
            "direction": "bearish",
            "text": f"🔴 BREAKOUT DOWN: closed at {last['close']:.2f}, "
                    f"below {BREAKOUT_LOOKBACK}-bar low ({recent_low:.2f})",
        })

    if prev["ema_fast"] <= prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]:
        signals.append({
            "direction": "bullish",
            "text": f"🟢 MA CROSS UP: EMA{EMA_FAST} > EMA{EMA_SLOW} at {last['close']:.2f}",
        })
    elif prev["ema_fast"] >= prev["ema_slow"] and last["ema_fast"] < last["ema_slow"]:
        signals.append({
            "direction": "bearish",
            "text": f"🔴 MA CROSS DOWN: EMA{EMA_FAST} < EMA{EMA_SLOW} at {last['close']:.2f}",
        })

    if last["rsi"] >= RSI_OVERBOUGHT:
        signals.append({
            "direction": "bearish",
            "text": f"⚠️ RSI OVERBOUGHT: RSI({RSI_PERIOD}) = {last['rsi']:.1f}",
        })
    elif last["rsi"] <= RSI_OVERSOLD:
        signals.append({
            "direction": "bullish",
            "text": f"⚠️ RSI OVERSOLD: RSI({RSI_PERIOD}) = {last['rsi']:.1f}",
        })

    return signals


def compute_3m_reference(direction: str) -> dict | None:
    """Pulls 3M candles and computes a tight SL from the recent 3M
    swing high/low, plus TP1 (1R -> breakeven-move reference) and
    TP2 (MIN_RR -> your checklist's minimum target)."""
    df3 = fetch_candles(STRUCTURE_INTERVAL, STRUCTURE_OUTPUT_SIZE)
    entry = df3.iloc[-1]["close"]
    lookback = df3.iloc[-(STRUCTURE_LOOKBACK + 1):-1]

    if direction == "bullish":
        swing_low = lookback["low"].min()
        sl = swing_low - SL_BUFFER
        risk = entry - sl
        if risk <= 0:
            return None
        tp1 = entry + risk        # 1R -> breakeven-move reference
        tp2 = entry + risk * MIN_RR
    else:
        swing_high = lookback["high"].max()
        sl = swing_high + SL_BUFFER
        risk = sl - entry
        if risk <= 0:
            return None
        tp1 = entry - risk
        tp2 = entry - risk * MIN_RR

    return {
        "entry": entry,
        "sl": sl,
        "risk": risk,
        "tp1_breakeven": tp1,
        "tp2": tp2,
    }


def send_telegram(message: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=20)
    if not resp.ok:
        print(f"Telegram send failed: {resp.text}", file=sys.stderr)


def main() -> None:
    df = fetch_candles(SIGNAL_INTERVAL, SIGNAL_OUTPUT_SIZE)
    df = compute_indicators(df)
    signals = check_signals(df)

    if not signals:
        print("No signal this run.")
        return

    lines = [f"XAUUSD signal ({SIGNAL_INTERVAL} chart)"]
    for sig in signals:
        lines.append(sig["text"])

    # Use the first signal's direction to compute the 3M reference.
    # (If signals disagree in direction, this flags it instead of guessing.)
    directions = {s["direction"] for s in signals}
    if len(directions) == 1:
        direction = directions.pop()
        ref = compute_3m_reference(direction)
        if ref:
            lines.append("")
            lines.append(f"3M reference ({direction}):")
            lines.append(f"Entry ~ {ref['entry']:.2f}")
            lines.append(f"SL (3M swing): {ref['sl']:.2f}")
            lines.append(f"Move to breakeven at: {ref['tp1_breakeven']:.2f}  (1R)")
            lines.append(f"TP2 (1:{MIN_RR}): {ref['tp2']:.2f}")
            lines.append("")
            lines.append("Manual step: confirm 3M liquidity sweep + MSS on")
            lines.append("chart before entering. This alert does not place")
            lines.append("trades or move your SL for you.")
    else:
        lines.append("")
        lines.append("⚠️ Mixed signal directions this run — check chart manually.")

    message = "\n".join(lines)
    send_telegram(message)
    print(message)


if __name__ == "__main__":
    main()
