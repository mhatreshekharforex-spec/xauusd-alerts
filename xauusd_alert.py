"""
XAUUSD signal alert -> Telegram
Checks three conditions on Gold (XAU/USD):
  1. Breakout of the recent N-bar high/low
  2. EMA9/EMA21 cross
  3. RSI(14) overbought/oversold
"""

import os
import sys
import requests
import pandas as pd

SYMBOL = "XAU/USD"
INTERVAL = "15min"
OUTPUT_SIZE = 60
BREAKOUT_LOOKBACK = 20
EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

TWELVE_DATA_API_KEY = os.environ["TWELVE_DATA_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]


def fetch_candles() -> pd.DataFrame:
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": OUTPUT_SIZE,
        "apikey": TWELVE_DATA_API_KEY,
        "order": "ASC",
    }
    resp = requests.get(url, params=params, timeout=20)
    data = resp.json()
    if "values" not in data:
        raise RuntimeError(f"Twelve Data error: {data}")
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


def check_signals(df: pd.DataFrame) -> list[str]:
    signals = []
    last = df.iloc[-1]
    prev = df.iloc[-2]

    lookback = df.iloc[-(BREAKOUT_LOOKBACK + 1):-1]
    recent_high = lookback["high"].max()
    recent_low = lookback["low"].min()
    if last["close"] > recent_high:
        signals.append(
            f"🟢 BREAKOUT UP: XAUUSD closed at {last['close']:.2f}, "
            f"above the last {BREAKOUT_LOOKBACK}-bar high ({recent_high:.2f})"
        )
    elif last["close"] < recent_low:
        signals.append(
            f"🔴 BREAKOUT DOWN: XAUUSD closed at {last['close']:.2f}, "
            f"below the last {BREAKOUT_LOOKBACK}-bar low ({recent_low:.2f})"
        )

    if prev["ema_fast"] <= prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]:
        signals.append(
            f"🟢 MA CROSS UP: EMA{EMA_FAST} crossed above EMA{EMA_SLOW} at {last['close']:.2f}"
        )
    elif prev["ema_fast"] >= prev["ema_slow"] and last["ema_fast"] < last["ema_slow"]:
        signals.append(
            f"🔴 MA CROSS DOWN: EMA{EMA_FAST} crossed below EMA{EMA_SLOW} at {last['close']:.2f}"
        )

    if last["rsi"] >= RSI_OVERBOUGHT:
        signals.append(f"⚠️ RSI OVERBOUGHT: RSI({RSI_PERIOD}) = {last['rsi']:.1f}")
    elif last["rsi"] <= RSI_OVERSOLD:
        signals.append(f"⚠️ RSI OVERSOLD: RSI({RSI_PERIOD}) = {last['rsi']:.1f}")

    return signals


def send_telegram(message: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=20)
    if not resp.ok:
        print(f"Telegram send failed: {resp.text}", file=sys.stderr)


def main() -> None:
    df = fetch_candles()
    df = compute_indicators(df)
    signals = check_signals(df)

    if not signals:
        print("No signal this run.")
        return

    header = f"XAUUSD signal ({INTERVAL} chart)\n"
    message = header + "\n".join(signals)
    send_telegram(message)
    print(message)


if __name__ == "__main__":
    main()
