"""
XAUUSD combined signal alert -> Telegram
Single script, single cron job. Runs FOUR independent checks every run
and sends a separate Telegram message for each one that fires:

  1) ALIGNMENT + SWEEP (1-minute chart):
     Breakout + EMA cross + RSI overbought/oversold ALL fire and agree
     on the 1-min chart, AND at least one of 30min/1h/4h shows a fresh
     liquidity sweep (wick through a recent swing high/low, close back
     inside) in the same direction.

  2) BULLISH DIVERGENCE + STRUCTURE (1-minute chart):
     Price Lower Low + RSI Higher Low (divergence) -> price closes above
     the swing high between them (structure shift / change of character)
     -> pullback with RSI landing back in 45-55 and price turning up.

  3) BEARISH DIVERGENCE + STRUCTURE: mirror of #2 (Higher High + RSI
     Lower High -> structure break down -> RSI 45-55 retest, turning down).

  4) RSI EXTREME -> 50 RETEST -> CONTINUATION (no price structure needed):
     Bearish: RSI overbought (>=70) -> crosses below 50 -> dips
     meaningfully -> retests 50 -> turns back down (continuation toward
     RSI ~30 expected). Bullish is the mirror (oversold -> crosses above
     50 -> rises -> retests 50 -> turns back up, continuation toward ~70).

     Fix vs earlier version: this only fires on the EXACT bar where RSI
     turns (the bar before it must have still been moving the OLD way),
     so it won't re-fire on every subsequent bar while RSI keeps drifting
     in the same direction inside the 45-55 zone.

SWING DETECTION for #2/#3 uses simple N-bar fractals, which have an
inherent N-bar confirmation lag -- a property of the pattern, not a bug.

STATE: this script is stateless -- each run re-scans the fetched window
fresh. Practical effect: for #1, if the 1-min signals stay aligned for
several consecutive bars, you may get more than one alert for the same
setup (no cooldown). #4 no longer has that issue after the fix above.
Say the word if you want a cooldown/dedupe layer added (needs state
persisted across runs, e.g. a small file committed back to the repo).

IMPORTANT: This script only sends alerts. It does NOT place trades, does
NOT move your stop loss automatically, and does NOT connect to any
broker. None of these patterns are a backtested or independently
validated edge -- treat every alert as a filter to watch and confirm on
your own chart, not a signal to trade blindly.

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

# --- Shared 1-minute chart settings ---
SIGNAL_INTERVAL = "1min"
SIGNAL_OUTPUT_SIZE = 150
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
    ("30min", 60, 20),
    ("1h", 60, 20),
    ("4h", 60, 20),
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


def check_htf_sweep(interval: str, outputsize: int, lookback: int, direction: str):
    df = fetch_candles(interval, outputsize)
    last = df.iloc[-1]
    window = df.iloc[-(lookback + 1):-1]
    swing_high = window["high"].max()
    swing_low = window["low"].min()

    if direction == "bullish":
        if last["low"] < swing_low and last["close"] > swing_low:
            return {"interval": interval, "level": swing_low, "wick": last["low"]}
    else:
        if last["high"] > swing_high and last["close"] < swing_high:
            return {"interval": interval, "level": swing_high, "wick": last["high"]}
    return None


def find_htf_sweep(direction: str) -> dict | None:
    for interval, outputsize, lookback in HTF_CONFIGS:
        sweep = check_htf_sweep(interval, outputsize, lookback, direction)
        if sweep:
            return sweep
    return None


def build_alignment_message(df: pd.DataFrame, direction: str, signals: list[dict], sweep: dict) -> str:
    lines = [f"XAUUSD signal ({SIGNAL_INTERVAL} chart, {direction}) -- ALIGNMENT + SWEEP"]
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

    break_rows = df[(df.index > i2) & (df["close"] > structure_high_price)]
    if break_rows.empty:
        return None
    break_idx = break_rows.index[0]

    last = df.iloc[-1]
    prev = df.iloc[-2]
    if last.name <= break_idx:
        return None
    if not (RSI_RETEST_LOW <= last["rsi"] <= RSI_RETEST_HIGH):
        return None
    if not (last["close"] > prev["close"]):
        return None

    return {
        "type": "bullish_divergence",
        "swing_low_1": (df["time"].iloc[i1], price1, rsi1),
        "swing_low_2": (df["time"].iloc[i2], price2, rsi2),
        "structure_level": structure_high_price,
        "structure_break_time": df["time"].iloc[break_idx],
        "retest_price": last["close"],
        "retest_rsi": last["rsi"],
    }


def check_bearish_divergence(df: pd.DataFrame, highs_idx, lows_idx):
    if len(highs_idx) < 2:
        return None
    i1, i2 = highs_idx[-2], highs_idx[-1]
    price1, price2 = df["high"].iloc[i1], df["high"].iloc[i2]
    rsi1, rsi2 = df["rsi"].iloc[i1], df["rsi"].iloc[i2]
    if not (price2 > price1 and rsi2 < rsi1):
        return None

    candidate_lows = [lo for lo in lows_idx if lo > i1]
    if not candidate_lows:
        return None
    structure_low_idx = candidate_lows[0]
    structure_low_price = df["low"].iloc[structure_low_idx]

    break_rows = df[(df.index > i2) & (df["close"] < structure_low_price)]
    if break_rows.empty:
        return None
    break_idx = break_rows.index[0]

    last = df.iloc[-1]
    prev = df.iloc[-2]
    if last.name <= break_idx:
        return None
    if not (RSI_RETEST_LOW <= last["rsi"] <= RSI_RETEST_HIGH):
        return None
    if not (last["close"] < prev["close"]):
        return None

    return {
        "type": "bearish_divergence",
        "swing_high_1": (df["time"].iloc[i1], price1, rsi1),
        "swing_high_2": (df["time"].iloc[i2], price2, rsi2),
        "structure_level": structure_low_price,
        "structure_break_time": df["time"].iloc[break_idx],
        "retest_price": last["close"],
        "retest_rsi": last["rsi"],
    }


def build_divergence_message(result: dict, df: pd.DataFrame) -> str:
    direction = "bullish" if result["type"] == "bullish_divergence" else "bearish"
    ref = compute_structure_reference(df, direction)

    if result["type"] == "bullish_divergence":
        t1, p1, r1 = result["swing_low_1"]
        t2, p2, r2 = result["swing_low_2"]
        lines = [
            "XAUUSD divergence + structure alert (1M) -- BULLISH",
            "",
            "🟢 Bullish RSI divergence found:",
            f"  Swing low 1: {p1:.2f} (RSI {r1:.1f}) at {t1}",
            f"  Swing low 2: {p2:.2f} (RSI {r2:.1f}) at {t2}  <- lower price, higher RSI",
            "",
            f"📐 Structure shift confirmed: closed above {result['structure_level']:.2f} "
            f"at {result['structure_break_time']} (Higher-Low / change of character)",
            "",
            f"🔁 Retest: price {result['retest_price']:.2f}, RSI {result['retest_rsi']:.1f} "
            f"(in 45-55 zone) and turning back up",
        ]
    else:
        t1, p1, r1 = result["swing_high_1"]
        t2, p2, r2 = result["swing_high_2"]
        lines = [
            "XAUUSD divergence + structure alert (1M) -- BEARISH",
            "",
            "🔴 Bearish RSI divergence found:",
            f"  Swing high 1: {p1:.2f} (RSI {r1:.1f}) at {t1}",
            f"  Swing high 2: {p2:.2f} (RSI {r2:.1f}) at {t2}  <- higher price, lower RSI",
            "",
            f"📐 Structure shift confirmed: closed below {result['structure_level']:.2f} "
            f"at {result['structure_break_time']} (Lower-High / change of character)",
            "",
            f"🔁 Retest: price {result['retest_price']:.2f}, RSI {result['retest_rsi']:.1f} "
            f"(in 45-55 zone) and turning back down",
        ]

    if ref:
        lines.append("")
        lines.append(f"1M structure reference ({direction}):")
        lines.append(f"Entry ~ {ref['entry']:.2f}")
        lines.append(f"SL (1M swing): {ref['sl']:.2f}")
        lines.append(f"Move to breakeven at: {ref['tp1_breakeven']:.2f}  (1R)")
        lines.append(f"TP2 (1:{MIN_RR}): {ref['tp2']:.2f}")

    lines.append("")
    lines.append("Manual step: confirm rejection/candle close on chart before")
    lines.append("entering. This is a discretionary pattern, not a validated edge.")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Alert #4: RSI extreme -> 50 retest -> continuation
# ---------------------------------------------------------------------

def check_bearish_rsi_extreme_retest(df: pd.DataFrame):
    """Fires only on the exact bar RSI turns down at the 50 retest --
    i.e. the bar before it must still have been rising."""
    window = df.iloc[-RETEST_SCAN_LOOKBACK:].reset_index(drop=True)
    rsi = window["rsi"]
    time = window["time"]

    state = 0
    peak_val = peak_time = None
    cross_time = None
    dip_val = dip_time = None

    for i in range(len(rsi)):
        r = rsi.iloc[i]
        if pd.isna(r):
            continue
        if state == 0:
            if r >= RSI_OVERBOUGHT_ZONE:
                peak_val, peak_time = r, time.iloc[i]
                state = 1
        elif state == 1:
            if r >= RSI_OVERBOUGHT_ZONE and r > peak_val:
                peak_val, peak_time = r, time.iloc[i]
            elif r < 50:
                cross_time = time.iloc[i]
                dip_val, dip_time = r, time.iloc[i]
                state = 2
        elif state == 2:
            if r < dip_val:
                dip_val, dip_time = r, time.iloc[i]
            if dip_val <= RSI_DIP_CONFIRM:
                state = 3
        elif state == 3:
            if RSI_RETEST_LOW <= r <= RSI_RETEST_HIGH and i == len(rsi) - 1 and i >= 2:
                prev_r = rsi.iloc[i - 1]
                prev_prev_r = rsi.iloc[i - 2]
                # must be the FIRST bar turning down: was rising, now falling
                if prev_r > prev_prev_r and r <= prev_r:
                    return {
                        "type": "bearish_extreme_retest",
                        "peak_rsi": peak_val, "peak_time": peak_time,
                        "cross_time": cross_time,
                        "dip_rsi": dip_val, "dip_time": dip_time,
                        "retest_rsi": r, "retest_time": time.iloc[i],
                        "price": window["close"].iloc[i],
                    }
    return None


def check_bullish_rsi_extreme_retest(df: pd.DataFrame):
    """Fires only on the exact bar RSI turns up at the 50 retest --
    i.e. the bar before it must still have been falling."""
    window = df.iloc[-RETEST_SCAN_LOOKBACK:].reset_index(drop=True)
    rsi = window["rsi"]
    time = window["time"]

    state = 0
    trough_val = trough_time = None
    cross_time = None
    rise_val = rise_time = None

    for i in range(len(rsi)):
        r = rsi.iloc[i]
        if pd.isna(r):
            continue
        if state == 0:
            if r <= RSI_OVERSOLD_ZONE:
                trough_val, trough_time = r, time.iloc[i]
                state = 1
        elif state == 1:
            if r <= RSI_OVERSOLD_ZONE and r < trough_val:
                trough_val, trough_time = r, time.iloc[i]
            elif r > 50:
                cross_time = time.iloc[i]
                rise_val, rise_time = r, time.iloc[i]
                state = 2
        elif state == 2:
            if r > rise_val:
                rise_val, rise_time = r, time.iloc[i]
            if rise_val >= RSI_RISE_CONFIRM:
                state = 3
        elif state == 3:
            if RSI_RETEST_LOW <= r <= RSI_RETEST_HIGH and i == len(rsi) - 1 and i >= 2:
                prev_r = rsi.iloc[i - 1]
                prev_prev_r = rsi.iloc[i - 2]
                # must be the FIRST bar turning up: was falling, now rising
                if prev_r < prev_prev_r and r >= prev_r:
                    return {
                        "type": "bullish_extreme_retest",
                        "trough_rsi": trough_val, "trough_time": trough_time,
                        "cross_time": cross_time,
                        "rise_rsi": rise_val, "rise_time": rise_time,
                        "retest_rsi": r, "retest_time": time.iloc[i],
                        "price": window["close"].iloc[i],
                    }
    return None


def build_retest_message(result: dict, df: pd.DataFrame) -> str:
    direction = "bullish" if result["type"] == "bullish_extreme_retest" else "bearish"
    ref = compute_structure_reference(df, direction)

    if result["type"] == "bearish_extreme_retest":
        lines = [
            "XAUUSD RSI structure alert (1M) -- BEARISH (50 retest)",
            "",
            f"📉 RSI was overbought: peak {result['peak_rsi']:.1f} at {result['peak_time']}",
            f"↓ Crossed below 50 at {result['cross_time']}",
            f"↓ Dipped to {result['dip_rsi']:.1f} at {result['dip_time']} (confirms real move down)",
            f"🔁 Now retesting 50: RSI {result['retest_rsi']:.1f} at {result['retest_time']}, "
            f"price {result['price']:.2f}, turning back down",
            "",
            "Bias: continuation toward RSI ~30.",
        ]
    else:
        lines = [
            "XAUUSD RSI structure alert (1M) -- BULLISH (50 retest)",
            "",
            f"📈 RSI was oversold: trough {result['trough_rsi']:.1f} at {result['trough_time']}",
            f"↑ Crossed above 50 at {result['cross_time']}",
            f"↑ Rose to {result['rise_rsi']:.1f} at {result['rise_time']} (confirms real move up)",
            f"🔁 Now retesting 50: RSI {result['retest_rsi']:.1f} at {result['retest_time']}, "
            f"price {result['price']:.2f}, turning back up",
            "",
            "Bias: continuation toward RSI ~70.",
        ]

    if ref:
        lines.append("")
        lines.append(f"1M structure reference ({direction}):")
        lines.append(f"Entry ~ {ref['entry']:.2f}")
        lines.append(f"SL (1M swing): {ref['sl']:.2f}")
        lines.append(f"Move to breakeven at: {ref['tp1_breakeven']:.2f}  (1R)")
        lines.append(f"TP2 (1:{MIN_RR}): {ref['tp2']:.2f}")

    lines.append("")
    lines.append("Manual step: confirm rejection/candle close on chart before")
    lines.append("entering. This is a discretionary pattern, not a validated edge.")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Telegram + main
# ---------------------------------------------------------------------

def send_telegram(message: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=20)
    if not resp.ok:
        print(f"Telegram send failed: {resp.text}", file=sys.stderr)


def main() -> None:
    df = fetch_candles(SIGNAL_INTERVAL, SIGNAL_OUTPUT_SIZE)
    df = compute_indicators(df)

    messages = []

    # --- Alert #1: alignment + HTF sweep ---
    signals = check_signals(df)
    direction = all_signals_agree(signals)
    if direction is not None:
        sweep = find_htf_sweep(direction)
        if sweep is not None:
            messages.append(build_alignment_message(df, direction, signals, sweep))

    # --- Alerts #2/#3: divergence + structure ---
    highs_idx, lows_idx = find_fractal_swings(df)
    bullish_div = check_bullish_divergence(df, highs_idx, lows_idx)
    bearish_div = check_bearish_divergence(df, highs_idx, lows_idx)
    if bullish_div:
        messages.append(build_divergence_message(bullish_div, df))
    if bearish_div:
        messages.append(build_divergence_message(bearish_div, df))

    # --- Alert #4: RSI extreme -> 50 retest ---
    bearish_retest = check_bearish_rsi_extreme_retest(df)
    bullish_retest = check_bullish_rsi_extreme_retest(df)
    if bearish_retest:
        messages.append(build_retest_message(bearish_retest, df))
    if bullish_retest:
        messages.append(build_retest_message(bullish_retest, df))

    if not messages:
        print("No signal this run.")
        return

    for message in messages:
        send_telegram(message)
        print(message)
        print("---")


if __name__ == "__main__":
    main()
