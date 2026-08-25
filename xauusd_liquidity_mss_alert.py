"""
XAUUSD Liquidity Sweep + 1m MSS Confirmation Alert Script
------------------------------------------------------------
Stage 1: Detects a liquidity sweep of Month/Week/Day/4H high or low
         on a closed 1-minute candle -> sends a Telegram alert with
         the price level to watch for MSS.
Stage 2: On subsequent 1m candle closes, checks whether price closes
         beyond the last swing point (MSS) -> sends confirmation alert.

Runs standalone (separate from your breakout/MA/RSI script).
Designed to run on a schedule via GitHub Actions.

Required environment variables (set as GitHub Actions secrets):
    TWELVE_DATA_API_KEY
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID

State is persisted to mss_state.json — your GitHub Actions workflow
must commit this file back to the repo after each run (same pattern
as your existing script), otherwise sweep flags reset every run.

API BUDGET NOTE (Twelve Data free plan: 8 req/min, 800 req/day):
This script makes 5 calls per run (month, week, day, 4h, 1min).
Running every 5 min = ~1440 calls/day, which exceeds the free daily
cap. Recommended: run this every 10-15 min, OR split into two
workflows -- one that refreshes HTF levels every few hours (cheap),
and one that polls only the 1m candle more frequently for MSS checks.
"""

import requests
import json
import os

API_KEY = os.environ["TWELVE_DATA_API_KEY"]
TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

SYMBOL = "XAU/USD"
STATE_FILE = "mss_state.json"
SWING_LOOKBACK = 3  # candles on each side to confirm a swing point


def fetch_candles(interval, outputsize=30):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": SYMBOL,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": API_KEY,
    }
    r = requests.get(url, params=params, timeout=15)
    data = r.json()
    if "values" not in data:
        raise RuntimeError(f"Twelve Data error for interval={interval}: {data}")
    candles = data["values"]
    candles.reverse()  # oldest -> newest
    for c in candles:
        c["open"] = float(c["open"])
        c["high"] = float(c["high"])
        c["low"] = float(c["low"])
        c["close"] = float(c["close"])
    return candles


def default_state():
    return {
        "swept": {
            k: False
            for k in [
                "month_high", "month_low",
                "week_high", "week_low",
                "day_high", "day_low",
                "4h_high", "4h_low",
            ]
        },
        "period_keys": {},
        "mss_pending": None,  # {"direction": "bullish"/"bearish", "level": float}
    }


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return default_state()


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=15)
    if resp.status_code != 200:
        print(f"Telegram send failed: {resp.text}")


def get_htf_levels():
    month = fetch_candles("1month", outputsize=2)[-2]  # last COMPLETED month
    week = fetch_candles("1week", outputsize=2)[-2]
    day = fetch_candles("1day", outputsize=2)[-2]
    h4 = fetch_candles("4h", outputsize=2)[-2]

    levels = {
        "month_high": month["high"], "month_low": month["low"],
        "week_high": week["high"], "week_low": week["low"],
        "day_high": day["high"], "day_low": day["low"],
        "4h_high": h4["high"], "4h_low": h4["low"],
    }
    period_keys = {
        "month": month["datetime"], "week": week["datetime"],
        "day": day["datetime"], "4h": h4["datetime"],
    }
    return levels, period_keys


def find_last_swing(candles, kind, lookback=SWING_LOOKBACK):
    """kind='low' -> last confirmed swing low (for bullish MSS target)
       kind='high' -> last confirmed swing high (for bearish MSS target)"""
    for i in range(len(candles) - lookback - 1, lookback, -1):
        window = candles[i - lookback: i + lookback + 1]
        c = candles[i]
        if kind == "low" and c["low"] == min(w["low"] for w in window):
            return c["low"]
        if kind == "high" and c["high"] == max(w["high"] for w in window):
            return c["high"]
    return None


def main():
    state = load_state()
    levels, period_keys = get_htf_levels()

    # Reset sweep flags whenever a new HTF period starts
    for period in ["month", "week", "day", "4h"]:
        prev_key = state["period_keys"].get(period)
        cur_key = period_keys[period]
        if prev_key != cur_key:
            state["swept"][f"{period}_high"] = False
            state["swept"][f"{period}_low"] = False
            state["period_keys"][period] = cur_key

    m1 = fetch_candles("1min", outputsize=30)
    last_closed = m1[-2]        # most recently CLOSED 1m candle
    prior_candles = m1[:-1]     # candles up to and including last_closed, for swing calc

    swept_this_run = []
    for period in ["month", "week", "day", "4h"]:
        high_key, low_key = f"{period}_high", f"{period}_low"
        level_high, level_low = levels[high_key], levels[low_key]

        # Sellside sweep: wick above HTF high, closes back below
        if not state["swept"][high_key] and last_closed["high"] > level_high and last_closed["close"] < level_high:
            state["swept"][high_key] = True
            swept_this_run.append(("sell", period, level_high))

        # Buyside sweep: wick below HTF low, closes back above
        if not state["swept"][low_key] and last_closed["low"] < level_low and last_closed["close"] > level_low:
            state["swept"][low_key] = True
            swept_this_run.append(("buy", period, level_low))

    # Stage 1: sweep alert(s) -- fires only on the candle that caused the sweep
    if swept_this_run:
        sell_sweeps = [s for s in swept_this_run if s[0] == "sell"]
        buy_sweeps = [s for s in swept_this_run if s[0] == "buy"]

        if sell_sweeps:
            names = ", ".join(f"{p.title()} High @ {lvl:.2f}" for _, p, lvl in sell_sweeps)
            swing_low = find_last_swing(prior_candles, "low")
            if swing_low is not None:
                state["mss_pending"] = {"direction": "bearish", "level": swing_low}
                send_telegram(
                    f"⚠️ SELL-SIDE LIQUIDITY SWEPT: {names}\n"
                    f"Watch 1m CLOSE below {swing_low:.2f} for bearish MSS."
                )

        if buy_sweeps:
            names = ", ".join(f"{p.title()} Low @ {lvl:.2f}" for _, p, lvl in buy_sweeps)
            swing_high = find_last_swing(prior_candles, "high")
            if swing_high is not None:
                state["mss_pending"] = {"direction": "bullish", "level": swing_high}
                send_telegram(
                    f"⚠️ BUY-SIDE LIQUIDITY SWEPT: {names}\n"
                    f"Watch 1m CLOSE above {swing_high:.2f} for bullish MSS."
                )

    # Stage 2: MSS confirmation -- only checked once a sweep is pending,
    # and only against candles that close AFTER the sweep candle.
    pending = state.get("mss_pending")
    if pending:
        level = pending["level"]
        if pending["direction"] == "bearish" and last_closed["close"] < level:
            send_telegram(
                f"✅ Bearish MSS confirmed — 1m closed at {last_closed['close']:.2f}, "
                f"below {level:.2f}. Short setup active."
            )
            state["mss_pending"] = None
        elif pending["direction"] == "bullish" and last_closed["close"] > level:
            send_telegram(
                f"✅ Bullish MSS confirmed — 1m closed at {last_closed['close']:.2f}, "
                f"above {level:.2f}. Long setup active."
            )
            state["mss_pending"] = None

    save_state(state)


if __name__ == "__main__":
    main()
