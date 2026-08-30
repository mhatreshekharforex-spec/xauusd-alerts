"""
XAUUSD Liquidity Sweep + 1m MSS Confirmation Alert Script (FIXED)
------------------------------------------------------------------
Stage 1: Detects a liquidity sweep of Month/Week/Day/4H high or low
         on a closed 1-minute candle -> sends a Telegram alert with
         the price level to watch for MSS.
Stage 2: On subsequent 1m candle closes, checks whether price closes
         beyond the last swing point (MSS) -> sends a confirmation alert
         with a full trade card.

WHAT CHANGED FROM THE PREVIOUS VERSION
---------------------------------------
The old version only ever looked at the single most-recently-closed 1m
candle (`m1[-2]`) on each run. Since this runs every ~15 min, any sweep
or MSS break that happened on one of the OTHER ~14 candles since the
last run was invisible -- never checked at all. That's why some PDH/PDL
and 4H sweeps never triggered an alert, and why the ones that did fire
were already several points stale.

This version tracks a cursor (`last_processed_time` in the state file)
and walks forward through EVERY new closed 1m candle since the last
successful run, in chronological order, running the same sweep/MSS
logic on each one. This does not eliminate the inherent lag of a
15-min schedule (you can still be up to ~15 min behind the live
market), but it guarantees no candle -- and no sweep/MSS event -- is
silently skipped between runs.

Also widened the 1m fetch window (60 candles instead of 30) to make
sure there's always enough history to cover a full cron gap plus swing
lookback padding, even if a run is delayed.

Required environment variables (set as GitHub Actions secrets):
    TWELVE_DATA_API_KEY
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID

State is persisted to mss_state.json -- your GitHub Actions workflow
must commit this file back to the repo after each run, otherwise
sweep flags and the processing cursor reset every run. See the
accompanying workflow.yml notes for hardening the git push step
(pull-before-push + concurrency group) so a failed push doesn't
silently lose a run's state update.

API BUDGET NOTE (Twelve Data free plan: 8 req/min, 800 req/day):
This script makes 5 calls per run (month, week, day, 4h, 1min).
Running every 15 min = ~480 calls/day, within the free daily cap.
"""

import requests
import json
import os
from datetime import datetime

API_KEY = os.environ["TWELVE_DATA_API_KEY"]
TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

SYMBOL = "XAU/USD"
STATE_FILE = "mss_state.json"
SWING_LOOKBACK = 3        # candles on each side to confirm a swing point
SL_BUFFER_PCT = 0.0005    # 0.05% buffer beyond the sweep candle's wick
RR_MULTIPLE = 2           # fixed Take Profit at 1:2 risk:reward
M1_OUTPUTSIZE = 60        # widened so a delayed run still has full history


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
        "mss_pending": None,
        "last_processed_time": None,  # NEW: cursor -- datetime string of the
                                       # newest 1m candle already checked
    }


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)
        # backfill for state files written by the old script version
        state.setdefault("last_processed_time", None)
        return state
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
       kind='high' -> last confirmed swing high (for bearish MSS target)
    `candles` should be all candles up to (not including) the candle
    being evaluated."""
    for i in range(len(candles) - lookback - 1, lookback, -1):
        window = candles[i - lookback: i + lookback + 1]
        c = candles[i]
        if kind == "low" and c["low"] == min(w["low"] for w in window):
            return c["low"]
        if kind == "high" and c["high"] == max(w["high"] for w in window):
            return c["high"]
    return None


def process_candle(state, levels, all_candles_up_to_and_incl_current):
    """Runs sweep-detection (Stage 1) and MSS-confirmation (Stage 2) logic
    for ONE closed 1m candle, exactly like the original script did for
    `last_closed`, but callable in a loop across several candles."""
    current = all_candles_up_to_and_incl_current[-1]
    prior_candles = all_candles_up_to_and_incl_current[:-1]

    swept_this_candle = []
    for period in ["month", "week", "day", "4h"]:
        high_key, low_key = f"{period}_high", f"{period}_low"
        level_high, level_low = levels[high_key], levels[low_key]

        if not state["swept"][high_key] and current["high"] > level_high and current["close"] < level_high:
            state["swept"][high_key] = True
            swept_this_candle.append(("sell", period, level_high))

        if not state["swept"][low_key] and current["low"] < level_low and current["close"] > level_low:
            state["swept"][low_key] = True
            swept_this_candle.append(("buy", period, level_low))

    # Stage 1: sweep alert(s)
    if swept_this_candle:
        sell_sweeps = [s for s in swept_this_candle if s[0] == "sell"]
        buy_sweeps = [s for s in swept_this_candle if s[0] == "buy"]

        if sell_sweeps:
            names = ", ".join(f"{p.title()} High @ {lvl:.2f}" for _, p, lvl in sell_sweeps)
            swing_low = find_last_swing(prior_candles, "low")
            widest_period = sell_sweeps[0][1]
            if swing_low is not None:
                state["mss_pending"] = {
                    "direction": "bearish",
                    "level": swing_low,
                    "sweep_extreme": current["high"],
                    "swept_period": widest_period,
                }
                send_telegram(
                    f"⚠️ SELL-SIDE LIQUIDITY SWEPT: {names}  [{current['datetime']}]\n"
                    f"Watch 1m CLOSE below {swing_low:.2f} for bearish MSS."
                )

        if buy_sweeps:
            names = ", ".join(f"{p.title()} Low @ {lvl:.2f}" for _, p, lvl in buy_sweeps)
            swing_high = find_last_swing(prior_candles, "high")
            widest_period = buy_sweeps[0][1]
            if swing_high is not None:
                state["mss_pending"] = {
                    "direction": "bullish",
                    "level": swing_high,
                    "sweep_extreme": current["low"],
                    "swept_period": widest_period,
                }
                send_telegram(
                    f"⚠️ BUY-SIDE LIQUIDITY SWEPT: {names}  [{current['datetime']}]\n"
                    f"Watch 1m CLOSE above {swing_high:.2f} for bullish MSS."
                )

    # Stage 2: MSS confirmation
    pending = state.get("mss_pending")
    if pending:
        level = pending["level"]
        sweep_extreme = pending.get("sweep_extreme")
        swept_period = pending.get("swept_period")

        if pending["direction"] == "bearish" and current["close"] < level:
            entry = current["close"]
            sl = sweep_extreme * (1 + SL_BUFFER_PCT)
            risk = sl - entry
            tp_rr = entry - RR_MULTIPLE * risk
            tp_htf = levels.get(f"{swept_period}_low") if swept_period else None

            msg = (
                f"✅ BEARISH MSS CONFIRMED — SHORT SETUP  [{current['datetime']}]\n"
                f"Entry: {entry:.2f}\n"
                f"Stop Loss: {sl:.2f}  (risk {risk:.2f})\n"
                f"TP (1:{RR_MULTIPLE} R:R): {tp_rr:.2f}\n"
            )
            if tp_htf is not None:
                msg += f"TP ({swept_period.title()} Low, HTF target): {tp_htf:.2f}\n"
            msg += f"Swept: {swept_period.title()} High"
            send_telegram(msg)
            state["mss_pending"] = None

        elif pending["direction"] == "bullish" and current["close"] > level:
            entry = current["close"]
            sl = sweep_extreme * (1 - SL_BUFFER_PCT)
            risk = entry - sl
            tp_rr = entry + RR_MULTIPLE * risk
            tp_htf = levels.get(f"{swept_period}_high") if swept_period else None

            msg = (
                f"✅ BULLISH MSS CONFIRMED — LONG SETUP  [{current['datetime']}]\n"
                f"Entry: {entry:.2f}\n"
                f"Stop Loss: {sl:.2f}  (risk {risk:.2f})\n"
                f"TP (1:{RR_MULTIPLE} R:R): {tp_rr:.2f}\n"
            )
            if tp_htf is not None:
                msg += f"TP ({swept_period.title()} High, HTF target): {tp_htf:.2f}\n"
            msg += f"Swept: {swept_period.title()} Low"
            send_telegram(msg)
            state["mss_pending"] = None


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

    m1 = fetch_candles("1min", outputsize=M1_OUTPUTSIZE)
    closed_candles = m1[:-1]  # drop the still-forming last candle, same as before

    last_processed = state.get("last_processed_time")
    if last_processed is None:
        # First ever run (or state was reset): don't replay full history,
        # just establish the cursor at the newest closed candle.
        new_candles = closed_candles[-1:]
    else:
        new_candles = [c for c in closed_candles if c["datetime"] > last_processed]

    if not new_candles:
        print("No new closed 1m candles since last run -- nothing to check.")
        save_state(state)
        return

    print(f"Processing {len(new_candles)} new candle(s): "
          f"{new_candles[0]['datetime']} -> {new_candles[-1]['datetime']}")

    # Walk forward chronologically so sweep-then-MSS ordering within the
    # same batch is respected (e.g. sweep on candle N, MSS confirm on N+3).
    for candle in new_candles:
        idx = closed_candles.index(candle)
        window_up_to_here = closed_candles[: idx + 1]
        process_candle(state, levels, window_up_to_here)

    state["last_processed_time"] = closed_candles[-1]["datetime"]
    save_state(state)


if __name__ == "__main__":
    main()
