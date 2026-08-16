"""
engine/backtest.py — Week 3: walk N3 (Range Fade / Credit) over real history.

Calls evaluate_n3() from engine/rules.py -- the SAME function run_daily.py
will call live -- against real bar_data assembled from data_layer.py's
fo_bars / vix / spot_index tables. This file supplies the CALLER and the
DATA; the rule itself lives in rules.py and is not reimplemented here
(Engineering Review: "one function, two callers... non-negotiable").

SCOPE OF v1 -- what this does NOT yet do:
    Every entered position is held to expiry and settled once against the
    NIFTY closing spot on the expiry date. It does NOT walk day-by-day
    between entry and expiry checking CLAUDE.md's "close if tested" rule
    (no adjustments on the core account -- close early if a strike is
    challenged), and therefore it never exercises the ambiguous-bar honesty
    field -- every outcome here has ambiguous_bar=0 by construction. This
    UNDERSTATES real risk: a spread that goes deep ITM intraday and recovers
    by expiry shows as a full win here, when "close if tested" would have
    exited it at a loss. v1.1 adds the daily walk; this version exists to
    get a first real expectancy number, with that gap disclosed, not hidden.

    Also does not filter event days (RBI/Budget) -- N3Entry.event_day is
    always False here; the vault has no machine-readable event calendar yet.

Run:  python engine/backtest.py 2025-08-14 2026-08-13
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from costs import expiry_cost                                    # noqa: E402
from data_layer import connect                                   # noqa: E402
from engine.rules import ChainRow, N3Entry, N3Params, N3Signal, evaluate_n3  # noqa: E402

NEAR_TERM_WINDOW_DAYS = 9   # excludes NSE's far-dated placeholder listings


def _params_hash(params: N3Params) -> str:
    return hashlib.sha256(json.dumps(params.__dict__, sort_keys=True).encode()).hexdigest()[:12]


def list_expiries(con: sqlite3.Connection, symbol: str, start: str, end: str) -> list[dt.date]:
    """Distinct NEAR-TERM expiries with real trading activity in [start,end].
    NSE bhavcopy also lists far-dated placeholder monthly/quarterly contracts
    years out with no real liquidity -- filtered out by the DTE window."""
    rows = con.execute("""
        SELECT DISTINCT trade_date, expiry FROM fo_bars
        WHERE symbol=? AND instr_type='index_option' AND trade_date BETWEEN ? AND ?
    """, (symbol, start, end)).fetchall()
    expiries = set()
    for trade_date, expiry in rows:
        td, ed = dt.date.fromisoformat(trade_date), dt.date.fromisoformat(expiry)
        if 0 < (ed - td).days <= NEAR_TERM_WINDOW_DAYS:
            expiries.add(ed)
    return sorted(expiries)


def candidate_entries(expiry: dt.date, params: N3Params) -> list[dt.date]:
    return sorted(expiry - dt.timedelta(days=d) for d in
                  range(params.dte_lo, params.dte_hi + 1)
                  if (expiry - dt.timedelta(days=d)).weekday() < 5)


def _get_vix(con, day: dt.date) -> float | None:
    r = con.execute("SELECT close FROM vix WHERE trade_date=?", (day.isoformat(),)).fetchone()
    return r[0] if r else None


def _get_spot(con, day: dt.date, index_name: str = "Nifty 50"):
    return con.execute(
        "SELECT open, high, low, close FROM spot_index WHERE index_name=? AND trade_date=?",
        (index_name, day.isoformat())).fetchone()


def _prior_trading_day(con, day: dt.date, index_name: str = "Nifty 50") -> dt.date | None:
    r = con.execute(
        "SELECT trade_date FROM spot_index WHERE index_name=? AND trade_date<? "
        "ORDER BY trade_date DESC LIMIT 1", (index_name, day.isoformat())).fetchone()
    return dt.date.fromisoformat(r[0]) if r else None


def _get_chain(con, symbol: str, day: dt.date, expiry: dt.date) -> tuple[ChainRow, ...]:
    rows = con.execute("""
        SELECT strike, opt_type, close, underlying, lot_size FROM fo_bars
        WHERE symbol=? AND trade_date=? AND expiry=? AND opt_type IS NOT NULL
          AND close IS NOT NULL AND underlying IS NOT NULL AND lot_size IS NOT NULL
    """, (symbol, day.isoformat(), expiry.isoformat())).fetchall()
    return tuple(ChainRow(r[0], r[1], r[2], r[3], r[4]) for r in rows)


def build_n3_entry(con, symbol: str, entry_date: dt.date, expiry: dt.date) -> N3Entry | None:
    """None whenever any required series is missing that day -- a silent
    skip is correct here (not every day has clean data), a wrong value would
    not be."""
    vix = _get_vix(con, entry_date)
    if vix is None:
        return None
    spot_row = _get_spot(con, entry_date)
    if spot_row is None:
        return None
    spot_close = spot_row[3]
    prior_day = _prior_trading_day(con, entry_date)
    if prior_day is None:
        return None
    prior_row = _get_spot(con, prior_day)
    if prior_row is None:
        return None
    _, prior_high, prior_low, _ = prior_row
    chain = _get_chain(con, symbol, entry_date, expiry)
    if not chain:
        return None
    return N3Entry(as_of_date=entry_date, expiry=expiry, vix=vix, spot=spot_close,
                   prior_day_low=prior_low, prior_day_high=prior_high, chain=chain)


def settle_at_expiry(con, symbol: str, sig: N3Signal) -> dict | None:
    """v1: hold-to-expiry only. NSE cash-settles index options against the
    closing level on expiry day -- see module docstring for what this
    doesn't yet check (early close-if-tested)."""
    spot_row = _get_spot(con, sig.expiry)
    if spot_row is None:
        return None
    settle_spot = spot_row[3]

    if sig.side == "bull_put":
        short_intrinsic = max(0.0, sig.short_strike - settle_spot)
        long_intrinsic = max(0.0, sig.long_strike - settle_spot)
    else:
        short_intrinsic = max(0.0, settle_spot - sig.short_strike)
        long_intrinsic = max(0.0, settle_spot - sig.long_strike)

    settle_pnl_ps = sig.net_credit - short_intrinsic + long_intrinsic
    gross_pnl = settle_pnl_ps * sig.lot_size
    exit_cost = expiry_cost(long_intrinsic * sig.lot_size)   # STT on exercised LONG leg only
    total_cost = sig.entry_cost + exit_cost
    net_pnl = gross_pnl - total_cost
    r_multiple = net_pnl / sig.max_loss if sig.max_loss else float("nan")
    return dict(settle_spot=settle_spot, gross_pnl=gross_pnl, exit_cost=exit_cost,
               total_cost=total_cost, net_pnl=net_pnl, r_multiple=r_multiple)


def run_n3_backtest(con, symbol: str, start: dt.date, end: dt.date,
                    params: N3Params = N3Params(), record: bool = True) -> dict:
    now = dt.datetime.now().isoformat(timespec="seconds")
    phash = _params_hash(params)
    run_id = None
    if record:
        run_id = con.execute(
            "INSERT INTO runs (started_at, kind, params_hash, status) VALUES (?,?,?,?)",
            (now, "backtest", phash, "running")).lastrowid
        con.commit()

    expiries = list_expiries(con, symbol, start.isoformat(), end.isoformat())
    trades, reject_reasons, no_data = [], {}, 0

    for expiry in expiries:
        for entry_date in candidate_entries(expiry, params):
            if not (start <= entry_date <= end):
                continue
            entry = build_n3_entry(con, symbol, entry_date, expiry)
            if entry is None:
                no_data += 1
                continue
            sig = evaluate_n3(entry, params)
            if not sig.entered:
                reject_reasons[sig.reason] = reject_reasons.get(sig.reason, 0) + 1
                continue
            outcome = settle_at_expiry(con, symbol, sig)
            if outcome is None:
                no_data += 1
                continue

            trade = dict(entry_date=entry_date.isoformat(), expiry=expiry.isoformat(),
                        side=sig.side, short_strike=sig.short_strike, long_strike=sig.long_strike,
                        net_credit=sig.net_credit, max_loss=sig.max_loss, **outcome)
            trades.append(trade)

            if record:
                pred_id = con.execute("""
                    INSERT OR IGNORE INTO predictions
                    (created_at, strategy, params_hash, symbol, as_of_date, direction,
                     entry, stop, target, expected_r, rationale)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """, (now, "N3_range_fade_credit", phash, symbol, entry_date.isoformat(),
                      sig.side, sig.net_credit, None, None, sig.expected_r_after_cost,
                      f"{sig.side} short={sig.short_strike} long={sig.long_strike} "
                      f"width={sig.width} expiry={sig.expiry}")).lastrowid
                if pred_id:
                    con.execute("""
                        INSERT INTO outcomes (prediction_id, resolved_at, exit_price, r_multiple,
                            gross_pnl, costs, net_pnl, ambiguous_bar, ambiguity_rule, note)
                        VALUES (?,?,?,?,?,?,?,?,?,?)
                    """, (pred_id, now, outcome["settle_spot"], outcome["r_multiple"],
                          outcome["gross_pnl"], outcome["total_cost"], outcome["net_pnl"],
                          0, None, "v1: hold-to-expiry only, no intraday close-if-tested check"))
                con.commit()

    if record:
        con.execute("UPDATE runs SET finished_at=?, status='ok', note=? WHERE id=?",
                    (dt.datetime.now().isoformat(timespec="seconds"),
                     f"{len(trades)} entered / {sum(reject_reasons.values())} rejected / "
                     f"{no_data} no-data of {len(expiries)} expiries", run_id))
        con.commit()

    return dict(n_expiries=len(expiries), trades=trades, reject_reasons=reject_reasons,
               no_data=no_data, params_hash=phash)


def summarize(result: dict) -> str:
    trades = result["trades"]
    n = len(trades)
    L = [f"N3 backtest -- {result['n_expiries']} expiries scanned, {n} entered, "
         f"{sum(result['reject_reasons'].values())} filter-rejected, "
         f"{result['no_data']} skipped for missing data"]
    L.append("\nHonesty section (Forward-Testing-Spec convention -- lead with it):")
    L.append(f"  Ambiguous bars: 0 of {n} (0.0%) -- v1 does not check intraday, see module docstring")
    if n == 0:
        L.append("  No trades entered -- nothing else to report.")
        L.append("\nRejection reasons:")
        for reason, cnt in sorted(result["reject_reasons"].items(), key=lambda x: -x[1]):
            L.append(f"  {cnt:>4}  {reason}")
        return "\n".join(L)

    wins = [t for t in trades if t["net_pnl"] > 0]
    gross = sum(t["gross_pnl"] for t in trades)
    costs = sum(t["total_cost"] for t in trades)
    net = sum(t["net_pnl"] for t in trades)
    avg_r = sum(t["r_multiple"] for t in trades) / n
    L.append(f"  Rejected entries: {sum(result['reject_reasons'].values())} -- "
             f"{dict(sorted(result['reject_reasons'].items(), key=lambda x: -x[1])[:3])}")
    L.append(f"  Total costs as % of gross P&L: {costs/gross*100:.1f}%" if gross else
             "  Total costs as % of gross P&L: n/a (gross=0)")
    L.append(f"\nWin rate: {len(wins)}/{n} ({len(wins)/n*100:.1f}%)")
    L.append(f"Average R: {avg_r:+.3f}")
    L.append(f"Gross P&L: Rs {gross:,.0f}   Costs: Rs {costs:,.0f}   Net P&L: Rs {net:,.0f}")
    return "\n".join(L)


if __name__ == "__main__":
    con = connect()
    if len(sys.argv) >= 3:
        s = dt.date.fromisoformat(sys.argv[1]); e = dt.date.fromisoformat(sys.argv[2])
    else:
        e = dt.date.today(); s = e - dt.timedelta(days=365)
    print(f"Running N3 backtest {s} .. {e}\n")
    result = run_n3_backtest(con, "NIFTY", s, e)
    print(summarize(result))
