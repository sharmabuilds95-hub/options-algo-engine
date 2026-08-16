"""
engine/rules.py — pure strategy functions. [WEEK 2]

WHY THIS FILE EXISTS (Engineering Review, 2026-08-13):
    "These share no code. The backtest of 'N2 ORB' and the live evaluation of
    'N2 ORB' are two different implementations of an English description...
    Fix: one function, two callers... non-negotiable."

Every function here is pure: no DB connection, no clock, no network. All
market state arrives as a plain dataclass, so the exact same call is made by
a bar-by-bar backtest loop walking history (engine/backtest.py, Week 3) and
by run_daily.py evaluating today's live bar. If the two callers can diverge,
the backtest is worthless by the review's own words.

WHY N3 FIRST, NOT N2 (deviation from the review's own worked example):
    N1 (Trend Buy), N2 (ORB) and N4 (Expiry Scalp) all key off intraday
    triggers (VWAP reclaim, a 9:15-11:15 premium range, a same-day pin) that
    need intraday bars. data_layer.py only has NSE bhavcopy, which is
    end-of-day (ADR-010) -- and Kite cannot supply history for expired
    contracts at all (see data_layer.py's own docstring). There is currently
    NO data source for backtesting N1/N2/N4. N3 (Range Fade/Credit) is a
    5-7 DTE credit spread entered and held to expiry -- entry-day close and
    expiry settlement are exactly what daily bhavcopy bars give you. It is
    also the one sleeve the vault already ran a real (if n=2) pipeline test
    against -- see [[2026-07-28 N3 Range Fade Credit - Backtest Blocked (Data
    Limitation)]]. So N3 is the correct Week-2 starting point, not N2.

OPEN ITEMS this file does NOT resolve (flagged, not silently decided):
    1. VIX is a required input (N3Entry.vix) but nothing in the vault ingests
       HISTORICAL India VIX yet -- data_layer.py has no vix table. Live scans
       pull VIX from Kite/investing.com; a backtest walking 2025-2026 has no
       source for it yet. This blocks engine/backtest.py from actually
       running N3 over history until a historical VIX ingester exists.
    2. The strategy note files N3 under "Nifty Intraday Option Direction
       Playbook" (₹50K intraday bucket, ₹1,250 cap) but Risk Constitution
       v1.4's own sleeve table classifies "Credit/swing, 1-7 DTE" as CORE
       (₹2,00,000 bucket, ₹2,000 cap) -- see line 135 of that file. N3 holds
       5-7 days, which also contradicts the "intraday never held overnight"
       standing rule. This file uses the CORE ₹2,000 cap because Risk
       Constitution v1.4 wins on any conflict (CLAUDE.md standing rule 5),
       but the strategy note's own filing/categorisation of N3 should be
       fixed to match -- flagged for the user, not changed here.
    3. The "inside prior day's range" filter is kept as literally documented
       even though the 2026-07-28 backtest note found it never co-occurred
       with the VIX band across 12 weeks tested and recommended loosening it.
       That recommendation has not been formally adopted into the strategy
       note, so this file does not silently adopt it either.
    4. Bias selection (bull_put vs bear_call) is NOT specified by the
       strategy note beyond "sell OTM credit spread (bull put OR bear
       call) -- NOT naked". This file uses "fade toward the prior day's
       range midpoint" as a documented, reviewable implementation choice --
       not a rule extracted from vault text. Flag to the user if a different
       bias rule was intended.
"""
from __future__ import annotations

import datetime as dt
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analytics import greeks, implied_vol          # noqa: E402
from costs import option_trade_cost, passes_cost_gate  # noqa: E402

Side = Literal["bull_put", "bear_call"]
OptType = Literal["CE", "PE"]


@dataclass(frozen=True)
class N3Params:
    """Nifty Intraday Option Direction Playbook -- N3 Range Fade / Credit,
    strategy note updated 2026-07-29, cap from Risk Constitution v1.4
    (halved 2026-08-13). See this module's docstring, open item 2, on why
    the cap used here is CORE (₹2,000) not the intraday (₹1,250) the
    strategy note's own filename implies."""
    vix_lo: float = 16.0
    vix_hi: float = 22.0
    width_min: float = 100.0
    width_max: float = 115.0
    width_step: float = 5.0
    delta_lo: float = 0.20
    delta_hi: float = 0.30
    dte_lo: int = 5
    dte_hi: int = 7
    min_rr: float = 1.0                  # Net Credit / (Wing Width - Net Credit) >= this
    max_loss_cap: float = 2000.0         # Risk Constitution v1.4, CORE bucket
    cost_gate_multiple: float = 5.0      # costs.passes_cost_gate default
    r: float = 0.0575                    # risk-free rate, matches analytics.py default
    # 2026-08-14: [[2026-08-14 N3 Range Fade Credit Backtest...]] found the DEFAULT (True) setting
    # produces zero trades over a full backtested year -- the compound filter (VIX band AND inside
    # prior-range AND exact DTE window) never co-occurs. This flag lets that one variable be tested
    # in isolation (per the vault's own "change ONE variable per iteration" rule) without touching
    # the strategy note's documented default. Still True by default -- loosening is an experiment
    # result to review, not yet an adopted rule change.
    require_inside_range: bool = True


@dataclass(frozen=True)
class ChainRow:
    strike: float
    opt_type: OptType
    close: float
    underlying: float
    lot_size: int


@dataclass(frozen=True)
class N3Entry:
    """One candidate entry day's bar_data -- everything evaluate_n3 needs,
    nothing it has to go fetch itself. Supplied by run_daily.py (live) or
    engine/backtest.py (history) -- same shape, same function, either way."""
    as_of_date: dt.date
    expiry: dt.date
    vix: float
    spot: float
    prior_day_low: float
    prior_day_high: float
    chain: tuple[ChainRow, ...]           # this underlying's rows for `expiry`, on `as_of_date`
    event_day: bool = False               # RBI/Budget day -- strategy note excludes these


@dataclass(frozen=True)
class N3Signal:
    entered: bool
    reason: str
    as_of_date: dt.date | None = None
    expiry: dt.date | None = None
    side: Side | None = None
    short_strike: float | None = None
    long_strike: float | None = None
    net_credit: float | None = None       # per share
    width: float | None = None
    max_loss: float | None = None         # per lot, rupees
    lot_size: int | None = None
    rr: float | None = None
    entry_cost: float | None = None
    expected_r_after_cost: float | None = None


def _year_fraction(entry_date: dt.date, expiry_date: dt.date) -> float:
    d1 = dt.datetime.combine(entry_date, dt.time(15, 30))
    d2 = dt.datetime.combine(expiry_date, dt.time(15, 30))
    return max((d2 - d1).total_seconds() / (365 * 24 * 3600), 1e-8)


def _delta_of(row: ChainRow, T: float, r: float) -> float:
    iv = implied_vol(row.close, row.underlying, row.strike, T, row.opt_type, r)
    if iv != iv:                          # NaN: quote at/below intrinsic, unusable
        return float("nan")
    return greeks(row.underlying, row.strike, T, iv, row.opt_type, r).delta


def _pick_short_strike(chain: tuple[ChainRow, ...], opt_type: OptType,
                       T: float, params: N3Params) -> ChainRow | None:
    """Closest |delta| to the middle of [delta_lo, delta_hi] among in-band rows."""
    target = (params.delta_lo + params.delta_hi) / 2
    best, best_dist = None, float("inf")
    for row in chain:
        if row.opt_type != opt_type:
            continue
        d = _delta_of(row, T, params.r)
        if d != d:
            continue
        ad = abs(d)
        if not (params.delta_lo <= ad <= params.delta_hi):
            continue
        dist = abs(ad - target)
        if dist < best_dist:
            best, best_dist = row, dist
    return best


def evaluate_n3(entry: N3Entry, params: N3Params = N3Params()) -> N3Signal:
    """The one and only N3 rule. See module docstring for the two things it
    depends on that are not yet built (historical VIX) or resolved
    (sleeve/cap classification)."""
    d = entry.as_of_date

    if entry.event_day:
        return N3Signal(False, "event day (RBI/Budget) -- excluded", as_of_date=d)
    if not (params.vix_lo <= entry.vix <= params.vix_hi):
        return N3Signal(False, f"VIX {entry.vix:.2f} outside [{params.vix_lo},{params.vix_hi}]",
                        as_of_date=d)
    if params.require_inside_range and not (entry.prior_day_low <= entry.spot <= entry.prior_day_high):
        return N3Signal(False, "outside prior day's range", as_of_date=d)

    dte = (entry.expiry - d).days
    if not (params.dte_lo <= dte <= params.dte_hi):
        return N3Signal(False, f"DTE {dte} outside [{params.dte_lo},{params.dte_hi}]", as_of_date=d)

    range_mid = (entry.prior_day_low + entry.prior_day_high) / 2
    side: Side = "bear_call" if entry.spot >= range_mid else "bull_put"
    opt_type: OptType = "CE" if side == "bear_call" else "PE"

    T = _year_fraction(d, entry.expiry)
    short = _pick_short_strike(entry.chain, opt_type, T, params)
    if short is None:
        return N3Signal(False, f"no strike in delta band [{params.delta_lo},{params.delta_hi}]",
                        as_of_date=d)

    by_strike = {(r.strike, r.opt_type): r for r in entry.chain}
    width = params.width_min
    chosen = None
    while width <= params.width_max + 1e-9:
        long_strike = short.strike - width if side == "bull_put" else short.strike + width
        long_row = by_strike.get((long_strike, opt_type))
        if long_row is not None:
            net_credit = short.close - long_row.close
            max_loss_ps = width - net_credit
            if max_loss_ps > 0:
                rr = net_credit / max_loss_ps
                lot_size = short.lot_size
                max_loss = max_loss_ps * lot_size
                if rr >= params.min_rr and max_loss <= params.max_loss_cap:
                    chosen = (long_row, net_credit, width, rr, max_loss, lot_size)
                    break
        width += params.width_step

    if chosen is None:
        return N3Signal(False, "no width clears both R:R>=1:1 and the "
                        f"Rs {params.max_loss_cap:,.0f} core cap", as_of_date=d,
                        expiry=entry.expiry, side=side, short_strike=short.strike)

    long_row, net_credit, width, rr, max_loss, lot_size = chosen
    sell_val = short.close * lot_size
    buy_val = long_row.close * lot_size
    cost = option_trade_cost(buy_val, sell_val, n_orders=2)
    gate_ok, gate_why = passes_cost_gate(net_credit * lot_size, cost.total,
                                         params.cost_gate_multiple)
    # R-multiple denominator is THIS position's own max loss, not the account-wide
    # cap -- the cap only bounds which trades are allowed in, per CLAUDE.md's
    # Net Credit / (Wing Width - Net Credit) formula (matches the vault's own
    # 2026-07-28 N3 note, where "R" = net_credit/max_loss, e.g. 3445/6305 = 0.546R).
    expected_r = (net_credit * lot_size - cost.total) / max_loss

    return N3Signal(
        entered=True,
        reason="ok" if gate_ok else f"filters pass but fails cost gate: {gate_why}",
        as_of_date=d, expiry=entry.expiry, side=side,
        short_strike=short.strike, long_strike=long_row.strike,
        net_credit=net_credit, width=width, max_loss=max_loss, lot_size=lot_size,
        rr=rr, entry_cost=cost.total, expected_r_after_cost=expected_r,
    )
