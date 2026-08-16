"""
walk.py — walk an option position day by day instead of holding to expiry.

WHAT THIS REPLACES
    backtest.py v1 settles every position at expiry. Its own docstring admits
    `ambiguous_bar` is "always 0 by construction". That makes it unable to
    evaluate ANY rule that fires during a trade's life -- profit targets, stop
    losses, short-strike breaches, delta resets -- which is most of the rules
    in the strategy batch under validation.

THE HONEST PROBLEM, AND HOW THIS HANDLES IT
    Daily bars give the RANGE of a day, not the PATH through it. If on some day
    both the profit target and the stop lie inside the day's range, the data
    cannot say which was hit first. Backtests routinely resolve this silently
    in the profitable direction, which manufactures edge out of nothing.

    This module refuses to guess. It computes BOTH outcomes:

        optimistic  -- every ambiguous bar resolved in the trade's favour
        pessimistic -- every ambiguous bar resolved against it

    and reports the count of ambiguous bars. A strategy whose optimistic and
    pessimistic results straddle zero has not been shown to work; it has been
    shown that daily data cannot answer the question. That is a real finding,
    not a failure of the tool.

BOUNDING A MULTI-LEG POSITION'S INTRADAY P&L -- AND WHY THE OBVIOUS WAY IS WRONG
    A first attempt took each leg's own worst extreme independently:
        worst case = short legs at their HIGH, long legs at their LOW
    That is WRONG, and an integration test on a real 2026-03-04 bear call
    spread proved it: the two legs were adjacent strikes (25150/25200 CE) on
    the same underlying, so "short at its high while long at its low" requires
    spot to be simultaneously up and down. The bound was so loose it reported a
    cap breach on a day the position was comfortably PROFITABLE.

    The legs are not independent -- they are all driven by one underlying:
        calls rise when spot rises;  puts fall when spot rises.
    So there are only TWO coherent intraday scenarios:
        spot-up   : CE legs at their HIGH, PE legs at their LOW
        spot-down : CE legs at their LOW,  PE legs at their HIGH
    best/worst are the max/min of those two. This is still an outer bound (the
    true extremes need not coincide exactly with spot's), but it is a far
    tighter and physically coherent one, and it does not manufacture
    impossible scenarios.

PURE CORE, THIN ADAPTER
    walk() takes already-fetched DayBar objects, so it is testable without a
    database. load_day_bars() is the only part that touches SQLite.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Iterable, Sequence

try:                                   # allow both `import walk` and package use
    from liquidity import assess
except ImportError:                    # pragma: no cover
    from .liquidity import assess      # type: ignore


# --------------------------------------------------------------------------
# position description
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Leg:
    """One leg. qty is NEGATIVE for short, POSITIVE for long, in lots."""
    label: str
    strike: float
    opt_type: str                 # "CE" | "PE"
    qty: int                      # lots; negative = short
    entry_price: float            # premium per unit at entry

    @property
    def is_short(self) -> bool:
        return self.qty < 0


@dataclass(frozen=True)
class LegBar:
    """One leg's prices on one day."""
    label: str
    close: float
    high: float
    low: float
    volume: int = 0
    open_interest: int = 0
    lot_size: int = 0


@dataclass(frozen=True)
class DayBar:
    """Everything needed to evaluate one day of a position's life."""
    date: str                     # ISO
    spot_close: float
    spot_high: float
    spot_low: float
    legs: tuple[LegBar, ...]
    lot_size: int
    dte: int


@dataclass
class WalkRules:
    """Exit rules. All optional -- None disables that rule.

    Defaults mirror positions.yaml's documented paper-book rules so the
    backtest and the live monitor evaluate the same conditions.
    """
    profit_target_pct_of_credit: float | None = 0.25
    stop_pct_of_max_loss: float | None = 0.50
    close_below_short_strike: bool = True
    time_stop_dte: int | None = 2
    hard_risk_cap: float | None = 2000.0     # Rs; overrides everything when breached


@dataclass
class DayMark:
    date: str
    dte: int
    pnl_close: float
    pnl_best: float
    pnl_worst: float
    spot_close: float
    ambiguous: bool = False
    theoretical_legs: int = 0
    triggered: str | None = None


@dataclass
class WalkResult:
    exit_date: str | None = None
    exit_reason: str = "expiry"
    days_held: int = 0

    pnl_optimistic: float = 0.0
    pnl_pessimistic: float = 0.0
    pnl_close_basis: float = 0.0

    ambiguous_bars: int = 0
    theoretical_bars: int = 0
    path: list[DayMark] = field(default_factory=list)

    @property
    def is_ambiguous(self) -> bool:
        return self.ambiguous_bars > 0

    @property
    def spread(self) -> float:
        """Width of the uncertainty band the daily data leaves behind."""
        return self.pnl_optimistic - self.pnl_pessimistic

    @property
    def has_data(self) -> bool:
        """False when the walk had no bars at all -- never score such a trade."""
        return self.exit_reason != "no_data"

    @property
    def verdict_is_safe(self) -> bool:
        """True only if optimistic and pessimistic agree on the SIGN.

        If they disagree, daily data genuinely cannot tell you whether this
        trade made or lost money, and no amount of presentation fixes that.
        """
        return (self.pnl_optimistic >= 0) == (self.pnl_pessimistic >= 0)

    def summary(self) -> str:
        amb = (f"  [{self.ambiguous_bars} AMBIGUOUS bar(s): daily data cannot "
               f"resolve order of touches]" if self.ambiguous_bars else "")
        warn = "" if self.verdict_is_safe else "  <- SIGN UNRESOLVED, do not score this trade"
        return (f"exit {self.exit_date} ({self.exit_reason}) after {self.days_held}d\n"
                f"  close-basis P&L Rs {self.pnl_close_basis:,.0f}\n"
                f"  optimistic      Rs {self.pnl_optimistic:,.0f}\n"
                f"  pessimistic     Rs {self.pnl_pessimistic:,.0f}"
                f"   (band Rs {self.spread:,.0f}){warn}{amb}")


# --------------------------------------------------------------------------
# P&L maths
# --------------------------------------------------------------------------

def position_pnl(legs: Sequence[Leg], prices: dict[str, float],
                 lot_size: int) -> float:
    """Mark-to-market P&L in rupees.

    Short leg gains when price falls; long leg gains when price rises. qty
    carries the sign, so one expression covers both.
    """
    total = 0.0
    for leg in legs:
        px = prices.get(leg.label)
        if px is None:
            continue
        total += leg.qty * (px - leg.entry_price) * lot_size
    return total


def pnl_envelope(legs: Sequence[Leg], bar: DayBar) -> tuple[float, float, float]:
    """(close, best, worst) P&L for the day.

    Built from the two COHERENT scenarios the underlying permits -- spot up and
    spot down -- rather than from each leg's independent extreme. See the module
    docstring for why the independent-extreme version is wrong (it can place
    adjacent strikes in physically impossible states and fabricate breaches).

    Still an outer bound, because a leg's own intraday high need not occur at
    the same instant as spot's. But it never invents an impossible scenario.
    """
    by_label = {lb.label: lb for lb in bar.legs}
    close_px = {lb.label: lb.close for lb in bar.legs}

    up_px: dict[str, float] = {}       # scenario: spot rallied intraday
    down_px: dict[str, float] = {}     # scenario: spot fell intraday
    for leg in legs:
        lb = by_label.get(leg.label)
        if lb is None:
            continue
        if leg.opt_type.upper() == "CE":
            up_px[leg.label], down_px[leg.label] = lb.high, lb.low
        else:                           # puts move inversely to spot
            up_px[leg.label], down_px[leg.label] = lb.low, lb.high

    pnl_up = position_pnl(legs, up_px, bar.lot_size)
    pnl_down = position_pnl(legs, down_px, bar.lot_size)
    return (position_pnl(legs, close_px, bar.lot_size),
            max(pnl_up, pnl_down), min(pnl_up, pnl_down))


def net_credit(legs: Sequence[Leg], lot_size: int) -> float:
    """Rupees received at entry (positive) or paid (negative)."""
    return sum(-leg.qty * leg.entry_price * lot_size for leg in legs)


def short_strikes(legs: Sequence[Leg]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {"CE": [], "PE": []}
    for leg in legs:
        if leg.is_short:
            out[leg.opt_type].append(leg.strike)
    return out


# --------------------------------------------------------------------------
# the walk
# --------------------------------------------------------------------------

def walk(legs: Sequence[Leg], bars: Iterable[DayBar], rules: WalkRules,
         max_loss: float) -> WalkResult:
    """Step through the position's life, first triggered rule wins.

    max_loss is the structure's defined risk at entry (the R denominator).
    """
    res = WalkResult()
    bars = list(bars)
    if not bars:
        # MUST NOT look like a flat trade. An empty bar list means we have no
        # data for this position's life (0-DTE entry, missing spot rows, a bad
        # date range) -- reporting "expiry, P&L 0" would silently inject a
        # break-even trade into the sample. Found by an integration test that
        # accidentally used expiry == entry_date.
        res.exit_reason = "no_data"
        return res

    credit = net_credit(legs, bars[0].lot_size)
    target = (rules.profit_target_pct_of_credit * credit
              if rules.profit_target_pct_of_credit is not None and credit > 0 else None)
    stop = (-rules.stop_pct_of_max_loss * max_loss
            if rules.stop_pct_of_max_loss is not None and max_loss > 0 else None)

    for i, bar in enumerate(bars):
        pnl_c, pnl_b, pnl_w = pnl_envelope(legs, bar)
        mark = DayMark(date=bar.date, dte=bar.dte, pnl_close=pnl_c,
                       pnl_best=pnl_b, pnl_worst=pnl_w, spot_close=bar.spot_close)

        mark.theoretical_legs = sum(
            1 for lb in bar.legs
            if not assess(lb.volume, lb.open_interest, lb.lot_size or bar.lot_size).price_is_real)
        if mark.theoretical_legs:
            res.theoretical_bars += 1

        hit_target = target is not None and pnl_b >= target
        hit_stop = stop is not None and pnl_w <= stop
        hit_cap = (rules.hard_risk_cap is not None
                   and pnl_w <= -abs(rules.hard_risk_cap))

        # Strike breach is evaluated on the CLOSE, per positions.yaml: an
        # intraday wick through a short strike that closes back inside is not
        # a breach, and treating it as one would fire constantly.
        breached = False
        if rules.close_below_short_strike:
            ss = short_strikes(legs)
            for k in ss["PE"]:
                if bar.spot_close < k:
                    breached = True
            for k in ss["CE"]:
                if bar.spot_close > k:
                    breached = True

        time_stop = (rules.time_stop_dte is not None and bar.dte <= rules.time_stop_dte)

        if hit_target and (hit_stop or hit_cap):
            # Both reachable within the same day's envelope -- unresolvable.
            mark.ambiguous = True
            res.ambiguous_bars += 1

        res.path.append(mark)

        trigger = None
        if hit_cap:
            trigger = "hard_risk_cap"
        elif hit_stop:
            trigger = "stop"
        elif hit_target:
            trigger = "profit_target"
        elif breached:
            trigger = "short_strike_breached"
        elif time_stop:
            trigger = "time_stop"

        if trigger:
            mark.triggered = trigger
            res.exit_date = bar.date
            res.exit_reason = trigger
            res.days_held = i + 1
            res.pnl_close_basis = pnl_c

            # Exit AT THE TRIGGER LEVEL, not at the day's close.
            # A resting profit-target order fills at the target and you are
            # out; the day's close is irrelevant to you. Using the close
            # instead biases results optimistically whenever price kept
            # running after the target was touched -- on the real 2026-03-04
            # spread that was Rs 611 booked against a Rs 306 target, a 2x
            # overstatement on a single trade.
            # Gaps can fill better (target) or worse (stop) than the level;
            # daily data cannot show that, so the level is the honest estimate.
            level = {
                "profit_target": target,
                "stop": stop,
                "hard_risk_cap": (-abs(rules.hard_risk_cap)
                                  if rules.hard_risk_cap is not None else None),
            }.get(trigger)
            exit_pnl = pnl_c if level is None else level

            if mark.ambiguous:
                # Unresolvable: report both resolutions rather than picking one.
                res.pnl_optimistic = target if target is not None else pnl_b
                worst_level = stop if stop is not None else None
                if hit_cap and rules.hard_risk_cap is not None:
                    worst_level = -abs(rules.hard_risk_cap)
                res.pnl_pessimistic = worst_level if worst_level is not None else pnl_w
            else:
                res.pnl_optimistic = exit_pnl
                res.pnl_pessimistic = exit_pnl
            return res

    last = bars[-1]
    pnl_c, pnl_b, pnl_w = pnl_envelope(legs, last)
    res.exit_date = last.date
    res.exit_reason = "expiry"
    res.days_held = len(bars)
    res.pnl_close_basis = pnl_c
    res.pnl_optimistic = pnl_c
    res.pnl_pessimistic = pnl_c
    return res


# --------------------------------------------------------------------------
# database adapter (the only impure part)
# --------------------------------------------------------------------------

def load_day_bars(con, symbol: str, legs: Sequence[Leg], entry_date: dt.date,
                  expiry: dt.date, leg_expiries: dict[str, dt.date] | None = None,
                  index_name: str = "Nifty 50") -> list[DayBar]:
    """Fetch one DayBar per trading day in (entry_date, expiry].

    leg_expiries lets a calendar/diagonal put its legs on different expiries;
    absent, every leg is assumed to sit on `expiry`.

    The range is EXCLUSIVE of entry_date (you enter at that day's close, so
    there is nothing to mark yet) and inclusive of expiry. A 0-DTE trade where
    entry_date == expiry therefore yields ZERO bars -- walk() reports that as
    exit_reason "no_data" rather than as a flat trade.

    A day is SKIPPED only if the spot row is missing. A missing LEG row is kept
    but carries volume=0, so it registers as theoretical rather than vanishing
    silently -- an absent leg is information, not noise.
    """
    leg_expiries = leg_expiries or {}
    rows = con.execute(
        """SELECT trade_date, close, high, low FROM spot_index
           WHERE index_name=? AND trade_date>? AND trade_date<=?
           ORDER BY trade_date""",
        (index_name, entry_date.isoformat(), expiry.isoformat())).fetchall()

    bars: list[DayBar] = []
    for r in rows:
        day = r[0]
        leg_bars: list[LegBar] = []
        lot_size = 0
        for leg in legs:
            exp = leg_expiries.get(leg.label, expiry)
            q = con.execute(
                """SELECT close, high, low, volume, open_interest, lot_size
                   FROM fo_bars WHERE symbol=? AND trade_date=? AND expiry=?
                     AND strike=? AND opt_type=?""",
                (symbol, day, exp.isoformat(), leg.strike, leg.opt_type)).fetchone()
            if q is None:
                leg_bars.append(LegBar(leg.label, leg.entry_price, leg.entry_price,
                                       leg.entry_price, 0, 0, 0))
            else:
                leg_bars.append(LegBar(leg.label, q[0], q[1], q[2],
                                       int(q[3] or 0), int(q[4] or 0), int(q[5] or 0)))
                lot_size = lot_size or int(q[5] or 0)
        dte = (expiry - dt.date.fromisoformat(day)).days
        bars.append(DayBar(date=day, spot_close=r[1], spot_high=r[2], spot_low=r[3],
                           legs=tuple(leg_bars), lot_size=lot_size, dte=dte))
    return bars
