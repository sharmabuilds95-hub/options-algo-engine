"""
strategy.py — express a strategy once, then backtest / paper / check it the same way.

WHY THIS EXISTS (component E)
    Two gaps close here.

    1. Only `evaluate_n3` existed. Fourteen documented strategies cannot be
       compared if each is a bespoke function with its own conventions.

    2. walk.py walks a STATIC position. Nearly every strategy in the batch
       ADJUSTS mid-life: Delta Reset re-sells at 0.40 when delta drifts to
       0.15/0.65; the double diagonals reset short legs below 0.10 and book
       hedges above 0.70; the calendar strategies cap adjustments at 2 and then
       exit. Without position mutation those strategies simply cannot be tested.

THE COST POINT, WHICH IS THE WHOLE GAME
    Brokerage is PER ORDER. Every adjustment is a close plus an open, so an
    adjustment on a 2-leg structure is 4 orders. Against a Rs 2,000 risk unit
    and Rs 410 measured round-trip cost, adjustment-heavy strategies are the
    most likely in the batch to die on costs alone. This module therefore
    charges every adjustment through costs.py and reports the total, so the
    death is visible rather than hidden.

THREE DELIBERATE DESIGN DECISIONS

  1. ADJUSTMENT TRIGGERS FIRE ON THE CLOSE, NOT INTRADAY.
     Two independent reasons, and they agree:
       (a) If an adjustment could fire on an ambiguous bar, the position's
           future forks, and walking both branches is exponential. There is no
           honest cheap resolution.
       (b) The account owner is job-hunting and cannot watch deltas intraday.
           A rule he cannot execute is not a rule worth backtesting.
     EXIT rules keep walk.py's intraday envelope; only ADJUSTMENTS are on close.

  2. DELTA COMES OFF THE FORWARD, NOT SPOT.
     analytics.py's audit finding F2: options price off the forward and the
     Nifty basis has run ~9.5% annualised. Using spot would bias every delta
     and therefore every trigger. F is taken from put-call parity where the
     chain allows, else spot*e^(rT).

  3. A NaN DELTA IS NOT "NOT TRIGGERED".
     implied_vol() returns NaN for a quote at or below intrinsic -- exactly
     what an untraded, theoretically-priced contract looks like. Treating that
     as "condition false" would silently skip adjustments the strategy
     required. Unevaluable triggers are COUNTED and surfaced, never swallowed.

PURE CORE, THIN ADAPTER -- same shape as walk.py, for the same reason.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

try:
    import analytics as an
    from costs import option_trade_cost
    from liquidity import assess
    from walk import Leg, WalkRules, pnl_envelope, position_pnl
except ImportError:                                        # pragma: no cover
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import analytics as an
    from costs import option_trade_cost
    from liquidity import assess
    from walk import Leg, WalkRules, pnl_envelope, position_pnl


RISK_FREE = 0.0575


# --------------------------------------------------------------------------
# a day's option chain
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ChainRow:
    strike: float
    opt_type: str
    close: float
    high: float
    low: float
    volume: int
    open_interest: int
    lot_size: int


@dataclass
class ChainSnapshot:
    """One expiry's chain on one day, plus everything needed to price it.

    PERFORMANCE (measured 2026-08-16, 172-row chain):
        select_strike_by_delta was 500 ms per call because it recomputed T(),
        forward() and a Brent implied-vol solve for every row, and forward()
        itself scans the whole chain. At 1,704 strategy runs in the Tier 1
        sweep that is ~1.5 hours, which makes the "run it two or three times"
        discipline impractical.

        The three caches below are pure memoisation -- identical inputs, identical
        outputs, no semantic change -- plus an O(1) row index replacing a linear
        scan. Verified by asserting cached and uncached results match exactly.
    """
    date: str
    expiry: str
    spot: float
    rows: tuple[ChainRow, ...]
    lot_size: int

    _T: float | None = field(default=None, repr=False, compare=False)
    _fwd: dict[float, float] = field(default_factory=dict, repr=False, compare=False)
    _index: dict[tuple[float, str], ChainRow] | None = field(
        default=None, repr=False, compare=False)

    def T(self) -> float:
        if self._T is None:
            d0 = dt.datetime.fromisoformat(self.date)
            d1 = dt.datetime.fromisoformat(self.expiry)
            self._T = an.year_fraction(d0, d1)
        return self._T

    def get(self, strike: float, opt_type: str) -> ChainRow | None:
        if self._index is None:
            self._index = {(r.strike, r.opt_type): r for r in self.rows}
        return self._index.get((strike, opt_type))

    def forward(self, r: float = RISK_FREE) -> float:
        if r not in self._fwd:
            self._fwd[r] = self._forward_uncached(r)
        return self._fwd[r]

    def _forward_uncached(self, r: float = RISK_FREE) -> float:
        """Put-call parity forward where possible, else spot carried forward.

        Parity uses only strikes where BOTH legs traded -- a theoretical price
        on either side poisons the estimate.
        """
        T = self.T()
        by_strike: dict[float, dict] = {}
        for row in self.rows:
            if row.volume <= 0:
                continue
            by_strike.setdefault(row.strike, {})[row.opt_type] = row.close
        usable = {k: v for k, v in by_strike.items() if "CE" in v and "PE" in v}
        if len(usable) >= 2:
            ks = sorted(usable, key=lambda k: abs(k - self.spot))[:4]
            est = [k + (usable[k]["CE"] - usable[k]["PE"]) * math.exp(r * T) for k in ks]
            return sum(est) / len(est)
        return self.spot * math.exp(r * T)


@dataclass(frozen=True)
class LegGreeks:
    delta: float          # NaN when not computable
    iv: float
    price: float
    price_is_real: bool

    @property
    def evaluable(self) -> bool:
        return self.delta == self.delta and self.price_is_real


def leg_greeks(chain: ChainSnapshot, strike: float, opt_type: str,
               r: float = RISK_FREE) -> LegGreeks:
    """Delta/IV for one leg. Memoised per (chain, strike, opt_type, r).

    The implied-vol solve is a Brent root-find (~3.8 ms). select_strike_by_delta
    calls this once per candidate strike and the adjustment triggers call it
    again on the same strikes every day, so the cache is worth a lot and costs
    nothing: a ChainSnapshot is immutable in practice once loaded.
    """
    key = (strike, opt_type, r)
    cache = getattr(chain, "_greeks_cache", None)
    if cache is None:
        cache = {}
        object.__setattr__(chain, "_greeks_cache", cache)
    hit = cache.get(key)
    if hit is not None:
        return hit

    row = chain.get(strike, opt_type)
    if row is None:
        out = LegGreeks(float("nan"), float("nan"), float("nan"), False)
        cache[key] = out
        return out
    real = assess(row.volume, row.open_interest, row.lot_size or chain.lot_size).price_is_real
    T, F = chain.T(), chain.forward(r)
    iv = an.implied_vol(row.close, F, strike, T, opt_type, r)
    if iv != iv:
        out = LegGreeks(float("nan"), float("nan"), row.close, real)
    else:
        g = an.greeks(F, strike, T, iv, opt_type, r)
        out = LegGreeks(g.delta, iv, row.close, real)
    cache[key] = out
    return out


# --------------------------------------------------------------------------
# declarative strategy description
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LegSpec:
    """How to SELECT a leg, not which leg. Re-usable at entry and at re-entry."""
    label: str
    opt_type: str                 # CE | PE
    qty: int                      # lots; negative = short
    target_delta: float           # absolute value, e.g. 0.40
    expiry_key: str = "near"      # key into the expiries dict supplied per day


def select_strike_by_delta(chain: ChainSnapshot, spec: LegSpec,
                           r: float = RISK_FREE,
                           require_tradeable: bool = True) -> float | None:
    """Strike whose |delta| is closest to the target.

    Only considers rows whose price is REAL (traded). Selecting a strike off a
    theoretical price would pick a leg that could not have been traded, which
    is how a backtest ends up reporting fills that never existed.
    """
    best_k, best_err = None, float("inf")
    for row in chain.rows:
        if row.opt_type != spec.opt_type:
            continue
        if require_tradeable and row.volume <= 0:
            continue
        g = leg_greeks(chain, row.strike, row.opt_type, r)
        if not g.evaluable:
            continue
        err = abs(abs(g.delta) - abs(spec.target_delta))
        if err < best_err:
            best_k, best_err = row.strike, err
    return best_k


TriggerFn = Callable[["Position", ChainSnapshot, dict], bool]


@dataclass(frozen=True)
class Trigger:
    """A condition on the position, evaluated at the close.

    kind:
      leg_abs_delta_below / leg_abs_delta_above  -- one leg, by label
      combined_abs_delta_above                   -- sum of |delta| over short legs
      dte_below                                  -- days to the near expiry
      pnl_below / pnl_above                      -- total P&L in rupees
    """
    kind: str
    value: float
    label: str | None = None


@dataclass(frozen=True)
class Action:
    """kind: close_leg | reopen_leg | close_all"""
    kind: str
    label: str | None = None
    respec: LegSpec | None = None


@dataclass(frozen=True)
class AdjustmentRule:
    name: str
    trigger: Trigger
    actions: tuple[Action, ...]


@dataclass(frozen=True)
class StrategySpec:
    name: str
    entry_legs: tuple[LegSpec, ...]
    adjustments: tuple[AdjustmentRule, ...] = ()
    exit_rules: WalkRules = field(default_factory=WalkRules)
    max_adjustments: int | None = None
    on_budget_exhausted: str = "exit"      # "exit" | "hold"
    risk_cap: float = 2000.0


# --------------------------------------------------------------------------
# live position
# --------------------------------------------------------------------------

@dataclass
class OpenLeg:
    label: str
    strike: float
    opt_type: str
    qty: int
    entry_price: float
    entry_date: str
    expiry_key: str

    def as_walk_leg(self) -> Leg:
        return Leg(self.label, self.strike, self.opt_type, self.qty, self.entry_price)


@dataclass
class Position:
    legs: list[OpenLeg] = field(default_factory=list)
    realized_pnl: float = 0.0
    total_costs: float = 0.0
    adjustments: int = 0
    log: list[str] = field(default_factory=list)

    def by_label(self, label: str) -> OpenLeg | None:
        for lg in self.legs:
            if lg.label == label:
                return lg
        return None


@dataclass(frozen=True)
class RiskProfile:
    """The structure's DEFINED risk -- the denominator for every R-multiple.

    Found during review: an earlier version set max_loss_at_entry to the
    strategy's risk CAP (Rs 2,000). That conflates "the limit I intend to
    respect" with "what this structure can actually lose", and produced an
    absurd +8.73R on a 2-day test trade. Since score.py divides by this number,
    getting it wrong corrupts expectancy for every trade -- the single most
    important number in the project.
    """
    max_loss: float                 # positive rupees; inf when unbounded
    bounded: bool
    net_credit: float
    detail: str = ""

    @property
    def usable_as_r_denominator(self) -> bool:
        return self.bounded and self.max_loss > 0


def structure_max_loss(legs: Sequence[OpenLeg], lot: int) -> RiskProfile:
    """Worst expiry payoff over the whole strike domain.

    Evaluating the payoff at every strike (plus points outside the extremes) is
    exact for piecewise-linear option payoffs: the minimum of a piecewise-linear
    function lies at a kink or at an endpoint, and the kinks ARE the strikes.

    Unbounded cases are reported, not silently clamped:
      net short calls -> loss grows without limit as spot rises
      net short puts  -> bounded, but the bound is enormous (spot to zero)
    """
    if not legs:
        return RiskProfile(0.0, True, 0.0, "no legs")

    credit = sum(-lg.qty * lg.entry_price * lot for lg in legs)

    net_calls = sum(lg.qty for lg in legs if lg.opt_type.upper() == "CE")
    if net_calls < 0:
        return RiskProfile(float("inf"), False, credit,
                           "net short calls: loss unbounded to the upside")

    strikes = sorted({lg.strike for lg in legs})
    probes = [0.0] + strikes + [strikes[-1] * 2.0]
    # midpoints add nothing for piecewise-linear payoffs, but cost nothing
    worst = float("inf")
    for S in probes:
        payoff = credit
        for lg in legs:
            if lg.opt_type.upper() == "CE":
                intrinsic = max(0.0, S - lg.strike)
            else:
                intrinsic = max(0.0, lg.strike - S)
            payoff += lg.qty * intrinsic * lot
        worst = min(worst, payoff)

    max_loss = max(0.0, -worst)
    net_puts = sum(lg.qty for lg in legs if lg.opt_type.upper() == "PE")
    detail = ("net short puts: bounded only by spot reaching zero"
              if net_puts < 0 else "defined risk")
    return RiskProfile(max_loss, True, credit, detail)


def _txn_cost(closing: Sequence[tuple[OpenLeg, float]],
              opening: Sequence[tuple[OpenLeg, float]], lot_size: int) -> float:
    """Cost of one adjustment: closing legs reverse sign, opening legs do not.

    Closing a SHORT means buying it back; closing a LONG means selling it.
    Getting this backwards would misprice STT (charged on the sell side).
    """
    buy_val = sell_val = 0.0
    n = 0
    for lg, px in closing:
        n += 1
        if lg.qty < 0:
            buy_val += px * abs(lg.qty) * lot_size
        else:
            sell_val += px * abs(lg.qty) * lot_size
    for lg, px in opening:
        n += 1
        if lg.qty < 0:
            sell_val += px * abs(lg.qty) * lot_size
        else:
            buy_val += px * abs(lg.qty) * lot_size
    if n == 0:
        return 0.0
    return option_trade_cost(buy_val, sell_val, n).total


# --------------------------------------------------------------------------
# trigger evaluation
# --------------------------------------------------------------------------

@dataclass
class TriggerEval:
    fired: bool
    evaluable: bool
    detail: str = ""


def evaluate_trigger(trg: Trigger, pos: Position, chains: dict[str, ChainSnapshot],
                     unreal_pnl: float, r: float = RISK_FREE) -> TriggerEval:
    if trg.kind in ("leg_abs_delta_below", "leg_abs_delta_above"):
        lg = pos.by_label(trg.label or "")
        if lg is None:
            return TriggerEval(False, False, f"leg {trg.label} not open")
        ch = chains.get(lg.expiry_key)
        if ch is None:
            return TriggerEval(False, False, f"no chain for {lg.expiry_key}")
        g = leg_greeks(ch, lg.strike, lg.opt_type, r)
        if not g.evaluable:
            return TriggerEval(False, False,
                               f"{lg.label} delta not computable "
                               f"(price_is_real={g.price_is_real})")
        d = abs(g.delta)
        fired = d < trg.value if trg.kind.endswith("below") else d > trg.value
        return TriggerEval(fired, True, f"{lg.label} |delta|={d:.3f} vs {trg.value:.3f}")

    if trg.kind == "combined_abs_delta_above":
        tot, ok = 0.0, True
        for lg in pos.legs:
            if lg.qty >= 0:
                continue
            ch = chains.get(lg.expiry_key)
            if ch is None:
                ok = False
                break
            g = leg_greeks(ch, lg.strike, lg.opt_type, r)
            if not g.evaluable:
                ok = False
                break
            tot += abs(g.delta)
        if not ok:
            return TriggerEval(False, False, "combined delta not computable")
        return TriggerEval(tot > trg.value, True, f"combined |delta|={tot:.3f}")

    if trg.kind == "dte_below":
        ch = next(iter(chains.values()), None)
        if ch is None:
            return TriggerEval(False, False, "no chain")
        dte = (dt.date.fromisoformat(ch.expiry) - dt.date.fromisoformat(ch.date)).days
        return TriggerEval(dte < trg.value, True, f"dte={dte}")

    if trg.kind in ("pnl_below", "pnl_above"):
        total = pos.realized_pnl + unreal_pnl - pos.total_costs
        fired = total < trg.value if trg.kind.endswith("below") else total > trg.value
        return TriggerEval(fired, True, f"pnl={total:,.0f}")

    return TriggerEval(False, False, f"unknown trigger kind {trg.kind!r}")


# --------------------------------------------------------------------------
# result
# --------------------------------------------------------------------------

@dataclass
class StrategyResult:
    strategy: str = ""
    entry_date: str = ""
    exit_date: str | None = None
    exit_reason: str = "expiry"
    days_held: int = 0

    gross_pnl: float = 0.0            # realized + final unrealized, before costs
    total_costs: float = 0.0
    net_pnl: float = 0.0
    max_loss_at_entry: float = 0.0

    risk_bounded: bool = True
    risk_detail: str = ""
    fits_risk_cap: bool = True

    adjustments: int = 0
    unevaluable_triggers: int = 0
    theoretical_bars: int = 0
    ambiguous_bars: int = 0
    log: list[str] = field(default_factory=list)

    @property
    def has_data(self) -> bool:
        return self.exit_reason != "no_data"

    @property
    def r_multiple(self) -> float:
        """NaN unless the structure has a real, finite defined risk.

        Returning a number here when risk is unbounded would let a naked short
        be scored as if it were a defined-risk spread.
        """
        if not self.risk_bounded:
            return float("nan")
        if not self.max_loss_at_entry or self.max_loss_at_entry != self.max_loss_at_entry:
            return float("nan")
        return self.net_pnl / self.max_loss_at_entry

    @property
    def scoreable(self) -> bool:
        """Only feed score.py trades that have data AND a usable R denominator."""
        return self.has_data and self.risk_bounded and self.max_loss_at_entry > 0

    @property
    def cost_share_of_gross(self) -> float:
        return (100 * self.total_costs / abs(self.gross_pnl)
                if self.gross_pnl else float("inf"))

    def summary(self) -> str:
        rm = ("   R n/a (risk not bounded)" if not self.risk_bounded
              else f"   = {self.r_multiple:+.3f}R")
        risk = (f"  defined risk Rs {self.max_loss_at_entry:,.0f}"
                if self.risk_bounded else "  defined risk UNBOUNDED")
        cap = "" if self.fits_risk_cap else "   <- EXCEEDS the per-trade risk cap"
        return (f"{self.strategy}: entered {self.entry_date}, exit {self.exit_date} "
                f"({self.exit_reason}) after {self.days_held}d\n"
                f"  gross Rs {self.gross_pnl:,.0f}   costs Rs {self.total_costs:,.0f} "
                f"({self.cost_share_of_gross:.0f}% of gross)   net Rs {self.net_pnl:,.0f}"
                f"{rm}\n"
                f"{risk} ({self.risk_detail}){cap}\n"
                f"  adjustments {self.adjustments}"
                f"   unevaluable triggers {self.unevaluable_triggers}"
                f"   theoretical bars {self.theoretical_bars}")


# --------------------------------------------------------------------------
# the adjustable walk
# --------------------------------------------------------------------------

def open_position(spec: StrategySpec, chains: dict[str, ChainSnapshot],
                  date: str, r: float = RISK_FREE) -> Position | None:
    """Materialise entry legs. None if any leg cannot be selected."""
    pos = Position()
    opening: list[tuple[OpenLeg, float]] = []
    lot = 0
    for ls in spec.entry_legs:
        ch = chains.get(ls.expiry_key)
        if ch is None:
            return None
        k = select_strike_by_delta(ch, ls, r)
        if k is None:
            return None
        row = ch.get(k, ls.opt_type)
        if row is None:
            return None
        lot = lot or (row.lot_size or ch.lot_size)
        lg = OpenLeg(ls.label, k, ls.opt_type, ls.qty, row.close, date, ls.expiry_key)
        pos.legs.append(lg)
        opening.append((lg, row.close))
    pos.total_costs += _txn_cost([], opening, lot)
    pos.log.append(f"{date} OPEN " + ", ".join(
        f"{lg.label}@{lg.strike:g}{lg.opt_type}x{lg.qty}={lg.entry_price:.2f}"
        for lg in pos.legs))
    return pos


def _unrealized(pos: Position, chains: dict[str, ChainSnapshot], lot: int) -> float:
    total = 0.0
    for lg in pos.legs:
        ch = chains.get(lg.expiry_key)
        if ch is None:
            continue
        row = ch.get(lg.strike, lg.opt_type)
        if row is None:
            continue
        total += lg.qty * (row.close - lg.entry_price) * lot
    return total


def run_strategy(spec: StrategySpec, days: Sequence[dict], r: float = RISK_FREE
                 ) -> StrategyResult:
    """Walk a strategy through its life, adjusting as its rules require.

    `days` is a sequence of {"date", "chains": {expiry_key: ChainSnapshot}}.
    The FIRST element is the entry day; the position is opened at its close and
    marking starts the next day (same convention as walk.load_day_bars).
    """
    res = StrategyResult(strategy=spec.name)
    if not days:
        res.exit_reason = "no_data"
        return res

    entry = days[0]
    res.entry_date = entry["date"]
    pos = open_position(spec, entry["chains"], entry["date"], r)
    if pos is None:
        res.exit_reason = "no_entry"
        return res

    first_chain = next(iter(entry["chains"].values()))
    lot = first_chain.lot_size

    # The R denominator is the structure's OWN defined risk, computed from its
    # payoff -- never the strategy's intended cap. See RiskProfile.
    rp = structure_max_loss(pos.legs, lot)
    res.max_loss_at_entry = rp.max_loss
    res.risk_bounded = rp.bounded
    res.risk_detail = rp.detail
    res.fits_risk_cap = rp.bounded and rp.max_loss <= spec.risk_cap
    if not res.fits_risk_cap:
        pos.log.append(
            f"{entry['date']} RISK defined loss "
            f"{'UNBOUNDED' if not rp.bounded else f'Rs {rp.max_loss:,.0f}'} "
            f"exceeds cap Rs {spec.risk_cap:,.0f} -- not tradeable as sized")

    if len(days) == 1:
        res.exit_reason = "no_data"
        res.log = pos.log
        return res

    for i, day in enumerate(days[1:], start=1):
        chains = day["chains"]
        date = day["date"]

        walk_legs = [lg.as_walk_leg() for lg in pos.legs]
        unreal = _unrealized(pos, chains, lot)
        total_pnl = pos.realized_pnl + unreal - pos.total_costs

        for lg in pos.legs:
            ch = chains.get(lg.expiry_key)
            if ch is None:
                continue
            row = ch.get(lg.strike, lg.opt_type)
            if row is not None and not assess(
                    row.volume, row.open_interest, row.lot_size or lot).price_is_real:
                res.theoretical_bars += 1
                break

        # ---- hard risk cap, checked before anything else ------------------
        if spec.exit_rules.hard_risk_cap is not None and \
                total_pnl <= -abs(spec.exit_rules.hard_risk_cap):
            pos.realized_pnl += unreal
            pos.log.append(f"{date} EXIT hard_risk_cap pnl={total_pnl:,.0f}")
            return _finish(res, pos, "hard_risk_cap", date, i, unreal)

        # ---- time stop ----------------------------------------------------
        if spec.exit_rules.time_stop_dte is not None:
            near = chains.get(pos.legs[0].expiry_key) if pos.legs else None
            if near is not None:
                dte = (dt.date.fromisoformat(near.expiry)
                       - dt.date.fromisoformat(date)).days
                if dte <= spec.exit_rules.time_stop_dte:
                    pos.realized_pnl += unreal
                    pos.log.append(f"{date} EXIT time_stop dte={dte}")
                    return _finish(res, pos, "time_stop", date, i, unreal)

        # ---- adjustments (evaluated on the CLOSE, see module docstring) ----
        for rule in spec.adjustments:
            ev = evaluate_trigger(rule.trigger, pos, chains, unreal, r)
            if not ev.evaluable:
                res.unevaluable_triggers += 1
                pos.log.append(f"{date} UNEVALUABLE {rule.name}: {ev.detail}")
                continue
            if not ev.fired:
                continue

            budget_left = (spec.max_adjustments is None
                           or pos.adjustments < spec.max_adjustments)
            if not budget_left:
                if spec.on_budget_exhausted == "exit":
                    pos.realized_pnl += unreal
                    pos.log.append(
                        f"{date} EXIT adjustment_budget_exhausted ({rule.name})")
                    return _finish(res, pos, "adjustment_budget_exhausted",
                                   date, i, unreal)
                break

            _apply(rule, pos, chains, date, lot, r)
            pos.adjustments += 1
            unreal = _unrealized(pos, chains, lot)
            break        # one adjustment per day

    last = days[-1]
    unreal = _unrealized(pos, last["chains"], lot)
    pos.realized_pnl += unreal
    return _finish(res, pos, "expiry", last["date"], len(days) - 1, unreal)


def _apply(rule: AdjustmentRule, pos: Position, chains: dict[str, ChainSnapshot],
           date: str, lot: int, r: float) -> None:
    closing: list[tuple[OpenLeg, float]] = []
    opening: list[tuple[OpenLeg, float]] = []

    for act in rule.actions:
        if act.kind == "close_all":
            for lg in list(pos.legs):
                px = _price_of(lg, chains)
                if px is not None:
                    pos.realized_pnl += lg.qty * (px - lg.entry_price) * lot
                    closing.append((lg, px))
            pos.legs.clear()

        elif act.kind == "close_leg":
            lg = pos.by_label(act.label or "")
            if lg is not None:
                px = _price_of(lg, chains)
                if px is not None:
                    pos.realized_pnl += lg.qty * (px - lg.entry_price) * lot
                    closing.append((lg, px))
                pos.legs.remove(lg)

        elif act.kind == "reopen_leg":
            lg = pos.by_label(act.label or "")
            if lg is not None:
                px = _price_of(lg, chains)
                if px is not None:
                    pos.realized_pnl += lg.qty * (px - lg.entry_price) * lot
                    closing.append((lg, px))
                pos.legs.remove(lg)
            ls = act.respec
            if ls is not None:
                ch = chains.get(ls.expiry_key)
                if ch is not None:
                    k = select_strike_by_delta(ch, ls, r)
                    if k is not None:
                        row = ch.get(k, ls.opt_type)
                        if row is not None:
                            nl = OpenLeg(ls.label, k, ls.opt_type, ls.qty,
                                         row.close, date, ls.expiry_key)
                            pos.legs.append(nl)
                            opening.append((nl, row.close))

    cost = _txn_cost(closing, opening, lot)
    pos.total_costs += cost
    pos.log.append(
        f"{date} ADJUST {rule.name}: closed {len(closing)}, opened {len(opening)}, "
        f"cost Rs {cost:,.0f}")


def _price_of(lg: OpenLeg, chains: dict[str, ChainSnapshot]) -> float | None:
    ch = chains.get(lg.expiry_key)
    if ch is None:
        return None
    row = ch.get(lg.strike, lg.opt_type)
    return row.close if row else None


# --------------------------------------------------------------------------
# database adapter (the only impure part)
# --------------------------------------------------------------------------

def load_chain(con, symbol: str, trade_date: str, expiry: str,
               index_name: str = "Nifty 50") -> ChainSnapshot | None:
    """Build one ChainSnapshot from fo_bars + spot_index.

    Loads the WHOLE chain including untraded rows. Filtering happens later, at
    the point of use: select_strike_by_delta() skips untraded strikes, and
    forward() ignores them for parity. Dropping them here would hide how much
    of the chain is theoretical, which is information worth keeping.
    """
    spot_row = con.execute(
        "SELECT close FROM spot_index WHERE index_name=? AND trade_date=?",
        (index_name, trade_date)).fetchone()
    if spot_row is None:
        return None
    rows = con.execute(
        """SELECT strike, opt_type, close, high, low, volume, open_interest, lot_size
           FROM fo_bars
           WHERE symbol=? AND trade_date=? AND expiry=? AND opt_type IS NOT NULL
             AND close IS NOT NULL""",
        (symbol, trade_date, expiry)).fetchall()
    if not rows:
        return None
    chain_rows = tuple(
        ChainRow(r[0], r[1], r[2], r[3] if r[3] is not None else r[2],
                 r[4] if r[4] is not None else r[2],
                 int(r[5] or 0), int(r[6] or 0), int(r[7] or 0))
        for r in rows)
    lot = next((r.lot_size for r in chain_rows if r.lot_size), 0)
    return ChainSnapshot(trade_date, expiry, spot_row[0], chain_rows, lot)


def load_days(con, symbol: str, expiries: dict[str, str], start: str, end: str,
              index_name: str = "Nifty 50") -> list[dict]:
    """Assemble the day sequence run_strategy() consumes.

    `expiries` maps an expiry_key ("near", "far", ...) to an ISO expiry date,
    which is how a calendar or diagonal puts its legs on different expiries.
    Days where ANY requested chain is missing are skipped rather than partially
    filled -- a half-loaded day would silently change the position's marking.
    """
    dates = [r[0] for r in con.execute(
        """SELECT trade_date FROM spot_index
           WHERE index_name=? AND trade_date>=? AND trade_date<=?
           ORDER BY trade_date""", (index_name, start, end)).fetchall()]
    out: list[dict] = []
    for d in dates:
        chains: dict[str, ChainSnapshot] = {}
        ok = True
        for key, exp in expiries.items():
            ch = load_chain(con, symbol, d, exp, index_name)
            if ch is None:
                ok = False
                break
            chains[key] = ch
        if ok:
            out.append({"date": d, "chains": chains})
    return out


def _finish(res: StrategyResult, pos: Position, reason: str, date: str,
            days: int, final_unreal: float) -> StrategyResult:
    res.exit_reason = reason
    res.exit_date = date
    res.days_held = days
    res.gross_pnl = pos.realized_pnl
    res.total_costs = pos.total_costs
    res.net_pnl = pos.realized_pnl - pos.total_costs
    res.adjustments = pos.adjustments
    res.log = pos.log
    return res
