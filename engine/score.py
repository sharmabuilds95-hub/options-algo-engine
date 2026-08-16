"""
score.py — the one place a strategy's performance is turned into a number.

WHY THIS EXISTS
    Measured expectancy on this account is -0.27R (n=19). The Risk Constitution
    states the job plainly: "the gap from -0.27R to +0.20R is the entire job."
    You cannot move a number you do not measure consistently, so every source of
    evidence -- historical trades, backtests, paper trades -- is scored HERE, by
    the same code, so results are comparable.

THE GOVERNING EQUATION (verified against the Risk Constitution's own table)

    net_monthly = N * (E * R_unit - C)

    N      = trades per month      (margin-capped: ~8 weeklies, ~2 monthlies)
    E      = gross expectancy in R
    R_unit = Rs 2,000              (Risk Constitution v1.4 core cap)
    C      = Rs 410                (measured cost per trade)

    BREAK-EVEN EXPECTANCY = C / R_unit = 410/2000 = +0.205R.
    Below that a strategy loses money mechanically, before the market acts.

    CAVEAT: the equation assumes every trade risks exactly R_unit. When actual
    risk varies per trade, use total_risk_deployed instead of N * R_unit.
    required_expectancy() carries the same assumption -- it is a planning tool,
    not a scorer.

THE MOST IMPORTANT DESIGN RULE HERE
    Report UNCERTAINTY, never a bare point estimate. At n=19 the confidence
    interval on expectancy spans "profitable" and "ruinous". A scorer that
    prints "-0.27R" and stops invites exactly the overfitting the validation
    programme exists to avoid.

    v2 NOTE ON NORMALITY: option-selling returns are strongly negatively skewed
    (many small wins, rare large losses). The t-interval assumes approximate
    normality of the MEAN, which is weak at small n for skewed data. So we
    compute BOTH a t-interval and a bootstrap percentile interval, and use the
    bootstrap for the significance call. Where they disagree, the data is
    skewed enough that the t-interval should not be trusted.

DELIBERATELY PURE
    No I/O, no database, no network. Feed it Trade objects.
"""

from __future__ import annotations

import datetime as dt
import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Iterable, Sequence

# --------------------------------------------------------------------------
# constants sourced from the Risk Constitution v1.4 / config.yaml
# --------------------------------------------------------------------------

RISK_UNIT_CORE = 2000.0
RISK_UNIT_INTRADAY = 1250.0
MEASURED_COST_PER_TRADE = 410.0

# Validation Gate (Risk Constitution): required before live at 50% size.
GATE_MIN_TRADES = 10
GATE_MIN_EXPECTANCY_R = 0.0
GATE_MAX_DRAWDOWN = 9000.0

# Default "minimum effect of interest" for sample-size planning: the break-even
# expectancy. Detecting anything smaller than break-even is pointless -- a
# strategy that is real but below +0.205R still loses money.
DEFAULT_MIN_EFFECT_R = MEASURED_COST_PER_TRADE / RISK_UNIT_CORE   # 0.205

# Minimum n before `significant` may be True.
#
# WHY: found while running the Tier 1 sweep. With n=3 all-negative trades, every
# bootstrap resample is negative, so the CI excludes zero and the scorer
# announced "significant: YES" on THREE observations. That is technically what
# the interval says and practically worthless -- three trades cannot establish
# an edge. Aligned with GATE_MIN_TRADES so the two notions of "enough evidence"
# do not drift apart.
MIN_N_FOR_SIGNIFICANCE = 10

_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045,
}


def t_crit_95(df: int) -> float:
    """Two-sided 95% t critical value. Avoids a scipy dependency."""
    if df < 1:
        return float("nan")
    return _T95.get(df, 1.96)


# --------------------------------------------------------------------------
# input
# --------------------------------------------------------------------------

@dataclass
class Trade:
    """One closed trade.

    `risk` is the DENOMINATOR that makes P&L an R-multiple, and it must be the
    risk PLANNED AT ENTRY (for a defined-risk spread: its max loss). Using the
    realised loss instead makes every loser exactly -1R and destroys the metric.
    `validate_trades()` checks for that specific mistake.
    """
    trade_id: str
    strategy: str
    entry_date: str                  # ISO yyyy-mm-dd
    exit_date: str                   # ISO yyyy-mm-dd
    risk: float                      # Rs planned risk at entry (defined max loss)
    gross_pnl: float                 # Rs, before costs
    costs: float = MEASURED_COST_PER_TRADE
    n_legs: int = 1
    account: str = "core"            # core | intraday
    vix_at_entry: float | None = None
    costs_are_modelled: bool = False  # True when costs came from costs.py, not the default
    note: str = ""

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.costs

    @property
    def r(self) -> float:
        """Net R-multiple -- the unit everything else is expressed in."""
        return self.net_pnl / self.risk if self.risk else float("nan")

    @property
    def r_gross(self) -> float:
        return self.gross_pnl / self.risk if self.risk else float("nan")

    @property
    def cap(self) -> float:
        return RISK_UNIT_INTRADAY if self.account == "intraday" else RISK_UNIT_CORE

    @property
    def breached_sizing(self) -> bool:
        """PLANNED risk exceeded the cap -- a rule violation committed at ENTRY.

        This is the breach that matters: it means the position was sized so it
        *could* lose more than the constitution allows, regardless of outcome.
        """
        return self.risk > self.cap

    @property
    def overran_cap(self) -> bool:
        """REALISED loss exceeded the cap despite compliant sizing.

        Distinct from breached_sizing: this is gap/slippage risk escaping a
        correctly-sized position, not a sizing error. Conflating the two (as v1
        did) hides which of the two problems you actually have.
        """
        return (-self.net_pnl) > self.cap


def validate_trades(trades: Sequence[Trade]) -> list[str]:
    """Structural problems that would silently corrupt the score. Cheap insurance."""
    problems: list[str] = []
    seen: set[str] = set()
    for t in trades:
        if t.trade_id in seen:
            problems.append(f"duplicate trade_id {t.trade_id!r} -- double counting")
        seen.add(t.trade_id)
        if t.risk <= 0:
            problems.append(f"{t.trade_id}: risk <= 0, R-multiple undefined")
        if t.exit_date < t.entry_date:
            problems.append(f"{t.trade_id}: exit_date precedes entry_date")
        if t.n_legs > 1 and not t.costs_are_modelled and t.costs == MEASURED_COST_PER_TRADE:
            problems.append(
                f"{t.trade_id}: {t.n_legs} legs but costs are the flat Rs "
                f"{MEASURED_COST_PER_TRADE:.0f} default. Brokerage is PER ORDER, so this "
                f"understates a multi-leg structure -- use costs.round_trip_cost()")
    losses = [t for t in trades if t.net_pnl < 0]
    if len(losses) >= 4 and all(abs(t.r + 1.0) < 1e-9 for t in losses):
        problems.append(
            "every loss is exactly -1.00R, which usually means `risk` was set from the "
            "realised loss rather than planned risk. Expectancy would be meaningless.")
    return problems


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

@dataclass
class GateReason:
    text: str
    blocking: bool


@dataclass
class ScoreCard:
    n: int = 0
    strategy: str = "ALL"

    expectancy_r: float = float("nan")
    expectancy_r_gross: float = float("nan")
    stdev_r: float = float("nan")

    # two intervals: disagreement is itself a signal that the data is skewed
    ci_t_low: float = float("nan")
    ci_t_high: float = float("nan")
    ci_boot_low: float = float("nan")
    ci_boot_high: float = float("nan")
    skewness: float = float("nan")
    skew_warning: bool = False

    significant: bool = False
    significance_suppressed_by_n: bool = False
    trades_needed: int | None = None
    min_effect_used: float = DEFAULT_MIN_EFFECT_R

    breakeven_r: float = float("nan")
    clears_breakeven: bool = False

    win_rate: float = float("nan")
    avg_win_r: float = float("nan")
    avg_loss_r: float = float("nan")
    profit_factor: float = float("nan")

    gross_pnl: float = 0.0
    total_costs: float = 0.0
    net_pnl: float = 0.0
    cost_drag_pct: float = float("nan")

    max_drawdown: float = 0.0
    sharpe_per_trade: float = float("nan")
    sizing_breaches: int = 0
    cap_overruns: int = 0

    trades_per_month: float = float("nan")
    active_months: int = 0
    span_days: int = 0

    gate_pass: bool = False
    gate_reasons: list[GateReason] = field(default_factory=list)
    data_warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        def r(x): return "  n/a" if x != x else f"{x:+.3f}"
        lines = [
            f"=== ScoreCard: {self.strategy}  (n={self.n}) ===",
            f"  expectancy      {r(self.expectancy_r)}R net",
            f"     95% CI (t)        {r(self.ci_t_low)} .. {r(self.ci_t_high)}",
            f"     95% CI (bootstrap){r(self.ci_boot_low)} .. {r(self.ci_boot_high)}"
            + (f"   <- skew {self.skewness:+.2f}, TRUST THE BOOTSTRAP"
               if self.skew_warning else ""),
            f"  gross expectancy{r(self.expectancy_r_gross)}R"
            f"   breakeven +{self.breakeven_r:.3f}R"
            f"  -> {'CLEARS' if self.clears_breakeven else 'BELOW BREAK-EVEN'}",
            f"  significant?    {'yes' if self.significant else 'NO - indistinguishable from zero'}"
            + (f"   [CI excludes zero but n={self.n} < {MIN_N_FOR_SIGNIFICANCE}, "
               f"too few to claim it]" if self.significance_suppressed_by_n else "")
            + ("" if self.trades_needed is None
               else f"   (~{self.trades_needed} trades to detect {self.min_effect_used:+.3f}R)"),
            f"  win rate        {self.win_rate:.1%}"
            f"   avg win {r(self.avg_win_r)}R   avg loss {r(self.avg_loss_r)}R"
            f"   PF {self.profit_factor:.2f}",
            f"  money           gross Rs {self.gross_pnl:,.0f}"
            f"   costs Rs {self.total_costs:,.0f}"
            f"   net Rs {self.net_pnl:,.0f}"
            f"   (cost drag {self.cost_drag_pct:.0f}% of gross)",
            f"  risk            maxDD Rs {self.max_drawdown:,.0f}"
            f"   sharpe/trade {self.sharpe_per_trade:.2f}"
            f"   sizing breaches {self.sizing_breaches}"
            f"   cap overruns {self.cap_overruns}",
            f"  throughput      {self.trades_per_month:.1f} trades/month"
            f"   ({self.active_months} active months, {self.span_days} day span)",
            f"  VALIDATION GATE {'PASS' if self.gate_pass else 'FAIL'}",
        ]
        for g in self.gate_reasons:
            lines.append(f"      {'x' if g.blocking else '-'} {g.text}")
        for w in self.data_warnings:
            lines.append(f"      ! DATA: {w}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# statistics helpers
# --------------------------------------------------------------------------

def _days_between(a: str, b: str) -> int:
    return (dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days


def bootstrap_ci(values: Sequence[float], iters: int = 5000,
                 seed: int = 20260815) -> tuple[float, float]:
    """95% percentile bootstrap CI for the mean.

    Makes no normality assumption, which matters because option-selling R
    distributions are negatively skewed. Seeded so results are reproducible --
    an unseeded CI that shifts between runs is not auditable.
    """
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[min(int(0.975 * iters), iters - 1)]
    return (lo, hi)


def trades_needed_for_significance(stdev_r: float,
                                   min_effect_r: float = DEFAULT_MIN_EFFECT_R,
                                   power: float = 0.80) -> int | None:
    """Trades required to detect `min_effect_r` at 5% two-sided, given power.

        n = (z_{alpha/2} + z_beta)^2 * sigma^2 / delta^2

    v2 FIX -- IMPORTANT: v1 passed the OBSERVED expectancy as the effect size.
    That is the post-hoc power fallacy: if you observe a small effect by chance
    you conclude you need a huge sample, and if you observe a large one by
    chance you conclude the opposite. The effect size must be chosen on
    substantive grounds BEFORE looking at the data. The default here is
    break-even expectancy (+0.205R), because an effect smaller than break-even
    is not worth detecting -- it still loses money.
    """
    if stdev_r != stdev_r or stdev_r <= 0 or min_effect_r == 0:
        return None
    z_beta = {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449}.get(power, 0.8416)
    n = (1.96 + z_beta) ** 2 * (stdev_r ** 2) / (min_effect_r ** 2)
    return max(2, math.ceil(n))


def sample_skewness(values: Sequence[float]) -> float:
    """Moment-based sample skewness g1 = m3 / m2^1.5.

    Used to decide whether the t-interval can be trusted. Option-selling
    returns are characteristically negatively skewed (many small credits, rare
    large debits), and the t-interval's normality assumption degrades badly in
    that shape at small n. |g1| > 1 is the conventional "substantially skewed"
    threshold.

    Chosen over the v2-draft approach of comparing t and bootstrap interval
    endpoints: that comparison needed an arbitrary tolerance and was not
    directly interpretable. Skewness measures the thing we actually care about.
    """
    n = len(values)
    if n < 3:
        return float("nan")
    mean = sum(values) / n
    m2 = sum((x - mean) ** 2 for x in values) / n
    if m2 <= 0:
        return 0.0
    m3 = sum((x - mean) ** 3 for x in values) / n
    return m3 / (m2 ** 1.5)


SKEW_WARN_THRESHOLD = 1.0


def max_drawdown(pnls: Sequence[float]) -> float:
    """Peak-to-trough decline of cumulative net P&L, as a positive number.

    LIMITATION (documented, not fixed): this walks trades in CLOSE order and so
    measures REALISED drawdown. The Risk Constitution permits 2 concurrent
    positions, whose simultaneous open losses are not captured here. True
    mark-to-market drawdown requires daily position marking and will be larger.
    Treat this as a LOWER BOUND on drawdown.
    """
    peak = cum = worst = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        worst = max(worst, peak - cum)
    return worst


# --------------------------------------------------------------------------
# the scorer
# --------------------------------------------------------------------------

def score(trades: Iterable[Trade], strategy: str = "ALL",
          cost_per_trade: float = MEASURED_COST_PER_TRADE,
          risk_unit: float = RISK_UNIT_CORE,
          min_effect_r: float = DEFAULT_MIN_EFFECT_R,
          bootstrap_iters: int = 5000) -> ScoreCard:
    ts = list(trades)
    sc = ScoreCard(n=len(ts), strategy=strategy)
    sc.breakeven_r = cost_per_trade / risk_unit if risk_unit else float("nan")
    sc.min_effect_used = min_effect_r
    if not ts:
        sc.gate_reasons.append(GateReason("no trades", True))
        return sc

    sc.data_warnings = validate_trades(ts)

    rs = [t.r for t in ts]
    sc.expectancy_r = statistics.fmean(rs)
    sc.expectancy_r_gross = statistics.fmean([t.r_gross for t in ts])
    sc.clears_breakeven = sc.expectancy_r_gross > sc.breakeven_r

    if len(rs) >= 2:
        sc.stdev_r = statistics.stdev(rs)
        se = sc.stdev_r / math.sqrt(len(rs))
        h = t_crit_95(len(rs) - 1) * se
        sc.ci_t_low, sc.ci_t_high = sc.expectancy_r - h, sc.expectancy_r + h
        sc.ci_boot_low, sc.ci_boot_high = bootstrap_ci(rs, iters=bootstrap_iters)

        # Significance from the BOOTSTRAP -- no normality assumption -- but only
        # once there are enough observations to mean anything (see
        # MIN_N_FOR_SIGNIFICANCE). The interval is still reported at any n.
        ci_excludes_zero = (sc.ci_boot_low > 0) or (sc.ci_boot_high < 0)
        sc.significant = ci_excludes_zero and len(rs) >= MIN_N_FOR_SIGNIFICANCE
        sc.significance_suppressed_by_n = ci_excludes_zero and not sc.significant

        # Skewed returns break the t-interval's normality assumption.
        sc.skewness = sample_skewness(rs)
        sc.skew_warning = (sc.skewness == sc.skewness
                           and abs(sc.skewness) > SKEW_WARN_THRESHOLD)

        if not sc.significant:
            sc.trades_needed = trades_needed_for_significance(sc.stdev_r, min_effect_r)
        sc.sharpe_per_trade = sc.expectancy_r / sc.stdev_r if sc.stdev_r else float("nan")

    wins = [x for x in rs if x > 0]
    losses = [x for x in rs if x <= 0]
    sc.win_rate = len(wins) / len(rs)
    sc.avg_win_r = statistics.fmean(wins) if wins else float("nan")
    sc.avg_loss_r = statistics.fmean(losses) if losses else float("nan")
    gain = sum(t.net_pnl for t in ts if t.net_pnl > 0)
    pain = -sum(t.net_pnl for t in ts if t.net_pnl <= 0)
    sc.profit_factor = (gain / pain) if pain > 0 else float("inf")

    sc.gross_pnl = sum(t.gross_pnl for t in ts)
    sc.total_costs = sum(t.costs for t in ts)
    sc.net_pnl = sum(t.net_pnl for t in ts)
    sc.cost_drag_pct = (100 * sc.total_costs / abs(sc.gross_pnl)
                        if sc.gross_pnl else float("inf"))

    ordered = sorted(ts, key=lambda t: t.exit_date)
    sc.max_drawdown = max_drawdown([t.net_pnl for t in ordered])
    sc.sizing_breaches = sum(1 for t in ts if t.breached_sizing)
    sc.cap_overruns = sum(1 for t in ts if t.overran_cap)

    # Throughput: count DISTINCT CALENDAR MONTHS with activity, not raw span.
    # v1 used first-entry-to-last-exit, which massively overstates throughput
    # when trades cluster (10 trades in one week then nothing for a quarter).
    months = {t.entry_date[:7] for t in ts}
    sc.active_months = len(months)
    sc.span_days = _days_between(ordered[0].entry_date, ordered[-1].exit_date)
    sc.trades_per_month = len(ts) / sc.active_months if sc.active_months else float("nan")

    # ---- Validation Gate ------------------------------------------------
    g = sc.gate_reasons
    if sc.n < GATE_MIN_TRADES:
        g.append(GateReason(f"only {sc.n} trades, gate needs {GATE_MIN_TRADES}", True))
    if not (sc.expectancy_r > GATE_MIN_EXPECTANCY_R):
        g.append(GateReason(
            f"expectancy {sc.expectancy_r:+.3f}R is not > {GATE_MIN_EXPECTANCY_R:+.2f}R", True))
    if sc.max_drawdown > GATE_MAX_DRAWDOWN:
        g.append(GateReason(
            f"max drawdown Rs {sc.max_drawdown:,.0f} exceeds Rs {GATE_MAX_DRAWDOWN:,.0f} "
            f"(and this is a LOWER bound -- concurrency not modelled)", True))
    if sc.sizing_breaches:
        g.append(GateReason(
            f"{sc.sizing_breaches} trade(s) sized above the per-trade risk cap", True))
    if sc.cap_overruns:
        g.append(GateReason(
            f"{sc.cap_overruns} trade(s) lost more than the cap despite compliant sizing "
            f"(gap/slippage, not a sizing error)", False))
    if not sc.significant:
        g.append(GateReason(
            "expectancy is not statistically distinguishable from zero", False))
    if sc.data_warnings:
        g.append(GateReason(
            f"{len(sc.data_warnings)} data-quality warning(s) -- see DATA lines", True))

    sc.gate_pass = not any(x.blocking for x in g)
    return sc


def score_by_strategy(trades: Iterable[Trade], **kw) -> dict[str, ScoreCard]:
    buckets: dict[str, list[Trade]] = {}
    for t in trades:
        buckets.setdefault(t.strategy, []).append(t)
    return {k: score(v, strategy=k, **kw) for k, v in sorted(buckets.items())}


def score_by_regime(trades: Iterable[Trade],
                    bands: Sequence[tuple[str, float, float]] = (
                        ("VIX <13", 0.0, 13.0),
                        ("VIX 13-16", 13.0, 16.0),
                        ("VIX 16-25", 16.0, 25.0),
                        ("VIX >25", 25.0, 999.0),
                    ), **kw) -> dict[str, ScoreCard]:
    """Condition results on volatility regime.

    Necessary because only ~21% of days sit in VIX 16-25. "This works" and
    "this works in-regime" are different claims.

    LIMITATION: buckets on VIX AT ENTRY. A weekly trade entered at VIX 13 that
    spikes to 20 mid-life is still counted as a VIX-13 trade. For multi-day
    positions this is a simplification, not a truth.
    """
    trades = list(trades)
    out: dict[str, ScoreCard] = {}
    for label, lo, hi in bands:
        sel = [t for t in trades
               if t.vix_at_entry is not None and lo <= t.vix_at_entry < hi]
        if sel:
            out[label] = score(sel, strategy=label, **kw)
    unknown = [t for t in trades if t.vix_at_entry is None]
    if unknown:
        out["VIX unknown"] = score(unknown, strategy="VIX unknown", **kw)
    return out


def required_expectancy(target_rupees: float, trades_per_month: float,
                        risk_unit: float = RISK_UNIT_CORE,
                        cost_per_trade: float = MEASURED_COST_PER_TRADE) -> float:
    """Invert the governing equation: what E does this target demand?

        net_monthly = N * (E * R_unit - C)  ->  E = (net_monthly/N + C) / R_unit

    Planning tool. Assumes every trade risks exactly R_unit.
    """
    if trades_per_month <= 0:
        return float("nan")
    return (target_rupees / trades_per_month + cost_per_trade) / risk_unit
