"""
engine/test_score.py — verification for the scoring engine (v2).

TEST 1 IS THE LOAD-BEARING ONE:
    required_expectancy() is derived from the governing equation
        net_monthly = N * (E * R_unit - C)
    The Risk Constitution contains a table of required expectancies derived
    INDEPENDENTLY of this code. If the equation is right it must reproduce all
    six cells. It does. That cross-check licenses reasoning from the equation.

TESTS 11-15 cover the v2 corrections, each of which fixes a real defect in v1:
    11  post-hoc power fallacy in the sample-size estimate
    12  t-interval unreliability on skewed data -> bootstrap
    13  sizing breach vs. execution overrun were conflated
    14  throughput inflated by clustered trades
    15  structural data problems passed silently
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score import (  # noqa: E402
    DEFAULT_MIN_EFFECT_R, MEASURED_COST_PER_TRADE, RISK_UNIT_CORE, Trade,
    bootstrap_ci, max_drawdown, required_expectancy, sample_skewness, score,
    score_by_regime, score_by_strategy, trades_needed_for_significance,
    validate_trades,
)

PASS: list[str] = []
FAIL: list[str] = []


def check(label: str, got, want, tol: float | None = None):
    if tol is not None:
        ok = (got == got) and (want == want) and abs(got - want) <= tol
    else:
        ok = got == want
    (PASS if ok else FAIL).append(label)
    extra = "" if ok else f"   (got {got!r}, want {want!r})"
    print(f"  [{'ok  ' if ok else 'FAIL'}] {label}{extra}")


def mk(pnl_gross: float, risk: float = RISK_UNIT_CORE, day: int = 1,
       strategy: str = "S", costs: float = 0.0, vix: float | None = None,
       account: str = "core", tid: str | None = None, month: int = 1) -> Trade:
    d = f"2026-{month:02d}-{day:02d}"
    return Trade(trade_id=tid or f"T{month}-{day}", strategy=strategy,
                 entry_date=d, exit_date=d, risk=risk, gross_pnl=pnl_gross,
                 costs=costs, vix_at_entry=vix, account=account,
                 costs_are_modelled=True)


print("=" * 96)
print("TEST 1 -- the governing equation reproduces the Risk Constitution's table")
print("=" * 96)
print("  (Constitution figures are 2dp; we check the raw value rounds to them)")
for target, n, want in [
    (6000, 8, 0.58), (6000, 12, 0.46), (6000, 20, 0.36),
    (12000, 8, 0.96), (12000, 12, 0.71), (12000, 20, 0.51),
]:
    got = required_expectancy(target, n)
    check(f"Rs {target:,}/mo at {n} trades -> {want:+.2f}R  (raw {got:.4f})",
          got, want, 0.005 + 1e-9)

print("\n" + "=" * 96)
print("TEST 2 -- break-even expectancy is cost/risk_unit, and is enforced")
print("=" * 96)
sc = score([mk(500, costs=410.0, day=i + 1) for i in range(5)])
check("breakeven_r == 410/2000", sc.breakeven_r, 0.205, 1e-9)
check("gross expectancy +0.25R", sc.expectancy_r_gross, 0.25, 1e-9)
check("clears breakeven", sc.clears_breakeven, True)
check("net expectancy +0.045R", sc.expectancy_r, 0.045, 1e-9)

sc2 = score([mk(300, costs=410.0, day=i + 1) for i in range(5)])
check("gross +0.15R does NOT clear breakeven", sc2.clears_breakeven, False)
check("...and net expectancy is negative", sc2.expectancy_r < 0, True)

print("\n" + "=" * 96)
print("TEST 3 -- expectancy, win rate, profit factor")
print("=" * 96)
ts = [mk(2000, day=i + 1) for i in range(3)] + [mk(-2000, day=i + 4) for i in range(2)]
sc = score(ts)
check("n", sc.n, 5)
check("win rate 60%", sc.win_rate, 0.6, 1e-9)
check("avg win +1R", sc.avg_win_r, 1.0, 1e-9)
check("avg loss -1R", sc.avg_loss_r, -1.0, 1e-9)
check("expectancy +0.2R", sc.expectancy_r, 0.2, 1e-9)
check("profit factor 6000/4000", sc.profit_factor, 1.5, 1e-9)
check("net pnl Rs 2,000", sc.net_pnl, 2000.0, 1e-9)

print("\n" + "=" * 96)
print("TEST 4 -- uncertainty is always reported")
print("=" * 96)
check("t CI low below mean", sc.ci_t_low < sc.expectancy_r, True)
check("t CI high above mean", sc.ci_t_high > sc.expectancy_r, True)
check("bootstrap CI also computed", sc.ci_boot_low == sc.ci_boot_low, True)
check("noisy +0.2R sample is NOT significant", sc.significant, False)
check("reports trades needed", isinstance(sc.trades_needed, int), True)

tight = [mk(1000, day=i + 1) for i in range(10)]
sc_t = score(tight)
check("zero-variance positive sample IS significant", sc_t.significant, True)
check("  its bootstrap CI low is above zero", sc_t.ci_boot_low > 0, True)

# Regression: found while running the Tier 1 sweep. Three all-negative trades
# make every bootstrap resample negative, so the CI excludes zero and the
# scorer announced "significant" on n=3. Technically what the interval says;
# practically worthless.
tiny = [mk(-1000, day=i + 1) for i in range(3)]
sc_tiny = score(tiny)
check("n=3 all-negative: CI does exclude zero", sc_tiny.ci_boot_high < 0, True)
check("  ...but significance is SUPPRESSED at n=3", sc_tiny.significant, False)
check("  ...and the suppression is reported, not hidden",
      sc_tiny.significance_suppressed_by_n, True)
check("n=10 with the same sign is allowed to be significant",
      score([mk(-1000, day=i + 1) for i in range(10)]).significant, True)

print("\n" + "=" * 96)
print("TEST 5 -- max drawdown (realised, close-ordered)")
print("=" * 96)
check("simple run-down", max_drawdown([100, -50, -30, 20]), 80.0, 1e-9)
check("all winners -> zero DD", max_drawdown([10, 20, 30]), 0.0, 1e-9)
check("loss first counts from zero peak", max_drawdown([-40, 10]), 40.0, 1e-9)
check("empty", max_drawdown([]), 0.0, 1e-9)

print("\n" + "=" * 96)
print("TEST 6 -- validation gate: 10 trades, expectancy > 0, DD within cap")
print("=" * 96)
sc = score([mk(1000, day=i + 1) for i in range(9)])
check("9 trades fails the >=10 requirement", sc.gate_pass, False)
check("  reason names the trade count",
      any("9 trades" in g.text for g in sc.gate_reasons), True)
check("10 profitable trades passes",
      score([mk(1000, day=i + 1) for i in range(10)]).gate_pass, True)
check("10 losing trades fails on expectancy",
      score([mk(-1000, day=i + 1) for i in range(10)]).gate_pass, False)

print("\n" + "=" * 96)
print("TEST 7 -- cost drag")
print("=" * 96)
sc = score([mk(1000, costs=410.0, day=i + 1) for i in range(10)])
check("gross Rs 10,000", sc.gross_pnl, 10000.0, 1e-9)
check("costs Rs 4,100", sc.total_costs, 4100.0, 1e-9)
check("net Rs 5,900", sc.net_pnl, 5900.0, 1e-9)
check("cost drag 41% of gross", sc.cost_drag_pct, 41.0, 1e-9)
print("        -> at Rs 1,000 gross/trade costs eat 41% of gross. This is the"
      "\n           strongest argument for fewer legs and fewer adjustments.")

print("\n" + "=" * 96)
print("TEST 8 -- grouping by strategy and by VIX regime")
print("=" * 96)
mixed = [mk(2000, day=1, strategy="A", vix=12.0),
         mk(-1000, day=2, strategy="A", vix=14.0),
         mk(3000, day=3, strategy="B", vix=18.0),
         mk(1000, day=4, strategy="B", vix=27.0)]
by_s = score_by_strategy(mixed)
check("two strategy buckets", sorted(by_s), ["A", "B"])
check("A expectancy +0.25R", by_s["A"].expectancy_r, 0.25, 1e-9)
check("B expectancy +1.0R", by_s["B"].expectancy_r, 1.0, 1e-9)
by_r = score_by_regime(mixed)
check("all four VIX buckets present",
      sorted(by_r), ["VIX 13-16", "VIX 16-25", "VIX <13", "VIX >25"])
check("missing VIX is bucketed, not dropped",
      "VIX unknown" in score_by_regime([mk(100, day=1)]), True)

print("\n" + "=" * 96)
print("TEST 9 -- degenerate inputs do not crash")
print("=" * 96)
empty = score([])
check("empty -> n=0", empty.n, 0)
check("empty -> gate fails", empty.gate_pass, False)
one = score([mk(1000)])
check("n=1 -> expectancy computed", one.expectancy_r == one.expectancy_r, True)
check("n=1 -> stdev nan (undefined)", math.isnan(one.stdev_r), True)
check("n=1 -> not significant", one.significant, False)
check("zero risk -> r is nan, no ZeroDivisionError",
      math.isnan(Trade("x", "s", "2026-01-01", "2026-01-01", 0.0, 100.0).r), True)

print("\n" + "=" * 96)
print("TEST 10 -- bootstrap CI is reproducible and sane")
print("=" * 96)
vals = [1.0, -1.0, 0.5, -0.5, 2.0, -1.0, 0.3, -0.8, 1.2, -1.0]
a = bootstrap_ci(vals)
b = bootstrap_ci(vals)
check("seeded -> identical across runs (auditable)", a, b)
check("CI brackets the sample mean",
      a[0] <= sum(vals) / len(vals) <= a[1], True)
check("n<2 -> nan", math.isnan(bootstrap_ci([1.0])[0]), True)

print("\n" + "=" * 96)
print("TEST 11 [v2 FIX] -- sample size uses a PRE-SET effect, not the observed one")
print("=" * 96)
print("  v1 bug: passing the observed expectancy is the post-hoc power fallacy.")
check("default min effect is break-even +0.205R", DEFAULT_MIN_EFFECT_R, 0.205, 1e-9)
n1 = trades_needed_for_significance(stdev_r=1.0)
n2 = trades_needed_for_significance(stdev_r=2.0)
check("doubling sigma ~quadruples n", n2 / n1, 4.0, 0.05)
check("n for sigma=1 at break-even effect is ~187", n1, 187)
# The estimate must NOT depend on what expectancy happened to be observed.
lucky = score([mk(2000, day=i + 1) for i in range(3)] + [mk(-1800, day=i + 4) for i in range(3)])
unlucky = score([mk(1800, day=i + 1) for i in range(3)] + [mk(-2000, day=i + 4) for i in range(3)])
check("same dispersion -> same trades_needed regardless of observed mean",
      lucky.trades_needed, unlucky.trades_needed)
check("explicit min_effect_r is honoured",
      score([mk(2000, day=i + 1) for i in range(3)] + [mk(-1800, day=i + 4) for i in range(3)],
            min_effect_r=0.5).trades_needed
      < lucky.trades_needed, True)

print("\n" + "=" * 96)
print("TEST 12 [v2 FIX] -- skewed data flags the t-interval as untrustworthy")
print("=" * 96)
print("  Detection is by sample skewness (|g1| > 1), not by comparing interval")
print("  endpoints -- skewness measures the thing we actually care about and")
print("  needs no arbitrary tolerance.")
check("symmetric data has ~zero skew", sample_skewness([1, -1, 1, -1, 2, -2]), 0.0, 1e-9)
check("n<3 -> nan", math.isnan(sample_skewness([1.0, 2.0])), True)
check("zero-variance -> 0.0 not a crash", sample_skewness([5.0, 5.0, 5.0]), 0.0, 1e-9)
# Option-selling shape: many small wins, one catastrophic loss.
skewed = [mk(300, day=i + 1) for i in range(19)] + [mk(-20000, day=20)]
sc_sk = score(skewed)
check("bootstrap CI computed on skewed sample", sc_sk.ci_boot_low == sc_sk.ci_boot_low, True)
check("skewness is strongly negative", sc_sk.skewness < -1.0, True)
check("skew warning raised", sc_sk.skew_warning, True)
print(f"        skewness {sc_sk.skewness:+.2f}")
print(f"        t  CI: {sc_sk.ci_t_low:+.3f} .. {sc_sk.ci_t_high:+.3f}")
print(f"        boot CI: {sc_sk.ci_boot_low:+.3f} .. {sc_sk.ci_boot_high:+.3f}")
print("        -> on this shape the t-interval's normality assumption fails.")
sym = [mk(1000, day=i + 1) for i in range(10)] + [mk(-1000, day=i + 11) for i in range(10)]
check("symmetric sample raises NO skew warning", score(sym).skew_warning, False)

print("\n" + "=" * 96)
print("TEST 13 [v2 FIX] -- sizing breach and execution overrun are different things")
print("=" * 96)
over_sized = mk(500, risk=3000.0)          # planned risk 3000 > 2000 cap, but WON
check("planned risk above cap is a SIZING breach", over_sized.breached_sizing, True)
check("  ...even though the trade was profitable", over_sized.overran_cap, False)
gapped = mk(-2500, risk=1800.0)            # sized fine, lost more than the cap
check("compliant sizing that lost > cap is an OVERRUN", gapped.overran_cap, True)
check("  ...and is NOT a sizing breach", gapped.breached_sizing, False)
intra = mk(-1400, risk=1200.0, account="intraday")
check("intraday uses the Rs 1,250 cap", intra.overran_cap, True)
sc_b = score([over_sized, gapped])
check("scorecard counts them separately",
      (sc_b.sizing_breaches, sc_b.cap_overruns), (1, 1))
check("sizing breach BLOCKS the gate",
      any(g.blocking and "sized above" in g.text for g in sc_b.gate_reasons), True)
check("overrun is reported but does NOT block",
      any((not g.blocking) and "despite compliant sizing" in g.text
          for g in sc_b.gate_reasons), True)

print("\n" + "=" * 96)
print("TEST 14 [v2 FIX] -- throughput counts active months, not raw span")
print("=" * 96)
print("  v1 bug: 10 trades in one week over a 9-day span read as ~34 trades/month.")
burst = [mk(100, day=i + 1, month=1, tid=f"B{i}") for i in range(10)]
sc_b = score(burst)
check("10 trades in one month -> 10.0/month", sc_b.trades_per_month, 10.0, 1e-9)
check("active_months == 1", sc_b.active_months, 1)
spread = ([mk(100, day=1, month=1, tid="X1")] + [mk(100, day=1, month=2, tid="X2")]
          + [mk(100, day=1, month=3, tid="X3")] + [mk(100, day=1, month=4, tid="X4")])
check("4 trades across 4 months -> 1.0/month", score(spread).trades_per_month, 1.0, 1e-9)
check("  active_months == 4", score(spread).active_months, 4)

print("\n" + "=" * 96)
print("TEST 15 [v2 FIX] -- structural data problems are caught, not silently scored")
print("=" * 96)
dup = [mk(100, day=1, tid="SAME"), mk(200, day=2, tid="SAME")]
check("duplicate trade_id detected",
      any("duplicate" in p for p in validate_trades(dup)), True)
bad_dates = [Trade("d", "s", "2026-02-01", "2026-01-01", 2000.0, 100.0)]
check("exit before entry detected",
      any("precedes" in p for p in validate_trades(bad_dates)), True)
multileg = [Trade(f"m{i}", "s", "2026-01-01", "2026-01-02", 2000.0, 500.0,
                  costs=MEASURED_COST_PER_TRADE, n_legs=4) for i in range(3)]
check("flat default cost on a 4-leg structure is flagged",
      any("PER ORDER" in p for p in validate_trades(multileg)), True)
# The classic ruinous mistake: risk set from the realised loss.
wrong_denom = [Trade(f"w{i}", "s", "2026-01-01", "2026-01-02",
                     risk=1000.0 + i, gross_pnl=-(1000.0 + i), costs=0.0)
               for i in range(5)]
check("every-loss-is-exactly--1R detected",
      any("planned risk" in p for p in validate_trades(wrong_denom)), True)
check("data warnings BLOCK the gate", score(dup).gate_pass, False)
check("clean data produces no warnings", validate_trades(burst), [])

print("\n" + "=" * 96)
print("DEMONSTRATION -- what the scorer says at the account's stated -0.27R")
print("=" * 96)
print("  SYNTHETIC reconstruction with that exact mean. NOT the real trade history.")
print("  Losses are VARIED (never worse than -1.0R) so the sample neither trips")
print("  the 'risk came from the realised loss' validator nor creates fake overruns.\n")
import random as _random  # noqa: E402

_wins = [0.69875] * 8
_losses = [-1.0, -0.95, -1.0, -0.92, -1.0, -0.98, -1.0, -0.90, -1.0, -0.97, -1.0]
pattern = _wins + _losses
_random.Random(42).shuffle(pattern)      # deterministic, realistic equity curve
assert len(pattern) == 19
assert abs(sum(pattern) / 19 - (-0.27)) < 1e-9, sum(pattern) / 19
demo = [Trade(f"D{i}", "historical", f"2026-{1 + i // 10:02d}-{1 + i % 10:02d}",
              f"2026-{1 + i // 10:02d}-{1 + i % 10:02d}", RISK_UNIT_CORE,
              p * RISK_UNIT_CORE, costs=0.0, costs_are_modelled=True)
        for i, p in enumerate(pattern)]
demo_sc = score(demo)
print(demo_sc)
check("reconstructed mean is -0.27R", demo_sc.expectancy_r, -0.27, 1e-9)
print(f"\n  Required for 2%/month at 8 trades: {required_expectancy(6000, 8):+.3f}R")
print(f"  Gap to close:                      "
      f"{required_expectancy(6000, 8) - demo_sc.expectancy_r:+.3f}R")

print("\n" + "=" * 96)
n_pass, n_fail = len(PASS), len(FAIL)
print(f"RESULT: {n_pass} passed, {n_fail} failed")
if FAIL:
    print("FAILURES:")
    for f in FAIL:
        print(f"   - {f}")
    raise SystemExit(1)
print("All checks green. TEST 1 ties the equation to the Risk Constitution;")
print("TESTS 11-15 verify each v2 correction actually fixes its v1 defect.")
print("=" * 96)
