"""
engine/test_strategy.py — verification for the strategy spec + adjustable walk.

Chains are synthesised with analytics.black76 so that implied_vol inverts them
exactly and every delta is known in advance. That makes trigger behaviour
checkable against hand-computed values rather than against whatever the code
happens to produce.

The tests that matter most:
  TEST 4  a NaN/untraded delta is UNEVALUABLE, never silently "not triggered"
  TEST 7  adjustments actually mutate the position and realise P&L
  TEST 8  every adjustment is CHARGED -- the mechanism that kills these strategies
  TEST 9  the adjustment budget is enforced and can force an exit
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import analytics as an  # noqa: E402
from strategy import (  # noqa: E402
    Action, AdjustmentRule, ChainRow, ChainSnapshot, LegSpec, OpenLeg, Position,
    StrategySpec, Trigger, _txn_cost, evaluate_trigger, leg_greeks,
    open_position, run_strategy, select_strike_by_delta, structure_max_loss,
)
from walk import WalkRules  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []
LOT = 65
R = 0.0575


def check(label: str, got, want, tol: float | None = None):
    if tol is not None:
        ok = (got == got) and (want == want) and abs(got - want) <= tol
    else:
        ok = got == want
    (PASS if ok else FAIL).append(label)
    extra = "" if ok else f"   (got {got!r}, want {want!r})"
    print(f"  [{'ok  ' if ok else 'FAIL'}] {label}{extra}")


def make_chain(date: str, expiry: str, spot: float, vol: float = 0.15,
               lo: int = -1500, hi: int = 1500, step: int = 50,
               volume: int = 5000, oi: int = 650_000,
               untraded: set[float] | None = None) -> ChainSnapshot:
    """Synthetic but internally consistent chain: prices come from black76 at
    the true forward, so implied_vol recovers `vol` and delta is predictable."""
    untraded = untraded or set()
    T = an.year_fraction(__import__("datetime").datetime.fromisoformat(date),
                         __import__("datetime").datetime.fromisoformat(expiry))
    F = spot * math.exp(R * T)
    atm = round(spot / step) * step
    rows = []
    for off in range(lo, hi + 1, step):
        k = float(atm + off)
        if k <= 0:
            continue
        for ot in ("CE", "PE"):
            px = an.black76(F, k, T, vol, ot, R)
            v = 0 if k in untraded else volume
            rows.append(ChainRow(k, ot, px, px * 1.10, px * 0.90, v, oi, LOT))
    return ChainSnapshot(date, expiry, spot, tuple(rows), LOT)


print("=" * 96)
print("TEST 1 -- forward comes from put-call parity, not spot")
print("=" * 96)
ch = make_chain("2026-03-02", "2026-03-10", 25000.0)
T = ch.T()
true_F = 25000.0 * math.exp(R * T)
check("parity recovers the forward", ch.forward(R), true_F, 1.0)
check("forward exceeds spot (positive carry)", ch.forward(R) > ch.spot, True)
print(f"        spot {ch.spot:,.1f}  forward {ch.forward(R):,.1f}  "
      f"basis {ch.forward(R) - ch.spot:+.1f} pts over {T*365:.0f} days")
print("        -> analytics.py audit finding F2: pricing off spot biases every")
print("           delta, and therefore every delta-based trigger.")

all_dead = make_chain("2026-03-02", "2026-03-10", 25000.0, volume=0)
check("no traded strikes -> falls back to spot*e^(rT)",
      all_dead.forward(R), 25000.0 * math.exp(R * all_dead.T()), 1e-6)

print("\n" + "=" * 96)
print("TEST 2 -- leg greeks invert the synthetic chain exactly")
print("=" * 96)
atm = round(25000 / 50) * 50
g = leg_greeks(ch, float(atm), "CE", R)
check("IV recovers the 0.15 used to build the chain", g.iv, 0.15, 1e-4)
check("price is real (traded)", g.price_is_real, True)
check("evaluable", g.evaluable, True)
expected = an.greeks(ch.forward(R), float(atm), T, 0.15, "CE", R).delta
check("delta matches analytics.greeks", g.delta, expected, 1e-9)
check("ATM call delta is near +0.5", 0.45 < g.delta < 0.60, True)
gp = leg_greeks(ch, float(atm), "PE", R)
check("ATM put delta is near -0.5", -0.60 < gp.delta < -0.40, True)
check("missing strike -> not evaluable",
      leg_greeks(ch, 99999.0, "CE", R).evaluable, False)

print("\n" + "=" * 96)
print("TEST 3 -- strike selection targets delta and skips untraded strikes")
print("=" * 96)
spec40 = LegSpec("short_pe", "PE", -1, 0.40)
k = select_strike_by_delta(ch, spec40, R)
check("found a strike", k is not None, True)
sel = leg_greeks(ch, k, "PE", R)
check("selected |delta| is close to 0.40", abs(abs(sel.delta) - 0.40) < 0.05, True)
print(f"        target 0.40 -> strike {k:g} PE, actual delta {sel.delta:+.3f}")
check("a 0.20-delta target picks a further OTM strike",
      select_strike_by_delta(ch, LegSpec("x", "PE", -1, 0.20), R) < k, True)

ch_gap = make_chain("2026-03-02", "2026-03-10", 25000.0, untraded={float(k)})
k2 = select_strike_by_delta(ch_gap, spec40, R)
check("untraded strike is skipped, another chosen", k2 != k, True)
check("  ...and the chosen one did trade",
      leg_greeks(ch_gap, k2, "PE", R).price_is_real, True)
check("require_tradeable=False may return the untraded one",
      select_strike_by_delta(ch_gap, spec40, R, require_tradeable=False) is not None, True)

print("\n" + "=" * 96)
print("TEST 4 [CRITICAL] -- an unevaluable trigger is NOT 'not triggered'")
print("=" * 96)
print("  implied_vol returns NaN for a quote at/below intrinsic -- exactly what")
print("  an untraded, theoretically-priced contract looks like. Reading that as")
print("  'condition false' would silently skip adjustments the strategy required.")
pos = Position(legs=[OpenLeg("short_pe", float(k), "PE", -1, 100.0, "2026-03-02", "near")])
ev = evaluate_trigger(Trigger("leg_abs_delta_below", 0.15, "short_pe"),
                      pos, {"near": ch}, 0.0, R)
check("live leg -> evaluable", ev.evaluable, True)
check("  0.40-delta leg does not fire a <0.15 trigger", ev.fired, False)

dead_pos = Position(legs=[OpenLeg("short_pe", float(k), "PE", -1, 100.0,
                                  "2026-03-02", "near")])
ev_dead = evaluate_trigger(Trigger("leg_abs_delta_below", 0.15, "short_pe"),
                           dead_pos, {"near": ch_gap}, 0.0, R)
check("untraded leg -> UNEVALUABLE", ev_dead.evaluable, False)
check("  ...and reports why", "not computable" in ev_dead.detail, True)
check("missing leg -> unevaluable",
      evaluate_trigger(Trigger("leg_abs_delta_below", 0.15, "nope"),
                       pos, {"near": ch}, 0.0, R).evaluable, False)
check("unknown trigger kind -> unevaluable, not a crash",
      evaluate_trigger(Trigger("nonsense", 1.0), pos, {"near": ch}, 0.0, R).evaluable,
      False)

print("\n" + "=" * 96)
print("TEST 5 -- the other trigger kinds")
print("=" * 96)
hi_trg = evaluate_trigger(Trigger("leg_abs_delta_above", 0.20, "short_pe"),
                          pos, {"near": ch}, 0.0, R)
check("0.40-delta leg fires a >0.20 trigger", hi_trg.fired, True)
comb = evaluate_trigger(Trigger("combined_abs_delta_above", 0.30),
                        pos, {"near": ch}, 0.0, R)
check("combined delta over short legs fires", comb.fired, True)
check("dte trigger reads the chain", evaluate_trigger(
    Trigger("dte_below", 10), pos, {"near": ch}, 0.0, R).fired, True)
check("dte trigger does not fire when far out", evaluate_trigger(
    Trigger("dte_below", 3), pos, {"near": ch}, 0.0, R).fired, False)
p = Position(realized_pnl=-3000.0)
check("pnl_below fires", evaluate_trigger(Trigger("pnl_below", -2000.0),
                                          p, {"near": ch}, 0.0, R).fired, True)
check("pnl_above does not", evaluate_trigger(Trigger("pnl_above", 0.0),
                                             p, {"near": ch}, 0.0, R).fired, False)

print("\n" + "=" * 96)
print("TEST 6 -- transaction costs put buys and sells on the right side")
print("=" * 96)
print("  STT is charged on the SELL side, so getting this backwards misprices it.")
short_leg = OpenLeg("s", 25000.0, "PE", -1, 100.0, "2026-03-02", "near")
long_leg = OpenLeg("l", 24900.0, "PE", 1, 60.0, "2026-03-02", "near")
c_close_short = _txn_cost([(short_leg, 90.0)], [], LOT)   # buying back
c_close_long = _txn_cost([(long_leg, 70.0)], [], LOT)     # selling out
check("closing a short costs > 0", c_close_short > 0, True)
check("closing a long costs > 0", c_close_long > 0, True)
check("selling attracts more tax than buying the same notional",
      c_close_long > _txn_cost([(OpenLeg("s2", 24900.0, "PE", -1, 70.0,
                                         "2026-03-02", "near"), 70.0)], [], LOT),
      True)
check("no transactions -> zero cost", _txn_cost([], [], LOT), 0.0, 1e-9)
two = _txn_cost([(short_leg, 90.0)], [(long_leg, 60.0)], LOT)
check("two orders cost more than one", two > c_close_short, True)

print("\n" + "=" * 96)
print("TEST 7 -- adjustments mutate the position and realise P&L")
print("=" * 96)
# Delta-Reset shape: short a 0.40-delta put; if |delta| drops below 0.30,
# close it and re-sell at 0.40.
respec = LegSpec("short_pe", "PE", -1, 0.40)
spec = StrategySpec(
    name="delta_reset_test",
    entry_legs=(respec,),
    adjustments=(AdjustmentRule(
        "reset_low_delta",
        Trigger("leg_abs_delta_below", 0.30, "short_pe"),
        (Action("reopen_leg", "short_pe", respec),)),),
    exit_rules=WalkRules(profit_target_pct_of_credit=None, stop_pct_of_max_loss=None,
                         close_below_short_strike=False, time_stop_dte=None,
                         hard_risk_cap=None),
    max_adjustments=None,
)
# Day 1 entry at 25,000; then the market rallies, so the short put's delta falls.
days = [
    dict(date="2026-03-02", chains={"near": make_chain("2026-03-02", "2026-03-20", 25000.0)}),
    dict(date="2026-03-03", chains={"near": make_chain("2026-03-03", "2026-03-20", 25400.0)}),
    dict(date="2026-03-04", chains={"near": make_chain("2026-03-04", "2026-03-20", 25800.0)}),
]
res = run_strategy(spec, days, R)
check("has data", res.has_data, True)
check("at least one adjustment fired", res.adjustments >= 1, True)
check("costs were charged", res.total_costs > 0, True)
check("net = gross - costs", res.net_pnl, res.gross_pnl - res.total_costs, 1e-6)
print("\n".join("        " + x for x in res.log))
print("\n" + res.summary())

print("\n" + "=" * 96)
print("TEST 7b [REVIEW FIX] -- R uses the structure's OWN defined risk")
print("=" * 96)
print("  Caught by reading TEST 7's output: it reported +8.73R on a 2-day trade.")
print("  The cause was max_loss_at_entry being set to the strategy's risk CAP")
print("  (Rs 2,000) rather than to what the structure can actually lose. Since")
print("  score.py divides by this, a wrong value corrupts expectancy everywhere.")

naked_put = [OpenLeg("p", 25000.0, "PE", -1, 200.0, "2026-03-02", "near")]
rp = structure_max_loss(naked_put, LOT)
check("naked short put is BOUNDED (spot can only reach zero)", rp.bounded, True)
check("  ...but the bound is enormous, not Rs 2,000",
      rp.max_loss > 1_000_000, True)
check("  ...and says so", "spot reaching zero" in rp.detail, True)
print(f"        naked short put max loss: Rs {rp.max_loss:,.0f}")

naked_call = [OpenLeg("c", 25000.0, "CE", -1, 200.0, "2026-03-02", "near")]
rpc = structure_max_loss(naked_call, LOT)
check("naked short call is UNBOUNDED", rpc.bounded, False)
check("  ...max_loss is inf", math.isinf(rpc.max_loss), True)
check("  ...not usable as an R denominator", rpc.usable_as_r_denominator, False)

# A real bull put spread: short 25000 PE at 200, long 24900 PE at 150.
spread = [OpenLeg("s", 25000.0, "PE", -1, 200.0, "2026-03-02", "near"),
          OpenLeg("l", 24900.0, "PE", 1, 150.0, "2026-03-02", "near")]
rs = structure_max_loss(spread, LOT)
check("spread net credit = (200-150)*65 = Rs 3,250", rs.net_credit, 3250.0, 1e-9)
check("spread max loss = (100 - 50)*65 = Rs 3,250", rs.max_loss, 3250.0, 1e-6)
check("  bounded", rs.bounded, True)
check("  usable as an R denominator", rs.usable_as_r_denominator, True)
check("no legs -> zero risk, no crash", structure_max_loss([], LOT).max_loss, 0.0, 1e-9)

res_fixed = run_strategy(spec, days, R)
check("the naked-put test strategy is now flagged over the cap",
      res_fixed.fits_risk_cap, False)
check("  ...and is NOT scoreable at face value", res_fixed.max_loss_at_entry > 2000, True)
print(f"        R is now {res_fixed.r_multiple:+.3f} against a real "
      f"Rs {res_fixed.max_loss_at_entry:,.0f} denominator,")
print(f"        not the +8.73R the assumed Rs 2,000 denominator produced.")

print("\n" + "=" * 96)
print("TEST 8 [THE POINT] -- every adjustment is charged, and it compounds")
print("=" * 96)
no_adj = StrategySpec(name="no_adjust", entry_legs=(respec,),
                      adjustments=(), exit_rules=spec.exit_rules)
r_no = run_strategy(no_adj, days, R)
check("a no-adjust variant makes zero adjustments", r_no.adjustments, 0)
check("...and therefore costs less", r_no.total_costs < res.total_costs, True)
print(f"        with {res.adjustments} adjustment(s): costs Rs {res.total_costs:,.0f}")
print(f"        with 0 adjustments:                 costs Rs {r_no.total_costs:,.0f}")
print(f"        each adjustment cost ~Rs "
      f"{(res.total_costs - r_no.total_costs) / max(1, res.adjustments):,.0f}")
print("        -> against a Rs 2,000 risk unit this is the mechanism that kills")
print("           adjustment-heavy strategies, independent of market view.")

print("\n" + "=" * 96)
print("TEST 9 -- the adjustment budget is enforced")
print("=" * 96)
capped = StrategySpec(
    name="two_adjustment_cap", entry_legs=(respec,),
    adjustments=spec.adjustments, exit_rules=spec.exit_rules,
    max_adjustments=0, on_budget_exhausted="exit")
long_days = days + [
    dict(date="2026-03-05", chains={"near": make_chain("2026-03-05", "2026-03-20", 26200.0)}),
    dict(date="2026-03-06", chains={"near": make_chain("2026-03-06", "2026-03-20", 26600.0)}),
]
r_cap = run_strategy(capped, long_days, R)
check("budget 0 -> no adjustments made", r_cap.adjustments, 0)
check("exhausted budget forces an exit",
      r_cap.exit_reason, "adjustment_budget_exhausted")
hold = StrategySpec(name="hold", entry_legs=(respec,), adjustments=spec.adjustments,
                    exit_rules=spec.exit_rules, max_adjustments=0,
                    on_budget_exhausted="hold")
check("on_budget_exhausted='hold' rides to expiry instead",
      run_strategy(hold, long_days, R).exit_reason, "expiry")

print("\n" + "=" * 96)
print("TEST 10 -- exits: hard cap and time stop")
print("=" * 96)
capped_risk = StrategySpec(
    name="cap", entry_legs=(respec,), adjustments=(),
    exit_rules=WalkRules(profit_target_pct_of_credit=None, stop_pct_of_max_loss=None,
                         close_below_short_strike=False, time_stop_dte=None,
                         hard_risk_cap=1.0))     # absurdly tight -> must fire
crash = [
    dict(date="2026-03-02", chains={"near": make_chain("2026-03-02", "2026-03-20", 25000.0)}),
    dict(date="2026-03-03", chains={"near": make_chain("2026-03-03", "2026-03-20", 23500.0)}),
]
check("hard risk cap fires on a big adverse move",
      run_strategy(capped_risk, crash, R).exit_reason, "hard_risk_cap")

ts = StrategySpec(name="ts", entry_legs=(respec,), adjustments=(),
                  exit_rules=WalkRules(profit_target_pct_of_credit=None,
                                       stop_pct_of_max_loss=None,
                                       close_below_short_strike=False,
                                       time_stop_dte=30, hard_risk_cap=None))
check("time stop fires when dte <= threshold",
      run_strategy(ts, days, R).exit_reason, "time_stop")

print("\n" + "=" * 96)
print("TEST 11 -- degenerate inputs")
print("=" * 96)
check("no days -> no_data", run_strategy(spec, [], R).exit_reason, "no_data")
check("entry day only -> no_data (nothing to mark)",
      run_strategy(spec, days[:1], R).exit_reason, "no_data")
check("  ...and has_data is False", run_strategy(spec, days[:1], R).has_data, False)
dead_day = [dict(date="2026-03-02",
                 chains={"near": make_chain("2026-03-02", "2026-03-20", 25000.0,
                                            volume=0)}),
            dict(date="2026-03-03",
                 chains={"near": make_chain("2026-03-03", "2026-03-20", 25000.0,
                                            volume=0)})]
check("entirely untraded chain -> cannot enter", run_strategy(spec, dead_day, R).exit_reason,
      "no_entry")
check("open_position returns None when a leg cannot be selected",
      open_position(spec, {"near": make_chain('2026-03-02', '2026-03-20', 25000.0,
                                              volume=0)}, "2026-03-02", R), None)
# risk_cap no longer drives the R denominator (that was the TEST 7b defect), so
# setting it to 0 must NOT zero out max_loss -- it only changes cap compliance.
zero_cap = run_strategy(
    StrategySpec(name="z", entry_legs=(respec,), exit_rules=spec.exit_rules,
                 risk_cap=0.0), days, R)
check("risk_cap=0 does not corrupt the R denominator",
      zero_cap.max_loss_at_entry > 0, True)
check("  ...it only marks the structure as over-cap", zero_cap.fits_risk_cap, False)
check("  ...and R stays a real number", math.isnan(zero_cap.r_multiple), False)

# A genuinely unbounded structure must refuse to produce an R at all.
naked_call_spec = StrategySpec(
    name="naked_call", entry_legs=(LegSpec("short_ce", "CE", -1, 0.30),),
    exit_rules=spec.exit_rules)
nc = run_strategy(naked_call_spec, days, R)
check("net short calls -> risk unbounded", nc.risk_bounded, False)
check("  ...r_multiple is nan", math.isnan(nc.r_multiple), True)
check("  ...and it is NOT scoreable", nc.scoreable, False)
check("a bounded structure IS scoreable", zero_cap.scoreable, True)

print("\n" + "=" * 96)
n_pass, n_fail = len(PASS), len(FAIL)
print(f"RESULT: {n_pass} passed, {n_fail} failed")
if FAIL:
    print("FAILURES:")
    for f in FAIL:
        print(f"   - {f}")
    raise SystemExit(1)
print("All checks green -- strategies are declarative, adjustments mutate the")
print("position and are CHARGED, and an unevaluable trigger is never silently false.")
print("=" * 96)
