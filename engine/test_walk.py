"""
engine/test_walk.py — verification for the day-by-day position walk.

The value of this module is that it REFUSES to guess when daily data cannot
answer the question. These tests check that:
  - the P&L envelope has the right signs (short profits as premium falls)
  - a day where both target and stop are reachable is flagged AMBIGUOUS
    rather than silently resolved in the profitable direction
  - rule precedence is deterministic
  - a trade whose optimistic and pessimistic outcomes differ in SIGN is
    marked unsafe to score
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from walk import (  # noqa: E402
    DayBar, Leg, LegBar, WalkRules, net_credit, pnl_envelope, position_pnl,
    short_strikes, walk,
)

PASS: list[str] = []
FAIL: list[str] = []
LOT = 65


def check(label: str, got, want, tol: float | None = None):
    if tol is not None:
        ok = (got == got) and (want == want) and abs(got - want) <= tol
    else:
        ok = got == want
    (PASS if ok else FAIL).append(label)
    extra = "" if ok else f"   (got {got!r}, want {want!r})"
    print(f"  [{'ok  ' if ok else 'FAIL'}] {label}{extra}")


def bar(day: str, close=80.0, high=120.0, low=60.0, spot=25000.0,
        dte=5, vol=5000, oi=650_000, label="short_ce") -> DayBar:
    return DayBar(date=day, spot_close=spot, spot_high=spot + 50, spot_low=spot - 50,
                  legs=(LegBar(label, close, high, low, vol, oi, LOT),),
                  lot_size=LOT, dte=dte)


SHORT_CE = Leg("short_ce", strike=25200.0, opt_type="CE", qty=-1, entry_price=100.0)
LONG_CE = Leg("long_ce", strike=25400.0, opt_type="CE", qty=1, entry_price=40.0)

print("=" * 96)
print("TEST 1 -- P&L signs: short gains as premium falls, long gains as it rises")
print("=" * 96)
check("short CE, premium 100 -> 80, 1 lot of 65 = +Rs 1,300",
      position_pnl([SHORT_CE], {"short_ce": 80.0}, LOT), 1300.0, 1e-9)
check("short CE, premium 100 -> 120 = -Rs 1,300",
      position_pnl([SHORT_CE], {"short_ce": 120.0}, LOT), -1300.0, 1e-9)
check("long CE, premium 40 -> 60 = +Rs 1,300",
      position_pnl([LONG_CE], {"long_ce": 60.0}, LOT), 1300.0, 1e-9)
check("long CE, premium 40 -> 20 = -Rs 1,300",
      position_pnl([LONG_CE], {"long_ce": 20.0}, LOT), -1300.0, 1e-9)
check("missing price for a leg is skipped, not crashed",
      position_pnl([SHORT_CE, LONG_CE], {"short_ce": 80.0}, LOT), 1300.0, 1e-9)
check("net_credit of a short at 100 = +Rs 6,500 received",
      net_credit([SHORT_CE], LOT), 6500.0, 1e-9)
check("net_credit of a long at 40 = -Rs 2,600 paid",
      net_credit([LONG_CE], LOT), -2600.0, 1e-9)
check("spread: short 100 / long 40 -> net credit Rs 3,900",
      net_credit([SHORT_CE, LONG_CE], LOT), 3900.0, 1e-9)

print("\n" + "=" * 96)
print("TEST 2 -- the envelope assigns each leg its correct extreme")
print("=" * 96)
c, b, w = pnl_envelope([SHORT_CE], bar("2026-01-02"))
check("close 80 -> +Rs 1,300", c, 1300.0, 1e-9)
check("BEST for a short uses the day's LOW (60) -> +Rs 2,600", b, 2600.0, 1e-9)
check("WORST for a short uses the day's HIGH (120) -> -Rs 1,300", w, -1300.0, 1e-9)
check("best >= close >= worst", (b >= c >= w), True)

long_bar = DayBar("2026-01-02", 25000.0, 25050.0, 24950.0,
                  (LegBar("long_ce", 50.0, 70.0, 30.0, 5000, 650_000, LOT),),
                  LOT, 5)
c2, b2, w2 = pnl_envelope([LONG_CE], long_bar)
check("BEST for a long uses the day's HIGH (70) -> +Rs 1,950", b2, 1950.0, 1e-9)
check("WORST for a long uses the day's LOW (30) -> -Rs 650", w2, -650.0, 1e-9)
print("        -> for a SINGLE leg, coherent-scenario and independent-extreme")
print("           bounds coincide. They diverge sharply for spreads (TEST 2b).")

print("\n" + "=" * 96)
print("TEST 2b [REGRESSION] -- legs are correlated; impossible states must not be priced")
print("=" * 96)
print("  Found by an integration test on a real 2026-03-04 bear call spread.")
print("  The first implementation put the short leg at its HIGH and the long leg")
print("  at its LOW simultaneously -- impossible for adjacent strikes on ONE")
print("  underlying -- and reported a cap breach on a PROFITABLE day.")
# Bear call spread: short the lower strike, long the higher. Both are calls, so
# both rise together when spot rises.
bull = Leg("s", 25150.0, "CE", -1, 139.35)
hedge = Leg("l", 25200.0, "CE", 1, 120.55)
spread_bar = DayBar("2026-03-04", 24480.5, 24600.0, 24400.0,
                    (LegBar("s", 64.85, 150.0, 60.0, 370_274, 380_000, LOT),
                     LegBar("l", 55.45, 130.0, 50.0, 879_050, 2_130_000, LOT)),
                    LOT, 6)
c3, b3, w3 = pnl_envelope([bull, hedge], spread_bar)
# coherent scenarios:
#   spot-up  : s=150, l=130 -> net 20 vs entry net 18.80 -> slight loss
#   spot-down: s=60,  l=50  -> net 10 -> profit
up = position_pnl([bull, hedge], {"s": 150.0, "l": 130.0}, LOT)
dn = position_pnl([bull, hedge], {"s": 60.0, "l": 50.0}, LOT)
check("best == the better coherent scenario", b3, max(up, dn), 1e-9)
check("worst == the worse coherent scenario", w3, min(up, dn), 1e-9)
# The impossible state the old code priced:
impossible = position_pnl([bull, hedge], {"s": 150.0, "l": 50.0}, LOT)
check("old bound priced an IMPOSSIBLE state far worse than reality",
      impossible < w3 - 3000, True)
print(f"        coherent worst : Rs {w3:>9,.0f}")
print(f"        impossible state: Rs {impossible:>9,.0f}  <- what the old code used")
print(f"        band narrowed from Rs {b3 - impossible:,.0f} to Rs {b3 - w3:,.0f}")
check("close-basis P&L is positive that day (spot fell, good for a bear call)",
      c3 > 0, True)

print("\n" + "=" * 96)
print("TEST 3 -- short_strikes extraction")
print("=" * 96)
ss = short_strikes([SHORT_CE, LONG_CE,
                    Leg("short_pe", 24800.0, "PE", -1, 90.0)])
check("one short CE", ss["CE"], [25200.0])
check("one short PE", ss["PE"], [24800.0])
check("long legs excluded", len(ss["CE"]) + len(ss["PE"]), 2)

print("\n" + "=" * 96)
print("TEST 4 -- AMBIGUOUS bar: both target and stop reachable the same day")
print("=" * 96)
print("  credit Rs 6,500, target 25% = Rs 1,625, max_loss Rs 2,000, stop = -Rs 1,000")
print("  day envelope: best +Rs 2,600 (>= target), worst -Rs 1,300 (<= stop)")
rules = WalkRules(profit_target_pct_of_credit=0.25, stop_pct_of_max_loss=0.50,
                  close_below_short_strike=False, time_stop_dte=None,
                  hard_risk_cap=None)
res = walk([SHORT_CE], [bar("2026-01-02")], rules, max_loss=2000.0)
check("flagged ambiguous", res.is_ambiguous, True)
check("exactly one ambiguous bar", res.ambiguous_bars, 1)
check("optimistic resolves to the target", res.pnl_optimistic, 1625.0, 1e-9)
check("pessimistic resolves to the stop", res.pnl_pessimistic, -1000.0, 1e-9)
check("uncertainty band is Rs 2,625", res.spread, 2625.0, 1e-9)
check("sign is UNRESOLVED -> unsafe to score", res.verdict_is_safe, False)
print("\n" + res.summary())

print("\n  A clean day resolves to a single number:")
calm = bar("2026-01-02", close=95.0, high=99.0, low=92.0)
res2 = walk([SHORT_CE], [calm], WalkRules(profit_target_pct_of_credit=0.25,
                                          stop_pct_of_max_loss=0.50,
                                          close_below_short_strike=False,
                                          time_stop_dte=None, hard_risk_cap=None),
            max_loss=2000.0)
check("no ambiguity", res2.ambiguous_bars, 0)
check("optimistic == pessimistic", res2.pnl_optimistic, res2.pnl_pessimistic)
check("safe to score", res2.verdict_is_safe, True)

print("\n" + "=" * 96)
print("TEST 5 -- rule precedence is deterministic")
print("=" * 96)
# hard cap beats everything
hard = walk([SHORT_CE], [bar("2026-01-02", high=200.0, low=60.0)],
            WalkRules(profit_target_pct_of_credit=0.25, stop_pct_of_max_loss=0.50,
                      close_below_short_strike=False, time_stop_dte=None,
                      hard_risk_cap=2000.0), max_loss=2000.0)
check("hard_risk_cap wins over stop and target", hard.exit_reason, "hard_risk_cap")

# breach on close, no target/stop reachable
breach = walk([SHORT_CE], [bar("2026-01-02", close=95.0, high=99.0, low=92.0,
                               spot=25500.0)],
              WalkRules(profit_target_pct_of_credit=None, stop_pct_of_max_loss=None,
                        close_below_short_strike=True, time_stop_dte=None,
                        hard_risk_cap=None), max_loss=2000.0)
check("spot 25,500 above short CE 25,200 -> breach", breach.exit_reason,
      "short_strike_breached")
inside = walk([SHORT_CE], [bar("2026-01-02", close=95.0, high=99.0, low=92.0,
                               spot=25000.0, dte=9)],
              WalkRules(profit_target_pct_of_credit=None, stop_pct_of_max_loss=None,
                        close_below_short_strike=True, time_stop_dte=2,
                        hard_risk_cap=None), max_loss=2000.0)
check("spot inside the short strike -> no breach", inside.exit_reason, "expiry")

ts = walk([SHORT_CE], [bar("2026-01-02", close=95.0, high=99.0, low=92.0, dte=1)],
          WalkRules(profit_target_pct_of_credit=None, stop_pct_of_max_loss=None,
                    close_below_short_strike=False, time_stop_dte=2,
                    hard_risk_cap=None), max_loss=2000.0)
check("dte 1 <= time_stop 2 -> time_stop", ts.exit_reason, "time_stop")

print("\n" + "=" * 96)
print("TEST 6 -- multi-day walk exits on the FIRST trigger")
print("=" * 96)
days = [bar("2026-01-02", close=95.0, high=99.0, low=92.0, dte=5),
        bar("2026-01-03", close=90.0, high=96.0, low=88.0, dte=4),
        bar("2026-01-06", close=40.0, high=60.0, low=35.0, dte=3),   # target hit
        bar("2026-01-07", close=30.0, high=35.0, low=25.0, dte=2)]
r = walk([SHORT_CE], days, WalkRules(profit_target_pct_of_credit=0.25,
                                     stop_pct_of_max_loss=None,
                                     close_below_short_strike=False,
                                     time_stop_dte=None, hard_risk_cap=None),
         max_loss=2000.0)
check("exits on day 3", r.exit_date, "2026-01-06")
check("reason is profit_target", r.exit_reason, "profit_target")
check("days_held 3", r.days_held, 3)
check("path records every day walked", len(r.path), 3)
check("later days are not evaluated", all(m.date != "2026-01-07" for m in r.path), True)
# Regression: exit must be booked AT THE TARGET, not at the day's close.
_target = 0.25 * net_credit([SHORT_CE], LOT)
check("books the TARGET (Rs 1,625), not the day's close", r.pnl_optimistic,
      _target, 1e-9)
check("  ...and the close that day was better, which is the bias avoided",
      r.pnl_close_basis > r.pnl_optimistic, True)
print(f"        target booked Rs {r.pnl_optimistic:,.0f} vs close-basis "
      f"Rs {r.pnl_close_basis:,.0f} -- using the close would overstate by "
      f"Rs {r.pnl_close_basis - r.pnl_optimistic:,.0f}")

stopped = walk([SHORT_CE], [bar("2026-01-02", close=160.0, high=170.0, low=155.0)],
               WalkRules(profit_target_pct_of_credit=None, stop_pct_of_max_loss=0.50,
                         close_below_short_strike=False, time_stop_dte=None,
                         hard_risk_cap=None), max_loss=2000.0)
check("a stop books the STOP level (-Rs 1,000), not the close",
      stopped.pnl_pessimistic, -1000.0, 1e-9)
check("  ...and the close was worse, so this is not a free pass either",
      stopped.pnl_close_basis < stopped.pnl_pessimistic, True)

print("\n  No trigger -> falls through to expiry:")
quiet = [bar(f"2026-01-{d:02d}", close=95.0, high=97.0, low=93.0, dte=9 - i)
         for i, d in enumerate((2, 3, 6))]
rq = walk([SHORT_CE], quiet, WalkRules(profit_target_pct_of_credit=0.25,
                                       stop_pct_of_max_loss=0.50,
                                       close_below_short_strike=False,
                                       time_stop_dte=None, hard_risk_cap=None),
          max_loss=2000.0)
check("exit_reason expiry", rq.exit_reason, "expiry")
check("uses the last bar", rq.exit_date, "2026-01-06")
check("days_held == all bars", rq.days_held, 3)

print("\n" + "=" * 96)
print("TEST 7 -- theoretical legs are counted, not silently trusted")
print("=" * 96)
dead = DayBar("2026-01-02", 25000.0, 25050.0, 24950.0,
              (LegBar("short_ce", 95.0, 97.0, 93.0, volume=0,
                      open_interest=650_000, lot_size=LOT),),
              LOT, 5)
rt = walk([SHORT_CE], [dead], WalkRules(profit_target_pct_of_credit=None,
                                        stop_pct_of_max_loss=None,
                                        close_below_short_strike=False,
                                        time_stop_dte=None, hard_risk_cap=None),
          max_loss=2000.0)
check("bar with an untraded leg is counted", rt.theoretical_bars, 1)
check("the mark records which legs were theoretical",
      rt.path[0].theoretical_legs, 1)
check("a traded leg is not counted",
      walk([SHORT_CE], [bar("2026-01-02", close=95.0, high=97.0, low=93.0)],
           WalkRules(None, None, False, None, None),
           max_loss=2000.0).theoretical_bars, 0)

print("\n" + "=" * 96)
print("TEST 8 -- degenerate inputs")
print("=" * 96)
empty = walk([SHORT_CE], [], WalkRules(), max_loss=2000.0)
check("no bars -> exit_date None", empty.exit_date, None)
check("no bars -> days_held 0", empty.days_held, 0)
# Regression: found by an integration test that used expiry == entry_date.
# "expiry, P&L 0" would silently inject a fake break-even trade into a sample.
check("no bars -> exit_reason 'no_data', NOT 'expiry'", empty.exit_reason, "no_data")
check("no bars -> has_data False", empty.has_data, False)
check("a real walk has_data True",
      walk([SHORT_CE], [bar("2026-01-02")], WalkRules(), max_loss=2000.0).has_data, True)
check("zero max_loss disables the stop, no crash",
      walk([SHORT_CE], [bar("2026-01-02")],
           WalkRules(profit_target_pct_of_credit=None, stop_pct_of_max_loss=0.5,
                     close_below_short_strike=False, time_stop_dte=None,
                     hard_risk_cap=None), max_loss=0.0).exit_reason, "expiry")
debit = [Leg("long_ce", 25400.0, "CE", 1, 40.0)]
check("net-debit structure disables the credit-based target (no crash)",
      walk(debit, [long_bar], WalkRules(profit_target_pct_of_credit=0.25,
                                        stop_pct_of_max_loss=None,
                                        close_below_short_strike=False,
                                        time_stop_dte=None, hard_risk_cap=None),
           max_loss=2600.0).exit_reason, "expiry")

print("\n" + "=" * 96)
n_pass, n_fail = len(PASS), len(FAIL)
print(f"RESULT: {n_pass} passed, {n_fail} failed")
if FAIL:
    print("FAILURES:")
    for f in FAIL:
        print(f"   - {f}")
    raise SystemExit(1)
print("All checks green -- the walk exits on the first trigger, and refuses to")
print("resolve a day where daily data genuinely cannot say what happened first.")
print("=" * 96)
