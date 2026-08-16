"""
engine/test_rules.py — Week 2 exit criterion: "each rule reproduces a known
historical signal by hand-check."

WHY THIS IS A SYNTHETIC FIXTURE, NOT THE TWO REAL 2026-07-28 N3 TRADES:
    Those two trades (Sell 23300PE/Buy 23150PE, Sell 24200CE/Buy 24350CE) used
    the width the strategy note had BEFORE the 2026-07-29 correction (150pt,
    now 100-115pt) and the ₹4,000 cap from BEFORE the 2026-08-13 halving (now
    ₹2,000). Replaying them through the current evaluate_n3 would correctly
    REJECT both under the corrected rules -- that's the rule working as
    intended, not a bug, but it means they can't serve as a "does the current
    rule reproduce this" fixture. Instead, this file prices a synthetic-but-
    real chain with analytics.py's own Black-76 pricer (already verified
    against live Kite quotes in test_analytics.py, 33/33 passing) and hand-
    checks evaluate_n3's filter/selection/cost arithmetic against that.

FINDING WORTH FLAGGING: at a realistic short-dated Nifty vol (18%, mid-VIX-
band) and the strategy note's own delta band (0.20-0.30) and width (100-
115pt), net_credit/(width-net_credit) comes out to ~0.29 -- similar order to
the two real trades' ~0.55-0.62 R (computed the same way, at the OLD wider
150pt width, which should have been more favourable to R:R, not less). Under
the CURRENT 1:1 minimum R:R gate (CLAUDE.md core rule), that means this
fixture -- built from realistic inputs, not cherry-picked -- would be
REJECTED by the live filter. Whether that generalises (N3 rarely clears its
own R:R gate as specified) is a real question for the Week-3/4 backtest to
answer with actual history, not something to resolve here.

Run:  python engine/test_rules.py   (from 9- Automation/)
"""
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.rules import ChainRow, N3Entry, N3Params, evaluate_n3

PASS, FAIL = [], []


def check(name, got, want, tol=None):
    ok = (got == want) if tol is None else abs(got - want) <= tol
    (PASS if ok else FAIL).append(name)
    flag = "PASS" if ok else "**FAIL**"
    print(f"  [{flag}] {name:56s} got {got!r:>16}  want {want!r}"
          f"{f' (+/-{tol})' if tol is not None else ''}")


# ---------------------------------------------------------------- fixture
# Priced with analytics.black76 at spot=23950, T=7/365, vol=0.18, r=0.0575 --
# see this module's docstring. Values hand-verified independently via the
# same call in a throwaway script; reproduced here as fixed literals so this
# test does not depend on analytics.py at import time for its expected values.
SPOT = 23950.0
VIX = 18.0
PRIOR_LOW, PRIOR_HIGH = 23900.0, 24100.0     # spot 23950 < mid 24000 -> bull_put
LOT = 65
AS_OF = dt.date(2026, 5, 12)
EXPIRY = dt.date(2026, 5, 19)                # 7 DTE

CHAIN = (
    ChainRow(23550.0, "PE", 88.0248, SPOT, LOT),   # target short leg, delta -0.2454
    ChainRow(23450.0, "PE", 65.3144, SPOT, LOT),   # target long leg, width 100
)

NET_CREDIT = 88.0248 - 65.3144                 # 22.7104
WIDTH = 100.0
MAX_LOSS_PS = WIDTH - NET_CREDIT               # 77.2896
RR = NET_CREDIT / MAX_LOSS_PS                  # 0.2938
MAX_LOSS = MAX_LOSS_PS * LOT                   # 5023.83
COST_TOTAL = 62.4015                           # option_trade_cost(...), hand-verified
EXPECTED_R = (NET_CREDIT * LOT - COST_TOTAL) / MAX_LOSS   # 0.2814


def entry(**overrides) -> N3Entry:
    base = dict(as_of_date=AS_OF, expiry=EXPIRY, vix=VIX, spot=SPOT,
                prior_day_low=PRIOR_LOW, prior_day_high=PRIOR_HIGH,
                chain=CHAIN, event_day=False)
    base.update(overrides)
    return N3Entry(**base)


print("=" * 96)
print("TEST 1 -- VIX filter rejects outside [16,22]")
print("=" * 96)
sig = evaluate_n3(entry(vix=25.0))
check("entered", sig.entered, False)
check("reason mentions VIX", "VIX" in sig.reason, True)

print("\n" + "=" * 96)
print("TEST 2 -- prior-day-range filter rejects spot outside range")
print("=" * 96)
sig = evaluate_n3(entry(spot=24200.0))
check("entered", sig.entered, False)
check("reason mentions range", "range" in sig.reason, True)

print("\n" + "=" * 96)
print("TEST 3 -- DTE filter rejects outside [5,7]")
print("=" * 96)
sig = evaluate_n3(entry(expiry=dt.date(2026, 5, 15)))     # 3 DTE
check("entered", sig.entered, False)
check("reason mentions DTE", "DTE" in sig.reason, True)

print("\n" + "=" * 96)
print("TEST 4 -- bias picks bull_put when spot below prior-range midpoint")
print("          strike selection picks the strike closest to target delta 0.25")
print("=" * 96)
sig = evaluate_n3(entry())        # default params: min_rr=1.0, cap=2000 -> rejects on R:R
check("entered (default 1:1 gate)", sig.entered, False)
check("reason mentions R:R/cap", ("R:R" in sig.reason) or ("cap" in sig.reason), True)
check("side inferred bull_put even on rejection", sig.side, "bull_put")
check("short strike selected = 23550 (closest to target delta)", sig.short_strike, 23550.0)

print("\n" + "=" * 96)
print("TEST 5 -- max-loss cap rejects even when R:R is relaxed to pass")
print("=" * 96)
sig = evaluate_n3(entry(), params=N3Params(min_rr=0.1, max_loss_cap=2000.0))
check("entered", sig.entered, False)
check("reason mentions cap", "cap" in sig.reason, True)

print("\n" + "=" * 96)
print("TEST 6 -- full pass-through arithmetic (R:R and cap relaxed to admit the")
print("          fixture -- see module docstring on why realistic defaults reject it)")
print("=" * 96)
sig = evaluate_n3(entry(), params=N3Params(min_rr=0.2, max_loss_cap=6000.0))
check("entered", sig.entered, True)
check("side", sig.side, "bull_put")
check("short_strike", sig.short_strike, 23550.0)
check("long_strike", sig.long_strike, 23450.0)
check("width", sig.width, WIDTH, 1e-6)
check("net_credit", sig.net_credit, NET_CREDIT, 1e-3)
check("rr", sig.rr, RR, 1e-3)
check("max_loss", sig.max_loss, MAX_LOSS, 0.5)
check("lot_size", sig.lot_size, LOT)
check("entry_cost", sig.entry_cost, COST_TOTAL, 0.5)
check("expected_r_after_cost", sig.expected_r_after_cost, EXPECTED_R, 1e-3)

print("\n" + "=" * 96)
print("TEST 7 -- event day short-circuits before any other filter")
print("=" * 96)
sig = evaluate_n3(entry(vix=999.0, event_day=True))
check("entered", sig.entered, False)
check("reason mentions event day", "event day" in sig.reason, True)

print("\n" + "=" * 96)
n_pass, n_fail = len(PASS), len(FAIL)
print(f"RESULT: {n_pass} passed, {n_fail} failed")
if FAIL:
    print("FAILURES:")
    for f in FAIL:
        print(f"   - {f}")
    raise SystemExit(1)
print("All checks green -- evaluate_n3's filters, strike selection, and cost")
print("arithmetic match hand-computed values.")
print("=" * 96)
