"""
engine/test_liquidity.py — verification for the liquidity filter.

The point of this module is to stop a backtest trusting NSE's THEORETICAL
prices for contracts that never traded (42.6% of NIFTY option rows). These
tests check that the filter actually catches that, that the units are handled
correctly (volume = contracts, OI = units), and that the cap arithmetic
survives the four lot-size changes in the data window.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from liquidity import (  # noqa: E402
    MIN_OI_LOTS, MIN_VOLUME_TRADE, assess, assess_structure, fits_cap,
    liquid_chain_sql, max_loss_of_spread, max_width_for_cap,
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


print("=" * 96)
print("TEST 1 -- volume=0 means the price is THEORETICAL, not traded")
print("=" * 96)
v = assess(volume=0, open_interest=500_000, lot_size=65)
check("volume=0 -> price_is_real False", v.price_is_real, False)
check("volume=0 -> not tradeable", v.is_tradeable, False)
check("volume=0 -> tier THEORETICAL", v.tier, "THEORETICAL")
check("  reason explains it is NSE's model price",
      any("theoretical" in r for r in v.reasons), True)
print("        -> huge open interest does NOT rescue a zero-volume row:")
print("           OI says the strike exists, volume says nobody traded it today.")

print("\n" + "=" * 96)
print("TEST 2 -- units: volume is CONTRACTS, open_interest is UNITS")
print("=" * 96)
v = assess(volume=500, open_interest=65_000, lot_size=65)
check("OI 65,000 units at lot 65 -> 1,000 lots", v.oi_lots, 1000.0, 1e-9)
check("volume is used as-is (contracts)", v.volume, 500)
check("tradeable", v.is_tradeable, True)
check("tier TRADEABLE", v.tier, "TRADEABLE")
# Same OI in units means far fewer lots when the lot size is bigger.
check("same OI at lot 25 -> 2,600 lots", assess(500, 65_000, 25).oi_lots, 2600.0, 1e-9)
check("lot_size missing -> oi_lots 0, flagged",
      (assess(500, 65_000, 0).oi_lots,
       any("lot_size missing" in r for r in assess(500, 65_000, 0).reasons)),
      (0.0, True))

print("\n" + "=" * 96)
print("TEST 3 -- THIN: it traded, but not enough to trust a 1-lot fill")
print("=" * 96)
v = assess(volume=5, open_interest=65_000, lot_size=65)
check("volume 5 -> price IS real", v.price_is_real, True)
check("volume 5 -> NOT tradeable", v.is_tradeable, False)
check("tier THIN", v.tier, "THIN")
check("  reason cites the volume floor",
      any(f"< {MIN_VOLUME_TRADE}" in r for r in v.reasons), True)
print("        -> this is the 5-10% OTM bucket, where the measured median")
print("           day's volume is ONE contract. That is where 0.10-delta legs sit.")

low_oi = assess(volume=500, open_interest=65 * 3, lot_size=65)
check("high volume but 3 lots OI -> not tradeable", low_oi.is_tradeable, False)
check("  reason cites open interest",
      any("open interest" in r for r in low_oi.reasons), True)
check(f"  boundary: exactly {MIN_OI_LOTS} lots OI passes",
      assess(500, 65 * MIN_OI_LOTS, 65).is_tradeable, True)

print("\n" + "=" * 96)
print("TEST 4 -- missing data is treated as illiquid, never as fine")
print("=" * 96)
check("volume None -> theoretical", assess(None, 100_000, 65).price_is_real, False)
check("OI None -> not tradeable", assess(500, None, 65).is_tradeable, False)
check("all None -> theoretical", assess(None, None, None).price_is_real, False)

print("\n" + "=" * 96)
print("TEST 5 -- a structure is only as liquid as its WORST leg")
print("=" * 96)
print("  The batch's calendars/diagonals buy a monthly hedge, and 50.9% of")
print("  >30 DTE rows never traded. Three good legs plus one dead one is a")
print("  structure you cannot actually put on.")
legs = [
    dict(label="short weekly CE", volume=5000, open_interest=650_000, lot_size=65),
    dict(label="short weekly PE", volume=4200, open_interest=520_000, lot_size=65),
    dict(label="long monthly CE", volume=800, open_interest=130_000, lot_size=65),
    dict(label="long monthly PE", volume=0, open_interest=65_000, lot_size=65),
]
s = assess_structure(legs)
check("4 legs assessed", s.n_legs, 4)
check("one leg theoretical", s.n_theoretical, 1)
check("three tradeable", s.n_tradeable, 3)
check("structure verdict is THEORETICAL", s.worst_tier, "THEORETICAL")
check("all_tradeable False", s.all_tradeable, False)
check("any_theoretical True", s.any_theoretical, True)
print("\n" + s.explain())

all_good = assess_structure(legs[:3])
check("\n  all-liquid structure -> TRADEABLE", all_good.worst_tier, "TRADEABLE")
check("  all_tradeable True", all_good.all_tradeable, True)
thin_mix = assess_structure(legs[:2] + [dict(label="thin", volume=5,
                                             open_interest=650_000, lot_size=65)])
check("  one thin leg -> structure THIN", thin_mix.worst_tier, "THIN")
check("  empty structure -> EMPTY", assess_structure([]).worst_tier, "EMPTY")

print("\n" + "=" * 96)
print("TEST 6 -- the Rs 2,000 cap forces very tight spreads at current lot size")
print("=" * 96)
check("lot 65, zero credit -> max width 30.77 pts",
      max_width_for_cap(65, 0.0), 2000 / 65, 1e-9)
check("lot 65, 20pt credit  -> max width 50.77 pts",
      max_width_for_cap(65, 20.0), 2000 / 65 + 20, 1e-9)
print(f"        lot 65, no credit : {max_width_for_cap(65, 0.0):.1f} pts")
print(f"        lot 75, no credit : {max_width_for_cap(75, 0.0):.1f} pts")
print(f"        lot 25, no credit : {max_width_for_cap(25, 0.0):.1f} pts")
print("        -> the SAME strategy was compliant at lot 25 and not at lot 75.")
print("           Any backtest hardcoding a lot size gets this wrong.")

check("max loss of a 100pt spread at 20 credit, lot 65",
      max_loss_of_spread(100, 20, 65), 80 * 65, 1e-9)
ok, ml = fits_cap(100, 20, 65)
check("  ...that is Rs 5,200, which FAILS the Rs 2,000 cap", (ok, ml), (False, 5200.0))
ok2, ml2 = fits_cap(50, 20, 65)
check("  a 50pt spread at 20 credit = Rs 1,950, PASSES", (ok2, ml2), (True, 1950.0))
check("credit exceeding width -> zero max loss, not negative",
      max_loss_of_spread(50, 80, 65), 0.0, 1e-9)
check("lot_size 0 -> nan, no ZeroDivisionError",
      math.isnan(max_width_for_cap(0, 10.0)), True)

print("\n" + "=" * 96)
print("TEST 7 -- SQL helper excludes theoretical rows")
print("=" * 96)
sql = liquid_chain_sql()
check("mentions volume", "volume" in sql, True)
check("guards against NULL", "IS NOT NULL" in sql, True)
check("threshold is parameterised", "volume >= 50" in liquid_chain_sql(50), True)

print("\n" + "=" * 96)
n_pass, n_fail = len(PASS), len(FAIL)
print(f"RESULT: {n_pass} passed, {n_fail} failed")
if FAIL:
    print("FAILURES:")
    for f in FAIL:
        print(f"   - {f}")
    raise SystemExit(1)
print("All checks green -- theoretical prices are rejected, units are handled")
print("correctly, and the cap arithmetic tracks the changing lot size.")
print("=" * 96)
