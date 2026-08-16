"""
test_analytics.py — verification against REAL market data, not synthetic fixtures.

Every expected value below is a live Kite observation from 2026-08-13 12:47-12:51
IST, or an independently derived figure from the analysis in
[[2026-08-13 Nifty Forward Analysis to 25-Aug Expiry]]. If any of these break,
the analytics module is wrong and the daily snapshot cannot be trusted.

Run:  python test_analytics.py
"""
import datetime as dt
import math

import numpy as np

from analytics import (Leg, analyse_position, expiry_pnl, fit_smile,
                       forward_from_parity, greeks, implied_distribution,
                       implied_vol, max_pain, percentile, prob_touch,
                       realised_vol, variance_risk_premium, zone_probabilities)
from costs import cost_as_r_multiple, option_trade_cost, passes_cost_gate

# ------------------------------------------------------------------ fixtures
SPOT = 24380.30
FUT = 24456.90
T = (12 + 2.7 / 24) / 365
LOT = 65

CHAIN = {   # strike: {CE, PE, CE_OI, PE_OI}   (Kite live, 2026-08-13 12:48 IST)
    23800: dict(CE=695.50, PE=22.00,  CE_OI=498225,  PE_OI=2487680),
    23900: dict(CE=597.85, PE=30.35,  CE_OI=145600,  PE_OI=1235455),
    24000: dict(CE=507.30, PE=42.95,  CE_OI=5682170, PE_OI=9200360),
    24100: dict(CE=424.90, PE=59.80,  CE_OI=1414985, PE_OI=2626195),
    24200: dict(CE=347.40, PE=82.45,  CE_OI=1320475, PE_OI=3075540),
    24300: dict(CE=276.00, PE=111.85, CE_OI=1672840, PE_OI=2581345),
    24400: dict(CE=213.50, PE=149.20, CE_OI=2370680, PE_OI=2616575),
    24500: dict(CE=160.80, PE=197.00, CE_OI=7278570, PE_OI=5951400),
    24600: dict(CE=116.05, PE=251.40, CE_OI=2788240, PE_OI=1589705),
    24700: dict(CE=80.60,  PE=315.50, CE_OI=2265835, PE_OI=1059565),
    24800: dict(CE=54.50,  PE=387.95, CE_OI=3146975, PE_OI=567125),
    24900: dict(CE=35.95,  PE=462.50, CE_OI=1797510, PE_OI=185120),
    25000: dict(CE=23.70,  PE=555.20, CE_OI=8101600, PE_OI=2098850),
    25100: dict(CE=14.80,  PE=633.25, CE_OI=1911325, PE_OI=63895),
    25200: dict(CE=9.75,   PE=719.75, CE_OI=1838525, PE_OI=114400),
}

CLOSES = [24102.9, 23824.1, 24021.65, 24056, 23946.25, 23865.75, 24005.85, 24175.7,
          24270.85, 24430.35, 24398.7, 23882.05, 23962.8, 24206.9, 24211, 24052.05,
          24078.5, 24072.75, 24334.3, 24238.5, 24187.7, 23996.25, 23869.6, 23767.45,
          23995.95, 23985.35, 24250.2, 24317.15, 24383.6, 24774.3, 24614.9, 24624.65,
          24636, 24570.65, 24583.8, 24471.7, 24435.95, 24379.7]

LEGS = [Leg(24300, "PE",  1, 127.75, LOT),
        Leg(24600, "PE", -1, 228.55, LOT),
        Leg(24600, "CE", -1, 285.75, LOT),
        Leg(24900, "CE",  1, 142.15, LOT)]

PASS, FAIL = [], []


def check(name, got, want, tol, unit=""):
    ok = abs(got - want) <= tol
    (PASS if ok else FAIL).append(name)
    flag = "PASS" if ok else "**FAIL**"
    print(f"  [{flag}] {name:52s} got {got:>12,.4f}{unit}  want {want:>12,.4f} (+/-{tol}{unit})")


print("=" * 96)
print("TEST 1 — forward from put-call parity should reproduce the traded future")
print("=" * 96)
f_pcp = forward_from_parity(CHAIN, T, atm=24450)
check("forward from parity vs traded future 24456.90", f_pcp, FUT, 10.0, " pts")

print("\n" + "=" * 96)
print("TEST 2 — implied vol round-trips through the pricer")
print("=" * 96)
for K, opt in [(24600, "CE"), (24600, "PE"), (24300, "PE"), (24900, "CE"), (24000, "PE")]:
    mkt = CHAIN[K][opt]
    iv = implied_vol(mkt, FUT, K, T, opt)
    back = greeks(FUT, K, T, iv, opt).price
    check(f"reprice {K}{opt} from its own IV ({iv*100:.2f}%)", back, mkt, 0.02, "")

print("\n" + "=" * 96)
print("TEST 3 — smile fit quality (must be tight or nothing downstream is trustworthy)")
print("=" * 96)
sm = fit_smile(CHAIN, FUT, T)
print(f"  IV(z) = {sm.c0*100:.3f}% + {sm.c1*100:.4f}%*z + {sm.c2*100:.5f}%*z^2   "
      f"[z=100*ln(K/F)], flat outside z in [{sm.z_min:.2f},{sm.z_max:.2f}]")
check("max smile fit error", sm.max_fit_error * 100, 0.0, 0.60, " vol pts")
atm_iv = sm.iv(FUT, FUT)
check("ATM implied vol", atm_iv * 100, 10.03, 0.30, "%")

print("\n" + "=" * 96)
print("TEST 4 — realised vol + the variance risk premium (audit finding F3)")
print("=" * 96)
check("RV(20d)", realised_vol(CLOSES, 20) * 100, 10.28, 0.40, "%")
check("RV(5d)  — the extraordinary one", realised_vol(CLOSES, 5) * 100, 2.95, 0.60, "%")
vrp = variance_risk_premium(atm_iv, CLOSES, 20)
print(f"  VRP = {vrp['vrp_vol_points']:+.2f} vol pts -> {vrp['verdict']}")
assert vrp["sell_premium_ok"] is False, "gate should REJECT selling premium here"
PASS.append("VRP gate correctly rejects selling premium")
print("  [PASS] VRP gate correctly rejects selling premium at IV<RV")

print("\n" + "=" * 96)
print("TEST 5 — max pain and OI structure")
print("=" * 96)
mp = max_pain(CHAIN)
check("max pain strike", mp["max_pain"], 24400, 0, "")
check("overall PCR (OI)", mp["pcr_oi"], 0.839, 0.01, "")
print(f"  heaviest CALL OI (resistance) {mp['top_call_oi']}   "
      f"heaviest PUT OI (support) {mp['top_put_oi']}")
assert mp["top_put_oi"][0] == 24000 and mp["top_call_oi"][0] == 25000
PASS.append("OI walls identified")
print("  [PASS] OI walls identified: 24000 put wall / 25000 call wall")

print("\n" + "=" * 96)
print("TEST 6 — position analytics must reproduce the market mark to the rupee")
print("=" * 96)
leg_ivs = {f"{K}{o}": implied_vol(CHAIN[K][o], FUT, K, T, o)
           for K, o in [(24300, "PE"), (24600, "PE"), (24600, "CE"), (24900, "CE")]}
pa = analyse_position(LEGS, SPOT, FUT, T, sm, leg_ivs)
check("mark-to-market vs Kite LTP sum", pa.mark, 1609.0, 25.0, "")
check("max profit at expiry", pa.max_profit, 15886.0, 5.0, "")
check("max loss at expiry", pa.max_loss, -3614.0, 5.0, "")
check("lower breakeven (exact, interpolated)", pa.breakevens[0], 24355.60, 0.05, "")
check("upper breakeven (exact, interpolated)", pa.breakevens[1], 24844.40, 0.05, "")
check("spot-frozen expiry value", pa.frozen_expiry_value, 1605.0, 20.0, "")
check("THETA REMAINING (the decisive number)", pa.theta_remaining, 0.0, 30.0, "")
print(f"  Greeks: delta {pa.delta:+.2f}/pt  gamma {pa.gamma:+.4f}  "
      f"vega {pa.vega:+.0f}/volpt  theta {pa.theta:+.0f}/day")
assert pa.gamma < 0 and pa.vega < 0 and pa.theta > 0
PASS.append("Greek signs correct for a short butterfly")
print("  [PASS] Greek signs correct (short gamma, short vega, positive theta)")

print("\n" + "=" * 96)
print("TEST 7 — expiry payoff at known points")
print("=" * 96)
for S, want in [(24600, 15886.0), (24300, -3614.0), (24900, -3614.0),
                (24500, 9386.0), (24400, 2886.0)]:
    check(f"expiry P&L at {S}", expiry_pnl(LEGS, S), want, 1.0, "")

print("\n" + "=" * 96)
print("TEST 8 — market-implied distribution (Breeden-Litzenberger)")
print("=" * 96)
ks, pdf = implied_distribution(sm, FUT, T, lo=22000, hi=27000, step=10.0)
check("density integrates to 1", float(pdf.sum() * 10.0), 1.0, 0.01, "")
zones = zone_probabilities(ks, pdf, [24300, 24355.60, 24844.40, 24900])
labels = ["<=24300 max loss", "24300-24355.6", "24355.6-24844.4 PROFIT",
          "24844.4-24900", ">=24900 max loss"]
for lbl, p in zip(labels, zones):
    print(f"    P({lbl:28s}) = {p*100:5.1f}%")
check("P(profit zone)", zones[2] * 100, 41.3, 3.0, "%")
check("implied median", percentile(ks, pdf, 0.50), 24460, 60, "")

print("\n" + "=" * 96)
print("TEST 9 — prob_touch exposes the broken breakeven rule (audit finding F1)")
print("=" * 96)
pt = prob_touch(24355.60, SPOT, atm_iv, T)
check("P(touch lower breakeven before expiry)", pt * 100, 95.6, 3.0, "%")
print("  -> a 'stop' with a 95% chance of firing is not a risk control. This is F1, quantified.")

print("\n" + "=" * 96)
print("TEST 10 — cost model reproduces the Rs 119 close-out")
print("=" * 96)
buy_prem = (251.70 + 116.05) * LOT     # buying back the two shorts
sell_prem = (111.65 + 35.90) * LOT     # selling the two longs
cb = option_trade_cost(buy_prem, sell_prem, n_orders=4)
print(f"  {cb}")
# Expected value updated 2026-08-13 from 119 -> 128.
# TWO corrections, both validated against the real broker statement:
#   1. STT on option sells is 0.15% from 2026-04-01 (Budget 2026), not 0.10%.
#      The old expected value was encoding that bug.
#   2. Brokerage carries a 1.05 fill factor (249 billable orders vs 237 order_ids).
# Cross-check: the statement's actual STT of Rs 1,823 sits between the all-0.10%
# bound (Rs 1,403) and the all-0.15% bound (Rs 2,105), implying ~60% of sell
# premium fell after the hike date. Consistent with the Jan-Aug trade span.
check("cost to close 4 legs (post-Apr-2026 STT)", cb.total, 128.0, 3.0, "")
check("cost as fraction of a Rs 4,000 risk unit",
      cost_as_r_multiple(410, 4000) * 100, 10.25, 0.2, "%")
ok, why = passes_cost_gate(expected_profit_at_target=15886 * 0.25, round_trip=cb.total * 2)
print(f"  cost gate on a 25%-of-credit target: {why}")

print("\n" + "=" * 96)
n_pass, n_fail = len(PASS), len(FAIL)
print(f"RESULT: {n_pass} passed, {n_fail} failed")
if FAIL:
    print("FAILURES:")
    for f in FAIL:
        print(f"   - {f}")
    raise SystemExit(1)
print("All checks green — analytics reproduces live market data and the manual analysis.")
print("=" * 96)
