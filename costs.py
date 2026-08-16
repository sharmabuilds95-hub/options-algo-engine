"""
costs.py — Zerodha transaction cost model.

Audit finding F8: the vault measures costs but never *gates* on them. The
measured average was Rs 410/trade = 10.3% of a Rs 4,000 risk unit, i.e. every
trade must clear +0.10R before it earns anything.

Rates current as of 2026-08-13. Each is dated so a stale rate is visible.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict

# --- CALIBRATION AGAINST THE REAL BROKER STATEMENT (2026-08-13) -----------
# Validated against Zerodha's own F&O P&L for 2026-01-22 -> 2026-08-03:
#     actual charges Rs 8,899.85   |   this model predicted Rs 8,189   (8.0% low)
# Two causes, both corrected below:
#   1. STT: exercised ITM options pay 0.125% on INTRINSIC value at expiry. That is
#      STT_OPT_EXERCISE, which was in the model but was not being applied to expiring
#      positions. Use expiry_cost() for anything held to expiry -- it is NOT optional.
#   2. Brokerage: billable order count (249) exceeded distinct order_ids (237). Partial
#      fills across orders bill separately, so brokerage estimated from order_id count
#      runs ~5% low. BROKERAGE_FILL_FACTOR corrects for it.
BROKERAGE_FILL_FACTOR = 1.05     # measured 249/237 on the real statement

# --- rate card (verify annually; each line dated) -------------------------
BROKERAGE_PER_ORDER = 20.00      # Zerodha F&O flat, min(0.03%, Rs 20). 2026-08-13
# --- STT: RATE CHANGED 2026-04-01 (Budget 2026). Verified 2026-08-13. -----
# Options SELL : 0.10%  -> 0.15% of premium
# Options EXER : 0.125% -> 0.15% of intrinsic
# The previous values in this file were the PRE-April-2026 rates and understated
# STT by 50%. Historical backtests spanning 1-Apr-2026 MUST use the dated helper
# below, not the flat constant, or costs before/after the change are wrong.
STT_OPT_SELL        = 0.001500   # 0.15% of premium, SELL side only (from 2026-04-01)
STT_OPT_EXERCISE    = 0.001500   # 0.15% of INTRINSIC on exercised ITM longs (from 2026-04-01)
STT_OPT_SELL_PRE_APR2026     = 0.001000
STT_OPT_EXERCISE_PRE_APR2026 = 0.001250
STT_HIKE_DATE = "2026-04-01"


def stt_rates_on(trade_date: str) -> tuple[float, float]:
    """(sell_rate, exercise_rate) applicable on a given YYYY-MM-DD. Use this in
    backtests; the module constants are the CURRENT rates only."""
    if trade_date < STT_HIKE_DATE:
        return STT_OPT_SELL_PRE_APR2026, STT_OPT_EXERCISE_PRE_APR2026
    return STT_OPT_SELL, STT_OPT_EXERCISE
TXN_CHARGE_NSE_OPT  = 0.00035030 # NSE options, on premium turnover. 2026-08-13
SEBI_CHARGE         = 0.00000100 # Rs 10 per crore
STAMP_DUTY_BUY      = 0.00003000 # 0.003% on buy-side premium
GST_RATE            = 0.18       # on brokerage + txn + SEBI


@dataclass
class CostBreakdown:
    brokerage: float
    stt: float
    txn: float
    sebi: float
    stamp: float
    gst: float
    total: float

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (f"Rs {self.total:,.2f}  [brk {self.brokerage:,.0f} + STT {self.stt:,.0f} "
                f"+ txn {self.txn:,.0f} + stamp {self.stamp:,.0f} + GST {self.gst:,.0f}]")


def option_trade_cost(buy_premium_value: float, sell_premium_value: float,
                      n_orders: int) -> CostBreakdown:
    """One direction of a multi-leg option trade.

    buy_premium_value  = sum(price * shares) for every leg BOUGHT
    sell_premium_value = sum(price * shares) for every leg SOLD
    n_orders           = executed orders (leg count for this direction)
    """
    turnover = buy_premium_value + sell_premium_value
    brokerage = BROKERAGE_PER_ORDER * n_orders * BROKERAGE_FILL_FACTOR
    stt = STT_OPT_SELL * sell_premium_value
    txn = TXN_CHARGE_NSE_OPT * turnover
    sebi = SEBI_CHARGE * turnover
    stamp = STAMP_DUTY_BUY * buy_premium_value
    gst = GST_RATE * (brokerage + txn + sebi)
    return CostBreakdown(brokerage, stt, txn, sebi, stamp, gst,
                         brokerage + stt + txn + sebi + stamp + gst)


def round_trip_cost(open_buy: float, open_sell: float,
                    close_buy: float, close_sell: float, n_legs: int) -> CostBreakdown:
    a = option_trade_cost(open_buy, open_sell, n_legs)
    b = option_trade_cost(close_buy, close_sell, n_legs)
    return CostBreakdown(*[x + y for x, y in
                           zip(a.as_dict().values(), b.as_dict().values())])


def expiry_cost(itm_long_intrinsic_value: float) -> float:
    """Held to expiry: no brokerage. STT 0.125% on intrinsic of exercised LONG
    legs only. Short legs assigned pay nothing. Usually far cheaper than closing
    -- which is exactly why a 'time stop' should not be dismissed on cost grounds."""
    return STT_OPT_EXERCISE * max(0.0, itm_long_intrinsic_value)


# --- the gate the audit asked for (F8) ------------------------------------
def passes_cost_gate(expected_profit_at_target: float, round_trip: float,
                     multiple: float = 5.0) -> tuple[bool, str]:
    """Reject trades too small to survive their own costs."""
    if round_trip <= 0:
        return True, "no cost modelled"
    ratio = expected_profit_at_target / round_trip
    ok = ratio >= multiple
    return ok, (f"expected profit Rs {expected_profit_at_target:,.0f} is {ratio:.1f}x "
                f"round-trip cost Rs {round_trip:,.0f} "
                f"({'PASS' if ok else f'FAIL — needs >= {multiple:.0f}x'})")


def cost_as_r_multiple(total_cost: float, risk_per_trade: float) -> float:
    """How much edge each trade must clear before earning anything."""
    return total_cost / risk_per_trade if risk_per_trade else float("nan")
