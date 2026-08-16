"""
liquidity.py — decide whether an option row's price can be trusted or traded.

WHY THIS EXISTS (measured on this vault's own data, 2026-08-15, 1,059,039 NIFTY
option rows spanning 2024-01-01 .. 2026-08-13):

    42.6% of NIFTY option rows have volume = 0.

    Those contracts did not trade at all that day. NSE still publishes a close
    and a settlement price for them -- a THEORETICAL value from its own model.
    Crucially, `close < 0.05` occurs in 0.0% of rows and `close = 0` in 0.0%:
    the price always LOOKS valid. Nothing in the row marks it as untraded
    except volume.

    Deriving implied vol and delta from a theoretical price and then reporting
    the result as a backtest outcome is how a strategy gets validated on
    numbers that never existed. This module exists to prevent that.

WHERE THE ILLIQUIDITY SITS -- it is exactly where these strategies trade:

    distance from spot   untraded    median volume (contracts)
      ATM +-1%             14.1%          897
      1-2%                 20.5%          362
      2-3%                 27.8%          127
      3-5%                 36.7%           19
      5-10%                47.6%            1
      >10%                 66.6%            0

    days to expiry       untraded
      0-1 DTE              12.0%
      2-7 DTE              14.2%
      8-30 DTE             34.7%
      >30 DTE              50.9%     <-- every calendar/diagonal hedge leg

    Two consequences for the strategy batch under validation:
      1. 0.10-delta legs sit ~5% OTM, where roughly half the rows never traded
         and the median day's volume is ONE contract.
      2. Half of all >30 DTE rows never traded. Every calendar and diagonal
         strategy in the batch buys a monthly hedge leg in exactly that bucket.

UNITS -- verified empirically, do not assume:
    volume        = CONTRACTS  (only 1.6% of rows divide evenly by lot_size)
    open_interest = UNITS      (99.8% of rows divide evenly by lot_size)
    So OI in lots = open_interest / lot_size.

LOT SIZE IS NOT CONSTANT. Measured from bhavcopy across the window:
    2024-01..03 = 50 | 2024-04..11 = 25 | 2024-12..2025-09 = 75 | 2025-10.. = 65
    A "1 lot" position therefore means up to 3x different rupee exposure
    depending on the date. Always read lot_size from the row, never hardcode.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# thresholds
#
# Chosen from the measured distribution above, not invented. Among rows that
# DID trade, volume percentiles are p25=38, p50=505, p75=7,921 contracts.
#   - MIN_VOLUME_TRUST: a price is only a real price if something traded.
#     This is a hard gate, not a preference.
#   - MIN_VOLUME_TRADE: 100 contracts sits near the 40th percentile of traded
#     rows. Below it, a 1-lot retail order is a material share of the day's
#     flow and the close is unlikely to be reachable in practice.
#   - MIN_OI_LOTS: 10 lots of open interest means the strike genuinely exists
#     in the book rather than being a stray print.
# All are overridable -- they are defaults with a rationale, not constants of
# nature. Tighten them and fewer strategies will survive; that is the point.
# --------------------------------------------------------------------------

MIN_VOLUME_TRUST = 1        # contracts; below this the price is theoretical
MIN_VOLUME_TRADE = 100      # contracts; below this a 1-lot fill is optimistic
MIN_OI_LOTS = 10            # lots of open interest


@dataclass(frozen=True)
class Verdict:
    """Three-state, because 'tradeable' and 'trustworthy' are different questions."""
    price_is_real: bool     # did it trade at all? if False, IV/delta are model output
    is_tradeable: bool      # could a 1-lot order realistically fill near this price?
    volume: int
    oi_lots: float
    reasons: tuple[str, ...] = ()

    @property
    def tier(self) -> str:
        if not self.price_is_real:
            return "THEORETICAL"
        return "TRADEABLE" if self.is_tradeable else "THIN"


def assess(volume: int | None, open_interest: int | None, lot_size: int | None,
           min_volume_trust: int = MIN_VOLUME_TRUST,
           min_volume_trade: int = MIN_VOLUME_TRADE,
           min_oi_lots: float = MIN_OI_LOTS) -> Verdict:
    """Classify one option row's liquidity.

    Missing data is treated as illiquid, never as fine. A NULL volume is not
    evidence of trading.
    """
    v = int(volume or 0)
    oi_units = int(open_interest or 0)
    ls = int(lot_size) if lot_size else 0
    oi_lots = (oi_units / ls) if ls else 0.0

    reasons: list[str] = []
    price_is_real = v >= min_volume_trust
    if not price_is_real:
        reasons.append(
            "volume=0: close/settle is NSE's theoretical price, not a traded one")
    if ls == 0:
        reasons.append("lot_size missing -- cannot express OI in lots")

    tradeable = price_is_real
    if price_is_real and v < min_volume_trade:
        tradeable = False
        reasons.append(f"volume {v} < {min_volume_trade} contracts: a 1-lot order is "
                       f"a material share of the day's flow")
    if price_is_real and oi_lots < min_oi_lots:
        tradeable = False
        reasons.append(f"open interest {oi_lots:.1f} lots < {min_oi_lots}")

    return Verdict(price_is_real=price_is_real, is_tradeable=tradeable,
                   volume=v, oi_lots=oi_lots, reasons=tuple(reasons))


# --------------------------------------------------------------------------
# structure-level assessment
# --------------------------------------------------------------------------

@dataclass
class StructureLiquidity:
    """A multi-leg structure is only as liquid as its WORST leg.

    This is the operative point for the strategies under validation: a
    double diagonal can have three perfectly liquid legs and one monthly hedge
    that never traded. The structure is not tradeable, and its backtested P&L
    is partly fiction, because of that one leg.
    """
    n_legs: int = 0
    n_theoretical: int = 0
    n_thin: int = 0
    n_tradeable: int = 0
    per_leg: list[Verdict] = field(default_factory=list)
    leg_labels: list[str] = field(default_factory=list)

    @property
    def any_theoretical(self) -> bool:
        return self.n_theoretical > 0

    @property
    def all_tradeable(self) -> bool:
        return self.n_legs > 0 and self.n_tradeable == self.n_legs

    @property
    def worst_tier(self) -> str:
        if self.n_theoretical:
            return "THEORETICAL"
        if self.n_thin:
            return "THIN"
        return "TRADEABLE" if self.n_legs else "EMPTY"

    def explain(self) -> str:
        head = (f"structure: {self.n_legs} legs -> {self.worst_tier} "
                f"({self.n_tradeable} tradeable, {self.n_thin} thin, "
                f"{self.n_theoretical} theoretical)")
        out = [head]
        for label, v in zip(self.leg_labels, self.per_leg):
            out.append(f"    {label:<28} {v.tier:<12} vol={v.volume:<8,} "
                       f"oi={v.oi_lots:,.0f} lots")
            for rsn in v.reasons:
                out.append(f"        - {rsn}")
        return "\n".join(out)


def assess_structure(legs: list[dict], **kw) -> StructureLiquidity:
    """legs: [{label, volume, open_interest, lot_size}, ...]"""
    s = StructureLiquidity(n_legs=len(legs))
    for leg in legs:
        v = assess(leg.get("volume"), leg.get("open_interest"),
                   leg.get("lot_size"), **kw)
        s.per_leg.append(v)
        s.leg_labels.append(str(leg.get("label", "?")))
        if not v.price_is_real:
            s.n_theoretical += 1
        elif not v.is_tradeable:
            s.n_thin += 1
        else:
            s.n_tradeable += 1
    return s


# --------------------------------------------------------------------------
# sizing arithmetic that the lot-size change makes non-obvious
# --------------------------------------------------------------------------

def max_width_for_cap(lot_size: int, net_credit_points: float,
                      risk_cap: float = 2000.0) -> float:
    """Widest spread (in index points) whose max loss fits the per-trade cap.

        max_loss = (width - net_credit) * lot_size <= risk_cap
        width <= risk_cap/lot_size + net_credit

    WHY THIS MATTERS MORE THAN IT LOOKS: at lot 65 and zero credit the widest
    compliant spread is 2000/65 = 30.8 points. Most strategies in the batch
    under validation use 100-500 point wings. They cannot be sized to the
    Rs 2,000 cap at 1 lot -- which is the same conclusion the triage reached
    from margin, arrived at independently from risk.

    And because lot_size ran 50 -> 25 -> 75 -> 65 across the backtest window,
    the SAME strategy was compliant at some dates and not at others. Any
    backtest that hardcodes a lot size will get this wrong.
    """
    if lot_size <= 0:
        return float("nan")
    return risk_cap / lot_size + net_credit_points


def max_loss_of_spread(width_points: float, net_credit_points: float,
                       lot_size: int) -> float:
    return max(0.0, (width_points - net_credit_points)) * lot_size


def fits_cap(width_points: float, net_credit_points: float, lot_size: int,
             risk_cap: float = 2000.0) -> tuple[bool, float]:
    ml = max_loss_of_spread(width_points, net_credit_points, lot_size)
    return ml <= risk_cap, ml


# --------------------------------------------------------------------------
# SQL helper
# --------------------------------------------------------------------------

LIQUIDITY_COLUMNS = "volume, open_interest, lot_size"


def liquid_chain_sql(min_volume: int = MIN_VOLUME_TRUST) -> str:
    """Predicate for chain queries that should only see genuinely traded rows.

    backtest.py's _get_chain currently filters on `close IS NOT NULL` only,
    which admits every theoretical price. Append this to that WHERE clause to
    exclude them.
    """
    return f"AND volume IS NOT NULL AND volume >= {int(min_volume)}"
