"""
analytics.py — the reusable quantitative brain.

Everything the vault needs to reason about options, none of it broker-specific:
Black-76 pricing, implied vol, Greeks, realised vol, the implied-vs-realised
variance risk premium (audit finding F3), max pain, and the market-implied
risk-neutral distribution via Breeden-Litzenberger (audit finding F9).

Pure functions, no I/O, no Kite dependency -> unit-testable and reusable in
backtests as well as the live daily snapshot.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

OptType = Literal["CE", "PE"]

TRADING_DAYS = 252
CAL_DAYS = 365


# --------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------
def year_fraction(now, expiry) -> float:
    """Calendar-time year fraction. Options decay in calendar time for carry
    and roughly trading time for vol; over <30d horizons the difference is
    immaterial and calendar time is the market convention."""
    return max((expiry - now).total_seconds() / (CAL_DAYS * 24 * 3600), 1e-8)


# --------------------------------------------------------------------------
# Black-76: options on the FORWARD, not spot.
# Audit finding F2 — the vault has always used spot. Options price off the
# forward, and the Nifty basis has been running ~9.5% annualised.
# --------------------------------------------------------------------------
def black76(F: float, K: float, T: float, vol: float, opt: OptType, r: float = 0.0575) -> float:
    if T <= 0 or vol <= 0:
        return max(0.0, (F - K) if opt == "CE" else (K - F)) * math.exp(-r * T)
    sq = vol * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * vol * vol * T) / sq
    d2 = d1 - sq
    df = math.exp(-r * T)
    if opt == "CE":
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


@dataclass
class Greeks:
    price: float
    delta: float      # per 1 point of forward, per share
    gamma: float
    vega: float       # per 1 vol POINT (not per 1.0 of vol), per share
    theta: float      # per calendar day, per share

    def scaled(self, qty: int) -> "Greeks":
        return Greeks(self.price, self.delta * qty, self.gamma * qty,
                      self.vega * qty, self.theta * qty)


def greeks(F: float, K: float, T: float, vol: float, opt: OptType, r: float = 0.0575) -> Greeks:
    if T <= 0 or vol <= 0:
        intrinsic = max(0.0, (F - K) if opt == "CE" else (K - F))
        return Greeks(intrinsic, 1.0 if intrinsic > 0 else 0.0, 0.0, 0.0, 0.0)
    sq = vol * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * vol * vol * T) / sq
    d2 = d1 - sq
    df = math.exp(-r * T)
    price = black76(F, K, T, vol, opt, r)
    delta = df * norm.cdf(d1) if opt == "CE" else -df * norm.cdf(-d1)
    gamma = df * norm.pdf(d1) / (F * sq)
    vega = df * F * norm.pdf(d1) * math.sqrt(T) / 100.0
    theta = (-df * F * norm.pdf(d1) * vol / (2 * math.sqrt(T))) / CAL_DAYS
    return Greeks(price, delta, gamma, vega, theta)


def implied_vol(mkt: float, F: float, K: float, T: float, opt: OptType,
                r: float = 0.0575) -> float:
    """Returns NaN when the quote is at/below intrinsic (stale or crossed)."""
    intrinsic = max(0.0, (F - K) if opt == "CE" else (K - F)) * math.exp(-r * T)
    if not (mkt > intrinsic + 1e-6) or T <= 0:
        return float("nan")
    try:
        return brentq(lambda v: black76(F, K, T, v, opt, r) - mkt, 1e-4, 5.0, xtol=1e-8)
    except (ValueError, RuntimeError):
        return float("nan")


def forward_from_parity(chain: dict[float, dict], T: float, r: float = 0.0575,
                        n_strikes: int = 4, atm: float | None = None) -> float:
    """F = K + (C-P)*e^{rT}, averaged over the strikes nearest ATM.
    Independent cross-check on the traded future."""
    ks = sorted(chain)
    if atm is None:
        atm = ks[len(ks) // 2]
    near = sorted(ks, key=lambda k: abs(k - atm))[:n_strikes]
    vals = [k + (chain[k]["CE"] - chain[k]["PE"]) * math.exp(r * T) for k in near]
    return float(np.mean(vals))


# --------------------------------------------------------------------------
# Volatility smile — fitted in z = 100*ln(K/F) so the normal equations are
# well conditioned, with FLAT extrapolation beyond the quoted strikes.
# (An earlier un-scaled fit blew up in the tails; see the audit's method note.)
# --------------------------------------------------------------------------
@dataclass
class Smile:
    c0: float
    c1: float
    c2: float
    z_min: float
    z_max: float
    F: float
    max_fit_error: float

    def iv(self, K: float, F: float | None = None) -> float:
        F = self.F if F is None else F
        z = float(np.clip(100.0 * math.log(K / F), self.z_min, self.z_max))
        return max(0.03, self.c0 + self.c1 * z + self.c2 * z * z)


def fit_smile(chain: dict[float, dict], F: float, T: float, r: float = 0.0575) -> Smile:
    """Fit the OTM wing of each side (the liquid, informative quotes)."""
    zs, vs = [], []
    for K in sorted(chain):
        opt: OptType = "CE" if K >= F else "PE"
        v = implied_vol(chain[K][opt], F, K, T, opt, r)
        if math.isfinite(v):
            zs.append(100.0 * math.log(K / F))
            vs.append(v)
    z = np.asarray(zs)
    v = np.asarray(vs)
    c2, c1, c0 = np.polyfit(z, v, 2)
    sm = Smile(c0, c1, c2, float(z.min()), float(z.max()), F, 0.0)
    sm.max_fit_error = float(np.max(np.abs([sm.iv(K, F) - vv for K, vv
                                            in zip(sorted(chain)[:len(v)], v)]))) if len(v) else 0.0
    # recompute error properly against the same strikes used in the fit
    errs = []
    for K in sorted(chain):
        opt = "CE" if K >= F else "PE"
        mv = implied_vol(chain[K][opt], F, K, T, opt, r)
        if math.isfinite(mv):
            errs.append(abs(sm.iv(K, F) - mv))
    sm.max_fit_error = float(max(errs)) if errs else 0.0
    return sm


# --------------------------------------------------------------------------
# Realised vol + the variance risk premium (audit finding F3)
# --------------------------------------------------------------------------
def realised_vol(closes: Sequence[float], window: int) -> float:
    c = np.asarray(closes[-(window + 1):], dtype=float)
    if len(c) < 3:
        return float("nan")
    lr = np.diff(np.log(c))
    return float(lr.std(ddof=1) * math.sqrt(TRADING_DAYS))


def variance_risk_premium(atm_iv: float, closes: Sequence[float], window: int = 20) -> dict:
    """THE missing filter. Positive => you are being paid to sell premium."""
    rv = realised_vol(closes, window)
    vrp = atm_iv - rv
    return {
        "atm_iv": atm_iv,
        f"rv{window}": rv,
        "vrp_vol_points": vrp * 100,
        "sell_premium_ok": bool(vrp * 100 >= 1.0),   # audit F3 recommended gate
        "verdict": ("PAID to sell vol" if vrp * 100 >= 1.0 else
                    "NOT paid — implied is not meaningfully above realised. Do not sell premium."),
    }


# --------------------------------------------------------------------------
# Open interest
# --------------------------------------------------------------------------
def max_pain(oi_chain: dict[float, dict]) -> dict:
    ks = sorted(oi_chain)
    payouts = {}
    for s in ks:
        payouts[s] = sum(max(0.0, s - k) * oi_chain[k]["CE_OI"] +
                         max(0.0, k - s) * oi_chain[k]["PE_OI"] for k in ks)
    mp = min(payouts, key=payouts.get)
    tot_ce = sum(oi_chain[k]["CE_OI"] for k in ks)
    tot_pe = sum(oi_chain[k]["PE_OI"] for k in ks)
    return {
        "max_pain": mp,
        "pcr_oi": tot_pe / tot_ce if tot_ce else float("nan"),
        "top_call_oi": sorted(ks, key=lambda k: -oi_chain[k]["CE_OI"])[:3],
        "top_put_oi": sorted(ks, key=lambda k: -oi_chain[k]["PE_OI"])[:3],
        "payouts": payouts,
    }


# --------------------------------------------------------------------------
# Market-implied risk-neutral distribution (Breeden-Litzenberger).
# Reads probabilities out of live prices instead of assuming lognormal, so it
# captures the put skew. Audit finding F9.
# --------------------------------------------------------------------------
def implied_distribution(smile: Smile, F: float, T: float, r: float = 0.0575,
                         lo: float | None = None, hi: float | None = None,
                         step: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    lo = lo or F * 0.90
    hi = hi or F * 1.10
    ks = np.arange(lo, hi + step, step)
    h = step
    c = np.array([black76(F, k, T, smile.iv(k, F), "CE", r) for k in
                  np.concatenate([ks - h, ks, ks + h])])
    n = len(ks)
    cm, c0, cp = c[:n], c[n:2 * n], c[2 * n:]
    pdf = np.maximum(0.0, (cp - 2 * c0 + cm) / (h * h) / math.exp(-r * T))
    z = pdf.sum() * step
    return ks, (pdf / z if z > 0 else pdf)


def zone_probabilities(ks: np.ndarray, pdf: np.ndarray, bounds: Sequence[float]) -> list[float]:
    """P of falling in each interval defined by bounds = [-inf, b1, b2, ..., +inf]."""
    step = float(ks[1] - ks[0])
    edges = [-np.inf, *bounds, np.inf]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        out.append(float(pdf[(ks > a) & (ks <= b)].sum() * step))
    return out


def percentile(ks: np.ndarray, pdf: np.ndarray, q: float) -> float:
    step = float(ks[1] - ks[0])
    cum = np.cumsum(pdf * step)
    idx = int(np.searchsorted(cum, q))
    return float(ks[min(idx, len(ks) - 1)])


# --------------------------------------------------------------------------
# Multi-leg position analytics
# --------------------------------------------------------------------------
@dataclass
class Leg:
    strike: float
    opt: OptType
    qty: int            # signed, in LOTS (+long / -short)
    entry: float        # per share
    lot_size: int = 65

    @property
    def shares(self) -> int:
        return self.qty * self.lot_size


@dataclass
class PositionAnalysis:
    mark: float
    frozen_expiry_value: float
    theta_remaining: float
    delta: float
    gamma: float
    vega: float
    theta: float
    breakevens: tuple[float, float] | None
    max_profit: float
    max_loss: float


def expiry_pnl(legs: Sequence[Leg], S: float) -> float:
    total = 0.0
    for lg in legs:
        payoff = max(0.0, S - lg.strike) if lg.opt == "CE" else max(0.0, lg.strike - S)
        total += lg.shares * (payoff - lg.entry)
    return total


def analyse_position(legs: Sequence[Leg], spot: float, F: float, T: float,
                     smile: Smile, leg_ivs: dict[str, float] | None = None,
                     r: float = 0.0575) -> PositionAnalysis:
    """leg_ivs keyed '24600PE' -> use each leg's OWN market IV where available.
    Using the OTM-side IV for an ITM leg misprices it; on a real butterfly that
    error was Rs 485. This is why the key exists."""
    basis = F - spot
    mark = d = g = v = th = 0.0
    for lg in legs:
        key = f"{int(lg.strike)}{lg.opt}"
        iv = (leg_ivs or {}).get(key) or smile.iv(lg.strike, F)
        gk = greeks(F, lg.strike, T, iv, lg.opt, r)
        mark += lg.shares * (gk.price - lg.entry)
        d += lg.shares * gk.delta
        g += lg.shares * gk.gamma
        v += lg.shares * gk.vega
        th += lg.shares * gk.theta

    frozen = expiry_pnl(legs, spot)

    # The expiry payoff is piecewise linear with kinks EXACTLY at the strikes,
    # so its extrema are always at a strike or at the ends. A plain arange grid
    # can straddle a strike and miss the true max by a few hundred rupees --
    # include every strike explicitly.
    strikes = sorted({lg.strike for lg in legs})
    grid = np.unique(np.concatenate([
        np.arange(spot * 0.85, spot * 1.15, 1.0),
        np.asarray(strikes, dtype=float),
    ]))
    pnl = np.array([expiry_pnl(legs, s) for s in grid])

    # Exact breakevens by linear interpolation across each sign change, rather
    # than snapping to whichever grid point happened to be nearest.
    bes_list: list[float] = []
    sign = np.sign(pnl)
    for i in np.where(np.diff(sign) != 0)[0]:
        x0, x1, y0, y1 = grid[i], grid[i + 1], pnl[i], pnl[i + 1]
        bes_list.append(float(x0 if y1 == y0 else x0 - y0 * (x1 - x0) / (y1 - y0)))
    bes = (bes_list[0], bes_list[-1]) if len(bes_list) >= 2 else None

    return PositionAnalysis(
        mark=mark,
        frozen_expiry_value=frozen,
        theta_remaining=frozen - mark,     # THE number: what is left from sitting still
        delta=d, gamma=g, vega=v, theta=th,
        breakevens=bes,
        max_profit=float(pnl.max()),
        max_loss=float(pnl.min()),
    )


def prob_touch(barrier: float, spot: float, vol: float, T: float) -> float:
    """Driftless reflection-principle approximation. Used to sanity-check
    whether a 'stop' is actually a stop or is ~certain to fire (audit F1)."""
    if T <= 0 or vol <= 0:
        return 0.0
    s = vol * math.sqrt(T)
    return float(min(1.0, 2 * norm.cdf(-abs(math.log(barrier / spot)) / s)))
