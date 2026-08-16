"""
regime.py -- tag every trading day with the market state it belonged to.

WHY THIS EXISTS
    Only ~21% of days in the option-data window sit in VIX 16-25. So
    "this strategy works" and "this strategy works IN REGIME" are different
    claims, and a backtest that reports only the first is not reporting a
    result -- it is reporting an average over regimes the strategy was never
    designed for. score.py already buckets on VIX AT ENTRY; this module is the
    thing that supplies those buckets, adds two more axes (trend, vol
    direction), and -- most importantly -- reports HOW MANY DAYS sit behind
    each bucket so nobody quotes a regime-conditioned number off n=7.

THREE AXES, and why each one

    1. VOLATILITY BAND   -- VIX <13 / 13-16 / 16-25 / >25.
       Same four bands score.py already uses. Kept identical on purpose:
       VIX_BANDS below is asserted equal to score.score_by_regime's defaults
       in test_regime.py, so the two can never silently drift apart.

    2. TREND             -- trending-up / trending-down / range-bound.
       A short strangle and a directional credit spread want opposite things
       from the same market. Without this axis, a strategy that only survives
       chop looks identical to one that survives everything.

    3. VOL DIRECTION     -- vol-rising / vol-flat / vol-falling.
       Vol crush is the #1 repeated risk across the whole documented strategy
       batch: these are premium sellers, and they live or die on whether VIX
       rises after entry. VIX LEVEL alone does not capture that -- entering at
       VIX 16 on the way up is a different trade from entering at VIX 16 on the
       way down.

================================================================================
MEASURED DATA -- every threshold below traces to one of these queries
================================================================================
Measured 2026-08-16 against 9- Automation/data/market.db.

COVERAGE, and the asymmetry that shapes the whole module
    vix                     1,138 rows   2022-01-03 .. 2026-08-13
    spot_index 'Nifty 50'     645 rows   2024-01-01 .. 2026-08-13
    -> 493 VIX days (43.3%) have NO spot row. For all of
       2022-01-03 .. 2023-12-29 the volatility band is knowable and the TREND
       IS NOT. Those days are tagged trend="unknown", never "range-bound".
       (Same 2024-01-01 floor as the option chain: spot_index was ingested
       alongside fo_bars. See architecture doc section 4.1a.)

1. VOLATILITY BAND -- observed frequency
    full VIX series (n=1,138)      median 14.02   min 9.15   max 31.98
        VIX <13      402   35.3%
        VIX 13-16    378   33.2%
        VIX 16-25    335   29.4%
        VIX >25       23    2.0%
    option-data window 2024-01-01+ (n=645)   median 13.76   max 27.89
        VIX <13      226   35.0%
        VIX 13-16    276   42.8%
        VIX 16-25    136   21.1%      <-- the ~21% the architecture doc cites
        VIX >25        7    1.1%
    The high-vol regime lives in 2022-2023, where there is no option chain.
    Any VIX>25 backtest claim rests on SEVEN days. Treat it as anecdote.

2. TREND -- why a VOL-NORMALISED drift, not a raw N-day return
    Correlation against VIX close, measured over the 625 usable spot days:
        r( |20d return| , VIX )        = +0.449
        r( 20d realised sd , VIX )     = +0.567
        r( |vol-normalised t| , VIX )  = +0.126     <-- near-independent
    A raw-return threshold would mechanically label high-VIX days "trending"
    -- a 3% month is noise at VIX 25 and a real move at VIX 10. That collapses
    axis 2 into axis 1 and makes the joint table meaningless. Normalising by
    realised vol removes the confound (r falls 0.449 -> 0.126).

    The statistic, exact definition:
        r_i  = daily % returns over the trailing N days      (N values)
        ret  = 100 * (close[t]/close[t-N] - 1)
        t    = ret / ( pstdev(r_i) * sqrt(N) )
    Under a driftless random walk t ~ N(0,1). So t is "how many random-walk
    standard deviations of drift did this window actually produce".

    Measured distribution of t at N=20 (n=625):
        p05 -1.58   p25 -0.39   p50 +0.22   p75 +0.70   p95 +1.46
        mean +0.152   sd 0.961
    (mean > 0 because the Nifty drifted UP across 2024-2026. That asymmetry is
    real and is deliberately NOT calibrated away -- see limitation 4.)

    WINDOW = 20 trading days. Chosen on measured persistence, because a label
    that flips every third day is not a regime. Mean run length here means
    total days / number of runs, over the 625 classifiable spot days:
        N= 5, |t|>1  -> mean run length 3.4 trading days
        N=10, |t|>1  -> mean run length 5.2
        N=20, |t|>1  -> mean run length 8.3     <-- outlives a weekly cycle
    Per label at N=20: trending-up 4.3d, range-bound 11.9d, trending-down 5.2d.
    20 trading days is also ~1 calendar month, matching the monthly hedge legs
    the calendar/diagonal strategies in the batch buy.

    THRESHOLD |t| > 1.0. Not a round number picked for looks: 1.0 is exactly
    one standard deviation of the random-walk null the statistic is measured
    against. Measured consequence at N=20 (n=625):
        trending-up      91   14.6%
        range-bound     451   72.2%
        trending-down    83   13.3%
    i.e. 27.9% of days lie outside the band, against the 31.7% the N(0,1) null
    predicts. Read that plainly: the Nifty's 20-day drift is barely
    distinguishable from noise most of the time. That is a finding about the
    market, not a defect in the threshold.
    Measured alternatives, if a caller wants more days in the trending buckets
    (TREND_T_THRESHOLD is overridable):
        |t|>0.75 -> up 22.1% / range 60.6% / down 17.3%, run length 6.3d
        |t|>0.50 -> up 35.7% / range 42.4% / down 21.9%, run length 5.2d

3. VOL DIRECTION -- percent, not points, and a 5-day window
    Points confound with level (a 1-point move at VIX 10 is not a 1-point move
    at VIX 25). Measured:
        r( |5d abs change in points| , VIX level ) = +0.416
        r( |5d percent change|       , VIX level ) = +0.159     <-- use percent

    WINDOW = 5 trading days = one week, matching the weekly cycle these
    strategies trade. It is also where the measured signal is cleanest --
    mean VIX change over the FOLLOWING 5 days, by trailing-window label:
        trailing  3d:  rising -0.59%   flat +0.59%   falling +0.96%
        trailing  5d:  rising -1.60%   flat +0.69%   falling +1.91%   <-- monotonic
        trailing 10d:  rising -1.34%   flat +1.48%   falling +0.85%   (not monotonic)

    THRESHOLD +-6.0%: the measured INTERQUARTILE band of the trailing 5-day
    VIX % change, i.e. "flat" is literally the middle half of the observed
    distribution. Measured quantiles (n=1,133 over the full VIX series):
        p05 -15.45   p25 -6.36   p50 -0.45   p75 +5.98   p95 +18.47   sd 11.41
    The two quartiles are -6.36 and +5.98 -- near-symmetric -- so a single
    symmetric 6.0 is used rather than fitting that 0.38-point asymmetry.
    Measured consequence over the 1,133 classifiable days (the first 5 are
    warm-up):
        vol-rising    282   24.9%     mean run 3.0 trading days
        vol-flat      556   49.1%     mean run 3.3
        vol-falling   295   26.0%     mean run 3.2
    Note the short runs: vol direction is the twitchiest of the three axes,
    which is expected -- VIX's own daily % change has sd 5.7%.

4. THE GAP GUARD -- a lookback must not silently span a hole in the data
    Measured calendar span of an N-trading-day lookback:
        N= 5 : median 7 days,  p95 10, max 12
        N=20 : median 29 days, p95 33, max 35
    Largest gap between adjacent rows: 5 calendar days in vix
    (2022-04-13 -> 2022-04-18), 4 in spot_index.
    Guard: a window is rejected as a data gap if it spans more than
    ceil(N*7/5) + 14 calendar days -- the ideal span plus a fortnight of
    holiday slack. That is 21 days at N=5 (measured max 12) and 42 at N=20
    (measured max 35). Comfortable on real data; a missing month fires it.

5. REFERENCE OUTPUT -- what tag_range + frequencies actually print today.
   Reproduce with `python engine/regime.py`. If these move, the DB changed.

   OPTION-DATA WINDOW 2024-01-01 .. 2026-08-13, 645 trading days
       band            <13 226 (35.0%) | 13-16 276 (42.8%)
                     16-25 136 (21.1%) |   >25   7 ( 1.1%)
       trend            up  91 (14.1%) | range 451 (69.9%)
                      down  83 (12.9%) | unknown 20 (3.1%, the warm-up)
       vol direction rising 163 (25.3%) | flat 310 (48.1%) | falling 172 (26.7%)
       fully known on all three axes: 625 of 645 (96.9%)
       joint band x trend x direction: 37 cells occupied, 27 of them under
       20 days and therefore NOT quotable.

       band x trend (marginalised over vol direction), * = under 20 days:
                        up     range      down   unknown   total
         VIX <13        41       168       14*        3*     226
         VIX 13-16      36       182        42       16*     276
         VIX 16-25      14*       99        22        1*     136
         VIX >25         0         2*        5*        0       7

       Read that bottom row before quoting anything about high volatility:
       "VIX >25 and trending-down" is FIVE DAYS.

   FULL VIX WINDOW 2022-01-03 .. 2026-08-13, 1,138 trading days
       band            <13 402 (35.3%) | 13-16 378 (33.2%)
                     16-25 335 (29.4%) |   >25  23 ( 2.0%)
       trend        unknown 513 (45.1%) -- 493 spot-less days + 20 warm-up,
                    one contiguous block
       fully known on all three axes: 625 of 1,138 (54.9%)

================================================================================
DESIGN RULES
================================================================================
PURE CORE + THIN DB ADAPTER
    Everything above the "DB adapter" banner is stdlib-only arithmetic on
    plain lists. It is testable, and tested, with no database at all. The
    adapter below it does nothing but fetch rows and hand them to the core.

MISSING DATA IS NEVER SILENTLY FINE
    There is no default regime. A date with no VIX row returns
    vol_band="unknown"; a date with no spot row (all of 2022-2023) returns
    trend="unknown"; a date without enough trailing history returns "unknown"
    for the affected axis. Every unknown carries a `reasons` string saying
    which input was missing. "unknown" is a first-class regime that shows up
    in the frequency table -- it is not filtered out, because a reader needs
    to see how much of the window could not be classified.

================================================================================
KNOWN LIMITATIONS -- what this classification CANNOT do
================================================================================
 1. IT IS A DAILY-CLOSE TAGGER. Intraday regime changes are invisible. The
    clearest case in this very database: 2024-06-04 (election result) traded
    VIX 18.80 to 31.71 and closed 26.75. The close-based band says
    "VIX >25"; it cannot say that the day also visited the 16-25 band and
    printed the highest VIX in the option-data window intraday.

 2. THE TREND MEASURE IS BLIND TO A ONE-DAY SHOCK, BY CONSTRUCTION. On
    2024-06-04 the Nifty fell 5.93% in a day, yet the 20-day t-stat was only
    -0.34 -> "range-bound". Correct arithmetic (the crash inflated the
    denominator's realised vol as much as the numerator's drift) and arguably
    the honest answer for a 20-day window, but if you want "was there a crash
    inside this trade", this axis will not tell you. Use the VIX band, or
    tag_holding_period()'s max_vix.

 3. IT IS BACKWARD-LOOKING AND MILDLY MEAN-REVERTING, SO DO NOT TRADE IT.
    Measured mean Nifty return over the FOLLOWING 5 trading days:
        after trending-up   (n=91)  -0.517%
        after range-bound  (n=446)  +0.251%
        after trending-down (n=83)  +0.051%
    The label describes the window that just happened. Using it as an entry
    signal would be trading a contrarian effect measured on 645 days --
    exactly the fake-winner trap the validation programme exists to avoid.
    It does separate forward REALISED MOVE a little (mean |5d move| after
    trending-down 1.69% vs range-bound 1.38%), which is the property that
    matters for a premium seller.

 4. THE TREND BUCKETS ARE NOT BALANCED, DELIBERATELY. 14.6% up vs 13.3% down
    is close, but the underlying t distribution is shifted positive
    (mean +0.152) because the Nifty rose across the sample. Thresholds are NOT
    re-centred to force equal buckets: doing so would define "trending-up"
    relative to a bull market and quietly hide the drift. The consequence is
    that these thresholds are calibrated to 2024-2026 Nifty and should be
    re-measured before being applied to another instrument or era.

 5. TREND IS STRUCTURALLY UNAVAILABLE BEFORE 2024-01-01 -- 43.3% of the VIX
    series. The genuinely high-vol regime (2022, max VIX 31.98) can be
    volatility-tagged but never trend-tagged, so a claim like "works in
    high-vol trending markets" cannot be tested at all on current data.

 6. VIX >25 IS SEVEN DAYS in the option-data window (1.1%). Any per-band
    result there is an anecdote with a percentage sign. thin_cells() flags it.

 7. THE JOINT TABLE IS SPARSE. 4 bands x 3 trends x 3 vol directions = 36
    cells over 645 usable days. Most cells are thin by construction; read the
    marginals, and check thin_cells() before quoting a joint number.

 8. NO REGIME-CHANGE DETECTION. Each day is classified independently from its
    own trailing window. There is no changepoint model, so a regime boundary
    is smeared across the length of the lookback (up to 20 trading days for
    trend). Run lengths are reported so this is at least visible.

 9. VOL DIRECTION USES THE CLOSE-TO-CLOSE CHANGE ONLY. A window that spiked
    and fully retraced reads as "flat", which for a seller who was margin-
    called mid-window is not flat at all.

10. NO CAUSAL CONTENT. These are descriptive labels. "Strategy X earns +0.3R
    in vol-falling regimes" is a statement about 282 days of one index over
    one 4.6-year sample, not a mechanism.
"""

from __future__ import annotations

import datetime as dt
import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

# --------------------------------------------------------------------------
# labels -- plain strings, so they land in dicts/CSV/SQL without conversion
# --------------------------------------------------------------------------

UNKNOWN = "unknown"

TREND_UP = "trending-up"
TREND_DOWN = "trending-down"
TREND_RANGE = "range-bound"

VOL_RISING = "vol-rising"
VOL_FLAT = "vol-flat"
VOL_FALLING = "vol-falling"

# MUST stay identical to score.score_by_regime()'s default bands.
# test_regime.py imports score.py and asserts equality, so drift is caught.
VIX_BANDS: tuple[tuple[str, float, float], ...] = (
    ("VIX <13", 0.0, 13.0),
    ("VIX 13-16", 13.0, 16.0),
    ("VIX 16-25", 16.0, 25.0),
    ("VIX >25", 25.0, 999.0),
)

# --------------------------------------------------------------------------
# thresholds -- see the MEASURED DATA block above for the query behind each.
# Defaults with a rationale, not constants of nature. All are overridable.
# --------------------------------------------------------------------------

TREND_LOOKBACK_DAYS = 20     # trading days; run length 8.3d vs 3.4d at N=5
TREND_T_THRESHOLD = 1.0      # 1 sd of the driftless random-walk null

VOL_DIR_LOOKBACK_DAYS = 5    # trading days; the only monotonic window measured
VOL_DIR_THRESHOLD_PCT = 6.0  # measured IQR of 5d VIX % change: -6.36 .. +5.98

# A joint cell thinner than this is not worth quoting. 20 trading days is
# 4 weekly expiry cycles -- below that a "regime-conditioned" weekly result
# rests on fewer trades than score.py's own 10-trade Validation Gate needs.
THIN_CELL_DAYS = 20


def max_window_span_days(n_trading_days: int) -> int:
    """Longest calendar span an n-trading-day window may cover before it is
    treated as spanning a hole in the data rather than a real window.

    ceil(n*7/5) is the ideal span (5 trading days per calendar week); +14 is a
    fortnight of holiday slack. Measured maxima on this DB: 12 days at n=5
    (allows 21) and 35 at n=20 (allows 42).
    """
    return math.ceil(n_trading_days * 7 / 5) + 14


# ==========================================================================
# PURE CORE -- no I/O, no database. Everything below takes plain numbers.
# ==========================================================================

def vol_band(vix_close: float | None,
             bands: Sequence[tuple[str, float, float]] = VIX_BANDS) -> str:
    """Volatility band from a VIX close. UNKNOWN for missing/invalid input.

    A None, a NaN or a non-positive VIX is missing data, not a low-vol day.
    """
    if vix_close is None:
        return UNKNOWN
    v = float(vix_close)
    if v != v or v <= 0:
        return UNKNOWN
    for label, lo, hi in bands:
        if lo <= v < hi:
            return label
    return UNKNOWN


def trend_tstat(closes: Sequence[float]) -> float:
    """Vol-normalised drift over the window described by `closes`.

        closes = [c_{t-N}, ..., c_t]   (N+1 values, oldest first)
        r_i    = 100 * (c_i / c_{i-1} - 1)        for i = 1..N
        ret    = 100 * (c_N / c_0 - 1)
        t      = ret / ( pstdev(r_i) * sqrt(N) )

    Under a driftless random walk t ~ N(0,1), so |t| is in units of "random
    walk standard deviations of drift". Returns NaN if the window is too short
    or contains a non-positive/missing close.

    Zero dispersion with non-zero drift is +-inf, which is the correct answer:
    a perfectly monotone series has infinitely significant drift.
    """
    if len(closes) < 3:
        return float("nan")
    cs = [float(c) if c is not None else float("nan") for c in closes]
    if any(c != c or c <= 0 for c in cs):
        return float("nan")
    n = len(cs) - 1
    daily = [100.0 * (cs[i] / cs[i - 1] - 1.0) for i in range(1, len(cs))]
    ret = 100.0 * (cs[-1] / cs[0] - 1.0)
    sd = statistics.pstdev(daily)
    if sd <= 0:
        if ret == 0:
            return 0.0
        return math.inf if ret > 0 else -math.inf
    return ret / (sd * math.sqrt(n))


def trend_return_pct(closes: Sequence[float]) -> float:
    """Plain % return across the window. Reported alongside the t-stat because
    a reader wants the human-readable move, not only the normalised one."""
    if len(closes) < 2:
        return float("nan")
    a, b = closes[0], closes[-1]
    if a is None or b is None or a != a or b != b or a <= 0:
        return float("nan")
    return 100.0 * (b / a - 1.0)


def classify_trend(t: float, threshold: float = TREND_T_THRESHOLD) -> str:
    """Map a t-stat to a trend label. NaN -> UNKNOWN, never range-bound.

    The distinction matters: "we measured this window and it was flat" and
    "we could not measure this window" must not share a bucket.
    """
    if t != t:
        return UNKNOWN
    if t > threshold:
        return TREND_UP
    if t < -threshold:
        return TREND_DOWN
    return TREND_RANGE


def vol_change_pct(vix_closes: Sequence[float]) -> float:
    """% change in VIX across the window (first to last). NaN if unusable."""
    if len(vix_closes) < 2:
        return float("nan")
    a, b = vix_closes[0], vix_closes[-1]
    if a is None or b is None:
        return float("nan")
    a, b = float(a), float(b)
    if a != a or b != b or a <= 0 or b <= 0:
        return float("nan")
    return 100.0 * (b / a - 1.0)


def classify_vol_direction(change_pct: float,
                           threshold: float = VOL_DIR_THRESHOLD_PCT) -> str:
    """Map a trailing VIX % change to a direction label. NaN -> UNKNOWN."""
    if change_pct != change_pct:
        return UNKNOWN
    if change_pct > threshold:
        return VOL_RISING
    if change_pct < -threshold:
        return VOL_FALLING
    return VOL_FLAT


@dataclass(frozen=True)
class Regime:
    """One trading day's market state on all three axes.

    Every axis can independently be UNKNOWN, and `reasons` says why. There is
    deliberately no "default" or "assumed" state anywhere in this dataclass.
    """
    date: str
    vol_band: str = UNKNOWN
    trend: str = UNKNOWN
    vol_direction: str = UNKNOWN

    vix_close: float | None = None
    spot_close: float | None = None
    trend_t: float = float("nan")
    trend_return_pct: float = float("nan")
    vix_change_pct: float = float("nan")

    reasons: tuple[str, ...] = ()

    @property
    def is_fully_known(self) -> bool:
        return UNKNOWN not in (self.vol_band, self.trend, self.vol_direction)

    @property
    def label(self) -> str:
        return f"{self.vol_band} | {self.trend} | {self.vol_direction}"

    @property
    def cell(self) -> tuple[str, str, str]:
        """Joint key, for the frequency table."""
        return (self.vol_band, self.trend, self.vol_direction)

    def __str__(self) -> str:
        def f(x, unit="", w=7, p=2):
            return (f"{'n/a':>{w}}" if x != x
                    else f"{x:+{w - len(unit)}.{p}f}{unit}")
        vix = "  n/a" if self.vix_close is None else f"{self.vix_close:5.2f}"
        out = (f"{self.date}  VIX {vix}  {self.label}"
               f"   [t {f(self.trend_t)}   spot {f(self.trend_return_pct, '%')}/20d"
               f"   VIX {f(self.vix_change_pct, '%')}/5d]")
        for r in self.reasons:
            out += f"\n           ! {r}"
        return out


def classify(date: str,
             vix_window: Sequence[float] | None,
             spot_window: Sequence[float] | None,
             vix_span_days: int | None = None,
             spot_span_days: int | None = None,
             trend_lookback: int = TREND_LOOKBACK_DAYS,
             vol_dir_lookback: int = VOL_DIR_LOOKBACK_DAYS,
             trend_threshold: float = TREND_T_THRESHOLD,
             vol_dir_threshold: float = VOL_DIR_THRESHOLD_PCT,
             bands: Sequence[tuple[str, float, float]] = VIX_BANDS) -> Regime:
    """The whole classification, from plain numbers. NO DATABASE.

    vix_window  : VIX closes ending on `date`, oldest first. The last element
                  IS the close on `date`. Needs vol_dir_lookback+1 elements
                  for the direction axis; 1 element is enough for the band.
    spot_window : spot closes ending on `date`, oldest first. Needs
                  trend_lookback+1 elements.
    *_span_days : calendar days actually covered by that window, for the gap
                  guard. Pass None to skip the guard (unit tests do).
    """
    reasons: list[str] = []
    vw = list(vix_window) if vix_window else []
    sw = list(spot_window) if spot_window else []

    # ---- axis 1: volatility band ----
    vix_close = vw[-1] if vw else None
    band = vol_band(vix_close, bands)
    if band == UNKNOWN:
        reasons.append("no usable VIX close for this date -- volatility band unknown")

    # ---- axis 3: vol direction ----
    v_chg = float("nan")
    if len(vw) < vol_dir_lookback + 1:
        reasons.append(
            f"only {len(vw)} VIX closes available, need {vol_dir_lookback + 1} "
            f"-- vol direction unknown")
    elif (vix_span_days is not None
          and vix_span_days > max_window_span_days(vol_dir_lookback)):
        reasons.append(
            f"VIX lookback spans {vix_span_days} calendar days, over the "
            f"{max_window_span_days(vol_dir_lookback)}-day limit -- treated as a "
            f"data gap, vol direction unknown")
    else:
        v_chg = vol_change_pct(vw[-(vol_dir_lookback + 1):])
        if v_chg != v_chg:
            reasons.append("VIX lookback contains an unusable value -- "
                           "vol direction unknown")
    direction = classify_vol_direction(v_chg, vol_dir_threshold)

    # ---- axis 2: trend ----
    t = float("nan")
    ret = float("nan")
    spot_close = sw[-1] if sw else None
    if not sw:
        reasons.append("no spot_index row for this date (the series starts "
                       "2024-01-01) -- trend unknown")
    elif len(sw) < trend_lookback + 1:
        reasons.append(
            f"only {len(sw)} spot closes available, need {trend_lookback + 1} "
            f"-- trend unknown")
    elif (spot_span_days is not None
          and spot_span_days > max_window_span_days(trend_lookback)):
        reasons.append(
            f"spot lookback spans {spot_span_days} calendar days, over the "
            f"{max_window_span_days(trend_lookback)}-day limit -- treated as a "
            f"data gap, trend unknown")
    else:
        win = sw[-(trend_lookback + 1):]
        t = trend_tstat(win)
        ret = trend_return_pct(win)
        if t != t:
            reasons.append("spot lookback contains an unusable close -- trend unknown")
    trend = classify_trend(t, trend_threshold)

    return Regime(date=date, vol_band=band, trend=trend, vol_direction=direction,
                  vix_close=vix_close, spot_close=spot_close,
                  trend_t=t, trend_return_pct=ret, vix_change_pct=v_chg,
                  reasons=tuple(reasons))


# --------------------------------------------------------------------------
# frequency reporting -- the part that stops a claim resting on n=7
# --------------------------------------------------------------------------

@dataclass
class RegimeFrequencies:
    n_days: int = 0
    first: str = ""
    last: str = ""
    by_vol_band: Counter = field(default_factory=Counter)
    by_trend: Counter = field(default_factory=Counter)
    by_vol_direction: Counter = field(default_factory=Counter)
    by_cell: Counter = field(default_factory=Counter)
    n_fully_known: int = 0
    # NESTED BY AXIS, not flat. UNKNOWN occurs on all three axes, so a flat
    # {label: run_length} silently overwrote it: the full-window run showed
    # "trend unknown, mean run 5.0d" when the 513 unknown trend days are ONE
    # contiguous block of 513. Found by eyeballing the integration output --
    # the unit tests could not see it because their synthetic days had a
    # distinct label per axis.
    run_lengths: dict[str, dict[str, float]] = field(default_factory=dict)

    def pct(self, count: int) -> float:
        return 100.0 * count / self.n_days if self.n_days else float("nan")

    def thin_cells(self, min_days: int = THIN_CELL_DAYS) -> list[tuple[tuple[str, str, str], int]]:
        """Joint cells with too few days to support a conditioned claim."""
        return sorted(((c, n) for c, n in self.by_cell.items() if n < min_days),
                      key=lambda x: -x[1])

    def _block(self, axis: str, title: str, counter: Counter,
               order: Sequence[str]) -> list[str]:
        runs = self.run_lengths.get(axis, {})
        out = [f"  {title}"]
        keys = [k for k in order if k in counter] + \
               sorted(k for k in counter if k not in order)
        for k in keys:
            n = counter[k]
            out.append(f"      {k:<14} {n:5d}  {self.pct(n):5.1f}%"
                       + (f"   mean run {runs[k]:6.1f}d" if k in runs else ""))
        return out

    def __str__(self) -> str:
        lines = [f"=== Regime frequencies: {self.n_days} trading days "
                 f"({self.first} .. {self.last}) ==="]
        lines += self._block("vol_band", "volatility band", self.by_vol_band,
                             [b[0] for b in VIX_BANDS] + [UNKNOWN])
        lines += self._block("trend", "trend", self.by_trend,
                             [TREND_UP, TREND_RANGE, TREND_DOWN, UNKNOWN])
        lines += self._block("vol_direction", "vol direction", self.by_vol_direction,
                             [VOL_RISING, VOL_FLAT, VOL_FALLING, UNKNOWN])
        lines.append(f"  fully known on all three axes: {self.n_fully_known} "
                     f"({self.pct(self.n_fully_known):.1f}%)")
        thin = self.thin_cells()
        lines.append(f"  joint cells: {len(self.by_cell)} occupied, "
                     f"{len(thin)} below {THIN_CELL_DAYS} days "
                     f"(= 4 weekly expiries) and NOT quotable")
        return "\n".join(lines)

    def joint_table(self, min_days: int = 1) -> str:
        """band x trend, marginalised over vol direction. The table a reader
        actually needs before quoting a regime-conditioned backtest number."""
        bands = [b[0] for b in VIX_BANDS] + [UNKNOWN]
        trends = [TREND_UP, TREND_RANGE, TREND_DOWN, UNKNOWN]
        grid: Counter = Counter()
        for (b, tr, _vd), n in self.by_cell.items():
            grid[(b, tr)] += n
        w = 14
        out = [f"  {'band':<11}" + "".join(f"{t:>{w}}" for t in trends) + f"{'total':>10}"]
        for b in bands:
            row = [grid[(b, t)] for t in trends]
            if sum(row) < min_days:
                continue
            cells = "".join(
                f"{(str(v) + ('*' if 0 < v < THIN_CELL_DAYS else '')):>{w}}" for v in row)
            out.append(f"  {b:<11}{cells}{sum(row):>10}")
        tot = [sum(grid[(b, t)] for b in bands) for t in trends]
        out.append(f"  {'total':<11}" + "".join(f"{v:>{w}}" for v in tot)
                   + f"{sum(tot):>10}")
        out.append(f"   (* = fewer than {THIN_CELL_DAYS} days; not enough for a "
                   f"regime-conditioned claim)")
        return "\n".join(out)


def _mean_run_lengths(labels: Sequence[str]) -> dict[str, float]:
    """Mean consecutive run length per label. A regime that flips daily is not
    a regime, so this is reported next to every frequency."""
    runs: dict[str, list[int]] = {}
    if not labels:
        return {}
    cur, length = labels[0], 1
    for x in labels[1:]:
        if x == cur:
            length += 1
        else:
            runs.setdefault(cur, []).append(length)
            cur, length = x, 1
    runs.setdefault(cur, []).append(length)
    return {k: statistics.fmean(v) for k, v in runs.items()}


def frequencies(regimes: Sequence[Regime]) -> RegimeFrequencies:
    """Measured frequency of every regime. UNKNOWN is counted, never dropped --
    a reader must see how much of the window could not be classified."""
    f = RegimeFrequencies(n_days=len(regimes))
    if not regimes:
        return f
    ordered = sorted(regimes, key=lambda r: r.date)
    f.first, f.last = ordered[0].date, ordered[-1].date
    for r in ordered:
        f.by_vol_band[r.vol_band] += 1
        f.by_trend[r.trend] += 1
        f.by_vol_direction[r.vol_direction] += 1
        f.by_cell[r.cell] += 1
        if r.is_fully_known:
            f.n_fully_known += 1
    f.run_lengths = {
        "vol_band": _mean_run_lengths([r.vol_band for r in ordered]),
        "trend": _mean_run_lengths([r.trend for r in ordered]),
        "vol_direction": _mean_run_lengths([r.vol_direction for r in ordered]),
    }
    return f


# ==========================================================================
# DB ADAPTER -- fetch rows, hand them to the pure core. Nothing else.
# ==========================================================================

DEFAULT_INDEX = "Nifty 50"


def _shift(iso: str, days: int) -> str:
    return (dt.date.fromisoformat(iso) + dt.timedelta(days=days)).isoformat()


def _span(a: str, b: str) -> int:
    return (dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days


def coverage(con) -> dict:
    """What the DB can and cannot classify. Call this before believing a
    regime-conditioned result -- it exposes the 2024-01-01 spot floor."""
    v = con.execute("SELECT COUNT(*) n, MIN(trade_date) a, MAX(trade_date) b "
                    "FROM vix").fetchone()
    s = con.execute("SELECT COUNT(*) n, MIN(trade_date) a, MAX(trade_date) b "
                    "FROM spot_index WHERE index_name=?", (DEFAULT_INDEX,)).fetchone()
    no_spot = con.execute(
        "SELECT COUNT(*) FROM vix v WHERE NOT EXISTS (SELECT 1 FROM spot_index s "
        "WHERE s.trade_date=v.trade_date AND s.index_name=?)",
        (DEFAULT_INDEX,)).fetchone()[0]
    return {"vix_days": v["n"], "vix_first": v["a"], "vix_last": v["b"],
            "spot_days": s["n"], "spot_first": s["a"], "spot_last": s["b"],
            "vix_days_without_spot": no_spot,
            "pct_untrendable": (100.0 * no_spot / v["n"]) if v["n"] else float("nan")}


def tag_range(con, start: str, end: str,
              index_name: str = DEFAULT_INDEX,
              trend_lookback: int = TREND_LOOKBACK_DAYS,
              vol_dir_lookback: int = VOL_DIR_LOOKBACK_DAYS,
              trend_threshold: float = TREND_T_THRESHOLD,
              vol_dir_threshold: float = VOL_DIR_THRESHOLD_PCT,
              bands: Sequence[tuple[str, float, float]] = VIX_BANDS) -> list[Regime]:
    """Tag every VIX trading day in [start, end].

    TWO QUERIES TOTAL, not two per day. Both pull enough warm-up history
    before `start` to fill the longest lookback, then the whole range is
    classified by walking the two series once.

    The day set is driven by the VIX table, because vol_band is the axis every
    strategy in the batch is gated on and it is the axis with the wider
    coverage. Days with VIX but no spot (all of 2022-2023) still appear -- with
    trend=UNKNOWN, which is the point.
    """
    warm = max(max_window_span_days(trend_lookback),
               max_window_span_days(vol_dir_lookback))
    q_start = _shift(start, -warm)

    vrows = con.execute(
        "SELECT trade_date, close FROM vix WHERE trade_date BETWEEN ? AND ? "
        "ORDER BY trade_date", (q_start, end)).fetchall()
    srows = con.execute(
        "SELECT trade_date, close FROM spot_index WHERE index_name=? "
        "AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
        (index_name, q_start, end)).fetchall()

    v_dates = [r["trade_date"] for r in vrows]
    v_close = [r["close"] for r in vrows]
    s_dates = [r["trade_date"] for r in srows]
    s_close = [r["close"] for r in srows]
    s_index = {d: i for i, d in enumerate(s_dates)}

    out: list[Regime] = []
    for i, d in enumerate(v_dates):
        if d < start:
            continue                                  # warm-up only

        vlo = max(0, i - vol_dir_lookback)
        vwin = v_close[vlo:i + 1]
        vspan = _span(v_dates[vlo], d) if len(vwin) == vol_dir_lookback + 1 else None

        j = s_index.get(d)
        if j is None:
            swin: list[float] = []
            sspan = None
        else:
            slo = max(0, j - trend_lookback)
            swin = s_close[slo:j + 1]
            sspan = _span(s_dates[slo], d) if len(swin) == trend_lookback + 1 else None

        out.append(classify(
            d, vwin, swin, vix_span_days=vspan, spot_span_days=sspan,
            trend_lookback=trend_lookback, vol_dir_lookback=vol_dir_lookback,
            trend_threshold=trend_threshold, vol_dir_threshold=vol_dir_threshold,
            bands=bands))
    return out


def tag_date(con, date: str, **kw) -> Regime:
    """Label a single date.

    A date with NO VIX row returns a fully-unknown Regime carrying the reason,
    rather than raising or silently returning a default. A backtest asking
    "what regime was 2023-01-15 in" must get "we do not know", because the
    spot series does not reach there.
    """
    got = tag_range(con, date, date, **kw)
    if got:
        return got[0]
    return Regime(date=date, reasons=(
        f"no vix row for {date} -- not a trading day, or outside the ingested "
        f"window; every axis is unknown",))


@dataclass
class HoldingPeriodRegime:
    """Regime across a trade's whole life, not just its entry day.

    score.py's documented limitation is that it buckets on VIX AT ENTRY only:
    "a weekly trade entered at VIX 13 that spikes to 20 is still counted as a
    VIX-13 trade". This is the structure that lets a caller fix that, by
    reporting what the position actually lived through.
    """
    entry_date: str
    exit_date: str
    n_days: int = 0
    entry: Regime | None = None
    exit: Regime | None = None
    max_vix: float | None = None
    min_vix: float | None = None
    max_vix_date: str = ""
    bands_visited: tuple[str, ...] = ()
    vix_change_pct: float = float("nan")
    vol_direction_realised: str = UNKNOWN
    trends_visited: tuple[str, ...] = ()

    @property
    def band_changed(self) -> bool:
        """True when entry-day bucketing would misdescribe the trade."""
        return len(self.bands_visited) > 1

    def __str__(self) -> str:
        mv = "n/a" if self.max_vix is None else f"{self.max_vix:.2f}"
        nv = "n/a" if self.min_vix is None else f"{self.min_vix:.2f}"
        chg = ("n/a" if self.vix_change_pct != self.vix_change_pct
               else f"{self.vix_change_pct:+.1f}%")
        return (f"{self.entry_date} -> {self.exit_date} ({self.n_days}d)  "
                f"entry {self.entry.label if self.entry else 'unknown'}\n"
                f"    VIX {nv} .. {mv} (max on {self.max_vix_date or 'n/a'}), "
                f"entry->exit {chg} -> {self.vol_direction_realised}\n"
                f"    bands visited: {', '.join(self.bands_visited) or 'none'}"
                + ("   <-- entry-day bucketing MISDESCRIBES this trade"
                   if self.band_changed else ""))


def tag_holding_period(con, entry_date: str, exit_date: str, **kw) -> HoldingPeriodRegime:
    """What the position actually lived through, entry close to exit close."""
    days = tag_range(con, entry_date, exit_date, **kw)
    h = HoldingPeriodRegime(entry_date=entry_date, exit_date=exit_date,
                            n_days=len(days))
    if not days:
        return h
    h.entry, h.exit = days[0], days[-1]
    vix = [(d.date, d.vix_close) for d in days if d.vix_close is not None]
    if vix:
        h.max_vix_date, h.max_vix = max(vix, key=lambda x: x[1])
        h.min_vix = min(v for _, v in vix)
        h.vix_change_pct = vol_change_pct([vix[0][1], vix[-1][1]])
        h.vol_direction_realised = classify_vol_direction(
            h.vix_change_pct, kw.get("vol_dir_threshold", VOL_DIR_THRESHOLD_PCT))
    # ordered-unique, so the sequence of states is readable
    h.bands_visited = tuple(dict.fromkeys(d.vol_band for d in days))
    h.trends_visited = tuple(dict.fromkeys(d.trend for d in days))
    return h


def tag_trades(con, trades: Iterable, **kw) -> dict[str, Regime]:
    """Regime at entry for a batch of score.Trade-like objects (anything with
    .trade_id and .entry_date). ONE tag_range call covering the whole span,
    then a dict lookup -- not one query per trade."""
    ts = list(trades)
    if not ts:
        return {}
    dates = sorted(t.entry_date for t in ts)
    tagged = {r.date: r for r in tag_range(con, dates[0], dates[-1], **kw)}
    out: dict[str, Regime] = {}
    for t in ts:
        out[t.trade_id] = tagged.get(t.entry_date, Regime(
            date=t.entry_date,
            reasons=(f"no vix row for entry_date {t.entry_date}", )))
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    import data_layer as dl

    con = dl.connect()
    print(coverage(con))
    regs = tag_range(con, "2024-01-01", "2026-08-13")
    f = frequencies(regs)
    print(f)
    print(f.joint_table())
