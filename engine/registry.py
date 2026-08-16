"""
registry.py -- component G: results registry with PRE-REGISTRATION.

WHY THIS EXISTS
    Architecture doc S4.5: "14 strategies x several parameter variants each, on
    ~167 expiries, is enough tests that some will look profitable BY CHANCE
    ALONE. This is the classic overfitting trap, and it is especially dangerous
    here because the plan explicitly involves 'build our own strategies from the
    winners.'"

    Concretely: at alpha = 0.05, running 40 independent tests on strategies that
    have NO edge whatsoever is expected to produce 2 "significant" winners. If
    you then build a strategy out of those two, you have built a strategy out of
    noise, and you will trade real money on it.

    Three defences, all implemented here:

      1. PRE-REGISTER. Write down the hypothesis, the exact parameters, the
         expected result and the pass threshold BEFORE the backtest runs.
         Write-once, enforced by the database, so the threshold cannot be moved
         to fit the answer afterwards.

      2. COUNT EVERY TEST, INCLUDING FAILURES. A winner found on the 40th test
         is a different claim from one found on the 2nd. The registry knows
         which it was and says so.

      3. CORRECT FOR MULTIPLICITY. Report the corrected significance threshold
         alongside the raw one, plus the number of false positives chance alone
         predicts at this test count.

WHY BENJAMINI-HOCHBERG IS THE DEFAULT (and Bonferroni is still printed)
    Both are implemented as pure functions; BH drives the default verdict.

    Bonferroni controls the FAMILY-WISE ERROR RATE: P(at least one false
    positive anywhere) <= alpha. It is the right bar for a single decisive
    claim -- "THIS strategy works". But it divides: at 40 tests the threshold
    is 0.05/40 = 0.00125. With ~52 weekly samples per test and the R-multiple
    standard deviations this account actually shows, essentially nothing can
    reach p < 0.00125. Bonferroni here would reject every strategy including
    the real ones, and we would learn nothing from the whole programme.

    Benjamini-Hochberg controls the FALSE DISCOVERY RATE: of the hypotheses I
    call winners, the expected fraction that are actually noise is <= alpha.
    That matches what this registry is actually for. The output of the backtest
    phase is not a live-trading decision -- it is a SHORTLIST that then goes to
    forward paper-testing under the Paper-Only Mandate. Paper-testing is a
    second, independent filter, so an occasional false positive costs paper
    time, not capital. Over-conservatism, by contrast, costs the entire
    programme its answer.

    So: BH decides the shortlist, Bonferroni is reported as the stricter bar a
    result must clear before anyone says "this strategy works" without
    qualification. Both numbers are always printed. Neither is a substitute for
    out-of-sample confirmation.

    BH assumes independence or positive regression dependency across tests.
    Variants of the same strategy on the same expiries are positively
    correlated, which is the case BH is known to tolerate (Benjamini-Yekutieli
    2001). It is NOT valid under arbitrary negative dependence. See limitations.

STRUCTURE
    Pure statistics (no DB, no I/O)  ..  t/bootstrap p-values, Bonferroni, BH.
    Pure domain logic (no DB)        ..  params_hash, evaluate_result.
    Storage (SQLite, data/registry.db)  register / record_result / summary.

    Everything above the storage line is testable without touching a database,
    which is how the multiple-comparisons maths gets checked against
    hand-computed numbers.

WRITE-ONCE, AND HOW IT IS ENFORCED
    Mirrors the anti-backfill design already in data_layer.py's `predictions`
    table, plus three additions that close holes that design leaves open:

      a) BEFORE UPDATE / BEFORE DELETE triggers on BOTH tables (RAISE(ABORT)).
      b) PRAGMA recursive_triggers = ON. Without it, SQLite does NOT fire
         DELETE triggers for rows removed by INSERT OR REPLACE -- so a plain
         `INSERT OR REPLACE` would silently overwrite a pre-registration while
         the delete trigger sat there doing nothing. This is a real hole and it
         is tested explicitly in test_registry.py.
      c) A trigger refusing any result whose run_at PRE-DATES its own
         pre-registration. Recording a "prediction" after you already know the
         answer is the exact fraud this module exists to prevent, so the
         database refuses it rather than trusting the caller's clock.

    Amendments are impossible by design. The legitimate path is `supersede()`,
    which writes a NEW pre-registration pointing at the old one. The original
    stays visible forever; that is the point.

SEPARATE DATABASE, DELIBERATELY
    data/registry.db, NOT data/market.db. market.db is ~3.2 GB of irreplaceable
    ingested market data. A registry is small, is written to constantly during
    experimentation, and gains nothing from sharing a file with it. Keeping them
    apart means no registry bug can ever put the market data at risk.

KNOWN LIMITATIONS  (honest; none of these are fixed)

    STATISTICS
    - The default p-value is a two-sided one-sample t-test on R-multiples.
      score.py deliberately does NOT use a t-interval for its own significance
      call, because option-selling R distributions are strongly negatively
      skewed and the t assumption degrades at small n. The t p-value here is a
      RANKING device for the correction, not a truth claim. Pass an explicit
      p-value from `bootstrap_p_value()` whenever the raw R-multiples are in
      hand; the API accepts one and prefers it.
    - Multiple-comparisons corrections assume the tests are, if not
      independent, at worst positively dependent. Parameter variants of one
      strategy walked over the SAME 167 expiries are heavily overlapping. The
      effective number of independent tests is smaller than m, so the
      correction is conservative in that direction -- but there is no attempt
      here to estimate the effective m.
    - The correction is only as honest as the declared `family`. Nothing stops
      a user from declaring a fresh family per test to shrink m to 1. That is
      p-hacking by taxonomy; the registry makes it visible (families are listed
      with their sizes) but cannot prevent it.
    - m counts pre-registrations that have been RUN. Tests abandoned before
      running are listed as pending but do not enlarge m. Arguably they should:
      a hypothesis you dropped after peeking is still a test.
    - No holdout enforcement. `data_window` and `holdout` are recorded as free
      text and never checked against what the backtest actually read.

    SCOPE
    - Nothing forces anyone to CALL this module. A backtest run outside the
      registry leaves no trace. The registry can prove what was registered; it
      cannot prove that nothing else was run.
    - Write-once is enforced by triggers inside the file. Anyone with the
      sqlite3 CLI can DROP them. `verify_integrity()` detects that on the next
      connect and refuses to open the registry, but a determined operator can
      still rewrite history and there is no cryptographic chain to stop them.
    - Deleting data/registry.db erases everything. No backup, no append-only
      log outside the file.
    - The ScoreCard is stored as a JSON blob with non-finite floats coerced to
      null, so a NaN read back becomes None. Key scalars are also promoted to
      real columns for querying; those keep NaN as NULL too.
    - `run_at` comes from the caller's clock. The pre-dating trigger stops the
      obvious backfill, but a wrong system clock is not detectable here.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import math
import random
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

HERE = Path(__file__).resolve().parent
DB_PATH = HERE.parent / "data" / "registry.db"

DEFAULT_ALPHA = 0.05

# Valid pre-registered expectations. Two values only, and they are about the
# THRESHOLD, not about the sign of the return -- "I expect this to clear the
# bar I just wrote down" or "I expect it not to". A prediction of failure that
# comes true is a real result and is scored as a confirmed hypothesis.
EXPECTATION_CLEARS = "clears"
EXPECTATION_FAILS = "fails"
EXPECTATIONS = (EXPECTATION_CLEARS, EXPECTATION_FAILS)


# ==========================================================================
# EXCEPTIONS
# ==========================================================================

class RegistryError(Exception):
    """Base for every registry refusal."""


class WriteOnceViolation(RegistryError):
    """Someone tried to change history."""


class DuplicatePreregistration(RegistryError):
    """This exact test is already pre-registered."""


class NoPreregistration(RegistryError):
    """A result was offered with no hypothesis behind it."""


class IntegrityCompromised(RegistryError):
    """The write-once guards are missing from the database file."""


# ==========================================================================
# PURE STATISTICS -- no database, no I/O, hand-checkable
# ==========================================================================

def _betacf(a: float, b: float, x: float,
            itmax: int = 300, eps: float = 3e-16) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def reg_incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b). Needed for the Student-t CDF.

    Implemented rather than imported because the whole engine is stdlib-only
    (architecture doc S11.5 -- "is this still a free build? YES").
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lfront = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
              + a * math.log(x) + b * math.log(1.0 - x))
    front = math.exp(lfront)
    # The continued fraction converges fast only on one side of the mode, so
    # use the symmetry I_x(a,b) = 1 - I_{1-x}(b,a) on the other side.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_test_p_two_sided(t_stat: float, df: int) -> float:
    """Two-sided p-value for a Student-t statistic. No scipy.

        p = I_{df/(df+t^2)} (df/2, 1/2)

    Verified in test_registry.py against the exact 95% critical values already
    tabulated in score.py: feeding t_crit_95(df) must return p = 0.05.
    """
    if df < 1 or t_stat != t_stat:
        return float("nan")
    if math.isinf(t_stat):
        return 0.0
    x = df / (df + t_stat * t_stat)
    return reg_incomplete_beta(df / 2.0, 0.5, x)


def t_stat_from_summary(mean: float, stdev: float, n: int) -> float:
    """One-sample t against a null of zero."""
    if n < 2 or stdev != stdev or stdev <= 0 or mean != mean:
        return float("nan")
    return mean / (stdev / math.sqrt(n))


def p_value_from_summary(mean: float, stdev: float, n: int) -> float:
    """Two-sided p for 'is this expectancy distinguishable from zero'.

    APPROXIMATION, deliberately flagged. See the module limitations: option
    R-multiples are skewed and the t assumption is weak at small n. Use
    bootstrap_p_value() when the raw per-trade R values are available.
    """
    return t_test_p_two_sided(t_stat_from_summary(mean, stdev, n), n - 1)


def bootstrap_p_value(values: Sequence[float], iters: int = 10000,
                      seed: int = 20260816) -> float:
    """Two-sided bootstrap p-value for 'mean differs from zero'.

    Resamples the MEAN-CENTRED sample, which is the null distribution, then
    asks how often a resampled mean is at least as extreme as the observed one.
    Makes no normality assumption, which is why score.py uses a bootstrap for
    its own significance call.

    The +1/+1 correction is standard: it keeps p strictly positive, because
    "p = 0" is never an honest statement about a finite resample.
    """
    n = len(values)
    if n < 2:
        return float("nan")
    obs = sum(values) / n
    centred = [v - obs for v in values]
    target = abs(obs)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(iters):
        m = sum(centred[rng.randrange(n)] for _ in range(n)) / n
        if abs(m) >= target:
            extreme += 1
    return (extreme + 1) / (iters + 1)


def bonferroni_threshold(alpha: float, m: int) -> float:
    """The per-test threshold that holds family-wise error at alpha.

    alpha / m. At m=1 it is alpha; at m=40 and alpha=0.05 it is 0.00125.
    """
    if m < 1:
        return float("nan")
    return alpha / m


def bonferroni_adjusted(p_values: Sequence[float]) -> list[float]:
    """min(1, m * p) for each test -- the same decision, expressed as q-values."""
    m = len(p_values)
    return [min(1.0, m * p) if p == p else float("nan") for p in p_values]


def benjamini_hochberg_adjusted(p_values: Sequence[float]) -> list[float]:
    """BH q-values, returned in the INPUT order.

        sort ascending, then walking DOWN from the largest:
            q_(i) = min( q_(i+1),  m/i * p_(i) ),  clipped to 1

    The downward-running minimum is what makes BH a STEP-UP procedure: once the
    largest surviving p-value is found at rank k, every test ranked below k is
    also declared significant, even one that fails its own i*alpha/m line.
    Tested explicitly.

    NaN p-values (an unscoreable run) sort last, get q = NaN, and can never be
    declared significant -- but they still count toward m. A test that ran and
    produced nothing is still a test, so it should still tighten the bar.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: (p_values[i] != p_values[i],  # NaN last
                                            p_values[i]))
    q = [float("nan")] * m
    running = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        p = p_values[idx]
        if p != p:
            q[idx] = float("nan")
            continue
        running = min(running, min(1.0, m * p / rank))
        q[idx] = running
    return q


@dataclass(frozen=True)
class TestOutcome:
    """One completed test, reduced to what the correction needs. Pure."""
    label: str
    p_value: float
    meets_threshold: bool = False


@dataclass
class CorrectionResult:
    method: str
    alpha: float
    m: int
    bonferroni_threshold: float = float("nan")
    bh_cutoff_p: float = 0.0          # largest raw p BH declares significant
    n_naive_significant: int = 0
    n_bonferroni_significant: int = 0
    n_bh_significant: int = 0
    significant_labels: tuple[str, ...] = ()
    expected_false_positives: float = 0.0
    adjusted: dict[str, float] = field(default_factory=dict)
    bonferroni_adjusted: dict[str, float] = field(default_factory=dict)

    @property
    def n_significant(self) -> int:
        return len(self.significant_labels)

    def __str__(self) -> str:
        bh_bar = (f"p <= {self.bh_cutoff_p:.5f} (data-dependent)"
                  if self.n_bh_significant else "no p-value is low enough")
        return (
            f"multiple comparisons: m={self.m} tests, alpha={self.alpha:.3f}, "
            f"method={self.method.upper()}\n"
            f"    raw threshold          p < {self.alpha:.5f}"
            f"   -> {self.n_naive_significant} significant\n"
            f"    Bonferroni (FWER)      p < {self.bonferroni_threshold:.5f}"
            f"   -> {self.n_bonferroni_significant} significant\n"
            f"    Benjamini-Hochberg     {bh_bar}"
            f"   -> {self.n_bh_significant} significant\n"
            f"    chance alone predicts ~{self.expected_false_positives:.1f} "
            f"false positive(s) at the raw threshold")


def correct(outcomes: Sequence[TestOutcome], alpha: float = DEFAULT_ALPHA,
            method: str = "bh") -> CorrectionResult:
    """Apply the multiple-comparisons correction to a family of tests. Pure.

    Both corrections are always computed; `method` only chooses which one
    populates `significant_labels`. Printing only the one that flatters the
    result would be exactly the behaviour this module exists to stop.
    """
    method = method.lower()
    if method not in ("bh", "bonferroni"):
        raise ValueError(f"unknown correction method {method!r}")
    m = len(outcomes)
    res = CorrectionResult(method=method, alpha=alpha, m=m)
    res.bonferroni_threshold = bonferroni_threshold(alpha, m)
    res.expected_false_positives = m * alpha
    if m == 0:
        return res

    ps = [o.p_value for o in outcomes]
    bh_q = benjamini_hochberg_adjusted(ps)
    bon_q = bonferroni_adjusted(ps)
    res.adjusted = {o.label: q for o, q in zip(outcomes, bh_q)}
    res.bonferroni_adjusted = {o.label: q for o, q in zip(outcomes, bon_q)}

    res.n_naive_significant = sum(1 for p in ps if p == p and p < alpha)
    res.n_bonferroni_significant = sum(
        1 for q in bon_q if q == q and q <= alpha)
    bh_sig = [o for o, q in zip(outcomes, bh_q) if q == q and q <= alpha]
    res.n_bh_significant = len(bh_sig)
    res.bh_cutoff_p = max((o.p_value for o in bh_sig), default=0.0)

    chosen = bh_sig if method == "bh" else [
        o for o, q in zip(outcomes, bon_q) if q == q and q <= alpha]
    res.significant_labels = tuple(o.label for o in chosen)
    return res


# ==========================================================================
# PURE DOMAIN LOGIC -- still no database
# ==========================================================================

def params_hash(params: dict) -> str:
    """Stable 12-hex-char fingerprint of a parameter set.

    Same construction as engine/backtest.py::_params_hash, so a hash computed
    there is comparable here. sort_keys makes it order-independent, which
    matters: {'a':1,'b':2} and {'b':2,'a':1} are the same experiment.
    """
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class PreRegistration:
    """A hypothesis, fixed in place BEFORE the evidence is seen."""
    prereg_id: str
    registered_at: str
    family: str
    strategy: str
    variant: str
    params: dict
    params_hash: str
    hypothesis: str
    expectation: str                 # clears | fails
    pass_expectancy_r: float
    pass_min_trades: int
    alpha: float = DEFAULT_ALPHA
    data_window: str = ""
    holdout: str = ""
    author: str = ""
    revision: int = 0
    supersedes: str | None = None
    note: str = ""

    def __str__(self) -> str:
        rev = f" rev{self.revision}" if self.revision else ""
        return (f"{self.prereg_id}{rev} [{self.family}] "
                f"{self.strategy}/{self.variant} "
                f"({self.params_hash})  expects to {self.expectation.upper()} "
                f">= {self.pass_expectancy_r:+.3f}R on >= {self.pass_min_trades} trades")


@dataclass(frozen=True)
class ResultCheck:
    """Did the evidence meet the bar that was written down beforehand?"""
    meets_threshold: bool
    hypothesis_confirmed: bool
    reasons: tuple[str, ...]

    def __str__(self) -> str:
        return ("PASS" if self.meets_threshold else "FAIL") + \
            (" (as predicted)" if self.hypothesis_confirmed else " (PREDICTION WRONG)")


def evaluate_result(prereg: PreRegistration, n: int,
                    expectancy_r: float) -> ResultCheck:
    """Compare a completed run against its own pre-registered threshold. Pure.

    Two separate verdicts, never conflated:
      meets_threshold      -- is this a candidate winner?
      hypothesis_confirmed -- did we correctly predict what would happen?

    The second is how you find out whether your priors are any good, and a
    correctly-predicted failure is a genuine result, not a non-event.
    """
    reasons: list[str] = []
    enough = n >= prereg.pass_min_trades
    if not enough:
        reasons.append(
            f"only {n} trades, pre-registered minimum was {prereg.pass_min_trades}")
    clears = expectancy_r == expectancy_r and expectancy_r >= prereg.pass_expectancy_r
    if expectancy_r != expectancy_r:
        reasons.append("expectancy is NaN -- unscoreable, treated as a failure")
    elif clears:
        reasons.append(
            f"expectancy {expectancy_r:+.3f}R >= threshold "
            f"{prereg.pass_expectancy_r:+.3f}R")
    else:
        reasons.append(
            f"expectancy {expectancy_r:+.3f}R < threshold "
            f"{prereg.pass_expectancy_r:+.3f}R")

    meets = bool(enough and clears)
    predicted_pass = (prereg.expectation == EXPECTATION_CLEARS)
    confirmed = (meets == predicted_pass)
    reasons.append(
        f"pre-registered expectation was {prereg.expectation.upper()} -- "
        + ("CONFIRMED" if confirmed else "WRONG"))
    return ResultCheck(meets_threshold=meets, hypothesis_confirmed=confirmed,
                       reasons=tuple(reasons))


# ==========================================================================
# STORAGE
# ==========================================================================

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- WRITE-ONCE pre-registrations. The hypothesis, the exact parameters and the
-- pass threshold, all fixed BEFORE the backtest runs. Anti-backfill is enforced
-- by the database, not by honour -- same design as data_layer.predictions.
--
-- UNIQUE (family, strategy, variant, params_hash, data_window, revision) means
-- the same experiment cannot be pre-registered twice at the same revision. That
-- is deliberate: silently re-registering and re-running until it passes is
-- p-hacking, so it is refused loudly.
--
-- `revision` is what makes supersede() possible without weakening that. A fresh
-- pre-registration is always revision 0; supersede() writes revision N+1 with a
-- `supersedes` pointer. So revising a threshold is allowed and NUMBERED, while
-- re-registering the identical experiment from scratch is still refused. (The
-- obvious alternative -- putting `supersedes` itself in the UNIQUE tuple --
-- does NOT work: SQLite treats NULLs as distinct, so every original would
-- collide-free with every other original and the duplicate guard would
-- silently do nothing.)
CREATE TABLE IF NOT EXISTS preregistrations (
    id                 INTEGER PRIMARY KEY,
    prereg_id          TEXT    NOT NULL UNIQUE,
    registered_at      TEXT    NOT NULL,
    family             TEXT    NOT NULL,
    strategy           TEXT    NOT NULL,
    variant            TEXT    NOT NULL DEFAULT 'base',
    params_json        TEXT    NOT NULL,
    params_hash        TEXT    NOT NULL,
    hypothesis         TEXT    NOT NULL,
    expectation        TEXT    NOT NULL CHECK (expectation IN ('clears','fails')),
    pass_expectancy_r  REAL    NOT NULL,
    pass_min_trades    INTEGER NOT NULL CHECK (pass_min_trades > 0),
    alpha              REAL    NOT NULL CHECK (alpha > 0 AND alpha < 1),
    data_window        TEXT    NOT NULL DEFAULT '',
    holdout            TEXT    NOT NULL DEFAULT '',
    author             TEXT    NOT NULL DEFAULT '',
    revision           INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
    supersedes         TEXT    REFERENCES preregistrations(prereg_id),
    note               TEXT    NOT NULL DEFAULT '',
    UNIQUE (family, strategy, variant, params_hash, data_window, revision)
);
CREATE TRIGGER IF NOT EXISTS preregistrations_no_update
BEFORE UPDATE ON preregistrations
BEGIN SELECT RAISE(ABORT,
  'pre-registrations are WRITE-ONCE (anti-backfill): use supersede()'); END;
CREATE TRIGGER IF NOT EXISTS preregistrations_no_delete
BEFORE DELETE ON preregistrations
BEGIN SELECT RAISE(ABORT,
  'pre-registrations are WRITE-ONCE (anti-backfill): they cannot be deleted'); END;

-- One completed run. Also write-once: editing a failure into a pass must be
-- impossible, or the whole exercise is theatre. MANY results may point at one
-- pre-registration -- re-running the same hypothesis is allowed but COUNTED and
-- surfaced, because repeated runs of one hypothesis is itself a p-hacking
-- signature.
CREATE TABLE IF NOT EXISTS results (
    id                   INTEGER PRIMARY KEY,
    run_id               TEXT    NOT NULL UNIQUE,
    prereg_id            TEXT    NOT NULL REFERENCES preregistrations(prereg_id),
    run_at               TEXT    NOT NULL,
    params_json          TEXT    NOT NULL,   -- params ACTUALLY used
    params_hash          TEXT    NOT NULL,
    params_match         INTEGER NOT NULL,   -- 0 = ran something else than registered
    n                    INTEGER NOT NULL,
    expectancy_r         REAL,
    expectancy_r_gross   REAL,
    stdev_r              REAL,
    ci_low               REAL,
    ci_high              REAL,
    win_rate             REAL,
    profit_factor        REAL,
    max_drawdown         REAL,
    net_pnl              REAL,
    sharpe_per_trade     REAL,
    p_value              REAL,
    p_value_source       TEXT    NOT NULL DEFAULT 't',   -- t | bootstrap | supplied
    scorecard_significant INTEGER NOT NULL DEFAULT 0,
    gate_pass            INTEGER NOT NULL DEFAULT 0,
    meets_threshold      INTEGER NOT NULL,
    hypothesis_confirmed INTEGER NOT NULL,
    reasons              TEXT    NOT NULL DEFAULT '',
    scorecard_json       TEXT    NOT NULL DEFAULT '{}',
    note                 TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ix_results_prereg ON results(prereg_id);
CREATE TRIGGER IF NOT EXISTS results_no_update
BEFORE UPDATE ON results
BEGIN SELECT RAISE(ABORT,
  'results are WRITE-ONCE: a recorded outcome cannot be edited'); END;
CREATE TRIGGER IF NOT EXISTS results_no_delete
BEFORE DELETE ON results
BEGIN SELECT RAISE(ABORT,
  'results are WRITE-ONCE: a recorded outcome cannot be deleted'); END;

-- THE ANTI-BACKFILL TRIGGER THAT MATTERS MOST. A result timestamped before its
-- own pre-registration is a prediction written after the answer was known.
CREATE TRIGGER IF NOT EXISTS results_no_predating
BEFORE INSERT ON results
WHEN NEW.run_at < (SELECT registered_at FROM preregistrations
                   WHERE prereg_id = NEW.prereg_id)
BEGIN SELECT RAISE(ABORT,
  'result run_at PRE-DATES its pre-registration (anti-backfill)'); END;
"""

GUARD_TRIGGERS = (
    "preregistrations_no_update",
    "preregistrations_no_delete",
    "results_no_update",
    "results_no_delete",
    "results_no_predating",
)


def verify_integrity(con: sqlite3.Connection) -> None:
    """Refuse to use a registry whose write-once guards have been removed.

    Cannot stop a determined operator dropping the triggers with the sqlite3
    CLI -- but it can make sure the next process to open the file NOTICES,
    rather than writing on top of tampered history as if nothing happened.
    """
    have = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger'").fetchall()}
    missing = [t for t in GUARD_TRIGGERS if t not in have]
    if missing:
        raise IntegrityCompromised(
            "write-once guards missing from registry database: "
            + ", ".join(missing)
            + " -- this file has been tampered with, or was created by an "
              "older schema. Do not trust its contents.")
    if not con.execute("PRAGMA recursive_triggers").fetchone()[0]:
        raise IntegrityCompromised(
            "recursive_triggers is OFF -- INSERT OR REPLACE would bypass the "
            "delete guard and silently overwrite a pre-registration.")


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    """Open (creating if needed) the registry. NEVER touches market.db."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    # MUST come after executescript: SQLite docs are explicit that REPLACE only
    # fires delete triggers when recursive_triggers is enabled. Without this
    # line `INSERT OR REPLACE INTO preregistrations ...` overwrites a row and
    # the no_delete trigger never runs. Tested in test_registry.py.
    con.execute("PRAGMA recursive_triggers = ON")
    con.execute("PRAGMA foreign_keys = ON")
    verify_integrity(con)
    return con


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _json_safe(obj: Any) -> Any:
    """Replace non-finite floats with None so the stored JSON stays valid JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _dump(obj: Any) -> str:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)
    return json.dumps(_json_safe(obj), sort_keys=True, default=str)


def _num(x: Any) -> float | None:
    """SQLite stores NaN as NULL anyway; be explicit about it."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ---- pre-registration ----------------------------------------------------

def register(con: sqlite3.Connection, *, family: str, strategy: str,
             params: dict, hypothesis: str,
             pass_expectancy_r: float, pass_min_trades: int,
             expectation: str = EXPECTATION_CLEARS,
             variant: str = "base", alpha: float = DEFAULT_ALPHA,
             data_window: str = "", holdout: str = "", author: str = "",
             revision: int = 0, supersedes: str | None = None, note: str = "",
             registered_at: str | None = None) -> PreRegistration:
    """Record a hypothesis BEFORE running the backtest. Write-once, forever.

    `family` is the multiple-comparisons universe -- the set of tests the
    correction will be applied across. It must be declared HERE, before any
    result is known, because choosing the family after the fact (to shrink m and
    loosen the threshold) is itself p-hacking.

    Raises DuplicatePreregistration if this exact experiment is already
    registered. That is a feature: re-running one hypothesis until it passes is
    the single easiest way to manufacture a fake winner.
    """
    if expectation not in EXPECTATIONS:
        raise ValueError(
            f"expectation must be one of {EXPECTATIONS}, got {expectation!r}")
    if pass_min_trades < 1:
        raise ValueError("pass_min_trades must be >= 1")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    if not hypothesis.strip():
        raise ValueError(
            "hypothesis text is required -- a pre-registration with no stated "
            "expectation is not a pre-registration")
    if revision and supersedes is None:
        raise ValueError(
            "revision > 0 requires `supersedes`. Otherwise bumping the revision "
            "would be a way to re-register the identical experiment and slip "
            "past the duplicate guard.")

    ph = params_hash(params)
    at = registered_at or _now()
    nid = con.execute(
        "SELECT COALESCE(MAX(id), 0) + 1 FROM preregistrations").fetchone()[0]
    prereg_id = f"PR-{nid:04d}"

    if supersedes is not None and get_preregistration(con, supersedes) is None:
        raise NoPreregistration(
            f"cannot supersede {supersedes!r}: no such pre-registration")

    try:
        con.execute("""
            INSERT INTO preregistrations
            (id, prereg_id, registered_at, family, strategy, variant,
             params_json, params_hash, hypothesis, expectation,
             pass_expectancy_r, pass_min_trades, alpha, data_window, holdout,
             author, revision, supersedes, note)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (nid, prereg_id, at, family, strategy, variant,
              _dump(params), ph, hypothesis, expectation,
              float(pass_expectancy_r), int(pass_min_trades), float(alpha),
              data_window, holdout, author, int(revision), supersedes, note))
        con.commit()
    except sqlite3.IntegrityError as e:
        con.rollback()
        existing = con.execute("""
            SELECT prereg_id FROM preregistrations
            WHERE family=? AND strategy=? AND variant=? AND params_hash=?
              AND data_window=? AND revision=?
        """, (family, strategy, variant, ph, data_window,
              int(revision))).fetchone()
        if existing:
            raise DuplicatePreregistration(
                f"{strategy}/{variant} with params {ph} on window "
                f"{data_window!r} (revision {revision}) is already "
                f"pre-registered as {existing['prereg_id']}. Re-running one "
                f"hypothesis until it passes is p-hacking. Use supersede() if "
                f"the design genuinely changed.") from e
        raise

    return _row_to_prereg(con.execute(
        "SELECT * FROM preregistrations WHERE prereg_id=?",
        (prereg_id,)).fetchone())


def supersede(con: sqlite3.Connection, prereg_id: str, *,
              reason: str, **changes) -> PreRegistration:
    """The ONLY legitimate way to revise a pre-registration.

    Writes a NEW pre-registration carrying `supersedes=prereg_id`. The original
    row survives untouched and still counts as a test that was registered. You
    cannot make an inconvenient hypothesis disappear; you can only add to the
    record that you changed your mind, and say why.
    """
    old = get_preregistration(con, prereg_id)
    if old is None:
        raise NoPreregistration(f"no pre-registration {prereg_id!r}")
    base = dict(
        family=old.family, strategy=old.strategy, variant=old.variant,
        params=old.params, hypothesis=old.hypothesis,
        expectation=old.expectation,
        pass_expectancy_r=old.pass_expectancy_r,
        pass_min_trades=old.pass_min_trades, alpha=old.alpha,
        data_window=old.data_window, holdout=old.holdout, author=old.author,
    )
    base.update(changes)
    base["supersedes"] = prereg_id
    # Next revision in this experiment's own chain. A revision bump is the only
    # thing that clears the duplicate-protection UNIQUE index, and it is only
    # reachable from here.
    base["revision"] = _next_revision(
        con, base["family"], base["strategy"], base["variant"],
        params_hash(base["params"]), base["data_window"])
    base["note"] = f"supersedes {prereg_id} (rev {old.revision}): {reason}"
    return register(con, **base)


def _next_revision(con: sqlite3.Connection, family: str, strategy: str,
                   variant: str, ph: str, data_window: str) -> int:
    row = con.execute("""
        SELECT COALESCE(MAX(revision), -1) + 1 FROM preregistrations
        WHERE family=? AND strategy=? AND variant=? AND params_hash=?
          AND data_window=?""",
        (family, strategy, variant, ph, data_window)).fetchone()
    return int(row[0])


def amend(con: sqlite3.Connection, prereg_id: str, **fields):
    """Always raises. Present so the answer to 'how do I edit this' is explicit."""
    raise WriteOnceViolation(
        f"pre-registrations are write-once; {prereg_id} cannot be amended. "
        f"Use supersede() -- it records the change instead of hiding it.")


def _row_to_prereg(row: sqlite3.Row | None) -> PreRegistration | None:
    if row is None:
        return None
    return PreRegistration(
        prereg_id=row["prereg_id"], registered_at=row["registered_at"],
        family=row["family"], strategy=row["strategy"], variant=row["variant"],
        params=json.loads(row["params_json"]), params_hash=row["params_hash"],
        hypothesis=row["hypothesis"], expectation=row["expectation"],
        pass_expectancy_r=row["pass_expectancy_r"],
        pass_min_trades=row["pass_min_trades"], alpha=row["alpha"],
        data_window=row["data_window"], holdout=row["holdout"],
        author=row["author"], revision=row["revision"],
        supersedes=row["supersedes"], note=row["note"])


def get_preregistration(con: sqlite3.Connection,
                        prereg_id: str) -> PreRegistration | None:
    return _row_to_prereg(con.execute(
        "SELECT * FROM preregistrations WHERE prereg_id=?",
        (prereg_id,)).fetchone())


def list_preregistrations(con: sqlite3.Connection, family: str | None = None,
                          strategy: str | None = None) -> list[PreRegistration]:
    sql = "SELECT * FROM preregistrations"
    where, args = [], []
    if family:
        where.append("family=?")
        args.append(family)
    if strategy:
        where.append("strategy=?")
        args.append(strategy)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id"
    return [_row_to_prereg(r) for r in con.execute(sql, args).fetchall()]


# ---- results -------------------------------------------------------------

_SC_FIELDS = (
    ("expectancy_r", "expectancy_r"),
    ("expectancy_r_gross", "expectancy_r_gross"),
    ("stdev_r", "stdev_r"),
    ("ci_low", "ci_boot_low"),
    ("ci_high", "ci_boot_high"),
    ("win_rate", "win_rate"),
    ("profit_factor", "profit_factor"),
    ("max_drawdown", "max_drawdown"),
    ("net_pnl", "net_pnl"),
    ("sharpe_per_trade", "sharpe_per_trade"),
)


def record_result(con: sqlite3.Connection, prereg_id: str, scorecard, *,
                  params: dict | None = None, p_value: float | None = None,
                  p_value_source: str | None = None,
                  run_at: str | None = None, note: str = "") -> dict:
    """Store a completed ScoreCard against its pre-registration.

    `scorecard` is anything with score.ScoreCard's attributes -- duck-typed on
    purpose so registry.py stays importable without pulling in the scorer.

    `params` is what was ACTUALLY run. If it differs from what was
    pre-registered, the row records both and flags params_match=0. Silently
    swapping parameters between registration and run would defeat the entire
    mechanism, so it is recorded rather than rejected: the deviation is data.

    A result whose run_at pre-dates the pre-registration is refused by the
    database, not by this function.
    """
    prereg = get_preregistration(con, prereg_id)
    if prereg is None:
        raise NoPreregistration(
            f"no pre-registration {prereg_id!r} -- every result must have a "
            f"hypothesis recorded BEFORE it. Call register() first.")

    used = prereg.params if params is None else params
    used_hash = params_hash(used)
    n = int(getattr(scorecard, "n", 0) or 0)
    exp_r = _num(getattr(scorecard, "expectancy_r", float("nan")))
    stdev = _num(getattr(scorecard, "stdev_r", float("nan")))

    if p_value is None:
        p_value = p_value_from_summary(
            exp_r if exp_r is not None else float("nan"),
            stdev if stdev is not None else float("nan"), n)
        p_value_source = p_value_source or "t"
    else:
        p_value_source = p_value_source or "supplied"

    check = evaluate_result(prereg, n,
                            exp_r if exp_r is not None else float("nan"))

    at = run_at or _now()
    nid = con.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM results").fetchone()[0]
    run_id = f"RUN-{nid:04d}"

    vals = {col: _num(getattr(scorecard, attr, None)) for col, attr in _SC_FIELDS}

    con.execute("""
        INSERT INTO results
        (id, run_id, prereg_id, run_at, params_json, params_hash, params_match,
         n, expectancy_r, expectancy_r_gross, stdev_r, ci_low, ci_high,
         win_rate, profit_factor, max_drawdown, net_pnl, sharpe_per_trade,
         p_value, p_value_source, scorecard_significant, gate_pass,
         meets_threshold, hypothesis_confirmed, reasons, scorecard_json, note)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (nid, run_id, prereg_id, at, _dump(used), used_hash,
          int(used_hash == prereg.params_hash), n,
          vals["expectancy_r"], vals["expectancy_r_gross"], vals["stdev_r"],
          vals["ci_low"], vals["ci_high"], vals["win_rate"],
          vals["profit_factor"], vals["max_drawdown"], vals["net_pnl"],
          vals["sharpe_per_trade"], _num(p_value), p_value_source,
          int(bool(getattr(scorecard, "significant", False))),
          int(bool(getattr(scorecard, "gate_pass", False))),
          int(check.meets_threshold), int(check.hypothesis_confirmed),
          " | ".join(check.reasons), _dump(scorecard), note))
    con.commit()

    return {"run_id": run_id, "prereg_id": prereg_id, "run_at": at,
            "meets_threshold": check.meets_threshold,
            "hypothesis_confirmed": check.hypothesis_confirmed,
            "p_value": p_value, "p_value_source": p_value_source,
            "params_match": used_hash == prereg.params_hash,
            "reasons": check.reasons}


def list_results(con: sqlite3.Connection, family: str | None = None,
                 strategy: str | None = None,
                 prereg_id: str | None = None) -> list[sqlite3.Row]:
    sql = """SELECT r.*, p.family, p.strategy, p.variant, p.expectation,
                    p.pass_expectancy_r, p.pass_min_trades, p.hypothesis
             FROM results r JOIN preregistrations p USING (prereg_id)"""
    where, args = [], []
    if family:
        where.append("p.family=?")
        args.append(family)
    if strategy:
        where.append("p.strategy=?")
        args.append(strategy)
    if prereg_id:
        where.append("r.prereg_id=?")
        args.append(prereg_id)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY r.id"
    return con.execute(sql, args).fetchall()


# ---- test counting -------------------------------------------------------

@dataclass
class TestCount:
    scope: str
    n_preregistered: int = 0
    n_runs: int = 0
    n_distinct_hypotheses_run: int = 0
    n_pending: int = 0            # registered, never run -- the silent failures
    n_passed: int = 0
    n_failed: int = 0
    n_reruns: int = 0             # runs beyond the first for a hypothesis
    n_param_deviations: int = 0
    n_superseded: int = 0

    @property
    def pass_rate(self) -> float:
        return self.n_passed / self.n_runs if self.n_runs else float("nan")

    def __str__(self) -> str:
        return (f"tests against {self.scope}: {self.n_runs} run "
                f"({self.n_passed} passed, {self.n_failed} failed), "
                f"{self.n_preregistered} pre-registered, "
                f"{self.n_pending} never run")


def test_count(con: sqlite3.Connection, family: str | None = None,
               strategy: str | None = None) -> TestCount:
    """How many tests have been run against this strategy / hypothesis family.

    The headline number the architecture doc asks for: "a winner found on the
    40th test is a very different claim from one found on the 2nd."
    """
    scope = family or strategy or "ALL"
    pres = list_preregistrations(con, family=family, strategy=strategy)
    rows = list_results(con, family=family, strategy=strategy)
    tc = TestCount(scope=scope, n_preregistered=len(pres), n_runs=len(rows))
    run_ids = [r["prereg_id"] for r in rows]
    tc.n_distinct_hypotheses_run = len(set(run_ids))
    tc.n_pending = len([p for p in pres if p.prereg_id not in set(run_ids)])
    tc.n_passed = sum(1 for r in rows if r["meets_threshold"])
    tc.n_failed = tc.n_runs - tc.n_passed
    tc.n_reruns = tc.n_runs - tc.n_distinct_hypotheses_run
    tc.n_param_deviations = sum(1 for r in rows if not r["params_match"])
    tc.n_superseded = sum(1 for p in pres if p.supersedes)
    return tc


def ordinal_of(con: sqlite3.Connection, run_id: str,
               family: str | None = None) -> tuple[int, int]:
    """(this run's position, total runs) within its family. 1-based.

    Exists so a result can never be quoted without its test number attached.
    """
    row = con.execute("""SELECT r.id, p.family FROM results r
                         JOIN preregistrations p USING (prereg_id)
                         WHERE r.run_id=?""", (run_id,)).fetchone()
    if row is None:
        raise KeyError(run_id)
    fam = family or row["family"]
    rows = list_results(con, family=fam)
    ids = [r["id"] for r in rows]
    return (ids.index(row["id"]) + 1, len(ids))


# ---- summary -------------------------------------------------------------

@dataclass
class RegistrySummary:
    scope: str = "ALL"
    counts: TestCount = field(default_factory=lambda: TestCount("ALL"))
    correction: CorrectionResult = field(
        default_factory=lambda: CorrectionResult("bh", DEFAULT_ALPHA, 0))
    alpha: float = DEFAULT_ALPHA
    alpha_note: str = ""
    winners: list[dict] = field(default_factory=list)
    losers: list[dict] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_hypotheses_confirmed: int = 0

    @property
    def survives_correction(self) -> list[str]:
        return list(self.correction.significant_labels)

    def __str__(self) -> str:
        c, mc = self.counts, self.correction
        lines = [
            f"=== REGISTRY SUMMARY: {self.scope} ===",
            f"  pre-registered      {c.n_preregistered}"
            f"   ({c.n_superseded} superseding an earlier one)",
            f"  tests RUN           {c.n_runs}"
            f"   over {c.n_distinct_hypotheses_run} distinct hypotheses"
            + (f"   [{c.n_reruns} RE-RUN of an already-tested hypothesis]"
               if c.n_reruns else ""),
            f"  passed threshold    {c.n_passed}"
            f"   failed {c.n_failed}"
            + (f"   pass rate {c.pass_rate:.0%}" if c.n_runs else ""),
            f"  prediction correct  {self.n_hypotheses_confirmed}/{c.n_runs}"
            f"   (how good our priors are, independent of pass/fail)",
            f"  NEVER RUN           {c.n_pending}"
            f"   pre-registered but no result recorded",
            "",
            "  " + str(mc).replace("\n", "\n  "),
            "",
            f"  SURVIVES {mc.method.upper()} CORRECTION: "
            + (", ".join(mc.significant_labels) if mc.significant_labels
               else "NOTHING"),
        ]
        if self.alpha_note:
            lines.append(f"  ! {self.alpha_note}")
        lines.append("")
        lines.append("  --- EVERY TEST, PASSES AND FAILURES ALIKE ---")
        for row in sorted(self.winners + self.losers,
                          key=lambda x: x["ordinal"]):
            lines.append(
                f"    #{row['ordinal']:>3}/{c.n_runs}  {row['run_id']}  "
                f"{row['label']:<34} n={row['n']:<4} "
                f"E={row['expectancy_r']:+.3f}R  p={row['p_value']:.4f}  "
                f"q={row['q_value']:.4f}  "
                f"{'PASS' if row['meets_threshold'] else 'fail'}"
                f"{'  <-- SURVIVES CORRECTION' if row['survives'] else ''}")
        if self.pending:
            lines.append("  --- PRE-REGISTERED, NEVER RUN (unreported tests) ---")
            for p in self.pending:
                lines.append(f"    {p}")
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


def summary(con: sqlite3.Connection, family: str | None = None,
            strategy: str | None = None, alpha: float | None = None,
            method: str = "bh") -> RegistrySummary:
    """Total tests run, how many passed, and the corrected threshold.

    Deliberately reports the failures and the never-run hypotheses in the same
    object as the winners. There is no code path here that returns only the
    winners -- that is the requirement, and it is met structurally rather than
    by convention.
    """
    scope = family or strategy or "ALL"
    s = RegistrySummary(scope=scope)
    s.counts = test_count(con, family=family, strategy=strategy)
    rows = list_results(con, family=family, strategy=strategy)

    pres = list_preregistrations(con, family=family, strategy=strategy)
    alphas = {p.alpha for p in pres}
    if alpha is not None:
        s.alpha = alpha
    elif len(alphas) == 1:
        s.alpha = alphas.pop()
    elif alphas:
        s.alpha = max(alphas)
        s.alpha_note = (f"pre-registrations disagree on alpha {sorted(alphas)}; "
                        f"using the LOOSEST ({s.alpha}) and flagging it")
    else:
        s.alpha = DEFAULT_ALPHA

    outcomes = [TestOutcome(label=r["run_id"],
                            p_value=(r["p_value"] if r["p_value"] is not None
                                     else float("nan")),
                            meets_threshold=bool(r["meets_threshold"]))
                for r in rows]
    s.correction = correct(outcomes, alpha=s.alpha, method=method)
    surviving = set(s.correction.significant_labels)
    s.n_hypotheses_confirmed = sum(1 for r in rows if r["hypothesis_confirmed"])

    for i, r in enumerate(rows, start=1):
        item = {
            "ordinal": i, "run_id": r["run_id"], "prereg_id": r["prereg_id"],
            "label": f"{r['strategy']}/{r['variant']}",
            "n": r["n"],
            "expectancy_r": r["expectancy_r"] if r["expectancy_r"] is not None
                            else float("nan"),
            "p_value": r["p_value"] if r["p_value"] is not None else float("nan"),
            "q_value": s.correction.adjusted.get(r["run_id"], float("nan")),
            "meets_threshold": bool(r["meets_threshold"]),
            "hypothesis_confirmed": bool(r["hypothesis_confirmed"]),
            "survives": r["run_id"] in surviving,
            "params_match": bool(r["params_match"]),
        }
        (s.winners if item["meets_threshold"] else s.losers).append(item)

    run_pres = {r["prereg_id"] for r in rows}
    s.pending = [str(p) for p in pres if p.prereg_id not in run_pres]

    # ---- warnings that must never be silently dropped -------------------
    if s.counts.n_pending:
        s.warnings.append(
            f"{s.counts.n_pending} pre-registered hypotheses have NO recorded "
            f"result. Either run them or supersede them -- an unreported test "
            f"is still a test.")
    if s.counts.n_reruns:
        s.warnings.append(
            f"{s.counts.n_reruns} run(s) repeat an already-tested hypothesis. "
            f"Repeated runs of one hypothesis is a p-hacking signature.")
    if s.counts.n_param_deviations:
        s.warnings.append(
            f"{s.counts.n_param_deviations} run(s) used parameters DIFFERENT "
            f"from what was pre-registered.")
    if s.counts.n_passed and s.counts.n_passed <= s.correction.expected_false_positives:
        s.warnings.append(
            f"{s.counts.n_passed} winner(s) found, but chance alone predicts "
            f"~{s.correction.expected_false_positives:.1f} at {s.counts.n_runs} "
            f"tests. This result set is INDISTINGUISHABLE FROM NOISE.")
    if s.counts.n_runs and not s.correction.significant_labels:
        s.warnings.append(
            "nothing survives the multiple-comparisons correction. Any 'winner' "
            "above is a candidate for forward paper-testing only, never a "
            "validated edge.")
    return s


def families(con: sqlite3.Connection) -> list[dict]:
    """Family sizes, so a suspiciously small correction universe is visible."""
    return [dict(r) for r in con.execute("""
        SELECT p.family,
               COUNT(DISTINCT p.prereg_id) AS n_preregistered,
               COUNT(r.id)                 AS n_runs,
               SUM(COALESCE(r.meets_threshold, 0)) AS n_passed
        FROM preregistrations p LEFT JOIN results r USING (prereg_id)
        GROUP BY p.family ORDER BY p.family""").fetchall()]


def threshold_table(alpha: float = DEFAULT_ALPHA,
                    ms: Sequence[int] = (1, 2, 5, 10, 20, 40, 84)) -> str:
    """How the corrected bar tightens as the programme runs more tests. Pure."""
    lines = [f"  tests (m) | Bonferroni p< | expected false positives at p<{alpha}",
             "  ----------+---------------+--------------------------------"]
    for m in ms:
        lines.append(f"  {m:>9} | {bonferroni_threshold(alpha, m):>13.5f} |"
                     f" {m * alpha:>6.2f}")
    return "\n".join(lines)


if __name__ == "__main__":                                   # pragma: no cover
    import sys
    con = connect()
    fam = sys.argv[1] if len(sys.argv) > 1 else None
    print(summary(con, family=fam))
    print()
    print("Family sizes (the declared correction universes):")
    for f in families(con):
        print(f"  {f}")
    print()
    print(threshold_table())
