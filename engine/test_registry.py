"""
engine/test_registry.py -- verification for the results registry (component G).

The registry exists to stop the programme fooling itself (architecture doc
S4.5). So these tests attack it the way a motivated user would:

  * do the multiple-comparisons corrections match HAND-COMPUTED numbers, and
    the published Benjamini-Hochberg 1995 worked example?
  * can a pre-registration be edited or deleted by ANY route -- UPDATE, DELETE,
    INSERT OR REPLACE, a back-dated result, the public API?
  * does the summary make it possible to report only the winners?

Every database in this file lives in a temp directory and is deleted at the
end. data/registry.db and data/market.db are never opened.
"""

from __future__ import annotations

import datetime as dt
import math
import random
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from registry import (  # noqa: E402
    DEFAULT_ALPHA, DuplicatePreregistration, IntegrityCompromised,
    NoPreregistration, PreRegistration, SCHEMA, TestOutcome, WriteOnceViolation,
    amend, benjamini_hochberg_adjusted, bonferroni_adjusted,
    bonferroni_threshold, bootstrap_p_value, connect, correct, evaluate_result,
    families, list_preregistrations, list_results, ordinal_of, params_hash,
    p_value_from_summary, record_result, reg_incomplete_beta, register,
    summary, supersede, t_stat_from_summary, t_test_p_two_sided, test_count,
    threshold_table, verify_integrity,
)
from score import Trade, score, t_crit_95  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []

TMPDIR = Path(tempfile.mkdtemp(prefix="registry_test_"))
_OPEN: list[sqlite3.Connection] = []


def check(label: str, got, want, tol: float | None = None):
    if tol is not None:
        ok = (got == got) and (want == want) and abs(got - want) <= tol
    else:
        ok = got == want
    (PASS if ok else FAIL).append(label)
    extra = "" if ok else f"   (got {got!r}, want {want!r})"
    print(f"  [{'ok  ' if ok else 'FAIL'}] {label}{extra}")


def new_db(name: str):
    con = connect(TMPDIR / f"{name}.db")
    _OPEN.append(con)
    return con


def raises(fn, *args, **kw) -> tuple[bool, str]:
    """Returns (did_it_raise, message)."""
    try:
        fn(*args, **kw)
    except Exception as e:                                   # noqa: BLE001
        return True, f"{type(e).__name__}: {e}"
    return False, "NO EXCEPTION -- the write succeeded"


try:
    # ======================================================================
    print("=" * 96)
    print("TEST 1 -- Student-t p-values, checked against score.py's own 95% table")
    print("=" * 96)
    print("  score.py already tabulates exact two-sided 95% critical values.")
    print("  Feeding t_crit_95(df) back in MUST return p = 0.05000. That is an")
    print("  independent hand-check of the incomplete-beta implementation.")
    for df in (1, 5, 10, 19, 29):
        check(f"t = t_crit_95({df}) = {t_crit_95(df):.3f} -> p = 0.0500",
              t_test_p_two_sided(t_crit_95(df), df), 0.05, 5e-4)
    check("I_x(a,b) at x=0 is 0", reg_incomplete_beta(2.0, 3.0, 0.0), 0.0, 1e-12)
    check("I_x(a,b) at x=1 is 1", reg_incomplete_beta(2.0, 3.0, 1.0), 1.0, 1e-12)
    check("I_0.5(1,1) = 0.5 (uniform)", reg_incomplete_beta(1.0, 1.0, 0.5),
          0.5, 1e-12)
    check("t=0 -> p=1 (no evidence at all)", t_test_p_two_sided(0.0, 10), 1.0, 1e-12)
    check("df<1 -> nan", math.isnan(t_test_p_two_sided(2.0, 0)), True)
    check("nan t -> nan p", math.isnan(t_test_p_two_sided(float('nan'), 10)), True)
    # hand arithmetic: mean 0.5, sd 1.0, n 25 -> t = 0.5 / (1/5) = 2.5
    check("t stat: mean .5, sd 1, n 25 -> 2.50",
          t_stat_from_summary(0.5, 1.0, 25), 2.5, 1e-12)
    # Cross-checked independently by Simpson integration of the t pdf
    # (2 * integral from 2.5 to infinity, df=24) = 0.0196541751165690.
    # The incomplete-beta route agrees to 1e-14, which is two unrelated
    # numerical methods landing on the same number.
    check("  ...p for t=2.5, df=24 = 0.01965418 (checked by quadrature)",
          p_value_from_summary(0.5, 1.0, 25), 0.0196541751165690, 1e-12)
    check("n<2 -> nan p", math.isnan(p_value_from_summary(0.5, 1.0, 1)), True)
    check("sd=0 -> nan p (not infinite significance)",
          math.isnan(p_value_from_summary(0.5, 0.0, 20)), True)

    # ======================================================================
    print("\n" + "=" * 96)
    print("TEST 2 -- Bonferroni: the arithmetic is alpha/m, and nothing else")
    print("=" * 96)
    check("alpha .05, m=1  -> 0.05000", bonferroni_threshold(0.05, 1), 0.05, 1e-15)
    check("alpha .05, m=40 -> 0.00125", bonferroni_threshold(0.05, 40), 0.00125, 1e-15)
    check("alpha .05, m=84 -> 0.000595 (14 strategies x 6 variants)",
          bonferroni_threshold(0.05, 84), 0.05 / 84, 1e-15)
    check("m<1 -> nan", math.isnan(bonferroni_threshold(0.05, 0)), True)
    adj = bonferroni_adjusted([0.001, 0.01, 0.02, 0.5])
    check("adjusted p = min(1, m*p): 0.001*4 = 0.004", adj[0], 0.004, 1e-12)
    check("  0.01*4 = 0.04", adj[1], 0.04, 1e-12)
    check("  0.5*4 clips to 1.0, never above", adj[3], 1.0, 1e-12)
    print("        -> at 40 tests Bonferroni demands p < 0.00125. With ~52 weekly")
    print("           samples per test that is effectively unreachable, which is")
    print("           exactly why BH is the default here and Bonferroni is the")
    print("           stricter bar reported alongside it.")

    # ======================================================================
    print("\n" + "=" * 96)
    print("TEST 3 -- Benjamini-Hochberg vs the PUBLISHED 1995 worked example")
    print("=" * 96)
    bh95 = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
            0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.0000]
    print("  Benjamini & Hochberg (1995) Table 1: m=15, alpha=0.05.")
    print("  Their published answer is 4 rejections (raw alpha would give 9).")
    q = benjamini_hochberg_adjusted(bh95)
    n_bh = sum(1 for x in q if x <= 0.05)
    n_raw = sum(1 for p in bh95 if p < 0.05)
    check("raw p<0.05 would reject 9", n_raw, 9)
    check("BH at alpha=0.05 rejects exactly 4", n_bh, 4)
    check("  the 4th smallest p (0.0095) is the cutoff", max(
        p for p, x in zip(bh95, q) if x <= 0.05), 0.0095, 1e-12)
    check("  the 5th (0.0201) is NOT rejected", q[4] > 0.05, True)
    # hand arithmetic on the largest: q_(15) = 15/15 * 1.0 = 1.0
    check("q for the largest p: 15/15 * 1.0 = 1.0", q[14], 1.0, 1e-12)
    # q_(1) = min over running: 15/1*0.0001 = 0.0015
    check("q for the smallest p: 15/1 * 0.0001 = 0.0015", q[0], 0.0015, 1e-12)

    print("\n  Step-up property -- BH rejects everything BELOW the cutoff rank,")
    print("  including tests that fail their own i*alpha/m line:")
    p2 = [0.040, 0.050]
    q2 = benjamini_hochberg_adjusted(p2)
    check("  m=2: p=0.05 sits at rank 2, crit 2*.05/2 = 0.050 -> passes",
          q2[1] <= 0.05, True)
    check("  p=0.04 FAILS its own rank-1 crit of 0.025 ...", 0.040 > 0.025, True)
    check("  ... but BH rejects it anyway (step-up)", q2[0] <= 0.05, True)

    print("\n  Hand-computable q-values, m=5, p = .01 .02 .03 .04 .05:")
    q3 = benjamini_hochberg_adjusted([0.01, 0.02, 0.03, 0.04, 0.05])
    print("     m/i * p_(i) = 5/1*.01 = 5/2*.02 = ... = 0.05 for every rank")
    for i, x in enumerate(q3):
        check(f"    q[{i}] = 0.05", x, 0.05, 1e-12)
    check("  order-preservation: input order is returned, not sorted order",
          benjamini_hochberg_adjusted([0.5, 0.001])[1], 0.002, 1e-12)
    check("  empty input -> empty output", benjamini_hochberg_adjusted([]), [])
    qn = benjamini_hochberg_adjusted([0.001, float("nan"), 0.9])
    check("  a NaN p-value gets q=NaN ...", math.isnan(qn[1]), True)
    check("  ... but still counts toward m: q = 3/1*0.001 = 0.003",
          qn[0], 0.003, 1e-12)

    # ======================================================================
    print("\n" + "=" * 96)
    print("TEST 4 -- correct(): both methods always computed, never just the kind one")
    print("=" * 96)
    outs = [TestOutcome(f"T{i}", p) for i, p in enumerate(bh95)]
    c = correct(outs, alpha=0.05, method="bh")
    check("m = 15", c.m, 15)
    check("naive significant = 9", c.n_naive_significant, 9)
    check("BH significant = 4", c.n_bh_significant, 4)
    # Bonferroni: 0.05/15 = 0.003333 -> p = .0001, .0004, .0019 pass; .0095 does not
    check("Bonferroni threshold 0.05/15 = 0.003333",
          c.bonferroni_threshold, 0.05 / 15, 1e-12)
    check("Bonferroni significant = 3 (0.0095 > 0.003333)",
          c.n_bonferroni_significant, 3)
    check("  method=bh drives the verdict -> 4 labels", len(c.significant_labels), 4)
    cb = correct(outs, alpha=0.05, method="bonferroni")
    check("  method=bonferroni -> 3 labels", len(cb.significant_labels), 3)
    check("  ...and BH count is STILL reported on the Bonferroni result",
          cb.n_bh_significant, 4)
    check("expected false positives = m*alpha = 15*0.05 = 0.75",
          c.expected_false_positives, 0.75, 1e-12)
    check("BH cutoff p is the largest rejected raw p", c.bh_cutoff_p, 0.0095, 1e-12)
    check("empty family -> m=0, no crash", correct([], 0.05).m, 0)
    ok, msg = raises(correct, outs, 0.05, "holm")
    check("unknown method is refused", ok, True)

    print("\n  How the bar tightens as the programme runs more tests:")
    print(threshold_table())
    print("  -> Bonferroni falls 40x between 1 and 40 tests. That is correct")
    print("     behaviour, and it is also why it alone would end the programme.")

    # ======================================================================
    print("\n" + "=" * 96)
    print("TEST 5 -- bootstrap p-value (the honest option when raw R values exist)")
    print("=" * 96)
    rng = random.Random(7)
    strong = [rng.gauss(1.0, 0.5) for _ in range(40)]
    noise = [rng.gauss(0.0, 1.0) for _ in range(40)]
    p_strong = bootstrap_p_value(strong, iters=2000)
    p_noise = bootstrap_p_value(noise, iters=2000)
    check("a clearly non-zero mean gets a small p", p_strong < 0.01, True)
    check("pure noise does not", p_noise > 0.05, True)
    check("seeded -> identical on re-run",
          bootstrap_p_value(strong, iters=2000), p_strong, 0.0)
    check("p is never exactly 0 (the +1/+1 correction)", p_strong > 0.0, True)
    check("n<2 -> nan", math.isnan(bootstrap_p_value([1.0])), True)
    print(f"        strong sample p = {p_strong:.5f}   noise sample p = {p_noise:.5f}")

    # ======================================================================
    print("\n" + "=" * 96)
    print("TEST 6 -- pure domain logic: params_hash and evaluate_result")
    print("=" * 96)
    check("params_hash is order-independent",
          params_hash({"a": 1, "b": 2}), params_hash({"b": 2, "a": 1}))
    check("...but value-sensitive",
          params_hash({"a": 1}) != params_hash({"a": 2}), True)
    check("12 hex chars", len(params_hash({"a": 1})), 12)

    pr = PreRegistration(
        prereg_id="PR-TEST", registered_at="2026-08-16T09:00:00",
        family="F", strategy="S", variant="base", params={}, params_hash="x",
        hypothesis="h", expectation="clears", pass_expectancy_r=0.205,
        pass_min_trades=10)
    r = evaluate_result(pr, n=20, expectancy_r=0.30)
    check("clears threshold with enough trades -> PASS", r.meets_threshold, True)
    check("  and the prediction ('clears') was right", r.hypothesis_confirmed, True)
    r = evaluate_result(pr, n=20, expectancy_r=0.10)
    check("below threshold -> FAIL", r.meets_threshold, False)
    check("  and the prediction was WRONG", r.hypothesis_confirmed, False)
    r = evaluate_result(pr, n=4, expectancy_r=0.90)
    check("great expectancy but only 4 trades -> still FAIL", r.meets_threshold, False)
    check("  reason names the pre-registered minimum",
          any("minimum was 10" in x for x in r.reasons), True)
    r = evaluate_result(pr, n=20, expectancy_r=float("nan"))
    check("NaN expectancy is a failure, never a pass", r.meets_threshold, False)
    check("  reason says unscoreable",
          any("unscoreable" in x for x in r.reasons), True)
    check("exactly at the threshold passes (>=, not >)",
          evaluate_result(pr, 10, 0.205).meets_threshold, True)

    pr_fail = PreRegistration(
        prereg_id="PR-TEST2", registered_at="2026-08-16T09:00:00",
        family="F", strategy="S", variant="base", params={}, params_hash="x",
        hypothesis="we expect this to die on costs", expectation="fails",
        pass_expectancy_r=0.205, pass_min_trades=10)
    r = evaluate_result(pr_fail, n=20, expectancy_r=0.05)
    check("predicted FAILS and it failed -> hypothesis CONFIRMED",
          (r.meets_threshold, r.hypothesis_confirmed), (False, True))
    r = evaluate_result(pr_fail, n=20, expectancy_r=0.90)
    check("predicted FAILS but it passed -> hypothesis WRONG",
          (r.meets_threshold, r.hypothesis_confirmed), (True, False))
    print("        -> a correctly-predicted failure is a real result. Keeping")
    print("           'did it pass' and 'were we right' separate is what makes")
    print("           the registry able to score our priors, not just strategies.")

    # ======================================================================
    print("\n" + "=" * 96)
    print("TEST 7 -- storage: register / record / refuse")
    print("=" * 96)
    con = new_db("basic")
    p1 = register(con, family="TIER1", strategy="DirectionalCreditSpread",
                  variant="delta30", params={"short_delta": 0.30, "wing": 150},
                  hypothesis="0.30-delta short clears break-even net of costs",
                  pass_expectancy_r=0.205, pass_min_trades=10,
                  data_window="2024-01-01..2026-08-13", author="test")
    check("prereg id is sequential and readable", p1.prereg_id, "PR-0001")
    check("params hashed at registration", p1.params_hash,
          params_hash({"short_delta": 0.30, "wing": 150}))
    check("round-trips out of the DB", get := (
        list_preregistrations(con)[0].hypothesis), p1.hypothesis)

    ok, msg = raises(register, con, family="TIER1",
                     strategy="DirectionalCreditSpread", variant="delta30",
                     params={"wing": 150, "short_delta": 0.30},
                     hypothesis="sneaking the same test in again",
                     pass_expectancy_r=-9.0, pass_min_trades=1,
                     data_window="2024-01-01..2026-08-13")
    check("re-registering the SAME experiment is refused", ok, True)
    check("  ...with a DuplicatePreregistration, naming the original",
          "PR-0001" in msg and "Duplicate" in msg, True)
    print(f"        {msg[:110]}...")

    p2 = register(con, family="TIER1", strategy="DirectionalCreditSpread",
                  variant="delta20", params={"short_delta": 0.20, "wing": 150},
                  hypothesis="a further-OTM short still clears break-even",
                  pass_expectancy_r=0.205, pass_min_trades=10,
                  data_window="2024-01-01..2026-08-13")
    check("a genuinely different variant registers fine", p2.prereg_id, "PR-0002")

    ok, _ = raises(register, con, family="F", strategy="S", params={},
                   hypothesis="   ", pass_expectancy_r=0.0, pass_min_trades=10)
    check("empty hypothesis text is refused", ok, True)
    ok, _ = raises(register, con, family="F", strategy="S", params={},
                   hypothesis="h", pass_expectancy_r=0.0, pass_min_trades=10,
                   expectation="maybe")
    check("invalid expectation is refused", ok, True)
    ok, _ = raises(register, con, family="F", strategy="S", params={},
                   hypothesis="h", pass_expectancy_r=0.0, pass_min_trades=0)
    check("pass_min_trades=0 is refused", ok, True)
    ok, _ = raises(register, con, family="F", strategy="S", params={},
                   hypothesis="h", pass_expectancy_r=0.0, pass_min_trades=10,
                   alpha=1.5)
    check("alpha outside (0,1) is refused", ok, True)
    ok, _ = raises(register, con, family="F", strategy="S", params={},
                   hypothesis="h", pass_expectancy_r=0.0, pass_min_trades=10,
                   supersedes="PR-9999")
    check("superseding a non-existent prereg is refused", ok, True)


    def make_trades(tag: str, n: int, mean_r: float, sd_r: float, seed: int,
                    risk: float = 2000.0):
        """Trades whose NET R is exactly the drawn value -- so the scorer's
        expectancy is controlled to the decimal and nothing is hand-waved."""
        g = random.Random(seed)
        base = dt.date(2024, 1, 8)
        out = []
        for i in range(n):
            rr = g.gauss(mean_r, sd_r)
            entry = base + dt.timedelta(days=7 * i)
            out.append(Trade(trade_id=f"{tag}-{i:03d}", strategy=tag,
                             entry_date=entry.isoformat(),
                             exit_date=(entry + dt.timedelta(days=4)).isoformat(),
                             risk=risk, gross_pnl=rr * risk + 410.0,
                             costs=410.0, n_legs=1))
        return out

    sc_good = score(make_trades("good", 40, 0.60, 0.9, seed=1),
                    strategy="good", bootstrap_iters=800)
    res = record_result(con, "PR-0001", sc_good)
    check("result recorded", res["run_id"], "RUN-0001")
    check("  meets the pre-registered threshold", res["meets_threshold"], True)
    check("  params default to the pre-registered ones -> match",
          res["params_match"], True)
    check("  p-value source defaults to the t approximation",
          res["p_value_source"], "t")

    sc_bad = score(make_trades("bad", 40, -0.10, 0.9, seed=2),
                   strategy="bad", bootstrap_iters=800)
    res2 = record_result(con, "PR-0002", sc_bad)
    check("a FAILING result is recorded just the same", res2["run_id"], "RUN-0002")
    check("  and is marked as not meeting the threshold",
          res2["meets_threshold"], False)

    ok, msg = raises(record_result, con, "PR-9999", sc_good)
    check("a result with NO pre-registration is refused", ok, True)
    check("  ...by name", "NoPreregistration" in msg, True)

    res3 = record_result(con, "PR-0002", sc_bad,
                         params={"short_delta": 0.25, "wing": 150},
                         note="ran different params by mistake")
    check("running params OTHER than registered is recorded, not hidden",
          res3["params_match"], False)

    res4 = record_result(con, "PR-0001", sc_good, p_value=0.0031,
                         p_value_source="bootstrap")
    check("an explicit (bootstrap) p-value is preferred over the t one",
          (res4["p_value"], res4["p_value_source"]), (0.0031, "bootstrap"))
    check("ordinal_of shows WHICH test this was", ordinal_of(con, "RUN-0001"),
          (1, 4))
    check("  the last one is #4 of 4", ordinal_of(con, "RUN-0004"), (4, 4))

    # ======================================================================
    print("\n" + "=" * 96)
    print("TEST 8 -- WRITE-ONCE: every route to changing history, blocked")
    print("=" * 96)
    wcon = new_db("writeonce")
    w1 = register(wcon, family="F", strategy="S", params={"k": 1},
                  hypothesis="original hypothesis, threshold +0.205R",
                  pass_expectancy_r=0.205, pass_min_trades=10,
                  registered_at="2026-08-16T12:00:00")

    ok, msg = raises(wcon.execute,
                     "UPDATE preregistrations SET pass_expectancy_r=-9.0 "
                     "WHERE prereg_id='PR-0001'")
    check("8a. UPDATE a pre-registration -> REFUSED", ok, True)
    check("    message says write-once", "WRITE-ONCE" in msg, True)
    print(f"        {msg}")

    ok, msg = raises(wcon.execute,
                     "DELETE FROM preregistrations WHERE prereg_id='PR-0001'")
    check("8b. DELETE a pre-registration -> REFUSED", ok, True)
    print(f"        {msg}")

    ok, msg = raises(wcon.execute, """
        INSERT OR REPLACE INTO preregistrations
        (id, prereg_id, registered_at, family, strategy, variant, params_json,
         params_hash, hypothesis, expectation, pass_expectancy_r,
         pass_min_trades, alpha)
        VALUES (1,'PR-0001','2026-08-16T12:00:00','F','S','base','{}','x',
                'REWRITTEN','clears',-9.0,1,0.05)""")
    check("8c. INSERT OR REPLACE (the sneaky one) -> REFUSED", ok, True)
    print(f"        {msg}")
    still = list_preregistrations(wcon)[0]
    check("    the original threshold is untouched", still.pass_expectancy_r,
          0.205, 1e-12)
    check("    the original hypothesis text is untouched",
          still.hypothesis.startswith("original"), True)

    ok, msg = raises(amend, wcon, "PR-0001", pass_expectancy_r=0.0)
    check("8d. the public amend() API -> always raises WriteOnceViolation", ok, True)
    check("    and points at supersede()", "supersede" in msg, True)

    ok, msg = raises(record_result, wcon, "PR-0001",
                     score(make_trades("x", 12, 0.5, 0.5, seed=3),
                           bootstrap_iters=400),
                     run_at="2026-08-15T09:00:00")
    check("8e. a result DATED BEFORE its pre-registration -> REFUSED", ok, True)
    check("    message names the anti-backfill rule", "anti-backfill" in msg, True)
    print(f"        {msg}")

    good_sc = score(make_trades("y", 12, 0.5, 0.5, seed=4), bootstrap_iters=400)
    record_result(wcon, "PR-0001", good_sc, run_at="2026-08-16T13:00:00")
    ok, msg = raises(wcon.execute, "UPDATE results SET meets_threshold=1")
    check("8f. UPDATE a recorded result -> REFUSED", ok, True)
    ok, msg = raises(wcon.execute, "DELETE FROM results")
    check("8g. DELETE a recorded result -> REFUSED", ok, True)
    ok, msg = raises(wcon.execute,
                     "INSERT INTO results (id, run_id, prereg_id, run_at, "
                     "params_json, params_hash, params_match, n, "
                     "meets_threshold, hypothesis_confirmed) "
                     "VALUES (99,'RUN-0099','PR-NOPE','2026-08-16T14:00:00',"
                     "'{}','x',1,10,1,1)")
    check("8h. a result pointing at a non-existent prereg -> FK REFUSED", ok, True)

    print("\n  8i. Why PRAGMA recursive_triggers is load-bearing, demonstrated:")
    raw = sqlite3.connect(TMPDIR / "raw.db")
    _OPEN.append(raw)
    raw.executescript(SCHEMA)          # schema only -- NO pragma, the default
    raw.execute("""INSERT INTO preregistrations
        (id,prereg_id,registered_at,family,strategy,variant,params_json,
         params_hash,hypothesis,expectation,pass_expectancy_r,pass_min_trades,alpha)
        VALUES (1,'PR-0001','2026-08-16T12:00:00','F','S','base','{}','x',
                'original','clears',0.205,10,0.05)""")
    raw.commit()
    hole, hole_msg = raises(raw.execute, """
        INSERT OR REPLACE INTO preregistrations
        (id,prereg_id,registered_at,family,strategy,variant,params_json,
         params_hash,hypothesis,expectation,pass_expectancy_r,pass_min_trades,alpha)
        VALUES (1,'PR-0001','2026-08-16T12:00:00','F','S','base','{}','x',
                'REWRITTEN','clears',-9.0,1,0.05)""")
    rewritten = raw.execute(
        "SELECT hypothesis FROM preregistrations").fetchone()[0]
    print(f"        without the pragma, REPLACE raised? {hole}"
          f"   stored hypothesis is now: {rewritten!r}")
    check("    the hole is REAL: default SQLite lets REPLACE bypass the guard",
          (hole, rewritten), (False, "REWRITTEN"))
    check("    connect() turns recursive_triggers ON, closing it",
          bool(wcon.execute("PRAGMA recursive_triggers").fetchone()[0]), True)
    ok, msg = raises(verify_integrity, raw)
    check("    verify_integrity() REFUSES a connection with it off", ok, True)
    print("        -> data_layer.py's `predictions` table had the same schema")
    print("           shape and the same hole. Fixed 2026-08-16: connect() now")
    print("           sets the pragma and verify_integrity() enforces it.")
    print("           Reproduced and regression-tested in test_data_layer.py.")

    print("\n  8j. Tampering (dropping the triggers) is detected on next open:")
    tcon = new_db("tamper")
    register(tcon, family="F", strategy="S", params={}, hypothesis="h",
             pass_expectancy_r=0.0, pass_min_trades=10)
    tcon.execute("DROP TRIGGER preregistrations_no_update")
    tcon.commit()
    ok, msg = raises(verify_integrity, tcon)
    check("    verify_integrity() raises IntegrityCompromised", ok, True)
    check("    naming the missing guard",
          "preregistrations_no_update" in msg, True)
    print(f"        {msg[:120]}")

    print("\n  8k. supersede() is the ONLY legitimate revision path:")
    s2 = supersede(wcon, "PR-0001", reason="cost model corrected to per-order",
                   pass_expectancy_r=0.250)
    check("    writes a NEW pre-registration", s2.prereg_id, "PR-0002")
    check("    that points back at the original", s2.supersedes, "PR-0001")
    check("    with the revised threshold", s2.pass_expectancy_r, 0.250, 1e-12)
    check("    numbered as revision 1", s2.revision, 1)
    check("    the ORIGINAL still exists, unchanged",
          list_preregistrations(wcon)[0].pass_expectancy_r, 0.205, 1e-12)
    check("    and still counts as a registered test",
          test_count(wcon).n_preregistered, 2)
    s3 = supersede(wcon, "PR-0002", reason="second thoughts",
                   pass_expectancy_r=0.300)
    check("    a chain is possible: revision 2", s3.revision, 2)
    check("    ...pointing at revision 1, not at the original",
          s3.supersedes, "PR-0002")
    ok, msg = raises(register, wcon, family="F", strategy="S", params={"k": 1},
                     hypothesis="dodging the duplicate guard via revision",
                     pass_expectancy_r=-9.0, pass_min_trades=1, revision=3)
    check("    revision>0 WITHOUT supersedes is refused (guard cannot be dodged)",
          ok, True)
    ok, msg = raises(register, wcon, family="F", strategy="S", params={"k": 1},
                     hypothesis="plain re-registration of the same experiment",
                     pass_expectancy_r=-9.0, pass_min_trades=1)
    check("    ...and a plain re-registration is still refused", ok, True)
    print("        -> you cannot make an inconvenient hypothesis disappear, and")
    print("           the revision counter that enables supersede() is not a")
    print("           back door into re-registering the same experiment.")

    # ======================================================================
    print("\n" + "=" * 96)
    print("TEST 9 -- test counting: 'the 40th test' vs 'the 2nd'")
    print("=" * 96)
    ccon = new_db("counting")
    for i in range(5):
        register(ccon, family="FAM", strategy=f"S{i}", params={"i": i},
                 hypothesis=f"hypothesis {i}", pass_expectancy_r=0.205,
                 pass_min_trades=10)
    # Unambiguous samples: mean +0.80R (sd 0.30) cannot fail a +0.205R bar,
    # mean -0.30R cannot clear it. No borderline arithmetic in a counting test.
    sc_p = score(make_trades("cp", 30, 0.80, 0.30, seed=5), bootstrap_iters=400)
    sc_f = score(make_trades("cf", 30, -0.30, 0.30, seed=6), bootstrap_iters=400)
    check("  the 'passing' sample really does clear +0.205R",
          sc_p.expectancy_r > 0.205, True)
    check("  the 'failing' sample really does not", sc_f.expectancy_r < 0.205, True)
    record_result(ccon, "PR-0001", sc_p)
    record_result(ccon, "PR-0002", sc_f)
    record_result(ccon, "PR-0002", sc_f, note="re-run of the same hypothesis")
    tc = test_count(ccon, family="FAM")
    check("5 pre-registered", tc.n_preregistered, 5)
    check("3 runs", tc.n_runs, 3)
    check("over 2 distinct hypotheses", tc.n_distinct_hypotheses_run, 2)
    check("1 of those runs is a RE-RUN (p-hacking signature)", tc.n_reruns, 1)
    check("3 pre-registered hypotheses were NEVER RUN", tc.n_pending, 3)
    check("pass count", tc.n_passed, 1)
    check("fail count", tc.n_failed, 2)
    check("pass rate is reported, not just the winner", tc.pass_rate, 1 / 3, 1e-12)
    check("per-strategy scoping works",
          test_count(ccon, strategy="S1").n_runs, 2)
    check("families() exposes the correction universe size",
          families(ccon)[0]["n_preregistered"], 5)
    print("        -> n_pending is the anti-cherry-picking number: 3 hypotheses")
    print("           were registered and never reported. The registry says so.")

    # ======================================================================
    print("\n" + "=" * 96)
    print("TEST 10 -- the summary cannot report only the winners")
    print("=" * 96)
    s = summary(ccon, family="FAM")
    check("summary carries the failures alongside the passes",
          len(s.winners) + len(s.losers), 3)
    check("summary carries the never-run hypotheses", len(s.pending), 3)
    check("summary warns about the unreported tests",
          any("NO recorded result" in w for w in s.warnings), True)
    check("summary warns about the re-run",
          any("p-hacking signature" in w for w in s.warnings), True)
    check("summary knows the corrected threshold",
          s.correction.bonferroni_threshold, 0.05 / 3, 1e-12)
    check("every result appears in the printed text",
          all(f"RUN-{i:04d}" in str(s) for i in (1, 2, 3)), True)
    empty = summary(new_db("empty"))
    check("an empty registry summarises without crashing", empty.counts.n_runs, 0)
    check("  and claims nothing", empty.survives_correction, [])

    # ======================================================================
    print("\n" + "=" * 96)
    print("INTEGRATION RUN -- 24 pre-registered tests, 23 of them pure noise")
    print("=" * 96)
    print("  Simulating exactly the S4.5 scenario: 8 strategies x 3 parameter")
    print("  variants, walked over the same window. TWENTY-THREE are drawn from")
    print("  a TRUE ZERO-EDGE distribution. ONE has a genuine +0.55R edge.")
    print("  A registry that works will (a) let some noise cross the raw bar,")
    print("  (b) cut most or all of it with the correction, and (c) never let")
    print("  the failures disappear.\n")

    icon = new_db("integration")
    WINDOW = "2024-01-01..2026-08-13 (645 days, 167 NIFTY expiries)"
    strategies = ["DirectionalCreditSpread", "DeltaReset", "UniqueATMSelling",
                  "DoubleDiagonal", "WeeklyCallSell", "IronFlyLowVol",
                  "RatioPutSpread", "BrokenButterfly"]
    variants = [("delta30", 0.30), ("delta20", 0.20), ("delta15", 0.15)]
    REAL_EDGE = ("IronFlyLowVol", "delta20")

    seed = 100
    recorded = []
    for st in strategies:
        for vname, dlt in variants:
            is_real = (st, vname) == REAL_EDGE
            register(icon, family="TIER1-SWEEP", strategy=st, variant=vname,
                     params={"short_delta": dlt, "wing_pts": 150, "dte": 7},
                     hypothesis=(f"{st} at {dlt} delta clears break-even "
                                 f"(+0.205R) net of the measured Rs 410 cost"),
                     expectation="clears", pass_expectancy_r=0.205,
                     pass_min_trades=10, alpha=0.05, data_window=WINDOW,
                     holdout="last 6 months held out", author="integration")
            seed += 1
            trues = 0.55 if is_real else 0.0
            trs = make_trades(f"{st}-{vname}", 52, trues, 1.10, seed=seed)
            sc = score(trs, strategy=f"{st}/{vname}", bootstrap_iters=800)
            # raw R values ARE available here, so use the honest bootstrap p
            bp = bootstrap_p_value([t.r for t in trs], iters=4000, seed=seed)
            pid = f"PR-{len(recorded) + 1:04d}"
            out = record_result(icon, pid, sc, p_value=bp,
                                p_value_source="bootstrap")
            recorded.append((st, vname, is_real, out))

    # Two more hypotheses pre-registered and deliberately NEVER RUN, to prove
    # the registry surfaces silent abandonment.
    for i, st in enumerate(["AdvancedPMCC", "IntradayAsymmetric"]):
        register(icon, family="TIER1-SWEEP", strategy=st, variant="base",
                 params={"note": "blocked on margin"},
                 hypothesis=f"{st} clears break-even", expectation="clears",
                 pass_expectancy_r=0.205, pass_min_trades=10, alpha=0.05,
                 data_window=WINDOW)

    isum = summary(icon, family="TIER1-SWEEP")
    print(isum)

    real_run = [o for st, v, real, o in recorded if real][0]
    real_ord = ordinal_of(icon, real_run["run_id"])
    print(f"\n  The ONE strategy with a genuine edge was {REAL_EDGE[0]}/"
          f"{REAL_EDGE[1]}, tested {real_ord[0]} of {real_ord[1]}.")
    print(f"  Its run {real_run['run_id']} meets_threshold="
          f"{real_run['meets_threshold']}, p={real_run['p_value']:.5f}, "
          f"survives correction={real_run['run_id'] in isum.survives_correction}")

    fake = [w for w in isum.winners
            if w["run_id"] != real_run["run_id"]]
    print(f"\n  FAKE WINNERS at the raw threshold: {len(fake)} zero-edge "
          f"strategies crossed +0.205R by chance.")
    for w in fake:
        print(f"    {w['label']:<34} E={w['expectancy_r']:+.3f}R  "
              f"p={w['p_value']:.4f}  q={w['q_value']:.4f}  "
              f"survives correction: {w['survives']}")

    print("\n  Corrected threshold as the sweep grows (same p-values, more tests):")
    all_p = [TestOutcome(f"T{i}", r["p_value"])
             for i, r in enumerate(list_results(icon, family="TIER1-SWEEP"))]
    print("     m | Bonferroni p< | naive sig | Bonferroni sig | BH sig")
    print("    ---+---------------+-----------+----------------+-------")
    for m in (1, 2, 4, 8, 16, 24):
        cm = correct(all_p[:m], alpha=0.05, method="bh")
        print(f"    {m:>2} | {cm.bonferroni_threshold:>13.5f} |"
              f" {cm.n_naive_significant:>9} | {cm.n_bonferroni_significant:>14} |"
              f" {cm.n_bh_significant:>6}")

    check("INTEGRATION: all 24 runs are on the record", isum.counts.n_runs, 24)
    check("INTEGRATION: 26 hypotheses pre-registered",
          isum.counts.n_preregistered, 26)
    check("INTEGRATION: the 2 abandoned ones are surfaced",
          isum.counts.n_pending, 2)
    check("INTEGRATION: passes + failures == runs",
          isum.counts.n_passed + isum.counts.n_failed, 24)
    check("INTEGRATION: the genuine edge crosses the raw bar",
          real_run["meets_threshold"], True)
    check("INTEGRATION: Bonferroni threshold is 0.05/24",
          isum.correction.bonferroni_threshold, 0.05 / 24, 1e-12)
    check("INTEGRATION: chance predicts 1.2 false positives at 24 tests",
          isum.correction.expected_false_positives, 1.2, 1e-12)
    check("INTEGRATION: the correction never keeps MORE than the raw bar",
          isum.correction.n_bh_significant <= isum.correction.n_naive_significant,
          True)
    check("INTEGRATION: Bonferroni is never looser than BH",
          isum.correction.n_bonferroni_significant
          <= isum.correction.n_bh_significant, True)
    check("INTEGRATION: threshold tightens monotonically with m",
          all(bonferroni_threshold(0.05, a) > bonferroni_threshold(0.05, b)
              for a, b in zip((1, 2, 4, 8, 16), (2, 4, 8, 16, 24))), True)

    _rows = list_results(icon, family="TIER1-SWEEP")
    passed_ids = {w["run_id"] for w in isum.winners}
    sig_ids = {r["run_id"] for r in _rows
               if r["p_value"] is not None and r["p_value"] < 0.05}
    print(f"\n  cleared the +0.205R economic bar : {sorted(passed_ids)}")
    print(f"  raw-significant (p < 0.05)       : {sorted(sig_ids)}")
    check("INTEGRATION: clearing the economic bar and being statistically "
          "significant are DIFFERENT sets", passed_ids != sig_ids, True)
    check("  at least one strategy cleared the bar without being significant",
          len(passed_ids - sig_ids) > 0, True)
    check("  at least one was significant without clearing the bar",
          len(sig_ids - passed_ids) > 0, True)

    print("\n  Reading the integration output:")
    print("   * NOTE the two filters are different and both are needed: a")
    print("     strategy can clear the +0.205R economic bar while its p-value")
    print("     says the result is indistinguishable from noise, and vice versa.")
    print("   * every zero-edge strategy that crossed +0.205R did so purely by")
    print("     chance. That is the S4.5 failure mode, reproduced on demand.")
    print("   * Bonferroni's bar falls from 0.05 to 0.00208 across the sweep --")
    print("     correct, and severe enough that it alone would reject the real")
    print("     edge too if the sample were smaller. Hence BH as the default.")
    print("   * the summary lists failures and never-run hypotheses in the same")
    print("     object as the winners. There is no winners-only code path.")

finally:
    for c in _OPEN:
        try:
            c.close()
        except Exception:                                    # noqa: BLE001
            pass
    shutil.rmtree(TMPDIR, ignore_errors=True)
    print(f"\n  [cleanup] temp registry databases removed: "
          f"{'gone' if not TMPDIR.exists() else 'STILL PRESENT -- ' + str(TMPDIR)}")

print("\n" + "=" * 96)
n_pass, n_fail = len(PASS), len(FAIL)
print(f"RESULT: {n_pass} passed, {n_fail} failed")
if FAIL:
    print("FAILURES:")
    for f in FAIL:
        print(f"   - {f}")
    raise SystemExit(1)
print("All checks green -- pre-registrations cannot be edited by any route,")
print("the multiple-comparisons maths matches the published worked example,")
print("and no summary path can report the winners without the failures.")
print("=" * 96)
