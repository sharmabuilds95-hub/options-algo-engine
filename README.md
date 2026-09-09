# options-algo-engine

A research engine for systematic options strategies on Indian equity derivatives (NSE), built around a single principle: **the main failure mode of systematic trading is fooling yourself, so the machinery should make that hard.**

This repository is the infrastructure layer — statistical validation, walk-forward backtesting, transaction-cost modelling, liquidity screening, and the test suite that covers them. It ships with one complete strategy implementation (`N3`) which, as documented below, **was tested and rejected**.

```bash
git clone <this-repo> && cd options-algo-engine
pip install -r requirements.txt
pytest                                 # all 7 suites, offline, no API keys
```

---

## Why this exists

Most retail backtests are broken in the same three ways: they test a strategy on the same data used to design it, they ignore transaction costs that exceed the edge, and they quietly try many variants until one looks good. Each of those produces a profitable-looking backtest and a losing strategy.

This engine is built to prevent all three — on myself, not on someone else.

## The parts that matter

### 1. Pre-registration with write-once integrity (`engine/registry.py`)

Before a strategy is backtested, its hypothesis and parameters are **pre-registered** and hashed. The registry enforces this at the type level:

- `WriteOnceViolation` — a registered hypothesis cannot be edited after the fact
- `NoPreregistration` — results cannot be recorded for a hypothesis that was never registered
- `DuplicatePreregistration` / `IntegrityCompromised` — guards against silently re-running the same idea until it passes

This is the anti-p-hacking layer. It exists so that "I predicted this before I saw the result" is a checkable claim rather than a memory.

### 2. Multiple-hypothesis-testing correction (`engine/registry.py`)

Test twenty strategies at p < 0.05 and one passes by chance. The registry implements the corrections properly:

- **Bonferroni** (`bonferroni_threshold`, `bonferroni_adjusted`) — conservative family-wise error control
- **Benjamini–Hochberg** (`benjamini_hochberg_adjusted`) — false-discovery-rate control, appropriate when testing many strategies
- Supporting statistics implemented from scratch: `reg_incomplete_beta`, two-sided t-tests from summary statistics, and a bootstrap p-value estimator (`bootstrap_p_value`, 10k iterations by default)

### 3. Walk-forward validation (`engine/walk.py`)

Strategies are evaluated out-of-sample on a rolling basis rather than on the fitted history. In-sample performance is treated as a diagnostic, not evidence.

### 4. Transaction-cost modelling (`costs.py`)

The full Indian F&O cost stack — brokerage, STT, GST, stamp duty and slippage — modelled per leg. This matters more than it sounds: the measured all-in cost per trade sets a **minimum effect size** below which an apparent edge is indistinguishable from friction. That threshold is computed and enforced in `engine/score.py` (`DEFAULT_MIN_EFFECT_R`), not left to judgement.

### 5. Gating (`engine/score.py`)

A strategy does not graduate on a good-looking equity curve. It has to clear explicit gates: minimum sample size, expectancy above the cost floor, statistical significance, and a drawdown ceiling. `GateReason` records *why* something was rejected.

### 6. Options structure machinery (`engine/strategy.py`)

Generic multi-leg options modelling: chain snapshots, per-leg Greeks, delta-based strike selection, position construction, structural max-loss computation (`structure_max_loss`), and trigger/adjustment-rule evaluation.

### 7. Liquidity screening (`engine/liquidity.py`) and regime classification (`engine/regime.py`)

A strategy that only works on illiquid strikes does not work. Liquidity filters run before a signal is considered tradeable.

---

## The included strategy is a rejected one

`engine/rules.py` implements **N3 — Range Fade / Credit**, complete with entry conditions, strike selection and signal evaluation.

It does not work, and that is why it is here.

> **Backtest, 2026-08-14.** 53 weekly expiries, 245 trading days (2025-08-14 → 2026-08-13), NSE bhavcopy and official NSE index archives.
> **Result: zero entries over a full year.** Not underperformance — the compound entry filter never fires. This confirmed and extended an earlier n=12-week finding at n=245 days.

The strategy was specified, implemented, tested against a full year of real market data, and rejected on the evidence. A framework that only ever ships winners is a framework that is hiding its failures, so the worked example here is a failure, kept deliberately.

---

## Test suite

Seven check suites, 2,201 lines, all runnable offline with no market data, no API keys, and no network access:

```bash
pytest
```

That reports 8 passing items: the seven suites, plus one guard that fails if suite discovery ever returns nothing.

| Suite | Checks | Covers |
|---|---:|---|
| `engine/test_registry.py` | 155 | pre-registration integrity, Bonferroni/BH correction, bootstrap p-values |
| `engine/test_score.py` | 87 | gating logic, expectancy, effect-size floor |
| `engine/test_strategy.py` | 69 | Greeks, strike selection, position construction, max-loss, triggers |
| `engine/test_walk.py` | 56 | walk-forward splitting and out-of-sample evaluation |
| `engine/test_liquidity.py` | 40 | liquidity filters |
| `test_analytics.py` | 33 | analytics and reporting |
| `engine/test_rules.py` | 25 | N3 entry logic |

465 individual checks in total. The registry suite is the largest because it attacks the write-once store through every route that could rewrite history: UPDATE, DELETE, INSERT OR REPLACE, back-dated results, and the public API. Its correction maths is checked against the worked example in the 1995 Benjamini-Hochberg paper.

Each suite is script-style. It asserts at module level, prints its own tally, and exits non-zero if any check fails. `test_suites.py` runs each one in a subprocess, so `pytest` covers them without the assertions being rewritten, and a failing suite prints its captured output so you can see which check broke. Any suite still runs on its own:

```bash
python engine/test_registry.py
```

Every suite keeps its databases in a temp directory, so running the tests never touches `data/registry.db` or `data/market.db`.

## Stack

Python 3.12 · NumPy · SciPy · Requests. Market data comes from free, no-auth official NSE sources (bhavcopy, index archives); no paid data feed is required to run the backtests.

## What is deliberately not in this repository

- **Live strategy parameters and the strategy registry database** — the research record and any working edge stay private.
- **Personal trading records, journal, position book, and account sizing.**
- **Credentials of any kind.** No secrets have ever been committed; `.secrets.yaml`, `.env` and token caches are gitignored at source.
- **The regime validation harness**, which requires a local market database not included here.

Risk-unit constants in `engine/score.py` are example defaults, not an account statement — change them in one line.

## License

MIT — see `LICENSE`.
