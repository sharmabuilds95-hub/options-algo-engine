"""
data_layer.py — SQLite store + NSE bhavcopy ingestion.  [WEEK 1]

WHY BHAVCOPY AND NOT KITE, verified 2026-08-13:
    Kite's instrument master contains ACTIVE contracts only. Searching for an
    expired contract (NIFTY26JUL24000CE, expired ~28-Jul-2026) returns EMPTY,
    while live/future contracts return full metadata. Without an instrument_token
    you cannot call get_historical_data. Therefore **Kite cannot supply option
    history for expired contracts**, which is 100% of what a backtest needs.

    NSE's daily F&O bhavcopy can. It is free, unauthenticated, archived for years,
    ~1.1 MB/day, and carries everything required:
        OHLC · settlement price · open interest · underlying price · LOT SIZE
    One day (2026-08-12) = 34,547 rows: 28,580 stock options, 5,326 index options,
    622 stock futures, 18 index futures.

    Cross-validated: bhavcopy UndrlygPric 24435.95 for 2026-08-12 exactly matches
    the Nifty close pulled independently from Kite.

DIVISION OF LABOUR (ADR-010):
    NSE bhavcopy -> ALL history (daily resolution). Free. No auth. No rate limit.
    Kite Connect -> live quotes, live chain, positions, margins, intraday bars.

Design rules (ADR-002): append-only, immutable, every row records where it came
from and when it was ingested. Nothing is ever UPDATEd.
"""
from __future__ import annotations

import datetime as dt
import io
import sqlite3
import zipfile
from pathlib import Path
from typing import Iterable

import requests

HERE = Path(__file__).parent
DB_PATH = HERE / "data" / "market.db"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
BHAV_URL = ("https://nsearchives.nseindia.com/content/fo/"
            "BhavCopy_NSE_FO_0_0_0_{yyyymmdd}_F_0000.csv.zip")

# bhavcopy FinInstrmTp codes
INSTR_TYPES = {"IDO": "index_option", "STO": "stock_option",
               "IDF": "index_future", "STF": "stock_future"}

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
-- NOTE: `PRAGMA recursive_triggers` is ALSO load-bearing for the write-once
-- guarantee on `predictions`, but it is a per-connection setting rather than a
-- schema property, so it is set in connect() and re-checked by
-- verify_integrity(). See the comment there for why it matters.

-- Every F&O contract's daily bar. THE backtest table.
-- Append-only: (trade_date, symbol, expiry, strike, opt_type) is unique.
CREATE TABLE IF NOT EXISTS fo_bars (
    id              INTEGER PRIMARY KEY,
    trade_date      TEXT    NOT NULL,
    symbol          TEXT    NOT NULL,          -- NIFTY, RELIANCE, ...
    instr_type      TEXT    NOT NULL,          -- index_option | stock_option | ...
    expiry          TEXT    NOT NULL,
    strike          REAL,                      -- NULL for futures
    opt_type        TEXT,                      -- CE | PE | NULL
    open            REAL, high REAL, low REAL, close REAL,
    settle          REAL,
    prev_close      REAL,
    underlying      REAL,                      -- spot at close, from NSE
    open_interest   INTEGER,
    oi_change       INTEGER,
    volume          INTEGER,
    trades          INTEGER,
    lot_size        INTEGER,                   -- as of THAT day. Handles the 75->65 change
    source          TEXT NOT NULL DEFAULT 'nse_bhavcopy',
    ingested_at     TEXT NOT NULL,
    UNIQUE (trade_date, symbol, expiry, strike, opt_type)
);
CREATE INDEX IF NOT EXISTS ix_fo_sym_date  ON fo_bars(symbol, trade_date);
CREATE INDEX IF NOT EXISTS ix_fo_expiry    ON fo_bars(symbol, expiry, trade_date);
CREATE INDEX IF NOT EXISTS ix_fo_chain     ON fo_bars(symbol, trade_date, expiry, strike);

-- Which days have been ingested, and did they succeed. Makes ingestion idempotent
-- and makes a GAP VISIBLE rather than silent.
CREATE TABLE IF NOT EXISTS ingest_log (
    trade_date  TEXT PRIMARY KEY,
    status      TEXT NOT NULL,     -- ok | empty | http_error | holiday
    rows        INTEGER NOT NULL DEFAULT 0,
    note        TEXT,
    ingested_at TEXT NOT NULL
);

-- India VIX daily close. From NSE's own "all indices" daily archive (the India
-- VIX row within it), NOT investing.com -- investing.com's ToS prohibits
-- storing/reproducing its data, and this is bulk storage for backtesting, not
-- a one-off live read. Same free/no-auth archive family as fo_bars, verified
-- 2026-08-14: matches investing.com's published closes exactly on spot-check
-- (12-Aug-2026: both show 11.69), confirming it's the same underlying series.
CREATE TABLE IF NOT EXISTS vix (
    trade_date  TEXT PRIMARY KEY,
    open        REAL, high REAL, low REAL, close REAL,
    points_change REAL,
    pct_change  REAL,
    source      TEXT NOT NULL DEFAULT 'nse_indices_archive',
    ingested_at TEXT NOT NULL
);

-- Separate log so a VIX gap is visible independently of the fo_bars gap --
-- the two archives are different files and can fail independently.
CREATE TABLE IF NOT EXISTS vix_ingest_log (
    trade_date  TEXT PRIMARY KEY,
    status      TEXT NOT NULL,     -- ok | no_vix_row | holiday | http_error
    note        TEXT,
    ingested_at TEXT NOT NULL
);

-- Daily spot-index OHLC (Nifty 50, etc). From the SAME "all indices" archive
-- file as vix above -- "inside yesterday's range" (N3, strategy note) needs
-- the underlying's own daily high/low, which fo_bars does not carry (its
-- `underlying` column is a single per-contract snapshot, not a day range).
CREATE TABLE IF NOT EXISTS spot_index (
    index_name  TEXT NOT NULL,
    trade_date  TEXT NOT NULL,
    open        REAL, high REAL, low REAL, close REAL,
    source      TEXT NOT NULL DEFAULT 'nse_indices_archive',
    ingested_at TEXT NOT NULL,
    PRIMARY KEY (index_name, trade_date)
);

CREATE TABLE IF NOT EXISTS spot_index_ingest_log (
    index_name  TEXT NOT NULL,
    trade_date  TEXT NOT NULL,
    status      TEXT NOT NULL,
    note        TEXT,
    ingested_at TEXT NOT NULL,
    PRIMARY KEY (index_name, trade_date)
);

-- Capital movements. Deposits/withdrawals must never be read as performance
-- (Risk Constitution: Capital Flows Ledger).
CREATE TABLE IF NOT EXISTS capital_flows (
    id          INTEGER PRIMARY KEY,
    flow_date   TEXT NOT NULL,
    flow_type   TEXT NOT NULL CHECK (flow_type IN ('deposit','withdrawal')),
    amount      REAL NOT NULL CHECK (amount > 0),
    note        TEXT,
    recorded_at TEXT NOT NULL
);

-- WRITE-ONCE predictions. Anti-backfill enforced by the database, not by honour.
-- The triggers below refuse UPDATE and DELETE.
--
-- The triggers ALONE are not sufficient, and the claim that used to sit here
-- ("makes UPDATE and DELETE physically impossible") was false for one route.
-- `INSERT OR REPLACE` resolves a UNIQUE conflict by deleting the old row and
-- inserting a new one, and SQLite fires DELETE triggers for that implicit
-- delete ONLY when recursive_triggers is ON -- which is OFF by default. So on
-- a default connection:
--     UPDATE            -> blocked by predictions_no_update   (correct)
--     DELETE            -> blocked by predictions_no_delete   (correct)
--     INSERT OR REPLACE -> SILENTLY REWROTE THE ROW           (the hole)
-- connect() now sets the pragma and verify_integrity() refuses to hand back a
-- connection without it, which is what makes "DB-enforced" true rather than
-- aspirational. Hole found and reproduced 2026-08-15 while building
-- engine/registry.py; regression test in test_data_layer.py.
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY,
    created_at    TEXT NOT NULL,
    strategy      TEXT NOT NULL,
    params_hash   TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    as_of_date    TEXT NOT NULL,
    direction     TEXT,
    entry         REAL, stop REAL, target REAL,
    expected_r    REAL,
    rationale     TEXT,
    UNIQUE (strategy, params_hash, symbol, as_of_date)
);
CREATE TRIGGER IF NOT EXISTS predictions_no_update
BEFORE UPDATE ON predictions
BEGIN SELECT RAISE(ABORT, 'predictions are write-once (anti-backfill)'); END;
CREATE TRIGGER IF NOT EXISTS predictions_no_delete
BEFORE DELETE ON predictions
BEGIN SELECT RAISE(ABORT, 'predictions are write-once (anti-backfill)'); END;

-- Outcomes are separate, and written later. One per prediction.
CREATE TABLE IF NOT EXISTS outcomes (
    prediction_id INTEGER PRIMARY KEY REFERENCES predictions(id),
    resolved_at   TEXT NOT NULL,
    exit_price    REAL,
    r_multiple    REAL,
    gross_pnl     REAL,
    costs         REAL,
    net_pnl       REAL,
    ambiguous_bar INTEGER NOT NULL DEFAULT 0,   -- Forward-Testing-Spec honesty field
    ambiguity_rule TEXT,
    note          TEXT
);

-- Every engine run, for reproducibility (ADR-002).
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    kind        TEXT NOT NULL,      -- daily | backtest | ingest
    git_sha     TEXT,
    params_hash TEXT,
    status      TEXT,
    note        TEXT
);
"""


class IntegrityCompromised(RuntimeError):
    """The write-once guards are missing from the database file."""


# Triggers that must exist for the anti-backfill guarantee on `predictions` to
# mean anything. Checked on every connect().
GUARD_TRIGGERS = ("predictions_no_update", "predictions_no_delete")


def _missing_guards(con: sqlite3.Connection) -> list[str]:
    have = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger'").fetchall()}
    return [t for t in GUARD_TRIGGERS if t not in have]


def _predictions_exists(con: sqlite3.Connection) -> bool:
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='predictions'"
    ).fetchone() is not None


def verify_integrity(con: sqlite3.Connection) -> None:
    """Refuse to use a database whose write-once guards have been removed.

    Cannot stop a determined operator dropping the triggers with the sqlite3
    CLI -- but it can make sure the next process to open the file NOTICES,
    rather than writing on top of tampered history as if nothing happened.
    Same pattern as engine/registry.py's verify_integrity().
    """
    missing = _missing_guards(con)
    if missing:
        raise IntegrityCompromised(
            "write-once guards missing from market database: "
            + ", ".join(missing)
            + " -- this file has been tampered with, or was created by an "
              "older schema. Do not trust its `predictions` contents.")
    if not con.execute("PRAGMA recursive_triggers").fetchone()[0]:
        raise IntegrityCompromised(
            "recursive_triggers is OFF -- INSERT OR REPLACE would bypass the "
            "delete guard and silently overwrite a prediction.")


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        # ORDER MATTERS. SCHEMA uses CREATE TRIGGER IF NOT EXISTS, so running it
        # first would SILENTLY RECREATE a guard that had been dropped -- the
        # file would be repaired and the tampering never reported. Anything
        # written while the guard was missing would then look pristine. So an
        # EXISTING predictions table is checked before the schema is reapplied;
        # a database that does not have the table yet is simply new, and there
        # is nothing to have tampered with.
        if _predictions_exists(con):
            missing = _missing_guards(con)
            if missing:
                raise IntegrityCompromised(
                    "write-once guards missing from market database: "
                    + ", ".join(missing)
                    + " -- this file has been tampered with, or was created by "
                      "an older schema. Refusing to reapply the schema over it, "
                      "because that would hide the fact. Do not trust its "
                      "`predictions` contents.")
        con.executescript(SCHEMA)
        # MUST be set per connection: recursive_triggers is a connection setting
        # and is NOT persisted in the database file, so every new connection
        # starts with it OFF regardless of what any previous connection did.
        # Without it, `INSERT OR REPLACE INTO predictions ...` silently deletes
        # and rewrites a row while predictions_no_delete sits there doing
        # nothing. Reproduced and regression-tested in test_data_layer.py.
        con.execute("PRAGMA recursive_triggers = ON")
        verify_integrity(con)
    except Exception:
        con.close()
        raise
    return con


def _f(v: str) -> float | None:
    v = (v or "").strip()
    try:
        return float(v)
    except ValueError:
        return None


def _i(v: str) -> int | None:
    f = _f(v)
    return int(f) if f is not None else None


def fetch_bhavcopy(day: dt.date, timeout: int = 60) -> bytes | None:
    """Download one day's F&O bhavcopy. Returns None on 403/404 (holiday/absent)."""
    url = BHAV_URL.format(yyyymmdd=day.strftime("%Y%m%d"))
    r = requests.get(url, headers={"User-Agent": UA, "Accept": "*/*"}, timeout=timeout)
    if r.status_code != 200 or len(r.content) < 1000:
        return None
    return r.content


def parse_bhavcopy(blob: bytes) -> list[dict]:
    """Parse the UDiFF CSV inside the zip into row dicts."""
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        text = z.read(name).decode("utf-8", errors="replace")

    lines = text.splitlines()
    hdr = [h.strip() for h in lines[0].split(",")]
    ix = {h: i for i, h in enumerate(hdr)}
    out: list[dict] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        c = line.split(",")
        if len(c) < len(hdr):
            continue
        itype = INSTR_TYPES.get(c[ix["FinInstrmTp"]].strip())
        if itype is None:
            continue
        opt = c[ix["OptnTp"]].strip() or None
        out.append(dict(
            trade_date=c[ix["TradDt"]].strip(),
            symbol=c[ix["TckrSymb"]].strip(),
            instr_type=itype,
            expiry=c[ix["XpryDt"]].strip(),
            strike=_f(c[ix["StrkPric"]]) if opt else None,
            opt_type=opt,
            open=_f(c[ix["OpnPric"]]), high=_f(c[ix["HghPric"]]),
            low=_f(c[ix["LwPric"]]), close=_f(c[ix["ClsPric"]]),
            settle=_f(c[ix["SttlmPric"]]),
            prev_close=_f(c[ix["PrvsClsgPric"]]),
            underlying=_f(c[ix["UndrlygPric"]]),
            open_interest=_i(c[ix["OpnIntrst"]]),
            oi_change=_i(c[ix["ChngInOpnIntrst"]]),
            volume=_i(c[ix["TtlTradgVol"]]),
            trades=_i(c[ix["TtlNbOfTxsExctd"]]),
            lot_size=_i(c[ix["NewBrdLotQty"]]),
        ))
    return out


def ingest_day(con: sqlite3.Connection, day: dt.date, force: bool = False) -> dict:
    """Idempotent: already-ingested days are skipped unless force=True."""
    ds = day.isoformat()
    if not force:
        row = con.execute("SELECT status, rows FROM ingest_log WHERE trade_date=?",
                          (ds,)).fetchone()
        if row and row["status"] == "ok":
            return {"date": ds, "status": "skipped", "rows": row["rows"]}

    now = dt.datetime.now().isoformat(timespec="seconds")
    blob = fetch_bhavcopy(day)
    if blob is None:
        con.execute("INSERT OR REPLACE INTO ingest_log VALUES (?,?,?,?,?)",
                    (ds, "holiday", 0, "no bhavcopy (weekend/holiday/not published)", now))
        con.commit()
        return {"date": ds, "status": "holiday", "rows": 0}

    rows = parse_bhavcopy(blob)
    con.executemany("""
        INSERT OR IGNORE INTO fo_bars
        (trade_date,symbol,instr_type,expiry,strike,opt_type,open,high,low,close,
         settle,prev_close,underlying,open_interest,oi_change,volume,trades,
         lot_size,source,ingested_at)
        VALUES (:trade_date,:symbol,:instr_type,:expiry,:strike,:opt_type,:open,:high,
                :low,:close,:settle,:prev_close,:underlying,:open_interest,:oi_change,
                :volume,:trades,:lot_size,'nse_bhavcopy',:ingested_at)
    """, [{**r, "ingested_at": now} for r in rows])
    con.execute("INSERT OR REPLACE INTO ingest_log VALUES (?,?,?,?,?)",
                (ds, "ok", len(rows), None, now))
    con.commit()
    return {"date": ds, "status": "ok", "rows": len(rows)}


def ingest_range(con: sqlite3.Connection, start: dt.date, end: dt.date,
                 progress: bool = True) -> dict:
    tot = {"ok": 0, "holiday": 0, "skipped": 0, "rows": 0}
    day = start
    while day <= end:
        if day.weekday() < 5:                      # skip weekends outright
            r = ingest_day(con, day)
            tot[r["status"]] = tot.get(r["status"], 0) + 1
            tot["rows"] += r["rows"]
            if progress:
                print(f"  {r['date']}  {r['status']:8s} {r['rows']:>7,} rows")
        day += dt.timedelta(days=1)
    return tot


# ---- India VIX (separate archive, separate log — see SCHEMA comment) -----
IDX_URL = "https://archives.nseindia.com/content/indices/ind_close_all_{ddmmyyyy}.csv"


def fetch_index_close_all(day: dt.date, timeout: int = 60) -> bytes | None:
    """Download one day's all-indices close file. Returns None on 404/absent
    (weekend/holiday) -- this archive is plain CSV, no zip, unlike bhavcopy."""
    url = IDX_URL.format(ddmmyyyy=day.strftime("%d%m%Y"))
    r = requests.get(url, headers={"User-Agent": UA, "Accept": "*/*"}, timeout=timeout)
    if r.status_code != 200 or len(r.content) < 100:
        return None
    return r.content


def parse_vix_row(blob: bytes) -> dict | None:
    """Pull just the 'India VIX' row out of the all-indices CSV. None if the
    row is missing that day (shouldn't happen on a trading day, but the
    ingest_log records it as a visible gap rather than a silent skip)."""
    text = blob.decode("utf-8", errors="replace")
    lines = text.splitlines()
    hdr = [h.strip() for h in lines[0].split(",")]
    ix = {h: i for i, h in enumerate(hdr)}
    for line in lines[1:]:
        if not line.strip():
            continue
        c = line.split(",")
        if len(c) < len(hdr) or c[ix["Index Name"]].strip() != "India VIX":
            continue
        return dict(
            open=_f(c[ix["Open Index Value"]]), high=_f(c[ix["High Index Value"]]),
            low=_f(c[ix["Low Index Value"]]), close=_f(c[ix["Closing Index Value"]]),
            points_change=_f(c[ix["Points Change"]]), pct_change=_f(c[ix["Change(%)"]]),
        )
    return None


def ingest_vix_day(con: sqlite3.Connection, day: dt.date, force: bool = False) -> dict:
    """Idempotent, mirrors ingest_day. Independent of fo_bars ingestion --
    call both when backfilling a range that needs to support N3 (or any
    other VIX-gated rule)."""
    ds = day.isoformat()
    if not force:
        row = con.execute("SELECT status FROM vix_ingest_log WHERE trade_date=?",
                          (ds,)).fetchone()
        if row and row["status"] == "ok":
            return {"date": ds, "status": "skipped"}

    now = dt.datetime.now().isoformat(timespec="seconds")
    blob = fetch_index_close_all(day)
    if blob is None:
        con.execute("INSERT OR REPLACE INTO vix_ingest_log VALUES (?,?,?,?)",
                    (ds, "holiday", "no all-indices file (weekend/holiday/not published)", now))
        con.commit()
        return {"date": ds, "status": "holiday"}

    row = parse_vix_row(blob)
    if row is None:
        con.execute("INSERT OR REPLACE INTO vix_ingest_log VALUES (?,?,?,?)",
                    (ds, "no_vix_row", "all-indices file present but no India VIX row", now))
        con.commit()
        return {"date": ds, "status": "no_vix_row"}

    con.execute("""
        INSERT OR IGNORE INTO vix (trade_date, open, high, low, close,
            points_change, pct_change, source, ingested_at)
        VALUES (?,?,?,?,?,?,?,'nse_indices_archive',?)
    """, (ds, row["open"], row["high"], row["low"], row["close"],
          row["points_change"], row["pct_change"], now))
    con.execute("INSERT OR REPLACE INTO vix_ingest_log VALUES (?,?,?,?)",
                (ds, "ok", None, now))
    con.commit()
    return {"date": ds, "status": "ok", "close": row["close"]}


def ingest_vix_range(con: sqlite3.Connection, start: dt.date, end: dt.date,
                     progress: bool = True) -> dict:
    tot = {"ok": 0, "holiday": 0, "skipped": 0, "no_vix_row": 0}
    day = start
    while day <= end:
        if day.weekday() < 5:
            r = ingest_vix_day(con, day)
            tot[r["status"]] = tot.get(r["status"], 0) + 1
            if progress:
                extra = f"  close={r['close']}" if r["status"] == "ok" else ""
                print(f"  {r['date']}  vix {r['status']:10s}{extra}")
        day += dt.timedelta(days=1)
    return tot


def vix_coverage(con) -> dict:
    r = con.execute("""SELECT COUNT(*) n, MIN(trade_date) a, MAX(trade_date) b
                       FROM vix""").fetchone()
    return {"rows": r["n"], "first": r["a"], "last": r["b"]}


# ---- Spot index OHLC (Nifty 50, etc) — same archive file as vix above ----
def parse_index_row(blob: bytes, index_name: str) -> dict | None:
    """Generic version of parse_vix_row for any row in the all-indices file."""
    text = blob.decode("utf-8", errors="replace")
    lines = text.splitlines()
    hdr = [h.strip() for h in lines[0].split(",")]
    ix = {h: i for i, h in enumerate(hdr)}
    for line in lines[1:]:
        if not line.strip():
            continue
        c = line.split(",")
        if len(c) < len(hdr) or c[ix["Index Name"]].strip() != index_name:
            continue
        return dict(
            open=_f(c[ix["Open Index Value"]]), high=_f(c[ix["High Index Value"]]),
            low=_f(c[ix["Low Index Value"]]), close=_f(c[ix["Closing Index Value"]]),
        )
    return None


def ingest_spot_index_day(con: sqlite3.Connection, day: dt.date,
                          index_name: str = "Nifty 50", force: bool = False) -> dict:
    """Idempotent, mirrors ingest_vix_day. Same source file, different row."""
    ds = day.isoformat()
    if not force:
        row = con.execute(
            "SELECT status FROM spot_index_ingest_log WHERE index_name=? AND trade_date=?",
            (index_name, ds)).fetchone()
        if row and row["status"] == "ok":
            return {"date": ds, "status": "skipped"}

    now = dt.datetime.now().isoformat(timespec="seconds")
    blob = fetch_index_close_all(day)
    if blob is None:
        con.execute("INSERT OR REPLACE INTO spot_index_ingest_log VALUES (?,?,?,?,?)",
                    (index_name, ds, "holiday", "no all-indices file", now))
        con.commit()
        return {"date": ds, "status": "holiday"}

    row = parse_index_row(blob, index_name)
    if row is None:
        con.execute("INSERT OR REPLACE INTO spot_index_ingest_log VALUES (?,?,?,?,?)",
                    (index_name, ds, "no_row", f"no '{index_name}' row in file", now))
        con.commit()
        return {"date": ds, "status": "no_row"}

    con.execute("""
        INSERT OR IGNORE INTO spot_index (index_name, trade_date, open, high, low, close,
            source, ingested_at) VALUES (?,?,?,?,?,?,'nse_indices_archive',?)
    """, (index_name, ds, row["open"], row["high"], row["low"], row["close"], now))
    con.execute("INSERT OR REPLACE INTO spot_index_ingest_log VALUES (?,?,?,?,?)",
                (index_name, ds, "ok", None, now))
    con.commit()
    return {"date": ds, "status": "ok", "close": row["close"]}


def ingest_spot_index_range(con: sqlite3.Connection, start: dt.date, end: dt.date,
                            index_name: str = "Nifty 50", progress: bool = True) -> dict:
    tot = {"ok": 0, "holiday": 0, "skipped": 0, "no_row": 0}
    day = start
    while day <= end:
        if day.weekday() < 5:
            r = ingest_spot_index_day(con, day, index_name)
            tot[r["status"]] = tot.get(r["status"], 0) + 1
            if progress:
                extra = f"  close={r['close']}" if r["status"] == "ok" else ""
                print(f"  {r['date']}  {index_name} {r['status']:10s}{extra}")
        day += dt.timedelta(days=1)
    return tot


# ---- convenience readers ------------------------------------------------
def option_chain(con, symbol: str, trade_date: str, expiry: str):
    return con.execute("""
        SELECT strike, opt_type, open, high, low, close, settle,
               open_interest, underlying, lot_size
        FROM fo_bars
        WHERE symbol=? AND trade_date=? AND expiry=? AND opt_type IS NOT NULL
        ORDER BY strike, opt_type
    """, (symbol, trade_date, expiry)).fetchall()


def contract_history(con, symbol: str, expiry: str, strike: float, opt_type: str):
    """The series a backtest walks. Works for EXPIRED contracts — the whole point."""
    return con.execute("""
        SELECT trade_date, open, high, low, close, settle, underlying,
               open_interest, lot_size
        FROM fo_bars
        WHERE symbol=? AND expiry=? AND strike=? AND opt_type=?
        ORDER BY trade_date
    """, (symbol, expiry, strike, opt_type)).fetchall()


def coverage(con) -> dict:
    r = con.execute("""SELECT COUNT(*) n, MIN(trade_date) a, MAX(trade_date) b,
                              COUNT(DISTINCT trade_date) d, COUNT(DISTINCT symbol) s
                       FROM fo_bars""").fetchone()
    return {"rows": r["n"], "first": r["a"], "last": r["b"],
            "trading_days": r["d"], "symbols": r["s"]}


if __name__ == "__main__":
    import sys
    con = connect()
    if len(sys.argv) >= 3:
        s = dt.date.fromisoformat(sys.argv[1]); e = dt.date.fromisoformat(sys.argv[2])
    else:
        e = dt.date.today(); s = e - dt.timedelta(days=7)

    print(f"Ingesting fo_bars {s} .. {e}")
    tot = ingest_range(con, s, e)
    print(f"\n{tot}")
    print(f"coverage: {coverage(con)}")

    print(f"\nIngesting vix {s} .. {e}")
    vtot = ingest_vix_range(con, s, e)
    print(f"\n{vtot}")
    print(f"vix_coverage: {vix_coverage(con)}")
