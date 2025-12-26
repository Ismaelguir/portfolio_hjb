# src/storage/schema.py

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS portfolio_state (
  id                INTEGER PRIMARY KEY CHECK (id = 1),
  symbol            TEXT NOT NULL,
  risk_free_rate    REAL NOT NULL,              -- r (annuel, ex 0.02)
  gamma             REAL NOT NULL,              -- aversion CRRA
  horizon_years     REAL NOT NULL,              -- T (années)
  pi_min            REAL NOT NULL,
  pi_max            REAL NOT NULL,

  shares_risky      REAL NOT NULL,              -- n
  cash_risk_free    REAL NOT NULL,              -- B

  last_ts_utc       TEXT NOT NULL,              -- ISO8601 UTC
  last_price        REAL NOT NULL,              -- S(last)
  last_wealth       REAL NOT NULL               -- X(last) = n*S + B
);

CREATE TABLE IF NOT EXISTS events (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_utc      TEXT NOT NULL,
  event_type  TEXT NOT NULL,                    -- 'INIT', 'UPDATE', 'REBALANCE', ...
  payload     TEXT                              -- JSON texte optionnel
);

CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts_utc);

CREATE TABLE IF NOT EXISTS prices (
  symbol        TEXT NOT NULL,
  granularity   TEXT NOT NULL,     -- 'DAILY' ou 'INTRADAY'
  ts_utc        TEXT NOT NULL,     -- ISO8601 UTC (ex: 2025-12-26T15:30:00+00:00)
  price         REAL NOT NULL,     -- S_t (close ou last)
  source        TEXT,              -- optionnel: 'CSV', 'ALPHAVANTAGE', etc.
  PRIMARY KEY (symbol, granularity, ts_utc)
);

CREATE INDEX IF NOT EXISTS idx_prices_sym_gran_ts
ON prices(symbol, granularity, ts_utc);

"""
