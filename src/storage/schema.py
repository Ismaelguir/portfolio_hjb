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

CREATE TABLE IF NOT EXISTS equity_curve (
  symbol          TEXT NOT NULL,
  ts_utc          TEXT NOT NULL,   -- timestamp ISO8601 UTC
  wealth          REAL NOT NULL,   -- X_t
  price           REAL NOT NULL,   -- S_t utilisé
  shares_risky    REAL NOT NULL,   -- n_t
  cash_risk_free  REAL NOT NULL,   -- B_t
  note            TEXT,
  PRIMARY KEY (symbol, ts_utc)
);

CREATE INDEX IF NOT EXISTS idx_equity_sym_ts
ON equity_curve(symbol, ts_utc);

CREATE TABLE IF NOT EXISTS rebalances (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol         TEXT NOT NULL,
  ts_utc         TEXT NOT NULL,      -- timestamp du rebalance (décision appliquée)
  price          REAL NOT NULL,      -- S_t utilisé
  wealth_before  REAL NOT NULL,      -- X juste avant rebalance (après valorisation)
  wealth_after   REAL NOT NULL,      -- X après rebalance (identique si pas de coûts)
  shares_risky   REAL NOT NULL,      -- n après rebalance
  cash_risk_free REAL NOT NULL,      -- B après rebalance
  pi             REAL NOT NULL,      -- pi appliqué (fixe ici)

  mu_annual      REAL,              -- estimé sur fenêtre, peut être NULL si pas assez d'historique
  sigma_annual   REAL,
  sigma_method   TEXT,
  ewma_lambda    REAL,
  gbm_correction INTEGER,
  window         INTEGER,
  annual_days    INTEGER,
  note           TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_reb_sym_ts ON rebalances(symbol, ts_utc);


CREATE INDEX IF NOT EXISTS idx_reb_sym_ts
ON rebalances(symbol, ts_utc);

CREATE TABLE IF NOT EXISTS run_config (
  id INTEGER PRIMARY KEY CHECK (id = 1),

  -- estimation windows
  mu_window INTEGER NOT NULL,
  sigma_window INTEGER NOT NULL,

  -- annualisation
  annual_days INTEGER NOT NULL,

  -- sigma estimator config
  sigma_method TEXT NOT NULL,       -- 'std' or 'ewma'
  ewma_lambda REAL,                 -- nullable if sigma_method='std'
  gbm_correction INTEGER NOT NULL,  -- 0/1

  -- HJB/grid config
  nx INTEGER NOT NULL,
  bc TEXT NOT NULL,
  x_min_factor REAL NOT NULL,
  x_max_factor REAL NOT NULL,
  xmin_floor REAL NOT NULL
);

INSERT OR IGNORE INTO run_config
(id, mu_window, sigma_window, annual_days, sigma_method, ewma_lambda, gbm_correction, nx, bc, x_min_factor, x_max_factor, xmin_floor)
VALUES
(1, 252, 60, 252, 'ewma', 0.94, 1, 400, 'NEUMANN', 0.1, 3.0, 1e-3);



"""
