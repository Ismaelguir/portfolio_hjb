from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import requests


BASE_URL = "https://www.alphavantage.co/query"

_TZ_MAP = {
    "US/Eastern": "America/New_York",
    "UTC": "UTC",
}


class AlphaVantageError(RuntimeError):
    pass


def _api_key() -> str:
    key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not key:
        raise AlphaVantageError("ALPHAVANTAGE_API_KEY manquante (mets-la dans .env).")
    return key


def _get_json(params: dict) -> dict:
    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    # juste après data = r.json() (ou équivalent)
    if "Error Message" in data:
        raise AlphaVantageError(data["Error Message"])
    if "Note" in data:
        raise AlphaVantageError(data["Note"])
    if "Information" in data:
        raise AlphaVantageError(data["Information"])
    if "Message" in data:
        raise AlphaVantageError(data["Message"])


    # erreurs / rate-limit typiques
    if isinstance(data, dict) and ("Error Message" in data):
        raise AlphaVantageError(data["Error Message"])
    if isinstance(data, dict) and ("Note" in data):
        raise AlphaVantageError(data["Note"])  # souvent limite de quota

    return data


def _parse_daily_ts(date_str: str) -> str:
    # stocké en UTC à minuit
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _parse_intraday_ts(ts_str: str, tz_label: str) -> str:
    tz_name = _TZ_MAP.get(tz_label, tz_label)
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc  # fallback (rare), mais on évite de crasher

    dt_local = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz)
    return dt_local.astimezone(timezone.utc).isoformat()


def fetch_daily_close(symbol: str, outputsize: str = "compact") -> list[tuple[str, float]]:
    """
    TIME_SERIES_DAILY -> liste (ts_utc_iso, close)
    outputsize: compact (≈100 derniers) ou full.
    """
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": outputsize,
        "apikey": _api_key(),
    }
    data = _get_json(params)

    series = data.get("Time Series (Daily)")
    if not series:
        raise AlphaVantageError("Réponse inattendue: pas de 'Time Series (Daily)'.")

    rows = []
    for day, fields in series.items():
        ts = _parse_daily_ts(day)
        close = float(fields["4. close"])
        rows.append((ts, close))

    rows.sort(key=lambda x: x[0])
    return rows


def fetch_intraday_close(symbol: str, interval: str = "60min", outputsize: str = "compact") -> list[tuple[str, float]]:
    """
    TIME_SERIES_INTRADAY -> liste (ts_utc_iso, close)
    interval: 1min/5min/15min/30min/60min.
    """
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": _api_key(),
    }
    data = _get_json(params)

    meta = data.get("Meta Data", {})
    tz_label = meta.get("6. Time Zone", meta.get("5. Time Zone", "UTC"))

    # clé dépend de l’intervalle: "Time Series (60min)" etc.
    key = f"Time Series ({interval})"
    if key not in data:
        raise AlphaVantageError(
            f"Réponse inattendue: clé '{key}' absente. "
            f"Clés dispo: {list(data.keys())}. "
            f"NB: TIME_SERIES_INTRADAY est un endpoint premium chez Alpha Vantage."
        )

    series = data.get(key)
    if not series:
        raise AlphaVantageError(f"Réponse inattendue: pas de '{key}'.")

    rows = []
    for ts_str, fields in series.items():
        ts = _parse_intraday_ts(ts_str, tz_label)
        close = float(fields["4. close"])
        rows.append((ts, close))

    rows.sort(key=lambda x: x[0])
    return rows
