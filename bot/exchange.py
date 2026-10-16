import time

import ccxt
import pandas as pd

from .config import Config

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def make_exchange(cfg: Config, authenticated: bool = True) -> ccxt.binance:
    params = {"enableRateLimit": True, "options": {"defaultType": "spot"}}
    if authenticated:
        params["apiKey"] = cfg.api_key
        params["secret"] = cfg.api_secret
    exchange = ccxt.binance(params)
    if cfg.testnet:
        exchange.set_sandbox_mode(True)
    return exchange


def to_frame(rows: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=OHLCV_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp").astype(float)


def fetch_ohlcv(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    return to_frame(exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit))


def fetch_history(exchange: ccxt.Exchange, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """Page through public klines going back `days` days."""
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    since = exchange.milliseconds() - days * 24 * 60 * 60 * 1000
    rows: list[list] = []
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        since = batch[-1][0] + tf_ms
        if len(batch) < 1000:
            break
        time.sleep(exchange.rateLimit / 1000)
    df = to_frame(rows)
    return df[~df.index.duplicated(keep="last")]
