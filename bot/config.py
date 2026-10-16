import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Config:
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    live_trading: bool = False

    symbol: str = "BTC/USDT"
    timeframe: str = "1h"

    fast_ema: int = 12
    slow_ema: int = 26
    rsi_period: int = 14
    rsi_max_entry: float = 70.0

    risk_per_trade: float = 0.01
    atr_period: int = 14
    stop_atr_mult: float = 2.0
    take_profit_atr_mult: float = 4.0
    max_position_pct: float = 0.25
    max_daily_loss_pct: float = 0.03

    paper_starting_balance: float = 1000.0
    fee_rate: float = 0.001
    slippage: float = 0.0005

    poll_seconds: int = 60

    def validate(self) -> None:
        if self.fast_ema >= self.slow_ema:
            raise ValueError("FAST_EMA must be smaller than SLOW_EMA")
        if not 0 < self.risk_per_trade <= 0.05:
            raise ValueError("RISK_PER_TRADE must be in (0, 0.05]")
        if not 0 < self.max_position_pct <= 1:
            raise ValueError("MAX_POSITION_PCT must be in (0, 1]")
        if self.stop_atr_mult <= 0 or self.take_profit_atr_mult <= 0:
            raise ValueError("ATR multipliers must be positive")


def load_config() -> Config:
    load_dotenv()
    env = os.environ.get
    cfg = Config(
        api_key=env("BINANCE_API_KEY", ""),
        api_secret=env("BINANCE_API_SECRET", ""),
        testnet=_bool(env("BINANCE_TESTNET", "true")),
        live_trading=_bool(env("LIVE_TRADING", "false")),
        symbol=env("SYMBOL", "BTC/USDT"),
        timeframe=env("TIMEFRAME", "1h"),
        fast_ema=int(env("FAST_EMA", "12")),
        slow_ema=int(env("SLOW_EMA", "26")),
        rsi_period=int(env("RSI_PERIOD", "14")),
        rsi_max_entry=float(env("RSI_MAX_ENTRY", "70")),
        risk_per_trade=float(env("RISK_PER_TRADE", "0.01")),
        atr_period=int(env("ATR_PERIOD", "14")),
        stop_atr_mult=float(env("STOP_ATR_MULT", "2.0")),
        take_profit_atr_mult=float(env("TAKE_PROFIT_ATR_MULT", "4.0")),
        max_position_pct=float(env("MAX_POSITION_PCT", "0.25")),
        max_daily_loss_pct=float(env("MAX_DAILY_LOSS_PCT", "0.03")),
        paper_starting_balance=float(env("PAPER_STARTING_BALANCE", "1000")),
        fee_rate=float(env("FEE_RATE", "0.001")),
        slippage=float(env("SLIPPAGE", "0.0005")),
        poll_seconds=int(env("POLL_SECONDS", "60")),
    )
    cfg.validate()
    return cfg
