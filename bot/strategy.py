from enum import Enum

import pandas as pd

from .config import Config
from .indicators import atr, ema, rsi


class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class EmaCrossStrategy:
    """Long-only EMA crossover.

    BUY when the fast EMA crosses above the slow EMA and RSI is not overbought.
    SELL when the fast EMA crosses below the slow EMA.
    Stops and take-profits are handled by the engine using ATR.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @property
    def warmup(self) -> int:
        return max(self.cfg.slow_ema, self.cfg.rsi_period, self.cfg.atr_period) * 3

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["ema_fast"] = ema(out["close"], self.cfg.fast_ema)
        out["ema_slow"] = ema(out["close"], self.cfg.slow_ema)
        out["rsi"] = rsi(out["close"], self.cfg.rsi_period)
        out["atr"] = atr(out, self.cfg.atr_period)

        above = out["ema_fast"] > out["ema_slow"]
        prev_above = above.shift(1, fill_value=False)
        cross_up = above & ~prev_above
        cross_down = ~above & prev_above

        out["signal"] = Signal.HOLD.value
        out.loc[cross_up & (out["rsi"] < self.cfg.rsi_max_entry), "signal"] = Signal.BUY.value
        out.loc[cross_down, "signal"] = Signal.SELL.value
        # Indicators are unreliable before warmup
        out.iloc[: self.warmup, out.columns.get_loc("signal")] = Signal.HOLD.value
        return out

    def latest(self, df: pd.DataFrame) -> tuple[Signal, pd.Series]:
        """Signal on the last *closed* candle (the final row is still forming)."""
        computed = self.compute(df)
        row = computed.iloc[-2]
        return Signal(row["signal"]), row
