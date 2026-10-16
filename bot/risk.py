from dataclasses import dataclass
from datetime import date

from .config import Config


@dataclass
class TradePlan:
    amount: float
    stop_price: float
    take_profit_price: float


class RiskManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._day: date | None = None
        self._day_start_equity = 0.0

    def plan_entry(self, equity: float, cash: float, price: float, atr_value: float) -> TradePlan | None:
        """Size a long so that hitting the stop loses ~risk_per_trade of equity."""
        if atr_value <= 0 or price <= 0 or equity <= 0:
            return None
        stop_distance = atr_value * self.cfg.stop_atr_mult
        stop_price = price - stop_distance
        if stop_price <= 0:
            return None

        amount = (equity * self.cfg.risk_per_trade) / stop_distance
        max_notional = min(equity * self.cfg.max_position_pct, cash / (1 + self.cfg.fee_rate))
        amount = min(amount, max_notional / price)
        if amount <= 0:
            return None
        return TradePlan(
            amount=amount,
            stop_price=stop_price,
            take_profit_price=price + atr_value * self.cfg.take_profit_atr_mult,
        )

    def update_day(self, today: date, equity: float) -> None:
        if self._day != today:
            self._day = today
            self._day_start_equity = equity

    def daily_loss_hit(self, equity: float) -> bool:
        if self._day_start_equity <= 0:
            return False
        loss = (self._day_start_equity - equity) / self._day_start_equity
        return loss >= self.cfg.max_daily_loss_pct
