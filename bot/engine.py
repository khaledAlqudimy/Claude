"""Live/paper trading loop."""
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import ccxt

from .broker import LiveBroker, PaperBroker
from .config import Config
from .exchange import fetch_ohlcv
from .risk import RiskManager
from .strategy import EmaCrossStrategy, Signal

log = logging.getLogger(__name__)


@dataclass
class Position:
    amount: float
    entry_price: float
    stop_price: float
    take_profit_price: float
    opened_at: str


class StateStore:
    """Persists the open position (and paper balances) so a restart keeps its stops."""

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text())

    def save(self, data: dict) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.path)


class TradingEngine:
    def __init__(self, cfg: Config, exchange: ccxt.Exchange, broker, state_path: Path = Path("state.json")):
        self.cfg = cfg
        self.exchange = exchange
        self.broker = broker
        self.strategy = EmaCrossStrategy(cfg)
        self.risk = RiskManager(cfg)
        self.store = StateStore(state_path)
        self.position: Position | None = None
        self.last_candle: str | None = None
        self._restore()

    def _restore(self) -> None:
        state = self.store.load()
        if state.get("position"):
            self.position = Position(**state["position"])
            log.info("Restored open position: %s", self.position)
        self.last_candle = state.get("last_candle")
        if isinstance(self.broker, PaperBroker) and "paper" in state:
            self.broker.cash = state["paper"]["cash"]
            self.broker.base = state["paper"]["base"]

    def _persist(self) -> None:
        state = {
            "position": asdict(self.position) if self.position else None,
            "last_candle": self.last_candle,
        }
        if isinstance(self.broker, PaperBroker):
            state["paper"] = {"cash": self.broker.cash, "base": self.broker.base}
        self.store.save(state)

    def _close(self, price: float, reason: str) -> None:
        fill = self.broker.market_sell(self.position.amount, price)
        if fill is None:
            log.error("Failed to close position (%s)", reason)
            return
        pnl = (fill.price - self.position.entry_price) * fill.amount - fill.fee
        log.info("SELL %.6f @ %.2f [%s] approx PnL %.2f", fill.amount, fill.price, reason, pnl)
        self.position = None

    def step(self) -> None:
        df = fetch_ohlcv(self.exchange, self.cfg.symbol, self.cfg.timeframe, limit=self.strategy.warmup + 100)
        price = float(self.exchange.fetch_ticker(self.cfg.symbol)["last"])
        cash, base = self.broker.balances()
        equity = cash + base * price
        self.risk.update_day(datetime.now(timezone.utc).date(), equity)

        # Stops are checked every tick, not just on candle close
        if self.position:
            if price <= self.position.stop_price:
                self._close(price, "stop-loss")
            elif price >= self.position.take_profit_price:
                self._close(price, "take-profit")

        signal, row = self.strategy.latest(df)
        candle_id = str(row.name)
        if candle_id != self.last_candle:
            self.last_candle = candle_id
            log.info(
                "Candle %s close=%.2f emaF=%.2f emaS=%.2f rsi=%.1f atr=%.2f -> %s | equity=%.2f",
                candle_id, row["close"], row["ema_fast"], row["ema_slow"], row["rsi"], row["atr"],
                signal.value, equity,
            )
            if signal == Signal.SELL and self.position:
                self._close(price, "signal")
            elif signal == Signal.BUY and not self.position:
                self._open(equity, cash, price, float(row["atr"]))

        self._persist()

    def _open(self, equity: float, cash: float, price: float, atr_value: float) -> None:
        if self.risk.daily_loss_hit(equity):
            log.warning("Daily loss limit reached; skipping entry")
            return
        plan = self.risk.plan_entry(equity, cash, price, atr_value)
        if plan is None:
            log.info("No viable position size")
            return
        fill = self.broker.market_buy(plan.amount, price)
        if fill is None:
            return
        # Re-anchor stops to the actual fill price
        offset = fill.price - price
        self.position = Position(
            amount=fill.amount,
            entry_price=fill.price,
            stop_price=plan.stop_price + offset,
            take_profit_price=plan.take_profit_price + offset,
            opened_at=datetime.now(timezone.utc).isoformat(),
        )
        log.info(
            "BUY %.6f @ %.2f stop=%.2f tp=%.2f",
            fill.amount, fill.price, self.position.stop_price, self.position.take_profit_price,
        )

    def run(self) -> None:
        mode = "LIVE" if isinstance(self.broker, LiveBroker) else "PAPER"
        log.info("Starting %s trading on %s %s (testnet=%s)", mode, self.cfg.symbol, self.cfg.timeframe, self.cfg.testnet)
        while True:
            try:
                self.step()
            except ccxt.NetworkError as e:
                log.warning("Network error, retrying next tick: %s", e)
            except ccxt.ExchangeError as e:
                log.error("Exchange error: %s", e)
            time.sleep(self.cfg.poll_seconds)
