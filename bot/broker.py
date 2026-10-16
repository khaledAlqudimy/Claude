"""Order execution. PaperBroker simulates fills; LiveBroker sends real orders through ccxt."""
import logging
from dataclasses import dataclass

import ccxt

from .config import Config

log = logging.getLogger(__name__)


@dataclass
class Fill:
    side: str
    amount: float
    price: float
    fee: float


class PaperBroker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cash = cfg.paper_starting_balance
        self.base = 0.0

    def balances(self) -> tuple[float, float]:
        return self.cash, self.base

    def market_buy(self, amount: float, price: float) -> Fill | None:
        fill_price = price * (1 + self.cfg.slippage)
        cost = amount * fill_price
        fee = cost * self.cfg.fee_rate
        if cost + fee > self.cash:
            log.warning("Paper: insufficient cash for buy")
            return None
        self.cash -= cost + fee
        self.base += amount
        return Fill("buy", amount, fill_price, fee)

    def market_sell(self, amount: float, price: float) -> Fill | None:
        amount = min(amount, self.base)
        if amount <= 0:
            return None
        fill_price = price * (1 - self.cfg.slippage)
        proceeds = amount * fill_price
        fee = proceeds * self.cfg.fee_rate
        self.base -= amount
        self.cash += proceeds - fee
        return Fill("sell", amount, fill_price, fee)


class LiveBroker:
    def __init__(self, cfg: Config, exchange: ccxt.Exchange):
        self.cfg = cfg
        self.exchange = exchange
        self.exchange.load_markets()
        self.market = self.exchange.market(cfg.symbol)

    def balances(self) -> tuple[float, float]:
        bal = self.exchange.fetch_balance()
        quote = float(bal.get(self.market["quote"], {}).get("free") or 0)
        base = float(bal.get(self.market["base"], {}).get("free") or 0)
        return quote, base

    def _prepare(self, amount: float, price: float) -> float | None:
        amount = float(self.exchange.amount_to_precision(self.cfg.symbol, amount))
        limits = self.market.get("limits", {})
        min_amount = (limits.get("amount") or {}).get("min") or 0
        min_cost = (limits.get("cost") or {}).get("min") or 0
        if amount <= 0 or amount < min_amount or amount * price < min_cost:
            log.warning("Order below exchange minimums (amount=%s, notional=%.2f)", amount, amount * price)
            return None
        return amount

    def _fill(self, side: str, order: dict, fallback_amount: float, fallback_price: float) -> Fill:
        fee = sum(float(f.get("cost") or 0) for f in order.get("fees") or [])
        return Fill(
            side,
            float(order.get("filled") or fallback_amount),
            float(order.get("average") or order.get("price") or fallback_price),
            fee,
        )

    def market_buy(self, amount: float, price: float) -> Fill | None:
        amount = self._prepare(amount, price)
        if amount is None:
            return None
        order = self.exchange.create_order(self.cfg.symbol, "market", "buy", amount)
        log.info("LIVE BUY order %s", order.get("id"))
        return self._fill("buy", order, amount, price)

    def market_sell(self, amount: float, price: float) -> Fill | None:
        _, base_free = self.balances()
        amount = self._prepare(min(amount, base_free), price)
        if amount is None:
            return None
        order = self.exchange.create_order(self.cfg.symbol, "market", "sell", amount)
        log.info("LIVE SELL order %s", order.get("id"))
        return self._fill("sell", order, amount, price)
