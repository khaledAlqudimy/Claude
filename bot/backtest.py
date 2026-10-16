"""Event-driven backtest that mirrors the live engine's rules."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .risk import RiskManager
from .strategy import EmaCrossStrategy, Signal


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    amount: float
    pnl: float
    reason: str


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity: pd.Series
    stats: dict = field(default_factory=dict)


def run_backtest(df: pd.DataFrame, cfg: Config) -> BacktestResult:
    """Signals act at the *next* candle's open; stops/TPs are checked intrabar.

    If a candle touches both stop and take-profit, the stop is assumed hit first (conservative).
    """
    strategy = EmaCrossStrategy(cfg)
    risk = RiskManager(cfg)
    data = strategy.compute(df)

    cash = cfg.paper_starting_balance
    pos = None  # dict(amount, entry, stop, tp, time, entry_fee)
    trades: list[Trade] = []
    equity_curve = []
    pending: str | None = None
    prev_atr = 0.0

    def sell(price: float, ts, reason: str):
        nonlocal cash, pos
        fill = price * (1 - cfg.slippage)
        proceeds = pos["amount"] * fill
        fee = proceeds * cfg.fee_rate
        cash += proceeds - fee
        pnl = proceeds - fee - pos["amount"] * pos["entry"] - pos["entry_fee"]
        trades.append(Trade(pos["time"], ts, pos["entry"], fill, pos["amount"], pnl, reason))
        pos = None

    for ts, row in data.iterrows():
        risk.update_day(ts.date(), cash + (pos["amount"] * row["open"] if pos else 0))

        # 1) Execute the order decided on the previous candle at this candle's open
        if pending == Signal.BUY.value and pos is None:
            equity = cash
            if not risk.daily_loss_hit(equity):
                plan = risk.plan_entry(equity, cash, row["open"], prev_atr)
                if plan:
                    fill = row["open"] * (1 + cfg.slippage)
                    cost = plan.amount * fill
                    fee = cost * cfg.fee_rate
                    if cost + fee <= cash:
                        offset = fill - row["open"]
                        cash -= cost + fee
                        pos = dict(amount=plan.amount, entry=fill, stop=plan.stop_price + offset,
                                   tp=plan.take_profit_price + offset, time=ts, entry_fee=fee)
        elif pending == Signal.SELL.value and pos is not None:
            sell(row["open"], ts, "signal")
        pending = None

        # 2) Intrabar stop / take-profit
        if pos is not None:
            if row["low"] <= pos["stop"]:
                sell(min(pos["stop"], row["open"]), ts, "stop-loss")
            elif row["high"] >= pos["tp"]:
                sell(max(pos["tp"], row["open"]), ts, "take-profit")

        # 3) Decide on candle close
        if row["signal"] != Signal.HOLD.value:
            pending = row["signal"]
        prev_atr = row["atr"]

        equity_curve.append(cash + (pos["amount"] * row["close"] if pos else 0))

    if pos is not None:
        sell(data["close"].iloc[-1], data.index[-1], "end")
        equity_curve[-1] = cash

    equity = pd.Series(equity_curve, index=data.index)
    return BacktestResult(trades, equity, _stats(trades, equity, data, cfg))


def _stats(trades: list[Trade], equity: pd.Series, data: pd.DataFrame, cfg: Config) -> dict:
    start = cfg.paper_starting_balance
    returns = equity.pct_change().dropna()
    periods_per_year = _periods_per_year(data.index)
    sharpe = (
        float(returns.mean() / returns.std() * np.sqrt(periods_per_year))
        if len(returns) > 1 and returns.std() > 0 else 0.0
    )
    drawdown = equity / equity.cummax() - 1
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gross_win = sum(t.pnl for t in wins)
    gross_loss = -sum(t.pnl for t in losses)
    return {
        "start_equity": start,
        "end_equity": float(equity.iloc[-1]),
        "total_return_pct": (float(equity.iloc[-1]) / start - 1) * 100,
        "buy_hold_return_pct": (data["close"].iloc[-1] / data["close"].iloc[0] - 1) * 100,
        "max_drawdown_pct": float(drawdown.min()) * 100,
        "sharpe": sharpe,
        "trades": len(trades),
        "win_rate_pct": len(wins) / len(trades) * 100 if trades else 0.0,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0,
    }


def _periods_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 1.0
    step = (index[1:] - index[:-1]).median().total_seconds()
    return 365 * 24 * 3600 / step if step > 0 else 1.0
