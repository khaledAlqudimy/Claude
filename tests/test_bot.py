from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from bot.backtest import run_backtest
from bot.broker import PaperBroker
from bot.config import Config
from bot.engine import TradingEngine
from bot.exchange import OHLCV_COLUMNS
from bot.indicators import atr, ema, rsi
from bot.risk import RiskManager
from bot.strategy import EmaCrossStrategy, Signal

CFG = Config()


def synthetic(n=1500, seed=7) -> pd.DataFrame:
    """Random walk with regime changes so the EMAs cross several times."""
    rng = np.random.default_rng(seed)
    drift = np.sin(np.linspace(0, 12 * np.pi, n)) * 0.002
    close = 30000 * np.exp(np.cumsum(drift + rng.normal(0, 0.006, n)))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.004, n)) * close
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": np.maximum(open_, close) + spread,
         "low": np.minimum(open_, close) - spread, "close": close, "volume": 1.0},
        index=idx,
    )


def test_ema_matches_pandas():
    s = pd.Series(np.arange(1, 50, dtype=float))
    assert ema(s, 10).iloc[-1] == pytest.approx(s.ewm(span=10, adjust=False).mean().iloc[-1])


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 60, dtype=float))
    assert rsi(up, 14).iloc[-1] == pytest.approx(100)
    down = pd.Series(np.arange(60, 1, -1, dtype=float))
    assert rsi(down, 14).iloc[-1] == pytest.approx(0)
    values = rsi(synthetic()["close"]).dropna()
    assert values.between(0, 100).all()


def test_atr_constant_range():
    df = pd.DataFrame({"high": [11.0] * 40, "low": [9.0] * 40, "close": [10.0] * 40})
    assert atr(df, 14).iloc[-1] == pytest.approx(2.0)


def test_strategy_emits_both_signals_and_respects_warmup():
    strat = EmaCrossStrategy(CFG)
    out = strat.compute(synthetic())
    assert (out["signal"] == "BUY").any() and (out["signal"] == "SELL").any()
    assert (out["signal"].iloc[: strat.warmup] == "HOLD").all()


def test_strategy_latest_uses_closed_candle():
    df = synthetic()
    strat = EmaCrossStrategy(CFG)
    _, row = strat.latest(df)
    assert row.name == df.index[-2]


def test_risk_sizing_loses_one_percent_at_stop():
    rm = RiskManager(replace(CFG, max_position_pct=1.0))
    plan = rm.plan_entry(equity=10_000, cash=10_000, price=100, atr_value=2)
    assert plan.stop_price == pytest.approx(96)
    assert plan.take_profit_price == pytest.approx(108)
    assert (100 - plan.stop_price) * plan.amount == pytest.approx(100)  # 1% of 10k


def test_risk_caps_position_size():
    rm = RiskManager(CFG)  # max 25%
    plan = rm.plan_entry(equity=10_000, cash=10_000, price=100, atr_value=0.01)
    assert plan.amount * 100 == pytest.approx(2500)


def test_risk_rejects_bad_inputs():
    rm = RiskManager(CFG)
    assert rm.plan_entry(1000, 1000, 100, 0) is None
    assert rm.plan_entry(1000, 1000, 100, 60) is None  # stop below zero


def test_daily_loss_limit():
    rm = RiskManager(CFG)
    rm.update_day(pd.Timestamp("2024-01-01").date(), 1000)
    assert not rm.daily_loss_hit(980)
    assert rm.daily_loss_hit(969)


def test_paper_broker_fees_and_balance():
    b = PaperBroker(replace(CFG, slippage=0))
    fill = b.market_buy(1, 100)
    assert b.cash == pytest.approx(1000 - 100 - 0.1)
    assert fill.fee == pytest.approx(0.1)
    b.market_sell(1, 110)
    assert b.base == 0
    assert b.cash == pytest.approx(900 - 0.1 + 110 - 0.11)
    assert b.market_buy(100, 100) is None  # insufficient cash


def test_config_validation():
    with pytest.raises(ValueError):
        replace(CFG, fast_ema=30, slow_ema=20).validate()
    with pytest.raises(ValueError):
        replace(CFG, risk_per_trade=0.5).validate()


def test_backtest_runs_and_accounting_is_consistent():
    cfg = CFG
    result = run_backtest(synthetic(), cfg)
    assert result.stats["trades"] > 0
    assert result.equity.iloc[-1] == pytest.approx(cfg.paper_starting_balance + sum(t.pnl for t in result.trades))
    assert result.stats["max_drawdown_pct"] <= 0
    for t in result.trades:
        assert t.exit_time >= t.entry_time


def test_backtest_no_lookahead():
    """Truncating future data must not change trades that closed before the cut."""
    df = synthetic()
    full = run_backtest(df, CFG).trades
    cut = df.index[1000]
    partial = run_backtest(df.loc[:cut], CFG).trades
    closed_before = [t for t in full if t.exit_time < cut]
    assert closed_before == [t for t in partial if t.exit_time < cut]


class FakeExchange:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.price = float(df["close"].iloc[-1])

    def fetch_ohlcv(self, symbol, timeframe, limit):
        rows = self.df.tail(limit).reset_index()
        rows["timestamp"] = rows["index"].astype("int64") // 10**6
        return rows[OHLCV_COLUMNS].values.tolist()

    def fetch_ticker(self, symbol):
        return {"last": self.price}


def _df_ending_with(signal: str) -> pd.DataFrame:
    df = synthetic()
    sig = EmaCrossStrategy(CFG).compute(df)["signal"]
    pos = np.where(sig.values == signal)[0][-1]
    return df.iloc[: pos + 2]  # signal candle closed, one forming candle after it


def test_engine_opens_and_persists_then_stops_out(tmp_path):
    df = _df_ending_with("BUY")
    ex = FakeExchange(df)
    state = tmp_path / "state.json"
    engine = TradingEngine(CFG, ex, PaperBroker(CFG), state)
    engine.step()
    assert engine.position is not None
    assert engine.position.stop_price < engine.position.entry_price < engine.position.take_profit_price

    # A restart restores the position and paper balances
    restored = TradingEngine(CFG, ex, PaperBroker(CFG), state)
    assert restored.position == engine.position
    assert restored.broker.base == pytest.approx(engine.broker.base)

    # Same candle again: no double entry; then price crashes through the stop
    ex.price = restored.position.stop_price * 0.99
    restored.step()
    assert restored.position is None
    assert restored.broker.base == pytest.approx(0)


def test_engine_ignores_buy_when_daily_loss_hit(tmp_path):
    ex = FakeExchange(_df_ending_with("BUY"))
    engine = TradingEngine(CFG, ex, PaperBroker(CFG), tmp_path / "s.json")
    engine.risk.update_day(pd.Timestamp.now(tz="UTC").date(), 10_000)  # pretend we started the day far higher
    engine.step()
    assert engine.position is None


def test_signal_enum_roundtrip():
    assert Signal("BUY") is Signal.BUY
