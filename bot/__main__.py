"""CLI entry point.

    python -m bot backtest --days 365
    python -m bot backtest --csv data/btc_1h.csv
    python -m bot paper
    python -m bot live --live
"""
import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .backtest import run_backtest
from .broker import LiveBroker, PaperBroker
from .config import load_config
from .engine import TradingEngine
from .exchange import fetch_history, make_exchange


def setup_logging() -> None:
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("logs/bot.log")],
    )


def cmd_backtest(args, cfg) -> None:
    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=["timestamp"], index_col="timestamp")
    else:
        # Historical klines are public; use the production endpoint for real data
        exchange = make_exchange(replace(cfg, testnet=False), authenticated=False)
        df = fetch_history(exchange, cfg.symbol, cfg.timeframe, args.days)
        if args.save:
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.save, index_label="timestamp")
    result = run_backtest(df, cfg)
    print(f"\nBacktest {cfg.symbol} {cfg.timeframe}  {df.index[0]} -> {df.index[-1]}  ({len(df)} candles)")
    for key, value in result.stats.items():
        print(f"  {key:22s} {value:,.2f}" if isinstance(value, float) else f"  {key:22s} {value}")
    if args.trades:
        for t in result.trades:
            print(f"  {t.entry_time} -> {t.exit_time}  {t.entry_price:.2f} -> {t.exit_price:.2f}  "
                  f"pnl={t.pnl:+.2f}  [{t.reason}]")


def cmd_paper(args, cfg) -> None:
    # Paper trading uses real mainnet prices; no keys are needed or sent
    exchange = make_exchange(replace(cfg, testnet=False), authenticated=False)
    TradingEngine(cfg, exchange, PaperBroker(cfg), Path(args.state)).run()


def cmd_live(args, cfg) -> None:
    if not (args.live and cfg.live_trading):
        sys.exit("Refusing to trade: live mode needs both LIVE_TRADING=true in .env and the --live flag.")
    if not cfg.api_key or not cfg.api_secret:
        sys.exit("BINANCE_API_KEY / BINANCE_API_SECRET are missing.")
    if not cfg.testnet:
        answer = input(f"REAL MONEY on Binance mainnet ({cfg.symbol}). Type 'YES' to continue: ")
        if answer.strip() != "YES":
            sys.exit("Aborted.")
    exchange = make_exchange(cfg)
    TradingEngine(cfg, exchange, LiveBroker(cfg, exchange), Path(args.state)).run()


def main() -> None:
    parser = argparse.ArgumentParser(prog="bot", description="Binance EMA-crossover trading bot")
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="Run a historical backtest")
    bt.add_argument("--days", type=int, default=365)
    bt.add_argument("--csv", help="Use a local CSV (timestamp,open,high,low,close,volume)")
    bt.add_argument("--save", help="Save downloaded candles to this CSV path")
    bt.add_argument("--trades", action="store_true", help="Print every trade")

    paper = sub.add_parser("paper", help="Trade live prices with simulated money")
    paper.add_argument("--state", default="state.paper.json")

    live = sub.add_parser("live", help="Send real orders (testnet or mainnet)")
    live.add_argument("--live", action="store_true", help="Confirm you want to place real orders")
    live.add_argument("--state", default="state.live.json")

    args = parser.parse_args()
    setup_logging()
    cfg = load_config()
    {"backtest": cmd_backtest, "paper": cmd_paper, "live": cmd_live}[args.command](args, cfg)


if __name__ == "__main__":
    main()
