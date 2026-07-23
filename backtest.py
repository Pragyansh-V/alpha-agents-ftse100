import json
import os
import re
import numpy as np
import pandas as pd
import yfinance as yf

# --- Configuration ---
RISK_FREE_ANNUAL = 0.045          # ~4.5% UK risk-free rate (Bank of England base ~2025)
TRADING_DAYS = 252
EVAL_PERIOD = "3mo"               # matches ground_truth.py for consistency
ALLOW_SHORTING = True             # SELL -> short (-1). Set False for long-only (SELL -> flat 0)

def decision_to_position(decision: str) -> int:
    """Map the swarm's verdict to a portfolio position weight."""
    if decision == "BUY":
        return 1
    if decision == "SELL":
        return -1 if ALLOW_SHORTING else 0
    return 0  # HOLD -> flat (in cash)

def compute_sharpe(daily_returns: pd.Series) -> float:
    """Annualised Sharpe ratio from a series of daily strategy returns."""
    if daily_returns.std() == 0 or len(daily_returns) == 0:
        return 0.0
    daily_rf = RISK_FREE_ANNUAL / TRADING_DAYS
    excess = daily_returns - daily_rf
    return float((excess.mean() / daily_returns.std()) * np.sqrt(TRADING_DAYS))

def compute_max_drawdown(cumulative: pd.Series) -> float:
    """Worst peak-to-trough decline of an equity curve."""
    rolling_peak = cumulative.cummax()
    drawdowns = (cumulative - rolling_peak) / rolling_peak
    return float(drawdowns.min())

def run_backtest():
    print("Initializing Portfolio Backtest Engine...")

    run_name = os.environ.get("RUN_NAME", "default_run")
    results_path = f"results/{run_name}.json"
    try:
        with open(results_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f" {results_path} not found. Run the Swarm first.")
        return

    # --- Build each ticker's daily strategy-return series ---
    per_ticker_returns = []
    holdings = []

    for asset in data:
        ticker = asset["ticker"]
        pm_text = asset.get("portfolio_decision", "")
        match = re.search(r'FINAL_DECISION:\s*(BUY|HOLD|SELL)', pm_text, re.IGNORECASE)
        if not match:
            print(f"Could not parse decision for {ticker}; skipping.")
            continue

        decision = match.group(1).upper()
        position = decision_to_position(decision)

        try:
            hist = yf.Ticker(ticker).history(period=EVAL_PERIOD).dropna(subset=["Close"])
            if hist.empty:
                print(f"No market data for {ticker}; skipping.")
                continue

            daily_stock_returns = hist["Close"].pct_change().dropna()
            # Strategy return = position * market return (short flips the sign)
            strat_returns = position * daily_stock_returns
            per_ticker_returns.append(strat_returns.reset_index(drop=True))

            total_return = (1 + strat_returns).prod() - 1
            holdings.append({
                "Ticker": ticker,
                "Decision": decision,
                "Position": position,
                "Period_Return_%": f"{total_return * 100:.2f}%"
            })
        except Exception as e:
            print(f"Failed to backtest {ticker}: {e}")

    if not per_ticker_returns:
        print(" No valid tickers to backtest.")
        return

    # --- Equal-weighted portfolio: average across all held positions daily ---
    portfolio_df = pd.concat(per_ticker_returns, axis=1)
    portfolio_daily = portfolio_df.mean(axis=1)  # equal weight
    portfolio_cumulative = (1 + portfolio_daily).cumprod()

    # --- Strategy metrics ---
    strat_total_return = float(portfolio_cumulative.iloc[-1] - 1)
    strat_sharpe = compute_sharpe(portfolio_daily)
    strat_mdd = compute_max_drawdown(portfolio_cumulative)

    # --- Benchmark: equal-weighted buy-and-hold of the SAME tickers ---
    bench_returns = []
    for asset in data:
        try:
            hist = yf.Ticker(asset["ticker"]).history(period=EVAL_PERIOD).dropna(subset=["Close"])
            if not hist.empty:
                bench_returns.append(hist["Close"].pct_change().dropna().reset_index(drop=True))
        except Exception:
            continue
    bench_daily = pd.concat(bench_returns, axis=1).mean(axis=1)
    bench_cumulative = (1 + bench_daily).cumprod()
    bench_total_return = float(bench_cumulative.iloc[-1] - 1)
    bench_sharpe = compute_sharpe(bench_daily)
    bench_mdd = compute_max_drawdown(bench_cumulative)

    # --- Report ---
    print("\n🏆 --- PORTFOLIO BACKTEST SUMMARY --- 🏆\n")
    print(pd.DataFrame(holdings).to_string(index=False))

    print(f"\n{'Metric':<22}{'Swarm Strategy':>18}{'Buy & Hold':>16}")
    print("-" * 56)
    print(f"{'Total Return':<22}{strat_total_return*100:>17.2f}%{bench_total_return*100:>15.2f}%")
    print(f"{'Sharpe Ratio':<22}{strat_sharpe:>18.3f}{bench_sharpe:>16.3f}")
    print(f"{'Max Drawdown':<22}{strat_mdd*100:>17.2f}%{bench_mdd*100:>15.2f}%")
    print(f"\nShorting enabled: {ALLOW_SHORTING}")

    # --- Persist so run_experiment.py can log it to MLflow ---
    metrics = {
        "run_name": run_name,
        "strategy": {"total_return": strat_total_return, "sharpe": strat_sharpe, "max_drawdown": strat_mdd},
        "benchmark": {"total_return": bench_total_return, "sharpe": bench_sharpe, "max_drawdown": bench_mdd},
        "shorting_enabled": ALLOW_SHORTING
    }
    os.makedirs("results", exist_ok=True)
    with open(f"results/{run_name}_backtest.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nBacktest metrics saved to results/{run_name}_backtest.json")

if __name__ == "__main__":
    run_backtest()