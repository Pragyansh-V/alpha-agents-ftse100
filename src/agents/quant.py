import yfinance as yf
import pandas as pd
from functools import lru_cache
from datetime import datetime, timedelta
from langchain_core.messages import AIMessage
from src.state import AgentState

@lru_cache(maxsize=None)
def _fetch_and_compute_metrics(ticker: str) -> dict:
    """Pure function of ticker — safe to cache for the lifetime of this run."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)

    raw_data = yf.download([ticker], start=start_date, end=end_date, progress=False)
    prices = raw_data['Adj Close'] if 'Adj Close' in raw_data.columns else raw_data['Close']
    returns = prices.pct_change().dropna()

    # --- TRUE MAX DRAWDOWN CALCULATION ---
    cumulative_returns = (1 + returns).cumprod()
    rolling_peak = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - rolling_peak) / rolling_peak
    true_max_drawdown = drawdowns.min()
    # -------------------------------------

    return {
        "volatility": round(float(returns.std().iloc[0]), 4),
        "max_drawdown": round(float(true_max_drawdown.iloc[0]), 4),
        "mean_daily_return": round(float(returns.mean().iloc[0]), 4)
    }

def quantitative_specialist_node(state: AgentState):
    """
    The Quant Agent: Fetches data, calculates TRUE mathematical risk,
    AND posts its findings to the debate ledger.
    """
    ticker = state["ticker"]
    current_round = state.get("debate_round", 1)

    print(f"\n[Quant Specialist] Round {current_round}: Executing analysis for {ticker}...")

    metrics = _fetch_and_compute_metrics(ticker)

    print(f"[Quant Specialist] Math complete.")

    quant_argument = (f"Quant Findings (Round {current_round}):\n"
                      f"Volatility (Daily Std Dev): {metrics['volatility']}\n"
                      f"True Max Drawdown (Peak-to-Trough): {metrics['max_drawdown']}\n"
                      f"Mean Daily Return: {metrics['mean_daily_return']}")

    new_message = AIMessage(content=quant_argument)

    return {
        "quant_metrics": metrics,
        "messages": [new_message],
    }