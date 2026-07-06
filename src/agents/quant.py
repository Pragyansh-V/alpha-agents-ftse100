import yfinance as yf
import pandas as pd
import ta
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

    # --- MOMENTUM INDICATORS (forward-predictive signals) ---
    # ta expects a 1-D Series; prices may be a single-column DataFrame, so squeeze it
    price_series = prices.squeeze()

    # RSI: >70 overbought (bearish), <30 oversold (bullish), ~50 neutral
    rsi = ta.momentum.RSIIndicator(close=price_series, window=14).rsi().iloc[-1]

    # MACD histogram: positive = bullish momentum building, negative = bearish
    macd = ta.trend.MACD(close=price_series)
    macd_hist = macd.macd_diff().iloc[-1]

    # Relative momentum vs FTSE 100: is this stock outperforming the index?
    try:
        ftse = yf.download('^FTSE', start=start_date, end=end_date, progress=False)
        ftse_prices = ftse['Adj Close'] if 'Adj Close' in ftse.columns else ftse['Close']
        ftse_mean_return = float(ftse_prices.pct_change().dropna().mean().iloc[0])
        stock_mean_return = float(returns.mean().iloc[0])
        relative_momentum = round(stock_mean_return - ftse_mean_return, 6)
    except Exception:
        relative_momentum = None  # graceful fallback if FTSE fetch fails
    # --------------------------------------------------------

    return {
        "volatility": round(float(returns.std().iloc[0]), 4),
        "max_drawdown": round(float(true_max_drawdown.iloc[0]), 4),
        "mean_daily_return": round(float(returns.mean().iloc[0]), 4),
        "rsi_14": round(float(rsi), 2),
        "macd_histogram": round(float(macd_hist), 4),
        "relative_momentum_vs_ftse": relative_momentum
    }

def quantitative_specialist_node(state: AgentState):
    """
    The Quant Agent: Fetches data, calculates TRUE mathematical risk + momentum,
    AND posts its findings to the debate ledger.
    """
    ticker = state["ticker"]
    current_round = state.get("debate_round", 1)

    print(f"\n[Quant Specialist] Round {current_round}: Executing analysis for {ticker}...")

    metrics = _fetch_and_compute_metrics(ticker)

    print(f"[Quant Specialist] Math complete.")

    # Interpret momentum signals in plain language so the LLM grounds correctly
    rsi_val = metrics['rsi_14']
    rsi_signal = "OVERBOUGHT (bearish)" if rsi_val > 70 else ("OVERSOLD (bullish)" if rsi_val < 30 else "NEUTRAL")
    macd_signal = "BULLISH momentum" if metrics['macd_histogram'] > 0 else "BEARISH momentum"
    rel_mom = metrics['relative_momentum_vs_ftse']
    rel_signal = "N/A" if rel_mom is None else ("OUTPERFORMING FTSE" if rel_mom > 0 else "UNDERPERFORMING FTSE")

    quant_argument = (f"Quant Findings (Round {current_round}):\n"
                      f"Volatility (Daily Std Dev): {metrics['volatility']}\n"
                      f"True Max Drawdown (Peak-to-Trough): {metrics['max_drawdown']}\n"
                      f"Mean Daily Return: {metrics['mean_daily_return']}\n"
                      f"RSI (14-day): {metrics['rsi_14']} → {rsi_signal}\n"
                      f"MACD Histogram: {metrics['macd_histogram']} → {macd_signal}\n"
                      f"Relative Momentum vs FTSE 100: {rel_mom} → {rel_signal}")

    new_message = AIMessage(content=quant_argument)

    return {
        "quant_metrics": metrics,
        "messages": [new_message],
    }