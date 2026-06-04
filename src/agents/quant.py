import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from langchain_core.messages import AIMessage
from src.state import AgentState

def quantitative_specialist_node(state: AgentState):
    """
    The Quant Agent: Fetches data, updates the strict dictionary, 
    AND posts its findings to the debate ledger.
    """
    ticker = state["ticker"]
    
    # 1. Read the current time on the clock
    current_round = state.get("debate_round", 1)
    
    print(f"\n[Quant Specialist] Round {current_round}: Executing analysis for {ticker}...")
    
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)
    
    raw_data = yf.download([ticker], start=start_date, end=end_date)
    prices = raw_data['Adj Close'] if 'Adj Close' in raw_data.columns else raw_data['Close']
        
    returns = prices.pct_change().dropna()
    
    metrics = {
        "volatility": round(float(returns.std().iloc[0]), 4),
        "max_drawdown_daily": round(float(returns.min().iloc[0]), 4),
        "mean_daily_return": round(float(returns.mean().iloc[0]), 4)
    }
    
    print(f"[Quant Specialist] Math complete.")
    
    # 2. Grab the microphone and speak into the debate ledger
    quant_argument = (f"Quant Findings (Round {current_round}):\n"
                      f"Volatility: {metrics['volatility']}\n"
                      f"Max Drawdown: {metrics['max_drawdown_daily']}\n"
                      f"Mean Return: {metrics['mean_daily_return']}")
    
    new_message = AIMessage(content=quant_argument)
    
    # 3. Return the dictionary, append the message, and advance the clock!
    return {
        "quant_metrics": metrics,
        "messages": [new_message],
        "debate_round": current_round + 1
    }