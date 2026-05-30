import pandas as pd
import yfinance as yf
import time

# --- CONFIGURATION ---
INPUT_CSV = "evaluation_data.csv"
OUTPUT_CSV = "ground_truth_results.csv"

# The historical window to evaluate the Swarm's decisions against
# (e.g., Evaluate Q1 2024 performance)
START_DATE = "2025-01-01"
END_DATE = "2025-07-01"

def fetch_actual_return(ticker):
    """Fetches historical data and calculates the % return over the period."""
    try:
        stock = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if stock.empty:
            return None
        
        # Get opening price on start date and closing price on end date
        open_price = float(stock['Open'].iloc[0])
        close_price = float(stock['Close'].iloc[-1])
        
        percent_return = ((close_price - open_price) / open_price) * 100
        return round(percent_return, 2)
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def grade_decision(pm_decision, actual_return):
    """Grades the PM's decision against the actual market outcome."""
    if actual_return is None:
        return "ERROR"
        
    actual_direction = "UP" if actual_return > 0 else "DOWN"
    
    # Ignore HOLDs for pure binary accuracy metrics
    if pm_decision == "HOLD":
        return "NEUTRAL"
        
    if pm_decision == "BUY" and actual_direction == "UP":
        return "TRUE POSITIVE (WIN)"
    elif pm_decision == "SELL" and actual_direction == "DOWN":
        return "TRUE NEGATIVE (WIN)"
    elif pm_decision == "BUY" and actual_direction == "DOWN":
        return "FALSE POSITIVE (LOSS)"
    elif pm_decision == "SELL" and actual_direction == "UP":
        return "FALSE NEGATIVE (LOSS)"
    
    return "UNKNOWN"

def run_evaluation():
    print(f"📊 Loading {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"❌ Could not find {INPUT_CSV}. Run backtest.py first.")
        return

    print(f"📈 Fetching historical market data from {START_DATE} to {END_DATE}...\n")
    
    results = []
    wins = 0
    losses = 0
    
    for index, row in df.iterrows():
        ticker = row['Ticker']
        pm_decision = row['PM_Decision']
        
        print(f"Evaluating {ticker}... (Swarm said: {pm_decision})")
        actual_return = fetch_actual_return(ticker)
        
        if actual_return is not None:
            grade = grade_decision(pm_decision, actual_return)
            
            if "WIN" in grade:
                wins += 1
            elif "LOSS" in grade:
                losses += 1
                
            results.append({
                "Ticker": ticker,
                "Swarm_Decision": pm_decision,
                "Actual_Return_%": actual_return,
                "Grade": grade
            })
            print(f"   -> Actual Return: {actual_return}% | Grade: {grade}")
        else:
            print(f"   -> Failed to fetch market data.")
            
        time.sleep(1) # Pacemaker for Yahoo Finance

    # Save final results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    
    # Calculate Academic Metrics
    total_actionable = wins + losses
    accuracy = (wins / total_actionable * 100) if total_actionable > 0 else 0
    
    print("\n========================================")
    print("🎓 DISSERTATION EVALUATION SUMMARY 🎓")
    print("========================================")
    print(f"Total Tickers Evaluated: {len(results_df)}")
    print(f"Total Actionable Decisions (Buy/Sell): {total_actionable}")
    print(f"Total Wins (Correct Predictions): {wins}")
    print(f"Total Losses (Incorrect Predictions): {losses}")
    print(f"Swarm Accuracy Rate: {round(accuracy, 2)}%")
    print(f"Detailed ledger saved to: {OUTPUT_CSV}")
    print("========================================\n")

if __name__ == "__main__":
    run_evaluation()