import csv
import time
from graph import alpha_agents_app 

# We will start with a sample of 10 FTSE 100 tickers. 
# We can expand this to 50 for the final dissertation run.
TICKERS = [
    "BP.L", "SHEL.L", "HSBA.L", "AZN.L", "ULVR.L",
    "RIO.L", "GSK.L", "DGE.L", "BATS.L", "GLEN.L"
]

CSV_FILENAME = "evaluation_data.csv"

def run_backtest():
    print(f"🚀 Starting AlphaAgents Historical Backtest for {len(TICKERS)} tickers...\n")
    
    # Initialize the CSV file and write the header row
    with open(CSV_FILENAME, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Ticker", "PM_Decision", "Auditor_Verdict", "Execution_Time_sec"])

    for ticker in TICKERS:
        print(f"--- Processing {ticker} ---")
        start_time = time.time()
        
        # Initialize LangGraph state for this specific ticker
        initial_state = {
            "ticker": ticker,
            "debate_round": 1,
            "messages": []
        }
        
        try:
            # 1. Trigger the Swarm
            final_state = alpha_agents_app.invoke(initial_state)
            
            # 2. Extract and parse the PM's final decision
            pm_raw = final_state.get("portfolio_decision", "")
            decision = "UNKNOWN"
            if "BUY" in pm_raw.upper(): decision = "BUY"
            elif "SELL" in pm_raw.upper(): decision = "SELL"
            elif "HOLD" in pm_raw.upper(): decision = "HOLD"
                
            # 3. Extract and parse the Auditor's verdict
            auditor_raw = final_state.get("audit_notes", "")
            verdict = "UNKNOWN"
            if "PASS" in auditor_raw.upper(): verdict = "PASS"
            elif "FAIL" in auditor_raw.upper(): verdict = "FAIL"
                
            execution_time = round(time.time() - start_time, 2)
            
            # 4. Append the results to the CSV safely
            with open(CSV_FILENAME, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([ticker, decision, verdict, execution_time])
                
            print(f"[✅] {ticker} completed in {execution_time}s -> {decision} | Auditor: {verdict}\n")
            
        except Exception as e:
            print(f"[❌] Error processing {ticker}: {e}\n")
            
        # A 2-second pacemaker to ensure yfinance doesn't rate-limit our Quant node
        time.sleep(2)

    print(f"🎉 Backtest complete! Data saved to {CSV_FILENAME}")

if __name__ == "__main__":
    run_backtest()