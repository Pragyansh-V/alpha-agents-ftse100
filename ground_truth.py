import json
import yfinance as yf
import pandas as pd
import re
import os

def evaluate_swarm():
    print("📊 Initializing JSON Ground Truth Evaluation Engine...")
    
    # 1. Load the Swarm's autonomous decisions
    try:
        run_name = os.environ.get("RUN_NAME", "default_run")
        results_path = f"results/{run_name}.json"
        with open(results_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ {results_path} not found. Run the Swarm first.")
        return
        
    results = []
    correct_predictions = 0
    
    for asset in data:
        ticker = asset['ticker']
        
        # Extract the exact SELL/HOLD/BUY decision using the new strict token
        pm_text = asset.get('portfolio_decision', '')
        
        # Updated Regex: ONLY looks for the exact FINAL_DECISION anchor
        match = re.search(r'FINAL_DECISION:\s*(BUY|HOLD|SELL)', pm_text, re.IGNORECASE)
        
        if not match:
            print(f"⚠️ Could not parse strict decision for {ticker}")
            continue
            
        ai_decision = match.group(1).upper()
        
        # 2. Fetch the Ground Truth (Actual Market Data for the last 3 months)
        try:
            stock = yf.Ticker(ticker)
            # Fetching historical data
            hist_data = stock.history(period="3mo").dropna(subset=['Close'])
            
            if hist_data.empty:
                print(f"⚠️ No market data found for {ticker}")
                continue
                
            start_price = float(hist_data['Close'].iloc[0])
            end_price = float(hist_data['Close'].iloc[-1])
            actual_return = (end_price - start_price) / start_price
            
            # Determine true optimal action based on a 5% threshold
            if actual_return > 0.05:
                ground_truth = "BUY"
            elif actual_return < -0.05:
                ground_truth = "SELL"
            else:
                ground_truth = "HOLD"
                
            # 3. Grade the AI
            is_correct = (ai_decision == ground_truth)
            if is_correct:
                correct_predictions += 1
                
            results.append({
                "Ticker": ticker,
                "Swarm_Decision": ai_decision,
                "Actual_Market": ground_truth,
                "Actual_Return_%": f"{actual_return * 100:.2f}%",
                "Graded_Result": "✅ PASS" if is_correct else "❌ FAIL"
            })
            
        except Exception as e:
            print(f"⚠️ Failed to evaluate {ticker}: {e}")

    # 4. Calculate and display final metrics
    total = len(results)
    accuracy = (correct_predictions / total) * 100 if total > 0 else 0
    
    print("\n🏆 --- DISSERTATION EVALUATION SUMMARY --- 🏆\n")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print(f"\n🧠 Final Swarm Predictive Accuracy: {accuracy:.2f}%\n")

if __name__ == "__main__":
    evaluate_swarm()