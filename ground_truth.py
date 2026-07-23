import json
import yfinance as yf
import pandas as pd
import re
import os

def grade_decision(ai_decision, ground_truth, actual_return):
    """
    Returns a tuple: (exact_match: bool, directional_credit: int)
    - directional_credit: 2 = exact, 1 = right direction wrong intensity, 0 = wrong direction
    """
    if ai_decision == ground_truth:
        return True, 2
    # Partial credit: model got the DIRECTION right even if intensity was off
    # (e.g. said SELL when truth was HOLD-but-negative, or BUY when HOLD-but-positive)
    if ai_decision == "SELL" and actual_return < 0:
        return False, 1
    if ai_decision == "BUY" and actual_return > 0:
        return False, 1
    if ai_decision == "HOLD" and abs(actual_return) < 0.10:
        return False, 1  # HOLD is "directionally reasonable" if move was modest
    return False, 0

def evaluate_swarm():
    print("Initializing JSON Ground Truth Evaluation Engine...")

    try:
        run_name = os.environ.get("RUN_NAME", "default_run")
        results_path = f"results/{run_name}.json"
        with open(results_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"{results_path} not found. Run the Swarm first.")
        return

    results = []
    correct_predictions = 0      # binary (exact match) — unchanged for fair comparison
    directional_points = 0       # new: rewards directional awareness
    max_directional = 0

    for asset in data:
        ticker = asset['ticker']
        pm_text = asset.get('portfolio_decision', '')
        match = re.search(r'FINAL_DECISION:\s*(BUY|HOLD|SELL)', pm_text, re.IGNORECASE)

        if not match:
            print(f"Could not parse strict decision for {ticker}")
            continue

        ai_decision = match.group(1).upper()

        try:
            stock = yf.Ticker(ticker)
            hist_data = stock.history(period="3mo").dropna(subset=['Close'])

            if hist_data.empty:
                print(f"No market data found for {ticker}")
                continue

            start_price = float(hist_data['Close'].iloc[0])
            end_price = float(hist_data['Close'].iloc[-1])
            actual_return = (end_price - start_price) / start_price

            if actual_return > 0.05:
                ground_truth = "BUY"
            elif actual_return < -0.05:
                ground_truth = "SELL"
            else:
                ground_truth = "HOLD"

            # --- Grade both ways ---
            is_correct, dir_credit = grade_decision(ai_decision, ground_truth, actual_return)
            if is_correct:
                correct_predictions += 1
            directional_points += dir_credit
            max_directional += 2

            results.append({
                "Ticker": ticker,
                "Swarm_Decision": ai_decision,
                "Actual_Market": ground_truth,
                "Actual_Return_%": f"{actual_return * 100:.2f}%",
                "Graded_Result": "PASS" if is_correct else "FAIL",
                "Directional": "🎯" if dir_credit == 2 else ("↗️" if dir_credit == 1 else "❌")
            })

        except Exception as e:
            print(f"Failed to evaluate {ticker}: {e}")

    total = len(results)
    accuracy = (correct_predictions / total) * 100 if total > 0 else 0
    directional_accuracy = (directional_points / max_directional) * 100 if max_directional > 0 else 0

    print("\n🏆 --- DISSERTATION EVALUATION SUMMARY --- 🏆\n")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print(f"\nFinal Swarm Predictive Accuracy (exact): {accuracy:.2f}%")
    print(f"Directional Accuracy (weighted): {directional_accuracy:.2f}%\n")

if __name__ == "__main__":
    evaluate_swarm()