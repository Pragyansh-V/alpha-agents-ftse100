import time
import os
import json
import pandas as pd
from dotenv import load_dotenv
from graph import alpha_agents_app

load_dotenv()

INPUT_CSV = "evaluation_data.csv"
RUN_NAME = os.environ.get("RUN_NAME", "default_run")
OUTPUT_JSON = f"results/{RUN_NAME}.json"
os.makedirs("results", exist_ok=True)

def run_batch_pipeline():
    # 1. Load your evaluation dataset to pull the tickers
    if not os.path.exists(INPUT_CSV):
        print(f"Could not find {INPUT_CSV}. Please run your data generation script first.")
        return
        
    df = pd.read_csv(INPUT_CSV)
    tickers = df['Ticker'].unique().tolist()
    
    print(f"Found {len(tickers)} unique tickers for evaluation: {tickers}")
    
    master_results = []

    # 2. Loop through each ticker sequentially
    for index, ticker in enumerate(tickers, 1):
        print(f"\n======================================== ({index}/{len(tickers)})")
        print(f"🏭 Starting Autonomous Swarm Processing for: {ticker}")
        print("========================================")
        
        initial_state = {
            "ticker": ticker, 
            "messages": [],
            "debate_round": 1, 
            "fundamental_analysis": "",
            "quant_metrics": {},
            "portfolio_decision": "",
            "audit_notes": ""
        }
        
        try:
            # Execute the LangGraph workflow
            final_state = alpha_agents_app.invoke(initial_state)
            
            # Serialize the LangChain message objects cleanly
            serialized_messages = []
            for msg in final_state.get('messages', []):
                if ':' in msg.content:
                    speaker, content = msg.content.split(':', 1)
                    serialized_messages.append({"speaker": speaker.strip(), "content": content.strip()})
                else:
                    serialized_messages.append({"speaker": "System", "content": msg.content})

            # Structure this company's profile
            company_payload = {
                "ticker": final_state["ticker"],
                "quant_metrics": final_state["quant_metrics"],
                "rag_context": final_state.get("rag_context", "No external context retrieved."),
                "portfolio_decision": final_state["portfolio_decision"],
                "audit_notes": final_state["audit_notes"],
                "messages": serialized_messages
            }
            
            master_results.append(company_payload)
            print(f"Successfully processed {ticker} and cached in local batch memory.")
            
        except Exception as e:
            print(f"Failed to process {ticker} due to execution runtime error: {e}")
            continue
        finally:
            time.sleep(2)  # spread token usage across the rolling 60s window

    # 3. Save the master array to disk
    with open(OUTPUT_JSON, "w") as f:
        json.dump(master_results, f, indent=4)
        
    print(f"\nBatch processing complete! Saved {len(master_results)} assets to {OUTPUT_JSON}")

if __name__ == "__main__":
    run_batch_pipeline()