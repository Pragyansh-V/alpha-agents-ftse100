import pandas as pd

def generate_dataset():
    # 10 specific FTSE 100 tickers for the controlled evaluation run
    tickers = [
        "RR.L", "LLOY.L", "TSCO.L", "AAL.L", "VOD.L", 
        "NG.L", "REL.L", "BA.L", "LGEN.L", "IHG.L"
    ]
    
    df = pd.DataFrame(tickers, columns=["Ticker"])
    
    # Save to the CSV file that run_swarm.py expects
    df.to_csv("evaluation_data.csv", index=False)
    print(f"Successfully generated evaluation_data.csv with {len(tickers)} FTSE100 tickers.")

if __name__ == "__main__":
    generate_dataset()