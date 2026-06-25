import os
import yfinance as yf

# The definitive 20-ticker master universe for all experimental runs
ALL_TICKERS = [
    # Baseline 10 (Runs 1, 2, 4)
    'RR.L', 'LLOY.L', 'TSCO.L', 'AAL.L', 'VOD.L', 
    'NG.L', 'REL.L', 'BA.L', 'LGEN.L', 'IHG.L',
    
    # Scalability Expansion +5 (Run 3)
    'BP.L', 'GSK.L', 'AZN.L', 'HSBA.L', 'RIO.L',
    
    # Final Scalability Expansion +5 (Run 5 / 20-Ticker Run)
    'SHEL.L', 'BARC.L', 'DGE.L', 'BATS.L', 'AV.L'
]

DATA_DIR = "./financial_reports"

def download_rag_context():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"📡 Fetching corporate profiles for ALL {len(ALL_TICKERS)} tickers from Yahoo Finance...")
    
    for ticker_symbol in ALL_TICKERS:
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            
            name = info.get("longName", ticker_symbol)
            summary = info.get("longBusinessSummary", "No corporate summary available.")
            sector = info.get("sector", "Unknown Sector")
            industry = info.get("industry", "Unknown Industry")
            
            # GLUING THE HEADER AND SUMMARY: Removed the \n boundary between the macro statement and the summary string
            file_content = (
                f"Company Name: {name} | Ticker: {ticker_symbol} | Sector: {sector} | Industry: {industry}\n"
                f"Macroeconomic data and risks for {ticker_symbol} | {summary}\n"
            )
            
            file_path = os.path.join(DATA_DIR, f"{ticker_symbol}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(file_content)
                
            print(f"✅ Saved RAG context for {ticker_symbol} -> {file_path}")
            
        except Exception as e:
            print(f"❌ Failed to fetch data for {ticker_symbol}: {e}")

if __name__ == "__main__":
    download_rag_context()