import os
import yfinance as yf

ACTIVE_TICKERS = ['RR.L', 'TSCO.L', 'AAL.L', 'BA.L', 'IHG.L']

DATA_DIR = "./financial_reports"

def format_number(value, prefix="", suffix="", billions=False):
    """Safely format numeric values, returning 'N/A' if missing."""
    if value is None:
        return "N/A"
    try:
        if billions:
            return f"{prefix}{float(value)/1e9:.2f}B{suffix}"
        return f"{prefix}{float(value):,.2f}{suffix}"
    except:
        return "N/A"

def download_rag_context():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"📡 Fetching enriched financial profiles for {len(ACTIVE_TICKERS)} tickers...")

    for ticker_symbol in ACTIVE_TICKERS:
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info

            # --- Identity ---
            name = info.get("longName", ticker_symbol)
            sector = info.get("sector", "Unknown Sector")
            industry = info.get("industry", "Unknown Industry")
            summary = info.get("longBusinessSummary", "No corporate summary available.")

            # --- Valuation ---
            market_cap = format_number(info.get("marketCap"), billions=True)
            pe_ratio = format_number(info.get("trailingPE"))
            forward_pe = format_number(info.get("forwardPE"))
            price_to_book = format_number(info.get("priceToBook"))
            ev_to_ebitda = format_number(info.get("enterpriseToEbitda"))

            # --- Financial Health ---
            revenue = format_number(info.get("totalRevenue"), billions=True)
            revenue_growth = format_number(info.get("revenueGrowth"), suffix="%") if info.get("revenueGrowth") else "N/A"
            gross_margins = format_number(info.get("grossMargins", 0) * 100, suffix="%")
            operating_margins = format_number(info.get("operatingMargins", 0) * 100, suffix="%")
            profit_margins = format_number(info.get("profitMargins", 0) * 100, suffix="%")
            debt_to_equity = format_number(info.get("debtToEquity"))
            current_ratio = format_number(info.get("currentRatio"))
            free_cashflow = format_number(info.get("freeCashflow"), billions=True)

            # --- Earnings & Growth ---
            earnings_growth = format_number(info.get("earningsGrowth", 0) * 100, suffix="%") if info.get("earningsGrowth") else "N/A"
            eps_trailing = format_number(info.get("trailingEps"))
            eps_forward = format_number(info.get("forwardEps"))
            return_on_equity = format_number(info.get("returnOnEquity", 0) * 100, suffix="%") if info.get("returnOnEquity") else "N/A"

            # --- Market Performance ---
            week_52_high = format_number(info.get("fiftyTwoWeekHigh"))
            week_52_low = format_number(info.get("fiftyTwoWeekLow"))
            current_price = format_number(info.get("currentPrice"))
            analyst_target = format_number(info.get("targetMeanPrice"))
            recommendation = info.get("recommendationKey", "N/A").upper()
            analyst_count = info.get("numberOfAnalystOpinions", "N/A")

            # --- Dividend ---
            dividend_yield = format_number(info.get("dividendYield", 0) * 100, suffix="%") if info.get("dividendYield") else "N/A"

            file_content = f"""Company Name: {name} | Ticker: {ticker_symbol} | Sector: {sector} | Industry: {industry}

BUSINESS OVERVIEW:
{summary}

VALUATION METRICS:
- Market Cap: {market_cap}
- Trailing P/E: {pe_ratio} | Forward P/E: {forward_pe}
- Price-to-Book: {price_to_book}
- EV/EBITDA: {ev_to_ebitda}

FINANCIAL HEALTH:
- Total Revenue: {revenue}
- Revenue Growth (YoY): {revenue_growth}
- Gross Margin: {gross_margins} | Operating Margin: {operating_margins} | Profit Margin: {profit_margins}
- Debt-to-Equity: {debt_to_equity}
- Current Ratio: {current_ratio}
- Free Cash Flow: {free_cashflow}

EARNINGS & GROWTH:
- EPS (Trailing): {eps_trailing} | EPS (Forward): {eps_forward}
- Earnings Growth: {earnings_growth}
- Return on Equity: {return_on_equity}

MARKET PERFORMANCE:
- Current Price: {current_price}
- 52-Week Range: {week_52_low} - {week_52_high}
- Analyst Mean Target: {analyst_target}
- Analyst Consensus: {recommendation} ({analyst_count} analysts)
- Dividend Yield: {dividend_yield}
"""

            file_path = os.path.join(DATA_DIR, f"{ticker_symbol}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(file_content)

            print(f"Saved enriched RAG context for {ticker_symbol}")

        except Exception as e:
            print(f"Failed to fetch data for {ticker_symbol}: {e}")

if __name__ == "__main__":
    download_rag_context()