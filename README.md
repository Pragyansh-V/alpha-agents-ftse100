# Reliable Multi-Agent Architectures for Equity Portfolio Management

MSc AI Dissertation Project — Heriot-Watt University, Edinburgh (2026)

## What this is
A multi-agent system for FTSE 100 equity analysis that extends the 
AlphaAgents framework (Wang et al., 2025) to the UK market with 
EU AI Act compliance built in.

## The problem it solves
Single LLM agents hallucinate in financial contexts and produce 
unauditable decisions — unacceptable under the EU AI Act (Article 14) 
and FCA operational resilience standards.

## Architecture
[Add your LangGraph diagram here]

Four specialised agents communicate via a Round Robin debate protocol:
- Fundamental Analyst — parses FTSE 100 annual reports via RAG
- Quantitative Specialist — executes Python for Sharpe Ratio, MDD
- Portfolio Manager — synthesises into investment thesis
- XAI Auditor — validates rationale fidelity using RAGAS metrics

## Tech stack
Python · LangGraph · LangChain · LlamaIndex · ChromaDB · 
yfinance · RAGAS · Google Colab (GPU)

## Key results
[Add when dissertation is complete]
- RAGAS faithfulness score: >0.85 target
- Sharpe Ratio vs FTSE 100 buy-and-hold baseline
- Cross-market comparison: S&P 500 vs FTSE 100

## Research report
[Link to your F21RP report PDF]
