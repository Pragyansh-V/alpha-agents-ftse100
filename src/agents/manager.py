import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.state import AgentState

def portfolio_manager_node(state: AgentState):
    ticker = state.get("ticker")
    print(f"\n[Portfolio Manager] Synthesizing final decision for {ticker}...")
    
    # Model configured for the baseline
    # analyst.py / manager.py / auditor.py
    llm = ChatGroq(
        model_name=os.environ.get("EXPERIMENT_MODEL", "llama-3.1-8b-instant"),
        temperature=float(os.environ.get("EXPERIMENT_TEMP", "0.0")),
        max_retries=6
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are the Lead Portfolio Manager for an institutional hedge fund. "
            "Review the multi-agent debate transcript and synthesize a final investment strategy.\n\n"
            "CRITICAL PROTOCOL (Reasoning First, Token Last):\n"
            "1. First, write a brief, high-signal rationale weighing the quantitative momentum against the macroeconomic RAG risks.\n"
            "2. You must break ties dynamically. Do not default to HOLD out of caution unless the data is explicitly neutral.\n"
            "3. Finally, on a new line at the very bottom of your response, output your definitive token action exactly like this:\n"
            "FINAL_DECISION: [BUY, HOLD, or SELL]"
        )),
        ("human", "Ticker: {ticker}\nDebate Transcript: {messages}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"ticker": ticker, "messages": state.get("messages", [])})
    response.content = f"Portfolio Manager: {response.content}"
    
    print("[Portfolio Manager] Decision finalized.")
    return {"messages": [response], "portfolio_decision": response.content}