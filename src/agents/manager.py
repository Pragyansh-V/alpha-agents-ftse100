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
            "You are the Lead Portfolio Manager for an institutional hedge fund managing "
            "a portfolio with LIMITED capital. You cannot buy every stock — capital allocated "
            "to one position is capital denied to another.\n\n"

            "Your task: review the multi-agent debate transcript (Quant signals, Fundamental "
            "Analyst critique, and the Round 2 Devil's Advocate objection) and issue a decision.\n\n"

            "DECISION DISCIPLINE:\n"
            "- BUY requires the quant lean AND fundamentals to jointly justify allocating scarce "
            "capital here over alternatives.\n"
            "- SELL when momentum and fundamentals are jointly negative.\n"
            "- HOLD only when signals genuinely conflict or are neutral — not as a hedge against uncertainty.\n"
            "- Do not apply the same verdict to every stock. Differentiate based on the specific evidence.\n\n"

            "REASONING PROTOCOL (follow in order):\n"
            "1. State the Devil's Advocate's single strongest objection.\n"
            "2. Either refute it with specific data from the transcript, or concede and adjust your verdict.\n"
            "3. Give a one-sentence rationale weighing quant momentum against RAG fundamental risk.\n"
            "4. On a NEW LINE at the very end, output exactly:\n"
            "FINAL_DECISION: [BUY, HOLD, or SELL]"
        )),
        ("human", "Ticker: {ticker}\nDebate Transcript: {messages}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"ticker": ticker, "messages": state.get("messages", [])})
    response.content = f"Portfolio Manager: {response.content}"
    
    print("[Portfolio Manager] Decision finalized.")
    return {"messages": [response], "portfolio_decision": response.content}