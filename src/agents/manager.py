from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.state import AgentState

def portfolio_manager_node(state: AgentState):
    ticker = state.get("ticker")
    print(f"\n[👔 Portfolio Manager] Synthesizing final decision for {ticker}...")
    
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the Lead Portfolio Manager. Review the debate transcript and make a definitive investment decision (BUY/HOLD/SELL) with clear justification."),
        ("human", "Ticker: {ticker}\nDebate Transcript: {messages}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"ticker": ticker, "messages": state.get("messages", [])})
    response.content = f"Portfolio Manager: {response.content}"
    
    print("[✅ Portfolio Manager] Decision finalized.")
    return {"messages": [response], "portfolio_decision": response.content}