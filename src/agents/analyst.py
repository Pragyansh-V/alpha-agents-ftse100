from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.state import AgentState

def fundamental_analyst_node(state: AgentState):
    ticker = state.get("ticker", "BP.L")
    quant_data = state.get("quant_metrics", {})
    current_round = state.get("debate_round", 1)
    
    print(f"\n[📖 Fundamental Analyst] Round {current_round}: Critiquing metrics for {ticker}...")

    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert Fundamental Analyst at a hedge fund. Critique the quantitative metrics provided, looking for macroeconomic risks."),
        ("human", "Ticker: {ticker}\nMetrics: {quant}\nPrevious Context: {messages}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"ticker": ticker, "quant": quant_data, "messages": state.get("messages", [])})
    response.content = f"Fundamental Analyst: {response.content}"
    
    print("[✅ Fundamental Analyst] Critique complete.")
    return {"messages": [response], "fundamental_analysis": response.content, "debate_round": current_round + 1}