from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.state import AgentState

def xai_auditor_node(state: AgentState):
    print(f"\n[⚖️ XAI Auditor] Auditing the PM's decision...")
    
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the Explainable AI (XAI) Auditor. Review the PM's final decision for bias, unmanaged risk, or logical inconsistency. Output a definitive PASS or FAIL verdict followed by your audit notes."),
        ("human", "PM Decision: {portfolio_decision}\nDebate Transcript: {messages}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"portfolio_decision": state.get("portfolio_decision", ""), "messages": state.get("messages", [])})
    response.content = f"XAI Auditor: {response.content}"
    
    print("[✅ XAI Auditor] Audit complete.")
    return {"messages": [response], "audit_notes": response.content}