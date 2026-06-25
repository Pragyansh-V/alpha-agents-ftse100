from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.state import AgentState

def xai_auditor_node(state: AgentState):
    print(f"\n[⚖️ XAI Auditor] Auditing the PM's decision...")
    
    # Model for rigorous compliance checking
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are the Explainable AI (XAI) Auditor for an institutional hedge fund. "
            "Review the Portfolio Manager's (PM) decision for bias, unmanaged risk, and logical inconsistency.\n\n"
            "COMPLIANCE MANDATE (EU AI Act):\n"
            "You must ensure this financial decision complies with Article 13 (Transparency) and Article 14 (Human Oversight). "
            "Your audit must explicitly confirm:\n"
            "1) The PM's logic is fully transparent, traceable to the RAG/Quant data, and devoid of 'black-box' reasoning.\n"
            "2) The output is formatted clearly so a human portfolio manager can meaningfully override the decision if necessary.\n\n"
            "REQUIRED OUTPUT FORMAT:\n"
            "VERDICT: [PASS or FAIL]\n"
            "COMPLIANCE NOTES: [Provide a concise, professional audit of the PM's transparency and risk management here]"
        )),
        ("human", "PM Decision: {portfolio_decision}\nDebate Transcript: {messages}")
    ])
    
    chain = prompt | llm
    
    # Safely extracting variables from the LangGraph state
    response = chain.invoke({
        "portfolio_decision": state.get("portfolio_decision", "NO DECISION FOUND"), 
        "messages": state.get("messages", [])
    })
    
    response.content = f"XAI Auditor: {response.content}"
    
    print("[✅ XAI Auditor] Audit complete.")
    return {"messages": [response], "audit_notes": response.content}