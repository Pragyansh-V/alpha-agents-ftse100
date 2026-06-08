from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.state import AgentState

def xai_auditor_node(state: AgentState):
    print(f"\n[⚖️ XAI Auditor] Auditing the PM's decision...")
    
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the Explainable AI (XAI) Auditor. Review the PM's final decision for bias, unmanaged risk, or logical inconsistency. Output a definitive PASS or FAIL verdict followed by your audit notes."),
        ("human", "PM Decision: {portfolio_decision}\nDebate Transcript: {messages}"),
        ("As the XAI & Compliance Auditor, you must ensure this financial decision complies with the EU AI Act (specifically Article 13 on Transparency and Article 14 on Human Oversight). Your audit must explicitly confirm: 1) The PM's logic is fully transparent and traceable to the provided data, devoid of unexplainable 'black-box' reasoning. 2) The output is formatted clearly so a human portfolio manager can meaningfully review and override the decision.")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"portfolio_decision": state.get("portfolio_decision", ""), "messages": state.get("messages", [])})
    response.content = f"XAI Auditor: {response.content}"
    
    print("[✅ XAI Auditor] Audit complete.")
    return {"messages": [response], "audit_notes": response.content}