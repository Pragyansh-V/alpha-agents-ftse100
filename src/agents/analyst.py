import os
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.state import AgentState

# Core RAG Imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def fundamental_analyst_node(state: AgentState):
    ticker = state.get("ticker", "BP.L")
    quant_data = state.get("quant_metrics", {})
    current_round = state.get("debate_round", 1)
    
    print(f"\n[📖 Fundamental Analyst] Round {current_round}: Critiquing metrics for {ticker}...")

    # --- LOCAL RAG LAYER ---
    print(f"[🔍 RAG Core] Querying FAISS database for {ticker} documents...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # Exact string matching query to prevent ticker collision (e.g., RR.L vs REL.L)
        retrieved_docs = vector_db.similarity_search(f"Ticker: {ticker} | Sector:", k=1)
        
        if retrieved_docs:
            retrieved_context = retrieved_docs[0].page_content
            print(f"[✅ RAG Core] Grounded context retrieved successfully.")
        else:
            retrieved_context = "No external matching reports found in local memory store."
            print("[⚠️ RAG Core] No matching documents found for this ticker.")
            
    except Exception as e:
        print(f"[⚠️ RAG Core] Local vector search failed: {e}")
        retrieved_context = "No external matching reports available due to a system indexing error."
    # -----------------------

    # analyst.py / manager.py / auditor.py
    llm = ChatGroq(model_name=os.environ.get("EXPERIMENT_MODEL", "llama-3.1-8b-instant"),
        temperature=float(os.environ.get("EXPERIMENT_TEMP", "0.0")),
        max_retries=6)
    
    # Injecting Grounded Intelligence seamlessly alongside pure numbers
    prompt = ChatPromptTemplate.from_messages([
        (
            "system", 
            "You are an expert Fundamental Analyst at a hedge fund. Critique the quantitative metrics "
            "provided, factoring in specific macroeconomic conditions and corporate intelligence."
        ),
        (
            "human", 
            "Ticker: {ticker}\n\n"
            "Hard Quantitative Metrics: {quant}\n\n"
            "Grounded Market Intelligence (RAG): {rag_context}\n\n"
            "Previous Boardroom Context: {messages}"
        )
    ])
    
    chain = prompt | llm
    
    response = chain.invoke({
        "ticker": ticker, 
        "quant": quant_data, 
        "rag_context": retrieved_context,
        "messages": state.get("messages", [])
    })
    
    # Safely create a new message object instead of mutating the old one to prevent LangChain serialization warnings
    safe_message = AIMessage(content=f"Fundamental Analyst: {response.content}")
    
    print("[✅ Fundamental Analyst] Critique complete.")
    return {
        "messages": [safe_message], 
        "fundamental_analysis": safe_message.content, 
        "debate_round": current_round + 1, 
        "rag_context": retrieved_context
    }