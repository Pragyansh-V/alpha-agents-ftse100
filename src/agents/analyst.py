import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.state import AgentState

# Core RAG Imports (Fixed typo)
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
        
        # Pull the top relevant chunk for this ticker
        retrieved_docs = vector_db.similarity_search(f"Macroeconomic data and risks for {ticker}", k=1)
        
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

    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2)
    
    # Updated to inject Grounded Intelligence seamlessly alongside pure numbers
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
    
    # Fed the new variable directly into the execution chain
    response = chain.invoke({
        "ticker": ticker, 
        "quant": quant_data, 
        "rag_context": retrieved_context,
        "messages": state.get("messages", [])
    })
    
    response.content = f"Fundamental Analyst: {response.content}"
    
    print("[✅ Fundamental Analyst] Critique complete.")
    return {"messages": [response], "fundamental_analysis": response.content, "debate_round": current_round + 1, "rag_context": retrieved_context}