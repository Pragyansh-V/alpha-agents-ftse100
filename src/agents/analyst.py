import os
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.state import AgentState

# Core RAG Imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Module-level RAG resources: loaded ONCE per process, not once per node call ---
try:
    _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    _vector_db = FAISS.load_local("./faiss_index", _embeddings, allow_dangerous_deserialization=True)
except Exception as e:
    print(f"[⚠️ RAG Core] Failed to load vector store at startup: {e}")
    _vector_db = None
# ------------------------------------------------------------------------------

def fundamental_analyst_node(state: AgentState):
    ticker = state.get("ticker", "BP.L")
    quant_data = state.get("quant_metrics", {})
    current_round = state.get("debate_round", 1)

    print(f"\n[📖 Fundamental Analyst] Round {current_round}: Critiquing metrics for {ticker}...")

    # --- LOCAL RAG LAYER ---
    print(f"[🔍 RAG Core] Querying FAISS database for {ticker} documents...")
    if _vector_db is not None:
        try:
            # Exact string matching query to prevent ticker collision (e.g., RR.L vs REL.L)
            retrieved_docs = _vector_db.similarity_search(f"Ticker: {ticker} | Sector:", k=1)

            if retrieved_docs:
                retrieved_context = retrieved_docs[0].page_content
                print(f"[✅ RAG Core] Grounded context retrieved successfully.")
            else:
                retrieved_context = "No external matching reports found in local memory store."
                print("[⚠️ RAG Core] No matching documents found for this ticker.")
        except Exception as e:
            print(f"[⚠️ RAG Core] Local vector search failed: {e}")
            retrieved_context = "No external matching reports available due to a system indexing error."
    else:
        retrieved_context = "No external matching reports available due to a system indexing error."
    
    # -----------------------

    llm = ChatGroq(
    model_name=os.environ.get("EXPERIMENT_MODEL", "llama-3.1-8b-instant"),
    temperature=float(os.environ.get("EXPERIMENT_TEMP", "0.0")),
    max_retries=6
    )

    
    # Round 2 = adversarial Devil's Advocate; Round 1 = standard fundamental critique
    if current_round == 2:
        system_message = (
            "You are a DEVIL'S ADVOCATE Fundamental Analyst at a hedge fund. "
            "Your job is to CHALLENGE the previous round's analysis. Identify every weakness, "
            "every risk that was understated, every bullish assumption that could be wrong. "
            "Scrutinise the quantitative momentum signals (RSI, MACD, relative momentum) for "
            "contradictions. Force the Portfolio Manager to defend their thesis against your "
            "strongest objection."
        )
    else:
        system_message = (
            "You are an expert Fundamental Analyst at a hedge fund. Critique the quantitative "
            "metrics provided, factoring in specific macroeconomic conditions and corporate intelligence."
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
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