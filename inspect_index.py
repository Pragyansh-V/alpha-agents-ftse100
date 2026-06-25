import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def audit_faiss():
    DB_DIR = "./faiss_index"
    
    if not os.path.exists(DB_DIR):
        print(f"❌ Error: The directory '{DB_DIR}' does not exist. Run build_vector_db.py first.")
        return

    print("📂 Loading local FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
    
    # We will test a mix of original tickers and our newly added tickers
    test_tickers = ['RR.L', 'TSCO.L', 'BP.L', 'AZN.L']
    
    print("\n🕵️ Running Semantic Integrity Search...")
    for ticker in test_tickers:
        query = f"Macroeconomic data and risks for {ticker}"
        docs = vector_db.similarity_search(query, k=1)
        
        print("=" * 60)
        print(f"📊 TARGET TICKER: {ticker}")
        if docs:
            print(f"📝 Document Source/Metadata: {docs.get('metadata', 'No metadata available') if hasattr(docs, 'get') else 'Available'}")
            print(f"📖 Retrieved Content Snippet:\n{docs[0].page_content[:400]}")
        else:
            print("❌ No matching content found in index!")
        print("=" * 60 + "\n")

if __name__ == "__main__":
    audit_faiss()