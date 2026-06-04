import streamlit as st
import json
import os

# Page configuration
st.set_page_config(page_title="AlphaAgents Boardroom Dashboard", page_icon="🏦", layout="wide")

st.title("🏦 AlphaAgents: Multi-Agent Boardroom Dashboard")
st.markdown("Select a ticker from the sidebar to inspect the automated quantitative analysis, local RAG memories, and final governance audit logs.")
st.divider()

MASTER_JSON = "master_debate_results.json"

# Check if the file exists
if not os.path.exists(MASTER_JSON):
    st.error(f"⚠️ `{MASTER_JSON}` not found. Please run `python run_swarm.py` to generate the multi-asset dataset first.")
    st.stop()

# Load the master list of dictionaries
with open(MASTER_JSON, "r") as f:
    master_data = json.load(f)

if not master_data:
    st.warning("The database file is empty. Run your processing script.")
    st.stop()

# --- SIDEBAR COMPONENT ---
st.sidebar.header("📂 Asset Portfolio")
# Map tickers to their index positions in the array
ticker_list = [asset.get("ticker", "UNKNOWN") for asset in master_data]
selected_ticker = st.sidebar.selectbox("Select Active Ticker:", ticker_list)

# Find the specific dictionary item matching the selected ticker
active_data = next((item for item in master_data if item["ticker"] == selected_ticker), master_data[0])

# --- DYNAMIC INCLUSIVE UX LAYOUT ---
col_debate, col_data = st.columns([2, 1])

with col_debate:
    st.subheader(f"🗣️ Debate Transcript: {active_data.get('ticker')}")
    
    for msg in active_data.get('messages', []):
        role = "assistant" if msg['speaker'] != "System" else "user"
        with st.chat_message(role):
            st.markdown(f"**{msg['speaker']}**")
            st.markdown(msg['content'])

with col_data:
    st.subheader("🧠 Swarm Intelligence Panel")
    
    # Expose the local RAG context
    with st.expander("📂 Grounded Memory (RAG)", expanded=True):
        st.markdown("*External context retrieved from local vector database during runtime:*")
        st.info(active_data.get("rag_context", "No external data pulled for this asset."))
        
    # Expose the hard numbers
    with st.expander("📈 Quantitative Metrics", expanded=False):
        st.markdown("*Mathematical metrics from the Python Quant engine:*")
        st.json(active_data.get('quant_metrics', {}))
        
    st.divider()
    
    st.subheader("⚖️ Final Execution Summary")
    
    # Display PM Decision
    with st.container():
        st.success(active_data.get('portfolio_decision', 'No decision written.'))
    
    # Display Auditor safety pass/fail
    with st.container():
        st.warning(active_data.get('audit_notes', 'No compliance audit details available.'))