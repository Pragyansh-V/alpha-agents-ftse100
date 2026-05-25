import streamlit as st
import json
import os

st.set_page_config(page_title="AlphaAgents Dashboard", layout="wide", page_icon="📈")

st.title("📈 AlphaAgents: Multi-Agent Hedge Fund")
st.markdown("An autonomous swarm architecture. **(Cached State)**")

# Check if the cache file exists
CACHE_FILE = "debate_results.json"

if not os.path.exists(CACHE_FILE):
    st.warning("⚠️ No data cache found. Please run `python graph.py` in your terminal to generate the debate.")
else:
    # Load the cached data
    with open(CACHE_FILE, "r") as f:
        data = json.load(f)

    st.success(f"Dashboard loaded successfully from cache. (Zero API calls made).")
    
    st.divider()
    
    # Top Row: The PM's Final Decision
    st.subheader(f"🏆 Portfolio Manager Verdict: {data['ticker']}")
    st.info(data['portfolio_decision'])
    
    st.divider()
    
    # Middle Row: The Data
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔢 Quantitative Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Daily Volatility", f"{data['quant_metrics']['volatility']:.4f}")
        m2.metric("Mean Return", f"{data['quant_metrics']['mean_daily_return']:.4f}")
        m3.metric("Max Drawdown", f"{data['quant_metrics']['max_drawdown_daily']:.4f}")
    
    with col2:
        st.subheader("⚖️ XAI Auditor Report")
        if "FAIL" in data['audit_notes']:
            st.error(data['audit_notes'])
        else:
            st.success(data['audit_notes'])
            
    st.divider()
    
    # Bottom Row: The Debate Transcript
    st.subheader("🗣️ The Debate Ledger")
    with st.expander("View Full Chronological Transcript"):
        for msg in data['messages']:
            st.markdown(f"**{msg['speaker']}**") 
            st.write(msg['content'])
            st.markdown("---")