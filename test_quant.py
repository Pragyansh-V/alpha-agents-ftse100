from src.agents.quant import quantitative_specialist_node

# 1. Create a "fake" LangGraph clipboard (State)
mock_state = {
    "ticker": "BP.L",  # We are asking it to analyze BP
    "messages": [],
    "fundamental_analysis": "",
    "quant_metrics": {},
    "portfolio_decision": "",
    "audit_notes": ""
}

# 2. Hand the clipboard to the Quant Agent
print("Testing Quant Node in isolation...")
result = quantitative_specialist_node(mock_state)

# 3. Inspect what the agent returns
print("\n--- Node Output ---")
print(result)