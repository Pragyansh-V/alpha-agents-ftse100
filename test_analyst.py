from dotenv import load_dotenv
from src.agents.analyst import fundamental_analyst_node

# 1. Load the hidden API keys from your .env file
load_dotenv()

# 2. Create the mock clipboard
mock_state = {
    "ticker": "BP.L", 
    "messages": [],
    "fundamental_analysis": "",
    "quant_metrics": {},
    "portfolio_decision": "",
    "audit_notes": ""
}

# 3. Run the node
print("Testing Fundamental Analyst Node...")
result = fundamental_analyst_node(mock_state)

print("\n--- Node Output ---")
print(result["fundamental_analysis"])