import os
import json
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

# Import the shared State schema
from src.state import AgentState

# Import your FOUR completed agents
from src.agents.quant import quantitative_specialist_node
from src.agents.analyst import fundamental_analyst_node
from src.agents.manager import portfolio_manager_node
from src.agents.auditor import xai_auditor_node

load_dotenv()

# --- The Traffic Cop (Conditional Router) ---
def debate_router(state: AgentState):
    """
    Checks the debate clock. If we hit 3 rounds, send to PM. 
    Otherwise, loop back to the Quant for rebuttal.
    """
    current_round = state.get("debate_round", 1)
    if current_round >= 3:
        print(f"\n[🚦 Router] Debate clock hit {current_round}. Routing to Portfolio Manager...")
        return "portfolio_manager"
    else:
        print(f"\n[🚦 Router] Debate round {current_round} complete. Looping back to Quant...")
        return "quant_specialist"

# --- Graph Definition ---
builder = StateGraph(AgentState)

# Add all nodes
builder.add_node("quant_specialist", quantitative_specialist_node)
builder.add_node("fundamental_analyst", fundamental_analyst_node)
builder.add_node("portfolio_manager", portfolio_manager_node)
builder.add_node("xai_auditor", xai_auditor_node)

# --- Cyclic Workflow ---
# START -> Quant
builder.add_edge(START, "quant_specialist")

# Quant -> Analyst
builder.add_edge("quant_specialist", "fundamental_analyst")

# Analyst -> ROUTER (Does it loop back to Quant, or go to PM?)
builder.add_conditional_edges(
    "fundamental_analyst", 
    debate_router,
    {
        "quant_specialist": "quant_specialist",   # Loop back
        "portfolio_manager": "portfolio_manager"  # Move forward to PM
    }
)

# PM -> Auditor -> END
builder.add_edge("portfolio_manager", "xai_auditor")
builder.add_edge("xai_auditor", END)

alpha_agents_app = builder.compile()

# --- Execution & Caching Block ---
if __name__ == "__main__":
    print("\n🚀 Initializing Round-Robin AlphaAgents Pipeline...")
    
    # Initialize the clock at Round 1
    initial_state = {
        "ticker": "BP.L", 
        "messages": [],
        "debate_round": 1, 
        "fundamental_analysis": "",
        "quant_metrics": {},
        "portfolio_decision": "",
        "audit_notes": ""
    }
    
    # Run the graph
    final_state = alpha_agents_app.invoke(initial_state)
    
    # --- Serialization and Caching Logic ---
    print("\n[Cache] Serializing data for frontend...")
    
    # Extract raw text from LangChain message objects
    serialized_messages = []
    for msg in final_state['messages']:
        # Split the string to separate the speaker from the content if possible
        if ':' in msg.content:
            speaker, content = msg.content.split(':', 1)
            serialized_messages.append({"speaker": speaker.strip(), "content": content.strip()})
        else:
            serialized_messages.append({"speaker": "System", "content": msg.content})

    # Build the final clean dictionary
    output_cache = {
        "ticker": final_state["ticker"],
        "quant_metrics": final_state["quant_metrics"],
        "rag_context": final_state.get("rag_context", "No external context retrieved."), # <-- ADD THIS LINE
        "portfolio_decision": final_state["portfolio_decision"],
        "audit_notes": final_state["audit_notes"],
        "messages": serialized_messages
    }
    # Save to a local JSON file
    with open("debate_results.json", "w") as f:
        json.dump(output_cache, f, indent=4)
        
    print("[Cache] Debate saved to debate_results.json. The UI is ready.")