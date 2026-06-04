from typing import Annotated, Sequence, TypedDict
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    The shared memory object passed between all nodes in the AlphaAgents graph.
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    ticker: str
    
    # NEW: The Clock to prevent infinite loops
    debate_round: int 
    
    fundamental_analysis: str
    quant_metrics: dict
    portfolio_decision: str
    audit_notes: str
    rag_context: str