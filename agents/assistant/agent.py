# Re-export root_agent from the parent package so ADK's loader can find it at:
#   agents/assistant/agent.py → root_agent
# All actual agent code lives in agents/agent.py
from agents.agent import root_agent  # noqa: F401
