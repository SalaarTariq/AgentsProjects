"""Pure routing functions for the LangGraph state machine.

Kept free of langchain imports so tests can exercise the real implementation
without the heavy LLM/Google dependency surface.
"""

from src.state import AgentState


def should_process(state: AgentState) -> str:
    images = state.get("generated_images", [])
    if not images:
        return "summary"
    return "process_images"


def should_publish(state: AgentState) -> str:
    if state.get("publish_to_instagram"):
        return "publish"
    return "summary"


def should_generate(state: AgentState) -> str:
    if state.get("error"):
        return "summary"
    return "generate_images"
