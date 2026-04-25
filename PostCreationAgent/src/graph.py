from langgraph.graph import StateGraph, START, END
from src.state import AgentState
from src.nodes import (
    analyze_style_node,
    enhance_prompt_node,
    generate_caption_node,
    generate_images_node,
    process_images_node,
    save_images_node,
    publish_node,
    summary_node,
)


def should_generate(state: AgentState) -> str:
    if state.get("error"):
        return "summary"
    return "generate_images"


def should_process(state: AgentState) -> str:
    images = state.get("generated_images", [])
    if not images:
        return "summary"
    return "process_images"


def should_publish(state: AgentState) -> str:
    if state.get("publish_to_instagram"):
        return "publish"
    return "summary"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("analyze_style", analyze_style_node)
    graph.add_node("enhance_prompt", enhance_prompt_node)
    graph.add_node("generate_caption", generate_caption_node)
    graph.add_node("generate_images", generate_images_node)
    graph.add_node("process_images", process_images_node)
    graph.add_node("save_images", save_images_node)
    graph.add_node("publish", publish_node)
    graph.add_node("summary", summary_node)

    # START -> analyze brand style
    graph.add_edge(START, "analyze_style")

    # analyze_style -> enhance_prompt + generate_caption (parallel-ish, sequential here)
    graph.add_edge("analyze_style", "enhance_prompt")
    graph.add_edge("enhance_prompt", "generate_caption")

    # generate_caption -> conditional: generate images or bail
    graph.add_conditional_edges("generate_caption", should_generate)

    # generate_images -> conditional: process or bail
    graph.add_conditional_edges("generate_images", should_process)

    # process_images -> save_images
    graph.add_edge("process_images", "save_images")

    # save_images -> conditional: publish or summary
    graph.add_conditional_edges("save_images", should_publish)

    # publish -> summary
    graph.add_edge("publish", "summary")

    # summary -> END
    graph.add_edge("summary", END)

    return graph


def compile_graph():
    graph = build_graph()
    return graph.compile()
