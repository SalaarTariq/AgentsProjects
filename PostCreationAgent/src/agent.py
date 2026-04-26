from pathlib import Path
from rich.console import Console
from src.state import AgentState
from src.graph import compile_graph, compile_design_graph
from src.style_analyzer import StyleAnalyzer
from src import config

console = Console()


class PostCreationAgent:
    def __init__(self):
        self.app = compile_graph()
        self.design_app = compile_design_graph()

    def create_post(
        self,
        prompt: str,
        count: int = 1,
        post_type: str = "feed",
        text_overlay: str | None = None,
        enhance: bool = True,
        publish: bool = False,
    ) -> list[Path]:
        console.print(f"\n[bold magenta]{'=' * 50}[/]")
        console.print(f"[bold magenta]  Post Creation Agent (LangGraph)[/]")
        console.print(f"[bold magenta]{'=' * 50}[/]")
        console.print(f"[dim]Prompt: {prompt}[/]")
        console.print(f"[dim]Images: {count} | Type: {post_type} | Publish: {publish}[/]\n")

        initial_state: AgentState = {
            "user_prompt": prompt,
            "post_type": post_type,
            "image_count": count,
            "enhance": enhance,
            "publish_to_instagram": publish,
        }

        if text_overlay:
            initial_state["text_overlay"] = text_overlay

        result = self.app.invoke(initial_state)
        return result.get("saved_paths", [])

    def design_post(
        self,
        lines: list[str],
        tagline: str | None = None,
        theme: str = "light",
        highlight_line: int = -2,
    ) -> list[Path]:
        """Generate a minimal branded post from user-provided text lines."""
        console.print(f"\n[bold magenta]{'=' * 50}[/]")
        console.print(f"[bold magenta]  Minimal Post Designer[/]")
        console.print(f"[bold magenta]{'=' * 50}[/]")
        console.print(f"[dim]Lines: {lines}[/]")
        console.print(f"[dim]Theme: {theme} | Tagline: {tagline}[/]\n")

        initial_state: AgentState = {
            "user_prompt": " ".join(lines),
            "design_lines": lines,
            "design_tagline": tagline,
            "design_theme": theme,
            "design_highlight_line": highlight_line,
        }

        result = self.design_app.invoke(initial_state)
        return result.get("saved_paths", [])

    def preview_post(
        self,
        prompt: str,
        count: int = 1,
        post_type: str = "feed",
    ) -> list[Path]:
        return self.create_post(prompt, count, post_type, publish=False)

    def analyze_brand(self):
        console.print("\n[bold cyan]Analyzing Brand Style...[/]\n")
        analyzer = StyleAnalyzer(config.REFERENCE_DIR)
        return analyzer.analyze_images()

    def get_status(self) -> dict:
        from src.image_generator import ImageGenerator
        generator = ImageGenerator()
        has_instagram = bool(config.INSTAGRAM_ACCESS_TOKEN and config.INSTAGRAM_ACCOUNT_ID)
        has_llm = bool(config.GOOGLE_API_KEY)
        providers = [p["name"] for p in generator.providers]

        return {
            "llm_configured": has_llm,
            "llm_model": config.LLM_MODEL,
            "instagram_configured": has_instagram,
            "available_providers": providers,
            "output_dir": str(config.OUTPUT_DIR),
            "reference_images": len(list(config.REFERENCE_DIR.glob("*"))) if config.REFERENCE_DIR.exists() else 0,
        }
