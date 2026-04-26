import sys
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.table import Table
from src.agent import PostCreationAgent
from src import config

console = Console()


def print_banner():
    console.print(Panel.fit(
        "[bold magenta]Instagram Post Creation Agent[/]\n"
        "[dim]LangGraph-powered agentic pipeline with multi-provider fallback[/]",
        border_style="magenta",
    ))


def print_status(agent: PostCreationAgent):
    status = agent.get_status()
    table = Table(title="Agent Status", border_style="cyan")
    table.add_column("Setting", style="bold")
    table.add_column("Value")

    table.add_row("LLM Configured", "[green]Yes[/]" if status["llm_configured"] else "[yellow]No (set GOOGLE_API_KEY)[/]")
    table.add_row("LLM Model", status["llm_model"])
    table.add_row("Instagram Connected", "[green]Yes[/]" if status["instagram_configured"] else "[yellow]No[/]")
    table.add_row("Image Providers", ", ".join(status["available_providers"]))
    table.add_row("Reference Images", str(status["reference_images"]))
    table.add_row("Output Directory", status["output_dir"])
    console.print(table)


def cmd_create(agent: PostCreationAgent):
    console.print("\n[bold cyan]Create New Post[/]\n")

    prompt = Prompt.ask("[bold]Describe the post you want to create")
    if not prompt.strip():
        console.print("[red]Prompt cannot be empty.[/]")
        return

    count = IntPrompt.ask("How many images", default=1)
    count = max(1, min(count, 10))

    post_type = Prompt.ask(
        "Post type",
        choices=["feed", "portrait", "story", "landscape"],
        default="feed",
    )

    text_overlay = None
    if Confirm.ask("Add text overlay?", default=False):
        text_overlay = Prompt.ask("  Text to overlay")

    enhance = Confirm.ask("Apply image enhancement?", default=True)
    publish = Confirm.ask("Publish to Instagram after generation?", default=False)

    agent.create_post(
        prompt=prompt,
        count=count,
        post_type=post_type,
        text_overlay=text_overlay,
        enhance=enhance,
        publish=publish,
    )


def cmd_preview(agent: PostCreationAgent):
    console.print("\n[bold cyan]Preview Post (Generate without publishing)[/]\n")

    prompt = Prompt.ask("[bold]Describe the post")
    if not prompt.strip():
        console.print("[red]Prompt cannot be empty.[/]")
        return

    count = IntPrompt.ask("How many images", default=1)
    post_type = Prompt.ask(
        "Post type",
        choices=["feed", "portrait", "story", "landscape"],
        default="feed",
    )

    agent.preview_post(prompt, count, post_type)


def cmd_analyze(agent: PostCreationAgent):
    console.print("\n[bold cyan]Analyzing Brand Style[/]\n")
    profile = agent.analyze_brand()

    table = Table(title="Brand Style Profile", border_style="green")
    table.add_column("Attribute", style="bold")
    table.add_column("Value")

    table.add_row("Dominant Colors", ", ".join(profile.color_palette[:5]))
    table.add_row("Brightness", f"{profile.avg_brightness:.2f}")
    table.add_row("Contrast", profile.contrast_level)
    table.add_row("Style Keywords", ", ".join(profile.style_keywords[:6]))
    table.add_row("Style Prompt", profile.style_prompt_suffix[:80] + "...")
    console.print(table)


def cmd_design(agent: PostCreationAgent):
    console.print("\n[bold cyan]Design Minimal Post[/]\n")
    console.print("[dim]Enter the text lines for your post. Empty line to finish.[/]\n")

    lines = []
    while True:
        line = Prompt.ask(f"  Line {len(lines) + 1} (empty to finish)", default="")
        if not line.strip():
            break
        lines.append(line.strip())

    if not lines:
        console.print("[red]No text lines provided.[/]")
        return

    tagline = Prompt.ask("Tagline (optional, press Enter to skip)", default="")
    tagline = tagline.strip() or None

    theme = Prompt.ask("Theme", choices=["light", "dark"], default="light")

    highlight = -2
    if Confirm.ask("Highlight a specific line in accent color?", default=False):
        highlight = IntPrompt.ask(f"  Which line (1-{len(lines)})", default=len(lines))
        highlight = highlight - 1  # convert to 0-indexed

    paths = agent.design_post(
        lines=lines,
        tagline=tagline,
        theme=theme,
        highlight_line=highlight,
    )

    if paths:
        console.print(f"\n[bold green]Post created: {paths[0]}[/]")


def cmd_batch(agent: PostCreationAgent):
    console.print("\n[bold cyan]Batch Post Creation[/]\n")
    console.print("[dim]Enter prompts one per line. Empty line to finish.[/]\n")

    prompts = []
    while True:
        line = Prompt.ask(f"  Prompt {len(prompts) + 1} (empty to finish)", default="")
        if not line.strip():
            break
        prompts.append(line.strip())

    if not prompts:
        console.print("[yellow]No prompts entered.[/]")
        return

    post_type = Prompt.ask(
        "Post type for all",
        choices=["feed", "portrait", "story", "landscape"],
        default="feed",
    )

    console.print(f"\n[bold]Generating {len(prompts)} posts...[/]\n")

    all_paths = []
    for i, prompt in enumerate(prompts):
        console.print(f"\n[bold]--- Post {i + 1}/{len(prompts)} ---[/]")
        paths = agent.create_post(prompt=prompt, count=1, post_type=post_type)
        all_paths.extend(paths)

    console.print(f"\n[bold green]Batch complete! Generated {len(all_paths)} images total.[/]")


def main():
    print_banner()

    agent = PostCreationAgent()

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "create":
            cmd_create(agent)
        elif command == "design":
            cmd_design(agent)
        elif command == "preview":
            cmd_preview(agent)
        elif command == "analyze":
            cmd_analyze(agent)
        elif command == "batch":
            cmd_batch(agent)
        elif command == "status":
            print_status(agent)
        elif command == "quick":
            prompt = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
            if not prompt:
                console.print("[red]Usage: python -m src.main quick <prompt>[/]")
                return
            agent.create_post(prompt=prompt, count=1, post_type="feed")
        else:
            console.print(f"[red]Unknown command: {command}[/]")
            show_help()
        return

    show_help()
    while True:
        console.print()
        choice = Prompt.ask(
            "[bold magenta]Command[/]",
            choices=["create", "design", "preview", "analyze", "batch", "status", "help", "quit"],
            default="create",
        )

        if choice == "create":
            cmd_create(agent)
        elif choice == "design":
            cmd_design(agent)
        elif choice == "preview":
            cmd_preview(agent)
        elif choice == "analyze":
            cmd_analyze(agent)
        elif choice == "batch":
            cmd_batch(agent)
        elif choice == "status":
            print_status(agent)
        elif choice == "help":
            show_help()
        elif choice == "quit":
            console.print("[dim]Goodbye![/]")
            break


def show_help():
    console.print("\n[bold]Available Commands:[/]")
    console.print("  [cyan]create[/]   - Create a new post (full LangGraph pipeline)")
    console.print("  [cyan]design[/]   - Design a minimal text post (you provide the lines)")
    console.print("  [cyan]preview[/]  - Generate images without publishing")
    console.print("  [cyan]analyze[/]  - Analyze brand style from reference images")
    console.print("  [cyan]batch[/]    - Create multiple posts at once")
    console.print("  [cyan]status[/]   - Show agent configuration status")
    console.print("  [cyan]help[/]     - Show this help")
    console.print("  [cyan]quit[/]     - Exit")
    console.print("\n[dim]Quick mode: python -m src.main quick <prompt>[/]")


if __name__ == "__main__":
    main()
