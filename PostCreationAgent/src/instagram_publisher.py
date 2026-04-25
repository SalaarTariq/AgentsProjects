import time
import requests
from pathlib import Path
from rich.console import Console
from src import config

console = Console()

GRAPH_API_BASE = "https://graph.facebook.com/v21.0"


class InstagramPublisher:
    def __init__(self):
        self.access_token = config.INSTAGRAM_ACCESS_TOKEN
        self.account_id = config.INSTAGRAM_ACCOUNT_ID

    def verify_connection(self) -> bool:
        if not self.access_token or not self.account_id:
            console.print("[red]Instagram credentials not configured.[/]")
            console.print("[dim]Set INSTAGRAM_ACCESS_TOKEN and INSTAGRAM_ACCOUNT_ID in .env[/]")
            return False

        try:
            url = f"{GRAPH_API_BASE}/{self.account_id}"
            params = {"fields": "id,username", "access_token": self.access_token}
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            console.print(f"[green]Connected to Instagram: @{data.get('username', 'unknown')}[/]")
            return True
        except Exception as e:
            console.print(f"[red]Instagram connection failed: {e}[/]")
            return False

    def publish_single(self, image_url: str, caption: str = "") -> dict:
        console.print("[bold]Publishing single image to Instagram...[/]")

        container_id = self._create_container(image_url, caption)
        if not container_id:
            return {"success": False, "error": "Failed to create media container"}

        self._wait_for_container(container_id)
        return self._publish_container(container_id)

    def publish_carousel(self, image_urls: list[str], caption: str = "") -> dict:
        if not image_urls:
            return {"success": False, "error": "No images provided"}

        if len(image_urls) < 2:
            console.print("[yellow]Carousel requires at least 2 images. Publishing as single.[/]")
            return self.publish_single(image_urls[0], caption)

        if len(image_urls) > 10:
            console.print("[yellow]Instagram allows max 10 images. Using first 10.[/]")
            image_urls = image_urls[:10]

        console.print(f"[bold]Publishing carousel ({len(image_urls)} images) to Instagram...[/]")

        children_ids = []
        for i, url in enumerate(image_urls):
            console.print(f"  Uploading image {i + 1}/{len(image_urls)}...")
            child_id = self._create_container(url, is_carousel_item=True)
            if child_id:
                children_ids.append(child_id)

        if len(children_ids) < 2:
            return {"success": False, "error": "Need at least 2 successful uploads for carousel"}

        url = f"{GRAPH_API_BASE}/{self.account_id}/media"
        payload = {
            "media_type": "CAROUSEL",
            "children": ",".join(children_ids),
            "caption": caption,
            "access_token": self.access_token,
        }
        resp = requests.post(url, data=payload, timeout=30)
        resp.raise_for_status()
        carousel_id = resp.json().get("id")

        if not carousel_id:
            return {"success": False, "error": "Failed to create carousel container"}

        self._wait_for_container(carousel_id)
        return self._publish_container(carousel_id)

    def _create_container(self, image_url: str, caption: str = "", is_carousel_item: bool = False) -> str | None:
        url = f"{GRAPH_API_BASE}/{self.account_id}/media"
        payload = {
            "image_url": image_url,
            "access_token": self.access_token,
        }

        if is_carousel_item:
            payload["is_carousel_item"] = "true"
        elif caption:
            payload["caption"] = caption

        try:
            resp = requests.post(url, data=payload, timeout=30)
            resp.raise_for_status()
            return resp.json().get("id")
        except Exception as e:
            console.print(f"  [red]Container creation failed: {e}[/]")
            return None

    def _wait_for_container(self, container_id: str, max_wait: int = 60):
        url = f"{GRAPH_API_BASE}/{container_id}"
        params = {"fields": "status_code", "access_token": self.access_token}

        for _ in range(max_wait // 5):
            try:
                resp = requests.get(url, params=params, timeout=15)
                status = resp.json().get("status_code")
                if status == "FINISHED":
                    return
                if status == "ERROR":
                    console.print(f"  [red]Container processing failed[/]")
                    return
            except Exception:
                pass
            time.sleep(5)

    def _publish_container(self, container_id: str) -> dict:
        url = f"{GRAPH_API_BASE}/{self.account_id}/media_publish"
        payload = {
            "creation_id": container_id,
            "access_token": self.access_token,
        }

        try:
            resp = requests.post(url, data=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            post_id = data.get("id", "")
            console.print(f"[green]Published successfully! Post ID: {post_id}[/]")
            return {"success": True, "post_id": post_id}
        except Exception as e:
            console.print(f"[red]Publishing failed: {e}[/]")
            return {"success": False, "error": str(e)}


class ImageHostHelper:
    """Hosts images for Instagram API (requires publicly accessible URLs)."""

    @staticmethod
    def get_public_url_instructions() -> str:
        return (
            "Instagram Graph API requires publicly accessible image URLs.\n"
            "Options to host your images:\n"
            "  1. Upload to your own server/CDN\n"
            "  2. Use a cloud storage service (S3, GCS, Azure Blob) with public access\n"
            "  3. Use a service like imgbb.com (free API)\n"
            "  4. Self-host with ngrok for testing\n"
            "\nSet up image hosting to enable automatic Instagram publishing."
        )
