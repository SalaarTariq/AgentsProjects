import io
import time
import urllib.parse
import requests
from PIL import Image
from rich.console import Console
from src import config

console = Console()


class ProviderExhaustedError(Exception):
    pass


class AllProvidersExhaustedError(Exception):
    pass


class ImageGenerator:
    def __init__(self):
        self.providers = self._build_provider_list()
        self._failed_providers: set[str] = set()

    def _build_provider_list(self) -> list[dict]:
        providers = []
        for name in config.PROVIDER_PRIORITY:
            if name == "pollinations":
                providers.append({"name": "pollinations", "fn": self._generate_pollinations})
            elif name == "huggingface" and config.HUGGINGFACE_API_KEY:
                providers.append({"name": "huggingface", "fn": self._generate_huggingface})
            elif name == "together" and config.TOGETHER_API_KEY:
                providers.append({"name": "together", "fn": self._generate_together})
            elif name == "stability" and config.STABILITY_API_KEY:
                providers.append({"name": "stability", "fn": self._generate_stability})
        if not any(p["name"] == "pollinations" for p in providers):
            providers.insert(0, {"name": "pollinations", "fn": self._generate_pollinations})
        return providers

    def generate(self, prompt: str, count: int = 1, size: tuple = (1080, 1080)) -> list[Image.Image]:
        images = []
        for i in range(count):
            console.print(f"\n[bold]Generating image {i + 1}/{count}...[/]")
            img = self._generate_single(prompt, size, seed_offset=i)
            if img:
                images.append(img)
                console.print(f"[green]  Image {i + 1} generated successfully![/]")
            else:
                console.print(f"[red]  Failed to generate image {i + 1}[/]")
        return images

    def _generate_single(self, prompt: str, size: tuple, seed_offset: int = 0) -> Image.Image | None:
        for provider in self.providers:
            if provider["name"] in self._failed_providers:
                continue
            try:
                console.print(f"  [dim]Trying {provider['name']}...[/]")
                img = provider["fn"](prompt, size, seed_offset)
                if img:
                    return img
            except ProviderExhaustedError:
                console.print(f"  [yellow]{provider['name']} rate limit reached, trying next...[/]")
                self._failed_providers.add(provider["name"])
            except Exception as e:
                console.print(f"  [yellow]{provider['name']} error: {e}, trying next...[/]")
        raise AllProvidersExhaustedError("All image generation providers have been exhausted.")

    def _generate_pollinations(self, prompt: str, size: tuple, seed_offset: int = 0) -> Image.Image:
        encoded = urllib.parse.quote(prompt)
        width, height = size
        seed = int(time.time()) + seed_offset
        url = (
            f"https://image.pollinations.ai/prompt/{encoded}"
            f"?width={width}&height={height}&seed={seed}&nologo=true"
        )
        response = requests.get(url, timeout=config.REQUEST_TIMEOUT)
        if response.status_code == 429:
            raise ProviderExhaustedError("Pollinations rate limit")
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "image" not in content_type and len(response.content) < 1000:
            raise ProviderExhaustedError("Pollinations returned non-image response")

        return Image.open(io.BytesIO(response.content)).convert("RGB")

    def _generate_huggingface(self, prompt: str, size: tuple, seed_offset: int = 0) -> Image.Image:
        model = "black-forest-labs/FLUX.1-schnell"
        url = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {config.HUGGINGFACE_API_KEY}"}
        payload = {"inputs": prompt}

        for attempt in range(config.MAX_RETRIES):
            response = requests.post(url, headers=headers, json=payload, timeout=config.REQUEST_TIMEOUT)

            if response.status_code == 429:
                raise ProviderExhaustedError("HuggingFace rate limit")

            if response.status_code == 503:
                wait = response.json().get("estimated_time", 30)
                console.print(f"  [dim]Model loading, waiting {wait:.0f}s...[/]")
                time.sleep(min(wait, 60))
                continue

            if response.status_code == 200:
                return Image.open(io.BytesIO(response.content)).convert("RGB")

            response.raise_for_status()

        raise ProviderExhaustedError("HuggingFace max retries exceeded")

    def _generate_together(self, prompt: str, size: tuple, seed_offset: int = 0) -> Image.Image:
        url = "https://api.together.xyz/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {config.TOGETHER_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "black-forest-labs/FLUX.1-schnell-Free",
            "prompt": prompt,
            "width": size[0],
            "height": size[1],
            "n": 1,
            "seed": int(time.time()) + seed_offset,
        }

        response = requests.post(url, headers=headers, json=payload, timeout=config.REQUEST_TIMEOUT)

        if response.status_code == 429:
            raise ProviderExhaustedError("Together AI rate limit")
        response.raise_for_status()

        data = response.json()
        if "data" in data and len(data["data"]) > 0:
            img_data = data["data"][0]
            if "url" in img_data:
                img_response = requests.get(img_data["url"], timeout=60)
                return Image.open(io.BytesIO(img_response.content)).convert("RGB")
            elif "b64_json" in img_data:
                import base64
                img_bytes = base64.b64decode(img_data["b64_json"])
                return Image.open(io.BytesIO(img_bytes)).convert("RGB")

        raise ProviderExhaustedError("Together AI returned no image data")

    def _generate_stability(self, prompt: str, size: tuple, seed_offset: int = 0) -> Image.Image:
        url = "https://api.stability.ai/v2beta/stable-image/generate/core"
        headers = {
            "Authorization": f"Bearer {config.STABILITY_API_KEY}",
            "Accept": "image/*",
        }
        data = {
            "prompt": prompt,
            "output_format": "png",
            "aspect_ratio": self._closest_aspect_ratio(size),
        }

        response = requests.post(url, headers=headers, files={"none": ""}, data=data, timeout=config.REQUEST_TIMEOUT)

        if response.status_code == 429 or response.status_code == 402:
            raise ProviderExhaustedError("Stability AI rate limit or credits exhausted")
        response.raise_for_status()

        return Image.open(io.BytesIO(response.content)).convert("RGB")

    def _closest_aspect_ratio(self, size: tuple) -> str:
        w, h = size
        ratio = w / h
        ratios = {
            "1:1": 1.0,
            "4:5": 0.8,
            "5:4": 1.25,
            "16:9": 1.78,
            "9:16": 0.5625,
            "3:2": 1.5,
            "2:3": 0.667,
        }
        closest = min(ratios.items(), key=lambda x: abs(x[1] - ratio))
        return closest[0]

    def reset_failed_providers(self):
        self._failed_providers.clear()
