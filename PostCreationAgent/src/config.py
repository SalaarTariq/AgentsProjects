import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
REFERENCE_DIR = BASE_DIR / "ideas"
OUTPUT_DIR = BASE_DIR / "postCreated"
OUTPUT_DIR.mkdir(exist_ok=True)

# LLM Configuration (free tier — Google Gemini)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")

# Image generation API keys
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY", "")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")

# Instagram Graph API
INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN", "")
INSTAGRAM_ACCOUNT_ID = os.getenv("INSTAGRAM_ACCOUNT_ID", "")

INSTAGRAM_SIZES = {
    "feed": (1080, 1080),
    "portrait": (1080, 1350),
    "story": (1080, 1920),
    "landscape": (1080, 608),
}

# Human-readable descriptions used by the LLM prompt builder.
POST_TYPE_DIMENSIONS = {
    "feed": "1080x1080 square (1:1)",
    "portrait": "1080x1350 vertical (4:5)",
    "story": "1080x1920 full-screen vertical (9:16)",
    "landscape": "1080x608 wide (1.91:1)",
}

POST_TYPE_CHOICES = tuple(INSTAGRAM_SIZES.keys())

DEFAULT_POST_TYPE = "feed"
MAX_IMAGES_PER_POST = 10
MAX_RETRIES = 3
REQUEST_TIMEOUT = 120

PROVIDER_PRIORITY = ["pollinations", "huggingface", "together", "stability"]

# Deep style analysis (local CLIP model — no API cost)
DEEP_STYLE_ENABLED = os.getenv("DEEP_STYLE", "true").lower() in ("1", "true", "yes")
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
