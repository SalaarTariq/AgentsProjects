import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
REFERENCE_DIR = BASE_DIR / "ideas"
OUTPUT_DIR = BASE_DIR / "output"
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

DEFAULT_POST_TYPE = "feed"
MAX_RETRIES = 3
REQUEST_TIMEOUT = 120

PROVIDER_PRIORITY = ["pollinations", "huggingface", "together", "stability"]
