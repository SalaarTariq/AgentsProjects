from src.style_analyzer import StyleProfile


class PromptEngineer:
    def __init__(self, style_profile: StyleProfile | None = None):
        self.style = style_profile

    def enhance_prompt(self, user_prompt: str, post_type: str = "feed") -> str:
        parts = [user_prompt.strip()]

        dimension_hints = {
            "feed": "square composition 1:1 aspect ratio",
            "portrait": "vertical composition 4:5 aspect ratio",
            "story": "vertical full-screen composition 9:16 aspect ratio",
            "landscape": "wide horizontal composition 1.91:1 aspect ratio",
        }
        parts.append(dimension_hints.get(post_type, dimension_hints["feed"]))

        if self.style and self.style.style_prompt_suffix:
            parts.append(self.style.style_prompt_suffix)

        parts.append("instagram post, high resolution, sharp details, no text watermark")

        return ", ".join(parts)

    def generate_caption(self, user_prompt: str) -> str:
        clean = user_prompt.strip().rstrip(".")
        caption = clean[0].upper() + clean[1:] if len(clean) > 1 else clean.upper()
        return caption

    def generate_hashtags(self, user_prompt: str, max_tags: int = 15) -> list[str]:
        words = user_prompt.lower().split()
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "for", "and", "nor", "but", "or", "yet", "so", "at", "by",
            "in", "of", "on", "to", "up", "it", "its", "with", "from",
            "as", "into", "about", "that", "this", "these", "those",
            "i", "we", "you", "he", "she", "they", "me", "him", "her",
            "us", "them", "my", "our", "your", "his", "their", "want",
            "need", "make", "create", "show", "post", "image", "picture",
        }
        keywords = [w.strip(".,!?;:'\"") for w in words if w.strip(".,!?;:'\"") not in stop_words]
        keywords = [k for k in keywords if len(k) > 2]

        seen = set()
        unique = []
        for k in keywords:
            if k not in seen:
                seen.add(k)
                unique.append(k)

        tags = [f"#{k}" for k in unique[:max_tags]]

        default_tags = ["#instagram", "#socialmedia", "#design"]
        for dt in default_tags:
            if len(tags) < max_tags and dt not in tags:
                tags.append(dt)

        return tags

    def build_full_caption(self, user_prompt: str) -> str:
        caption = self.generate_caption(user_prompt)
        hashtags = self.generate_hashtags(user_prompt)
        return f"{caption}\n\n{' '.join(hashtags)}"
