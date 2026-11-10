"""Secret scrubbing utility ensuring API keys are never exposed in logs or errors."""

import re

# Matches common API keys including Groq, OpenAI, Anthropic, Bearer tokens
SECRET_PATTERNS = [
    re.compile(r"gsk_[a-zA-Z0-9_-]{20,}", re.IGNORECASE),
    re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]{20,}", re.IGNORECASE),
    re.compile(r"api[-_]?key['\"]?\s*[:=]\s*['\"]?[a-zA-Z0-9_\-]{16,}['\"]?", re.IGNORECASE),
]


def sanitize_secrets(text: str) -> str:
    """Scrub sensitive API tokens and credentials from text strings."""
    if not text:
        return ""

    sanitized = text
    for pattern in SECRET_PATTERNS:
        sanitized = pattern.sub("[REDACTED_API_KEY]", sanitized)
    return sanitized
