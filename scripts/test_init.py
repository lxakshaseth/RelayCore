"""Smoke test script for RelayCore configuration and healthcheck."""

import asyncio
import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from app.api.v1.endpoints.health import get_health
from app.core.config import get_settings


async def main():
    settings = get_settings()
    print(f"Loaded config: App={settings.app_name}, Env={settings.app_env}")
    health = await get_health(settings)
    print("Health response:")
    print(health.model_dump_json(indent=2))
    print("Smoke test passed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
