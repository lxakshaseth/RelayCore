"""Pytest fixtures and configuration."""

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import app


@pytest.fixture
def test_settings() -> Settings:
    """Fixture providing isolated settings for testing."""
    return Settings(
        APP_ENV="testing",
        DEBUG=True,
        LOG_LEVEL="DEBUG",
        GROQ_API_KEY="mock_groq_test_key",
    )


@pytest.fixture
def client(test_settings: Settings) -> TestClient:
    """Synchronous test client fixture for standard endpoint verification."""
    app.dependency_overrides[get_settings] = lambda: test_settings
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


try:
    from httpx import ASGITransport, AsyncClient

    @pytest.fixture
    async def async_client(test_settings: Settings):
        """Async test client fixture for concurrent request validation."""
        app.dependency_overrides[get_settings] = lambda: test_settings
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            yield ac
        app.dependency_overrides.clear()
except ImportError:
    pass
