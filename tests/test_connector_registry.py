"""The connectors the app offers, in the order the Connectors page shows them."""

from app.services.connectors import CONNECTORS, get_connector


def test_the_offered_connectors():
    assert [c.id for c in CONNECTORS] == [
        "cloudflare",
        "digitalocean",
        "openrouter",
        "gemini",
        "groq",
        "nvidia",
        "cohere",
        "runpod",
    ]
    assert get_connector("runpod").display_name
