import pytest
from src import ElizaChatbot

@pytest.fixture
def bot():
    return ElizaChatbot()

def test_reflection(bot):
    assert bot.reflect("i am happy") == "you are happy"
    assert bot.reflect("my boss") == "your boss"

def test_pattern_matching(bot):
    response = bot.respond("I feel sad today")
    assert "feel sad today" in response or "sad today" in response

def test_fallback_response(bot):
    response = bot.respond("xyzabc nonsense input 123")
    assert response in bot.FALLBACK_RESPONSES
