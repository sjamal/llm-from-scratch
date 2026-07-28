import re
import random
from typing import Dict, List, Tuple

class ElizaChatbot:
    """A pattern-matching chatbot inspired by Joseph Weisenbaum's ELIZA (1966)."""
    
    PRONOUN_REFLECTIONS: Dict[str, str] = {
        "am": "are",
        "was": "were",
        "i": "you",
        "i'd": "you would",
        "i've": "you have",
        "i'll": "you will",
        "my": "your",
        "are": "am",
        "you've": "I have",
        "you'll": "I will",
        "your": "my",
        "yours": "mine",
        "you": "me",
        "me": "you"
    }

    MATCH_RULES: List[Tuple[str, List[str]]] = [
        (
            r"i need (.*)",
            [
                "Why do you need {0}?",
                "Would getting {0} really help you?",
                "Are you sure you need {0}?"
            ]
        ),
        (
            r"i feel (.*)",
            [
                "Why do you feel {0}?",
                "Does feeling {0} happen often?",
                "How does feeling {0} affect your daily life?"
            ]
        ),
        (
            r"my (.*)",
            [
                "Tell me more about your {0}.",
                "How does your {0} make you feel?",
                "Why do you bring up your {0}?"
            ]
        ),
        (
            r"hello|hi|hey",
            [
                "Hello! How can I help you today?",
                "Hi there. What is on your mind?",
                "Greetings! How are you feeling right now?"
            ]
        )
    ]

    FALLBACK_RESPONSES: List[str] = [
        "Please go on.",
        "Tell me more about that.",
        "How does that make you feel?",
        "Can you elaborate on that?"
    ]

    def reflect(self, fragment: str) -> str:
        """Swap pronouns in a string fragment to reverse perspective (I -> you, my -> your)."""
        words = fragment.lower().split()
        reflected_words = [self.PRONOUN_REFLECTIONS.get(w, w) for w in words]
        return " ".join(reflected_words)

    def respond(self, user_input: str) -> str:
        """Process user input against pattern rules or return a fallback response."""
        cleaned = user_input.strip().rstrip(".!?")
        
        for pattern, responses in self.MATCH_RULES:
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match:
                response_template = random.choice(responses)
                if match.groups():
                    captured = self.reflect(match.group(1))
                    return response_template.format(captured)
                return response_template

        return random.choice(self.FALLBACK_RESPONSES)

if __name__ == "__main__":
    bot = ElizaChatbot()
    print("Eliza Initialized. (Type 'quit' to exit)")
    while True:
        user_msg = input("> ")
        if user_msg.lower() == "quit":
            break
        print(f"Eliza: {bot.respond(user_msg)}")
