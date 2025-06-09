from typing import List, Dict

# Simple sentiment lexicon
LEXICON = {
    "good": 0.5,
    "bad": -0.5,
    "great": 0.8,
    "terrible": -0.8,
    "apple": 0.3,
    "love": 0.9,
    "hate": -0.9,
    "happy": 0.7,
    "sad": -0.7,
    "cool": 0.6,
    "amazing": 0.9,
    "best": 1.0,
    "friends": 0.7,
    "like": 0.4,
    "dont": -0.3,
    "not": -0.5,
}

def decode(text: str) -> List[Dict]:
    """
    Decodes a text string into a list of percept packets.
    Splits text on whitespace and maps tokens to sentiment values using a static lexicon.
    """
    packets = []
    for token in text.lower().split():
        packets.append({
            "type": "token",
            "value": token,
            "valence": LEXICON.get(token, 0.0)
        })
    return packets

def encode(action: str) -> str:
    """
    Formats the chosen action as a plain text string for output.
    """
    return f"Action: {action}" 