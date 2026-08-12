"""Builds a fast token-id to text mapping cache from the model's vocabulary.

Qwen utilizes a byte-level Byte-Pair Encoding (BPE) similar to GPT-2. Raw bytes 
are mapped to printable unicode characters in the vocabulary. To avoid using the 
slow, native decode() method during logit masking, we reverse this mapping and 
cache the utf-8 decoded strings.
"""

from __future__ import annotations

import json
from typing import Any


def build_byte_to_unicode_mapping() -> dict[int, str]:
    """
    Creates a dictionary mapping raw byte values (0-255) to printable unicode
    characters. This matches the specific GPT-2/Qwen BPE encoding scheme.
    """
    # 1. Define base printable ASCII characters
    printable_bytes = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    
    byte_to_unicode_map = {byte: chr(byte) for byte in printable_bytes}
    
    # 2. Map unprintable/control bytes to new printable unicode characters
    offset = 0
    for byte in range(256):
        if byte not in byte_to_unicode_map:
            byte_to_unicode_map[byte] = chr(256 + offset)
            offset += 1
            
    return byte_to_unicode_map


class Vocabulary:
    """A fast cache for resolving token IDs back to their actual string representations."""
    
    def __init__(self, vocab_path: str) -> None:
        # Step 1: Load the raw vocabulary (Token String -> Token ID)
        with open(vocab_path, encoding="utf-8") as file_handle:
            token_to_id: dict[str, int] = json.load(file_handle)

        # Step 2: Prepare the reverse mapping (Unicode char -> Original Raw Byte)
        forward_mapping = build_byte_to_unicode_mapping()
        unicode_to_byte = {unicode_char: byte for byte, unicode_char in forward_mapping.items()}

        # Step 3: Build the fast lookup cache (Token ID -> Decoded UTF-8 String)
        self.id_to_text: dict[int, str] = {}
        
        for token_str, token_id in token_to_id.items():
            try:
                # Convert the BPE representation back to original bytes
                raw_bytes = bytes(unicode_to_byte[char] for char in token_str)
                # Decode the bytes into a standard UTF-8 python string
                self.id_to_text[token_id] = raw_bytes.decode("utf-8")
            except (KeyError, UnicodeDecodeError):
                # Skip tokens that do not form valid standalone UTF-8 sequences.
                # We strictly enforce ASCII/valid UTF-8 for JSON generation anyway.
                continue

    @classmethod
    def from_model(cls, ai_model: Any) -> Vocabulary:
        """Instantiates the vocabulary cache directly from the loaded AI model."""
        return cls(ai_model.get_path_to_vocab_file())

    def decode(self, token_ids: list[int]) -> str:
        """Reconstructs text from a sequence of token IDs using the local cache."""
        return "".join(self.id_to_text.get(token_id, "") for token_id in token_ids)
