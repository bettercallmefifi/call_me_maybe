"""Builds a fast token-id to text mapping cache from the model's vocabulary."""

from __future__ import annotations

import json
from typing import Any


def build_byte_to_unicode_mapping() -> dict[int, str]:
    """Matches the specific GPT-2/Qwen BPE encoding scheme."""
    printable_bytes = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )

    byte_to_unicode_map = {byte: chr(byte) for byte in printable_bytes}

    offset = 0
    for byte in range(256):
        if byte not in byte_to_unicode_map:
            byte_to_unicode_map[byte] = chr(256 + offset)
            offset += 1

    return byte_to_unicode_map


class Vocabulary:
    """A fast cache for resolving token IDs."""

    def __init__(self, vocab_path: str) -> None:
        with open(vocab_path, encoding="utf-8") as file_handle:
            token_to_id: dict[str, int] = json.load(file_handle)

        forward_mapping = build_byte_to_unicode_mapping()
        unicode_to_byte = {unicode_char: byte for byte, unicode_char in forward_mapping.items()}

        self.id_to_text: dict[int, str] = {}

        for token_str, token_id in token_to_id.items():
            try:
                raw_bytes = bytes(unicode_to_byte[char] for char in token_str)
                self.id_to_text[token_id] = raw_bytes.decode("utf-8")
            except (KeyError, UnicodeDecodeError):
                continue

    @classmethod
    def from_model(cls, ai_model: Any) -> Vocabulary:
        return cls(ai_model.get_path_to_vocab_file())

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.id_to_text.get(token_id, "") for token_id in token_ids)
