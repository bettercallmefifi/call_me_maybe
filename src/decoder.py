"""Constrained decoding engine: Forces the LLM to output valid JSON (Regex-Free)."""

from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np

from .validater import FunctionCall, FunctionDefinition
from .tokenizer import Vocabulary

MAX_TOKEN_LENGTH = 256
VALID_STRING_ESCAPES = set('"\\/bfnrt')


class DecodeError(RuntimeError):
    pass


def is_number_valid(
    text_chunk: str,
    allow_decimals: bool,
    termination_char: str
) -> bool:
    """
    Pure Python validation of JSON numbers.
    Replaces regular expressions
    with explicit string state checking.
    """
    if termination_char in text_chunk:
        number_part = text_chunk[:text_chunk.index(termination_char)]
        is_complete = True
    else:
        number_part = text_chunk
        is_complete = False

    if not number_part:
        return False

    # Standalone minus sign is valid ONLY as a prefix being typed
    if number_part == "-":
        return not is_complete

    # 1. Check for illegal characters
    for char in number_part:
        if char not in "0123456789.-":
            return False

    # 2. Minus sign rules: can only be at the absolute beginning
    if "-" in number_part and not number_part.startswith("-"):
        return False

    # 3. Decimal point rules
    if "." in number_part:
        if not allow_decimals:
            return False
        if number_part.count(".") > 1:
            return False
        # If the number is complete, it cannot end with a dot (e.g., "12.")
        if is_complete and number_part.endswith("."):
            return False

    # 4. JSON strict leading zeros rule (e.g., "01" is invalid, "0.1" is valid)
    int_part = number_part.split(".")[0].lstrip("-")
    if len(int_part) > 1 and int_part.startswith("0"):
        return False

    return True


def find_string_closure(text_chunk: str) -> int:
    """Analyzes a string being generated token by token."""
    index = 1
    while index < len(text_chunk):
        char = text_chunk[index]
        if char == '"':
            return index
        if char == "\\":
            if index + 1 >= len(text_chunk):
                return -1
            if text_chunk[index + 1] not in VALID_STRING_ESCAPES:
                return -2
            index += 2
            continue
        if ord(char) < 0x20:
            return -2
        index += 1
    return -1


def format_available_functions(
    functions_list: list[FunctionDefinition]
) -> str:
    formatted_lines = []
    for func in functions_list:
        args_str = ", ".join(
            f"{name}: {dtype}" for name, dtype in func.ordered_parameters()
            )
        formatted_lines.append(
            f"- {func.name}({args_str}): {func.description}"
            )
    return "\n".join(formatted_lines)


def construct_system_prompt(
    user_request: str,
    functions_list: list[FunctionDefinition]
) -> str:
    return (
        "You are a strict data-extraction engine. "
        "Convert the user request into exactly one function call.\n"
        "CRITICAL: You must extract values EXACTLY as they appear in the request. "
        "Do not modify, summarize, or change any characters. "
        "Keep exact file paths and variables. "
        "Reply with only a JSON object of the form "
        '{"name": <function>, "parameters": {<arguments>}}.\n\n'
        f"Available functions:\n{format_available_functions(functions_list)}\n\n"
        f"Request: {user_request}\n"
        "Answer: "
    )


class ConstrainedDecoder:
    def __init__(self, ai_model: Any) -> None:
        self._model = ai_model
        self._vocab_cache = Vocabulary.from_model(ai_model).id_to_text

    def generate(
        self,
        prompt: str,
        functions_list: list[FunctionDefinition]
    ) -> FunctionCall:
        current_context = construct_system_prompt(prompt, functions_list)
        functions_map = {func.name: func for func in functions_list}

        forced_start = current_context + '{"name": "'
        selected_func_name = self.decode_choice(forced_start, list(functions_map.keys()))
        target_function = functions_map[selected_func_name]

        generation_state = current_context + f'{{"name": "{selected_func_name}", "parameters": {{'
        extracted_params: dict[str, Any] = {}

        ordered_params = target_function.ordered_parameters()
        for i, (param_name, param_type) in enumerate(ordered_params):
            is_last_param = (i == len(ordered_params) - 1)

            separator = ", " if i > 0 else ""
            generation_state += f'{separator}"{param_name}": '

            parsed_value, raw_written_text = self.decode_value(
                generation_state, param_type, is_last=is_last_param
            )

            extracted_params[param_name] = parsed_value
            generation_state += raw_written_text

        return FunctionCall(prompt=prompt, name=selected_func_name, parameters=extracted_params)

    def decode_value(
        self,
        context: str,
        param_type: str,
        is_last: bool
    ) -> tuple[Any, str]:
        if param_type == "string":
            return self.decode_string(context)

        if param_type == "boolean":
            boolean_str = self.decode_choice(context, ["true", "false"])
            return (boolean_str == "true"), boolean_str

        termination_char = "}" if is_last else ","
        number_str = self.decode_number(context, param_type == "number", termination_char)

        parsed_number = float(number_str) if param_type == "number" else int(number_str)
        return parsed_number, number_str

    def decode_choice(
        self,
        context: str,
        valid_options: list[str]
    ) -> str:
        generated_text = ""
        while generated_text not in valid_options:
            def is_valid_prefix(new_token: str) -> bool:
                combined_text = generated_text + new_token
                return any(option.startswith(combined_text) for option in valid_options)

            generated_text += self.get_most_likely_valid_token(
                context + generated_text,
                is_valid_prefix
                )
            if len(generated_text) > MAX_TOKEN_LENGTH:
                raise DecodeError(f"Runaway choice: {generated_text!r}")
        return generated_text

    def decode_string(
        self,
        context: str
    ) -> tuple[str, str]:
        generated_piece = '"'
        while find_string_closure(generated_piece) < 0:
            def is_structurally_sound(new_token: str) -> bool:
                return find_string_closure(generated_piece + new_token) != -2

            generated_piece += self.get_most_likely_valid_token(
                context + generated_piece,
                is_structurally_sound
                )
            if len(generated_piece) > MAX_TOKEN_LENGTH:
                raise DecodeError(f"Runaway string: {generated_piece!r}")

        closure_index = find_string_closure(generated_piece)
        clean_text = generated_piece[: closure_index + 1]
        return json.loads(clean_text), clean_text

    def decode_number(
        self,
        context: str,
        allow_decimals: bool,
        termination_char: str
    ) -> str:
        generated_piece = ""
        while termination_char not in generated_piece:
            def is_valid_number_step(new_token: str) -> bool:
                return is_number_valid(
                    generated_piece + new_token,
                    allow_decimals,
                    termination_char
                    )

            generated_piece += self.get_most_likely_valid_token(
                context + generated_piece,
                is_valid_number_step
                )
            if len(generated_piece) > MAX_TOKEN_LENGTH:
                raise DecodeError(f"Runaway number: {generated_piece!r}")

        return generated_piece[: generated_piece.index(termination_char)]

    def get_most_likely_valid_token(
        self,
        current_context: str,
        validation_func: Callable[[str], bool]
    ) -> str:
        input_ids = self._model.encode(current_context).tolist()[0]
        logits = np.asarray(self._model.get_logits_from_input_ids(input_ids))

        sorted_token_ids = np.argsort(logits)[::-1]

        for token_id in sorted_token_ids:
            token_text = self._vocab_cache.get(int(token_id))
            if token_text and validation_func(token_text):
                return token_text

        raise DecodeError("Model failed to find any structurally valid token.")


def decode_all(
    prompts_list: list[str],
    functions_list: list[FunctionDefinition],
    ai_model: Any,
    on_result: Callable[[int, FunctionCall], None] | None = None
) -> list[FunctionCall]:
    decoder_engine = ConstrainedDecoder(ai_model)
    collected_results: list[FunctionCall] = []

    for idx, prompt in enumerate(prompts_list):
        try:
            generated_call = decoder_engine.generate(prompt, functions_list)
        except DecodeError as error:
            print(f"[!] Warning: Prompt {idx + 1} could not be decoded: {error}")
            generated_call = FunctionCall(prompt=prompt, name=functions_list[0].name, parameters={})

        collected_results.append(generated_call)
        if on_result:
            on_result(idx, generated_call)

    return collected_results
