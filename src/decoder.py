"""Constrained decoding engine: Forces the LLM to output valid JSON matching a schema."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

import numpy as np

from .schemas import FunctionCall, FunctionDefinition
from .tokenizer import Vocabulary

# Safety limit to prevent infinite loops during generation
MAX_TOKEN_LENGTH = 256

# Regular expressions for strict number validation
# FULL: The number is complete. PREFIX: The number is currently being typed.
PATTERN_INT_COMPLETE = re.compile(r"-?(0|[1-9][0-9]*)")
PATTERN_INT_PARTIAL = re.compile(r"-?(0|[1-9][0-9]*)?")
PATTERN_FLOAT_COMPLETE = re.compile(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?")
PATTERN_FLOAT_PARTIAL = re.compile(r"-?((0|[1-9][0-9]*)(\.[0-9]*)?)?")

VALID_STRING_ESCAPES = set('"\\/bfnrt')


class DecodeError(RuntimeError):
    """Raised when the constrained decoding process fails or gets stuck."""
    pass


def is_number_valid(text_chunk: str, allow_decimals: bool, termination_char: str) -> bool:
    """Checks if the generated text is a structurally valid number or prefix."""
    complete_pattern = PATTERN_FLOAT_COMPLETE if allow_decimals else PATTERN_INT_COMPLETE
    partial_pattern = PATTERN_FLOAT_PARTIAL if allow_decimals else PATTERN_INT_PARTIAL
    
    if termination_char in text_chunk:
        # If we hit the end character (e.g., ',' or '}'), the number before it must be complete
        number_part = text_chunk[:text_chunk.index(termination_char)]
        return bool(complete_pattern.fullmatch(number_part))
    
    # Otherwise, it just needs to be a valid partial number
    return bool(partial_pattern.fullmatch(text_chunk))


def find_string_closure(text_chunk: str) -> int:
    """
    Analyzes a string being generated.
    Returns:
        Index of the closing quote if successfully closed.
        -1 if still open but valid.
        -2 if structurally invalid (e.g., bad escape sequence).
    """
    index = 1  # Skip the opening quote
    while index < len(text_chunk):
        char = text_chunk[index]
        if char == '"':
            return index
        if char == "\\":
            if index + 1 >= len(text_chunk):
                return -1  # Wait for the next token to complete the escape
            if text_chunk[index + 1] not in VALID_STRING_ESCAPES:
                return -2  # Invalid escape sequence
            index += 2
            continue
        if ord(char) < 0x20:
            return -2  # Raw control characters are invalid in JSON strings
        index += 1
    return -1


def format_available_functions(functions_list: list[FunctionDefinition]) -> str:
    """Formats the function schemas into a readable text block for the prompt."""
    formatted_lines = []
    for func in functions_list:
        args_str = ", ".join(f"{name}: {dtype}" for name, dtype in func.ordered_parameters())
        formatted_lines.append(f"- {func.name}({args_str}): {func.description}")
    return "\n".join(formatted_lines)


def construct_system_prompt(user_request: str, functions_list: list[FunctionDefinition]) -> str:
    """Builds the minimal prompt to guide the LLM."""
    return (
        "You are a function-calling engine. "
        "Convert the user request into exactly one function call.\n"
        "Reply with only a JSON object of the form "
        '{"name": <function>, "parameters": {<arguments>}}.\n\n'
        f"Available functions:\n{format_available_functions(functions_list)}\n\n"
        f"Request: {user_request}\n"
        "Answer: "
    )


class ConstrainedDecoder:
    """The core engine that manipulates logits to force valid JSON output."""
    
    def __init__(self, ai_model: Any) -> None:
        self._model = ai_model
        # Pre-load the vocabulary cache for O(1) text lookups
        self._vocab_cache = Vocabulary.from_model(ai_model).id_to_text

    def generate(self, prompt: str, functions_list: list[FunctionDefinition]) -> FunctionCall:
        """Generates a complete, guaranteed valid FunctionCall object."""
        current_context = construct_system_prompt(prompt, functions_list)
        functions_map = {func.name: func for func in functions_list}

        # 1. Force the function name selection
        forced_start = current_context + '{"name": "'
        selected_func_name = self._decode_choice(forced_start, list(functions_map.keys()))
        target_function = functions_map[selected_func_name]

        # 2. Force the parameters structure
        generation_state = current_context + f'{{"name": "{selected_func_name}", "parameters": {{'
        extracted_params: dict[str, Any] = {}
        
        ordered_params = target_function.ordered_parameters()
        for i, (param_name, param_type) in enumerate(ordered_params):
            is_last_param = (i == len(ordered_params) - 1)
            
            # Inject parameter key explicitly
            separator = ", " if i > 0 else ""
            generation_state += f'{separator}"{param_name}": '
            
            # Ask LLM to generate the valid value
            parsed_value, raw_written_text = self._decode_value(
                generation_state, param_type, is_last=is_last_param
            )
            
            extracted_params[param_name] = parsed_value
            generation_state += raw_written_text

        return FunctionCall(prompt=prompt, name=selected_func_name, parameters=extracted_params)

    def _decode_value(self, context: str, param_type: str, is_last: bool) -> tuple[Any, str]:
        """Routes the decoding process based on the expected data type."""
        if param_type == "string":
            return self._decode_string(context)
        
        if param_type == "boolean":
            boolean_str = self._decode_choice(context, ["true", "false"])
            return (boolean_str == "true"), boolean_str
            
        termination_char = "}" if is_last else ","
        number_str = self._decode_number(context, param_type == "number", termination_char)
        
        parsed_number = float(number_str) if param_type == "number" else int(number_str)
        return parsed_number, number_str

    def _decode_choice(self, context: str, valid_options: list[str]) -> str:
        """Forces the LLM to output one of the explicitly provided string options."""
        generated_text = ""
        while generated_text not in valid_options:
            def is_valid_prefix(new_token: str) -> bool:
                combined_text = generated_text + new_token
                return any(option.startswith(combined_text) for option in valid_options)

            generated_text += self._get_most_likely_valid_token(context + generated_text, is_valid_prefix)
            
            if len(generated_text) > MAX_TOKEN_LENGTH:
                raise DecodeError(f"Runaway choice generation: {generated_text!r}")
                
        return generated_text

    def _decode_string(self, context: str) -> tuple[str, str]:
        """Forces the LLM to output a structurally valid JSON string."""
        generated_piece = '"'
        while find_string_closure(generated_piece) < 0:
            def is_structurally_sound(new_token: str) -> bool:
                return find_string_closure(generated_piece + new_token) != -2

            generated_piece += self._get_most_likely_valid_token(context + generated_piece, is_structurally_sound)
            
            if len(generated_piece) > MAX_TOKEN_LENGTH:
                raise DecodeError(f"Runaway string generation: {generated_piece!r}")
                
        closure_index = find_string_closure(generated_piece)
        clean_text = generated_piece[: closure_index + 1]  # Discard trailing garbage
        return json.loads(clean_text), clean_text

    def _decode_number(self, context: str, allow_decimals: bool, termination_char: str) -> str:
        """Forces the LLM to output a valid integer or float."""
        generated_piece = ""
        while termination_char not in generated_piece:
            def is_valid_number_step(new_token: str) -> bool:
                return is_number_valid(generated_piece + new_token, allow_decimals, termination_char)

            generated_piece += self._get_most_likely_valid_token(context + generated_piece, is_valid_number_step)
            
            if len(generated_piece) > MAX_TOKEN_LENGTH:
                raise DecodeError(f"Runaway number generation: {generated_piece!r}")
                
        return generated_piece[: generated_piece.index(termination_char)]

    def _get_most_likely_valid_token(self, current_context: str, validation_func: Callable[[str], bool]) -> str:
        """
        The core of constrained decoding:
        Calculates probabilities (logits) for the next token and selects the
        highest-scoring token that satisfies the validation function.
        """
        input_ids = self._model.encode(current_context).tolist()[0]
        logits = np.asarray(self._model.get_logits_from_input_ids(input_ids))
        
        # Sort token IDs by probability (highest to lowest)
        sorted_token_ids = np.argsort(logits)[::-1]
        
        for token_id in sorted_token_ids:
            token_text = self._vocab_cache.get(int(token_id))
            # If the token is valid text AND passes our JSON/Schema constraints
            if token_text and validation_func(token_text):
                return token_text
                
        raise DecodeError("Model failed to find any structurally valid token.")


def decode_all(
    prompts_list: list[str],
    functions_list: list[FunctionDefinition],
    ai_model: Any,
    on_result: Callable[[int, FunctionCall], None] | None = None,
) -> list[FunctionCall]:
    """Processes a batch of prompts through the constrained decoder."""
    decoder_engine = ConstrainedDecoder(ai_model)
    collected_results: list[FunctionCall] = []
    
    for idx, prompt in enumerate(prompts_list):
        try:
            generated_call = decoder_engine.generate(prompt, functions_list)
        except DecodeError as error:
            print(f"[!] Warning: Prompt {idx + 1} could not be decoded: {error}")
            # Fallback to an empty call to maintain list alignment
            generated_call = FunctionCall(prompt=prompt, name=functions_list[0].name, parameters={})
            
        collected_results.append(generated_call)
        
        if on_result:
            on_result(idx, generated_call)
            
    return collected_results