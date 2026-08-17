"""Entry point for the function calling engine."""

import argparse
import json
import sys
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from .decoder import DecodeError, decode_all
from .validater import FunctionCall, FunctionDefinition

FUNCTION_VALIDATOR = TypeAdapter(list[FunctionDefinition])

DEFAULT_FUNCTIONS_FILE = Path("data/input/functions_definition.json")
DEFAULT_PROMPTS_FILE = Path("data/input/function_calling_tests.json")
DEFAULT_RESULTS_FILE = Path("data/output/function_calling_results.json")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call Me Maybe - Constrained Decoding Engine")
    parser.add_argument("--functions_definition", type=Path, default=DEFAULT_FUNCTIONS_FILE)
    parser.add_argument("--input", type=Path, default=DEFAULT_PROMPTS_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    return parser.parse_args()


def read_functions(file_path: Path) -> list[FunctionDefinition]:
    content = json.loads(file_path.read_text(encoding="utf-8"))
    return FUNCTION_VALIDATOR.validate_python(content)


def read_prompts(file_path: Path) -> list[str]:
    content = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(content, list):
        raise ValueError("Input file must contain a JSON array.")

    extracted_prompts = []
    for item in content:
        if not isinstance(item, dict) or "prompt" not in item:
            raise ValueError("Each item must be an object with a 'prompt' key.")
        extracted_prompts.append(str(item["prompt"]))

    return extracted_prompts


def run_engine() -> int:
    args = parse_arguments()

    try:
        functions_list = read_functions(args.functions_definition)
        prompts_list = read_prompts(args.input)
    except (FileNotFoundError, json.JSONDecodeError, ValidationError, ValueError) as error:
        print(f"[!] Input Error: {error}", file=sys.stderr)
        return 1

    if not functions_list:
        print("[!] Error: No function definitions provided.", file=sys.stderr)
        return 1

    try:
        from llm_sdk import Small_LLM_Model
        print(f"[*] Initializing model: {args.model} ...", file=sys.stderr)
        ai_model = Small_LLM_Model(model_name=args.model)
    except Exception as error:
        print(f"[!] Failed to load model '{args.model}': {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[!] Model download interrupted by user (Ctrl+C). Exiting safely.", file=sys.stderr)
        return 1

    def display_progress(index: int, call: FunctionCall) -> None:
        print(
            f"[Prompt {index + 1}/{len(prompts_list)}] Generated: {call.name}({call.parameters})",
            file=sys.stderr
            )

    try:
        decoded_results = decode_all(
            prompts_list,
            functions_list,
            ai_model,
            on_result=display_progress
            )
    except DecodeError as error:
        print(f"[!] Decoding Engine Error: {error}", file=sys.stderr)
        return 1

    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_data = [result.model_dump() for result in decoded_results]
        args.output.write_text(
            json.dumps(
                output_data,
                indent=2,
                ensure_ascii=False
                ) + "\n", encoding="utf-8"
                )
    except OSError as error:
        print(f"[!] Could not save output file: {error}", file=sys.stderr)
        return 1

    print(
        f"[*] Successfully saved {len(decoded_results)} function calls to "
        f"{args.output}", file=sys.stderr
        )
    return 0


if __name__ == "__main__":
    sys.exit(run_engine())
