"""Run the original SearchQA skillopt-loop prompt with an OpenAI-compatible OpenCode optimizer."""
import argparse
import json
import os
from pathlib import Path


ENV_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch", type=int, default=20)
    args = parser.parse_args()
    if args.rounds < 1 or not 1 <= args.batch <= 400:
        parser.error("rounds must be positive and batch must be between 1 and 400")

    def setting(name):
        return os.getenv(f"OPTIMIZER_OPENAI_COMPATIBLE_{name}") or os.getenv(f"OPENAI_COMPATIBLE_{name}")

    model = setting("MODEL")
    base_url = setting("BASE_URL")
    api_key = setting("API_KEY")
    if not all((model, base_url, api_key)):
        parser.error("set OPENAI_COMPATIBLE_MODEL, BASE_URL and API_KEY (or optimizer-specific equivalents)")

    provider_id = "skillopt-lite"
    selected = f"{provider_id}/{model}"
    env = os.environ.copy()
    env["OPTIMIZER_OPENAI_COMPATIBLE_API_KEY"] = api_key
    cap = setting("MAX_TOKENS")
    if cap:
        if not cap.isascii() or not cap.isdecimal() or int(cap) < 1:
            parser.error("optimizer token cap must be a positive integer")
        env["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] = cap
    config = {
        "enabled_providers": [provider_id],
        "model": selected,
        "small_model": selected,
        "provider": {provider_id: {
            "npm": "@ai-sdk/openai-compatible",
            "env": ["OPTIMIZER_OPENAI_COMPATIBLE_API_KEY"],
            "options": {"baseURL": base_url},
            "models": {model: {"name": model, "tool_call": True,
                               "interleaved": {"field": "reasoning_content"}}},
        }},
    }
    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
    prompt_file = ENV_DIR / ".github/prompts/skillopt-loop.prompt.md"
    prompt = prompt_file.read_text(encoding="utf-8").split("---\n", 2)[2]
    prompt = prompt.replace("${input:rounds:3}", str(args.rounds)).replace("${input:batch:20}", str(args.batch))
    prompt += (
        "\n\nOpenCode host: use read/edit/bash for the corresponding tools in this prompt. "
        "OpenCode returns shell results directly. For long evaluations, start the command once "
        "in the background with its output and exit status captured, then wait "
        "for that same command to finish and continue the loop. "
        "Resolve the improve prompt at .github/prompts/skillopt-improve.prompt.md.\n"
    )
    os.chdir(ENV_DIR)
    os.execvpe("opencode", ["opencode", "run", "--pure", "--agent", "build", "--model", selected, prompt], env)


if __name__ == "__main__":
    main()
