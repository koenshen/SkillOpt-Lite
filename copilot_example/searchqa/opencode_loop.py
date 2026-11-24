"""Run the original SearchQA skillopt-loop prompt with an OpenAI-compatible OpenCode optimizer."""
import argparse
import json
import os
import shlex
import shutil
import tempfile
from pathlib import Path

if __package__:
    from .dev_gate import add_dev_options
else:
    from dev_gate import add_dev_options


ENV_DIR = Path(__file__).resolve().parent


def mild_loop(prompt, args):
    """Give this host its own cwd so all native workspace/ references agree."""
    if args.dev_run_dir:
        run_dir = args.dev_run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        root = ENV_DIR / "workspaces"
        root.mkdir(exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix=f"mild-{args.dev_order}-", dir=root))
    shutil.copytree(ENV_DIR / ".github", run_dir / ".github")
    wrapper = run_dir / "run.sh"
    phase_command = ["bash", str(ENV_DIR / "run.sh"), "--dev-early-stop", "--dev-order", args.dev_order,
                     "--dev-batch-size", str(args.dev_batch_size), "--workspace", str(run_dir / "workspace")]
    wrapper.write_text('#!/usr/bin/env bash\nexec ' + shlex.join(phase_command) + ' "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)
    baseline = "bash run.sh --split val --eval_limit 0 --limit 0"
    prompt = prompt.replace(baseline, baseline + " --dev-round 0")
    prompt = prompt.replace("gate-val(full val)", "gate-val(batched candidate dev)")
    prompt = prompt.replace("gate            | `val`   | full (200)",
                            "gate            | `val`   | up to 200")
    prompt = prompt.replace(
        'Parse\n  `hard=<X> soft=<Y>` from the included output, then **immediately',
        'Read the completed phase output (including a certified partial `Gate: reject`), then **immediately',
    )
    prompt = prompt.replace("val_hard=<cand_acc>  val_soft=<cand_soft>",
                            "val_hard=<score or EM interval>  val_soft=<score or unmeasured>")
    start = prompt.index("### 3. Gate:")
    end = prompt.index("### 5. Produce", start)
    prompt = prompt[:start] + (ENV_DIR / "mild_gate_prompt.md").read_text(encoding="utf-8") + "\n\n" + prompt[end:]
    start = prompt.index("- **Gate uses")
    end = prompt.index("- **Train rollouts", start)
    prompt = prompt[:start] + "- Use the mild dev gate in step 3 for every candidate.\n" + prompt[end:]
    prompts = run_dir / ".github" / "prompts"
    (prompts / "skillopt-loop.prompt.md").write_text(prompt, encoding="utf-8")
    improve = prompts / "skillopt-improve.prompt.md"
    improve.write_text(improve.read_text(encoding="utf-8").replace(
        "--split val --eval_limit 0 --limit 0", '--split val --eval_limit 0 --limit 0 --dev-round "$r"'
    ), encoding="utf-8")
    print(f"Mild dev loop directory: {run_dir}", flush=True)
    return prompt, run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch", type=int, default=20)
    add_dev_options(parser)
    parser.add_argument("--dev-run-dir", type=Path, help="New isolated host directory for a mild-dev run")
    args = parser.parse_args()
    if args.rounds < 1 or not 1 <= args.batch <= 400:
        parser.error("rounds must be positive and batch must be between 1 and 400")
    if args.dev_batch_size < 1 or (args.dev_run_dir and not args.dev_early_stop):
        parser.error("dev batch size must be positive; --dev-run-dir requires --dev-early-stop")

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
    cwd = ENV_DIR
    if args.dev_early_stop:
        prompt, cwd = mild_loop(prompt, args)
    prompt += (
        "\n\nOpenCode host: use read/edit/bash for the corresponding tools in this prompt. "
        "OpenCode returns shell results directly. For long evaluations, start the command once "
        "in the background with its output and exit status captured, then wait "
        "for that same command to finish and continue the loop. "
        "Resolve the improve prompt at .github/prompts/skillopt-improve.prompt.md.\n"
    )
    os.chdir(cwd)
    os.execvpe("opencode", ["opencode", "run", "--pure", "--agent", "build", "--model", selected, prompt], env)


if __name__ == "__main__":
    main()
