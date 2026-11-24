"""Run the original Lite skillopt-loop prompt with an OpenAI-compatible OpenCode optimizer."""
import argparse
import json
import os
import shlex
import shutil
import tempfile
from pathlib import Path

if __package__:
    from .fixed_benchmarks import SPLITS
    from .dev_gate import add_dev_options
else:
    from fixed_benchmarks import SPLITS
    from dev_gate import add_dev_options


ENV_DIR = Path(__file__).resolve().parent


def render_task(text, benchmark):
    train_size = SPLITS[benchmark]["train"][0]
    test_size = SPLITS[benchmark]["test"][0]
    text = text.replace("copilot_example/searchqa/", "./").replace("searchqa", benchmark)
    text = text.replace("400 training IDs", f"{train_size} training IDs").replace("1400", str(test_size))
    if benchmark == "bird":
        text = text.replace("same answer-tag pattern", "same SQL failure")
        start = "   - If hard=0 but the answer looks plausible,"
        if start in text:
            end = text.index("\n\n", text.index(start))
            text = text[:text.index(start)] + (
                "   - If EX=0, compare the predicted and reference SQL and execution notes. "
                "Distinguish SQL errors, timeouts, and wrong result rows.") + text[end:]
        text = text.replace("the `<answer>` tag instruction", "the SQL output format")
        text = text.replace("answer-format / span-length", "query structure / SQL literals")
        start = "; increment `noop_streak`. **EXCEPTION**:"
        if start in text:
            end = text.index("\n\n", text.index(start))
            text = text[:text.index(start)] + "; increment `noop_streak`. Hard and soft both equal EX." + text[end:]
        start = "- F1 vs EM gap:"
        if start in text:
            end = text.index("\n- Train", text.index(start))
            text = text[:text.index(start)] + "- Report recurrent SQL execution errors and result mismatches." + text[end:]
        text = text.replace("hard=EM, soft=F1", "hard=EX, soft=EX").replace("(EM)", "(EX)").replace("(F1)", "(EX)")
        text = text.replace("so EM is fairly stable", "so EX is fairly stable")
        text = text.replace("one-point EM dead band, two-point F1 exception,",
                            "one-point EX dead band (hard and soft are both EX),")
        text = text.replace("EM interval", "EX interval").replace("full candidate EM, F1", "full candidate EX")
        text = text.replace("and EM bounds, mark F1 unmeasured", "and EX bounds, mark the exact score unmeasured")
    return text


def isolated_loop(prompt, args):
    """Give the host its own cwd, task instructions and phase wrapper."""
    if args.dev_run_dir:
        run_dir = args.dev_run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        root = ENV_DIR / "workspaces"
        root.mkdir(exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix=f"{args.benchmark}-{args.dev_order if args.dev_early_stop else 'full'}-", dir=root))
    shutil.copytree(ENV_DIR / ".github", run_dir / ".github")
    wrapper = run_dir / "run.sh"
    phase_command = ["bash", str(ENV_DIR / "run.sh"), "--benchmark", args.benchmark,
                     "--workspace", str(run_dir / "workspace")]
    if args.dev_early_stop:
        phase_command += ["--dev-early-stop", "--dev-order", args.dev_order,
                          "--dev-batch-size", str(args.dev_batch_size)]
    wrapper.write_text('#!/usr/bin/env bash\nexec ' + shlex.join(phase_command) + ' "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)
    prompts = run_dir / ".github" / "prompts"
    improve = prompts / "skillopt-improve.prompt.md"
    improve_text = improve.read_text(encoding="utf-8")
    if args.dev_early_stop:
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
        improve_text = improve_text.replace(
            "--split val --eval_limit 0 --limit 0", '--split val --eval_limit 0 --limit 0 --dev-round "$r"')
    if args.benchmark != "searchqa":
        prompt = render_task(prompt, args.benchmark)
        improve_text = render_task(improve_text, args.benchmark)
        context = run_dir / ".github/copilot-instructions.md"
        if args.benchmark == "bird":
            shutil.copyfile(ENV_DIR / "bird-instructions.md", context)
        else:
            original = render_task(context.read_text(encoding="utf-8"), args.benchmark)
            task_details = original[original.index("## Task summary"):]
            context.write_text(f"# Lite {args.benchmark} run\n\n"
                               "Edit workspace/skill.md using the passed and failed training samples in "
                               "workspace/.skillopt/samples/. Keep snapshots in workspace/.skillopt/history/. "
                               "Use the supplied run.sh for all phases.\n\n" + task_details, encoding="utf-8")
    (prompts / "skillopt-loop.prompt.md").write_text(prompt, encoding="utf-8")
    improve.write_text(improve_text, encoding="utf-8")
    print(f"Lite {args.benchmark} loop directory: {run_dir}", flush=True)
    return prompt, run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=tuple(SPLITS), default="searchqa")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch", type=int, default=20)
    add_dev_options(parser)
    parser.add_argument("--run-dir", "--dev-run-dir", dest="dev_run_dir", type=Path,
                        help="New isolated host directory for a benchmark run")
    args = parser.parse_args()
    if args.rounds < 1 or not 1 <= args.batch <= SPLITS[args.benchmark]["train"][0]:
        parser.error("rounds must be positive and batch must fit the training split")
    if args.dev_batch_size < 1:
        parser.error("dev batch size must be positive")

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
    if args.dev_early_stop or args.benchmark != "searchqa" or args.dev_run_dir:
        prompt, cwd = isolated_loop(prompt, args)
        env["SKILLOPT_LITE_ISOLATED_LOOP"] = "1"
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
