"""Run a Lite train, dev or test phase on a fixed QA or text-to-SQL benchmark."""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

if __package__:
    from .fixed_benchmarks import DEFAULT_DB_ROOT, SPLITS, check_databases, load_splits
    from .dev_gate import add_dev_options, evaluate_round
else:
    from fixed_benchmarks import DEFAULT_DB_ROOT, SPLITS, check_databases, load_splits
    from dev_gate import add_dev_options, evaluate_round

ENV_DIR = Path(__file__).resolve().parent
LITE_ROOT = ENV_DIR.parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=tuple(SPLITS), default="searchqa")
    parser.add_argument("--db-root", type=Path, default=os.getenv("BIRD_DB_ROOT", str(DEFAULT_DB_ROOT)),
                        help="Root of the BIRD SQLite databases")
    parser.add_argument("--check-data", action="store_true", help="Verify all manifest IDs without model calls")
    parser.add_argument("--skill", type=Path, help="Current skill (default: workspace/skill.md)")
    parser.add_argument("--workspace", type=Path, help="Override the model-specific workspace")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--limit", type=int, default=20, help="Training samples to export (0: all)")
    parser.add_argument("--eval_limit", type=int, default=0, help="Training batch size (0: full split)")
    parser.add_argument("--target_model", default=(
        os.getenv("SKILLOPT_LITE_TARGET_MODEL") or os.getenv("TARGET_OPENAI_COMPATIBLE_MODEL")
        or os.getenv("OPENAI_COMPATIBLE_MODEL") or "gpt-5.4"
    ))
    parser.add_argument("--reasoning", default="medium")
    parser.add_argument("--workers", type=int, default=os.getenv("SKILLOPT_WORKERS", "16"))
    parser.add_argument("--max_completion_tokens", type=int,
                        help="Answer budget, limited by the configured target cap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_dir", type=Path, help="Materialized split directory for the selected benchmark")
    add_dev_options(parser)
    parser.add_argument("--dev-round", type=int, help="Mild dev gate: 0 for initial validation, 1+ for candidates")
    args = parser.parse_args()
    if args.dev_batch_size < 1:
        parser.error("dev batch size must be positive")
    if args.dev_round is not None and (not args.dev_early_stop or args.dev_round < 0 or args.split != "val"):
        parser.error("--dev-round requires --dev-early-stop, --split val and a nonnegative round")
    if args.dev_early_stop and args.split == "val" and args.dev_round is None and not args.check_data:
        parser.error("specify --dev-round 0 for the initial dev or the candidate's round number")
    if args.limit < 0 or args.eval_limit < 0 or args.workers < 1:
        parser.error("limits must be nonnegative and workers must be positive")
    cap = os.getenv("TARGET_OPENAI_COMPATIBLE_MAX_TOKENS") or os.getenv("OPENAI_COMPATIBLE_MAX_TOKENS")
    if cap:
        try:
            target_cap = int(cap)
        except ValueError:
            parser.error("target token cap must be an integer")
        if args.max_completion_tokens is None:
            args.max_completion_tokens = target_cap
        else:
            args.max_completion_tokens = min(args.max_completion_tokens, target_cap)
    if args.max_completion_tokens is not None and args.max_completion_tokens < 1:
        parser.error("target token budget must be positive")
    if args.split != "train" and args.eval_limit:
        parser.error("fixed dev and test use the entire split; use --eval_limit 0")
    return args


def model_short_name(model):
    for name in ("gpt-5.5", "gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4",
                 "gpt-5-nano", "gpt-5-mini", "gpt-4o-mini", "gpt-4o", "gpt-5"):
        if model.startswith(name):
            return name
    if model.lower() in {"qwen/qwen3.5-9b", "qwen/qwen3.5-27b", "qwen3.5-9b", "qwen3.5-27b"}:
        return model.rsplit("/", 1)[-1].lower()
    return model.replace("/", "-")


def prepare_workspace(args):
    short_name = model_short_name(args.target_model)
    if short_name in {"", ".", ".."}:
        raise ValueError("expected a non-empty model name")
    default_workspace = ENV_DIR / "workspaces" / short_name
    if args.dev_early_stop:
        default_workspace = ENV_DIR / "workspaces" / f"{short_name}-dev-{args.dev_order}-b{args.dev_batch_size}"
    if args.benchmark != "searchqa":
        default_workspace = default_workspace.parent / args.benchmark / default_workspace.name
    workspace = (args.workspace or default_workspace).resolve()
    initialize_workspace(workspace, args.benchmark)
    link = ENV_DIR / "workspace"
    publish_link = (args.benchmark == "searchqa" and not args.dev_early_stop
                    and os.environ.get("SKILLOPT_LITE_ISOLATED_LOOP") != "1")
    if publish_link and link.resolve() != workspace:
        if link.is_symlink():
            link.unlink()
        link.symlink_to(workspace, target_is_directory=True)
    skill_path = (args.skill or workspace / "skill.md").resolve(strict=True)
    return workspace, skill_path


def initialize_workspace(workspace, benchmark):
    """Materialize the same starting skill for the host and standalone phase runner."""
    workspace.mkdir(parents=True, exist_ok=True)
    marker = workspace / ".benchmark"
    if marker.exists() and marker.read_text().strip() != benchmark:
        raise ValueError(f"Workspace belongs to another benchmark: {workspace}")
    if benchmark != "searchqa" and not marker.exists():
        if any(workspace.iterdir()):
            raise ValueError(f"Use an empty workspace for a new {benchmark} run: {workspace}")
        marker.write_text(benchmark + "\n", encoding="utf-8")
    for directory in ("history", "samples/passed", "samples/failed", "_eval_run"):
        (workspace / ".skillopt" / directory).mkdir(parents=True, exist_ok=True)
    local_skill = workspace / "skill.md"
    if not local_skill.exists():
        seed_dir = ENV_DIR / "skills"
        if benchmark != "searchqa":
            seed_dir = seed_dir / benchmark
        shutil.copyfile(seed_dir / "initial.md", local_skill)


def run_evaluation(rows, args, skill_path, out):
    # Both Lite and other projects provide a package named skillopt.
    sys.path.insert(0, str(LITE_ROOT))
    from skillopt.config import load_config
    from skillopt.envs.searchqa.rollout import run_batch
    from skillopt.model import get_token_summary, set_reasoning_effort, set_target_backend, set_target_deployment
    from skillopt.utils.scoring import compute_score

    config = load_config(str(LITE_ROOT / "configs/searchqa/default.yaml"))["env"]
    answer_budget = args.max_completion_tokens or config["max_completion_tokens"]
    set_target_backend("openai_chat")
    set_target_deployment(args.target_model)
    set_reasoning_effort(None if args.target_model.startswith("gpt-4o") else args.reasoning)
    skill = skill_path.read_text(encoding="utf-8")

    def rollout(batch):
        if args.benchmark == "bird":
            from skillopt.envs.bird.rollout import run_batch as bird_batch
            return bird_batch(batch, str(out), skill, str(args.db_root), workers=args.workers,
                              max_completion_tokens=answer_budget)
        return run_batch(
            batch, str(out), skill, workers=args.workers,
            max_turns=config["max_turns"], max_completion_tokens=answer_budget,
            exec_timeout=config["exec_timeout"], task_timeout=config["exec_timeout"],
        )

    if args.dev_early_stop and args.split == "val":
        settings = dict(order=args.dev_order, batch_size=args.dev_batch_size,
                        target_model=args.target_model, reasoning=args.reasoning,
                        max_completion_tokens=answer_budget, max_turns=config["max_turns"],
                        workers=args.workers)
        summary = evaluate_round(rows, out, args.dev_round, settings, skill, rollout, get_token_summary)
        if summary["stopped"]:
            metric = "EX" if args.benchmark == "bird" else "EM"
            print(f"Gate: reject (certified), evaluated={summary['n_evaluated']}/{summary['n_total']}, "
                  f"{metric} interval=[{summary['lower_bound']:.4f}, {summary['upper_bound']:.4f}], "
                  "full score unmeasured", flush=True)
        else:
            print(f"Results: hard={summary['hard']:.4f}  soft={summary['soft']:.4f}  "
                  f"(n={summary['n_total']})\nGate: {summary['action']}", flush=True)
        return
    ids = [row["id"] for row in rows]
    (out / "item_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    results = rollout(rows)
    by_id = {row["id"]: row for row in results}
    if len(results) != len(ids) or set(by_id) != set(ids):
        raise RuntimeError(f"Incomplete or duplicate {args.split} results in {out}")
    if not any(row.get("agent_ok") for row in results):
        raise RuntimeError(f"All {len(rows)} {args.benchmark} requests failed; see {out / 'results.jsonl'}")
    hard, soft = compute_score(results)
    summary = dict(split=args.split, skill=str(skill_path), n_items=len(rows), hard=hard, soft=soft,
                   target_model=args.target_model, seed=args.seed, workers=args.workers,
                   max_completion_tokens=answer_budget, token_usage=get_token_summary())
    if args.benchmark != "searchqa":
        summary["benchmark"] = args.benchmark
        summary["metric"] = "EX" if args.benchmark == "bird" else "EM/F1"
    (out / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Results: hard={hard:.4f}  soft={soft:.4f}  (n={len(rows)})", flush=True)
    return [by_id[i] for i in ids]


def main():
    args = parse_args()
    splits = load_splits(args.split_dir, args.benchmark)
    if args.benchmark == "bird":
        check_databases(splits, args.db_root)
    if args.check_data:
        print(f"Fixed {args.benchmark} IDs verified:", ", ".join(f"{s}={len(rows)}" for s, rows in splits.items()))
        return
    rows = splits[args.split]
    if args.split == "train" and 0 < args.eval_limit < len(rows):
        rows = random.Random(args.seed).sample(rows, args.eval_limit)
    workspace, skill_path = prepare_workspace(args)
    eval_root = workspace / ".skillopt/_eval_run"
    if args.dev_early_stop and args.split == "val":
        out = eval_root / f"val_round_{args.dev_round:04d}"
    else:
        out = Path(tempfile.mkdtemp(prefix=f"{args.split}_", dir=eval_root))
    results = run_evaluation(rows, args, skill_path, out)
    if skill_path != (workspace / "skill.md").resolve():
        shutil.copyfile(skill_path, workspace / "skill.md")
    if args.split == "train":
        # Lite's exporter expects question, context, expected answer and agent output.
        feedback_file = out / "train_feedback.jsonl"
        with feedback_file.open("w", encoding="utf-8") as feedback:
            for item, result in zip(rows, results, strict=True):
                if args.benchmark == "bird":
                    row = dict(result, question=item["question"], prompt=item["prompt"], db_id=item["db_id"],
                               evidence=item.get("evidence", ""), correct_text=item["SQL"],
                               predicted_text=result.get("predicted_sql", "") or result["response"])
                else:
                    row = dict(result, question=item["question"], context=item["context"],
                               correct_text=" | ".join(item["answers"]), predicted_text=result["response"])
                feedback.write(json.dumps(row, ensure_ascii=False) + "\n")
        subprocess.run([sys.executable, str(ENV_DIR.parent / "make_samples.py"),
                        "--results", str(feedback_file), "--workspace", str(workspace),
                        "--env", args.benchmark, "--limit", str(args.limit)], check=True)
    print(f"Workspace: {workspace}\nEvaluation: {out}")


if __name__ == "__main__":
    main()
