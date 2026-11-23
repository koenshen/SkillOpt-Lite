"""Run one SkillOpt-Lite SearchQA phase against the fixed SkillOpt manifests."""
import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ENV_DIR = Path(__file__).resolve().parent
LITE_ROOT = ENV_DIR.parents[1]
FIXED_SPLITS = {
    "train": (400, "cded018312f19f27e1da3e019931fb4d18fbbdbc4fd6d1651b5a52f29dedd268"),
    "val": (200, "d2913aff1ba10bd9a577e79c24121108218b30136c0d5a7002eed36133babafe"),
    "test": (1400, "7fca98256e96c837eee2612da9b9d29fb9d4e5cb908c751ca9bfdfd2d10ab6fb"),
}


def load_splits(split_dir):
    data = Path(split_dir or os.getenv("SEARCHQA_SPLIT_DIR") or LITE_ROOT.parent / "SkillOpt/data/searchqa_split")
    splits = {}
    for name, (size, fingerprint) in FIXED_SPLITS.items():
        path = data / name / "items.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        ids = [row["id"] for row in rows]
        digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
        if len(rows) != size or digest != fingerprint:
            raise ValueError(f"{path} must contain the {size} fixed SearchQA IDs in manifest order")
        for row in rows:
            if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("question", "context")):
                raise ValueError(f"Missing question or context for {name}/{row['id']}")
            golds = row.get("answers")
            if not isinstance(golds, list) or not golds or not all(isinstance(g, str) for g in golds):
                raise ValueError(f"Missing reference answers for {name}/{row['id']}")
        splits[name] = rows
    return splits


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--split_dir", type=Path, help="Materialized SkillOpt SearchQA split directory")
    args = parser.parse_args()
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
    workspace = (args.workspace or ENV_DIR / "workspaces" / short_name).resolve()
    for directory in ("history", "samples/passed", "samples/failed", "_eval_run"):
        (workspace / ".skillopt" / directory).mkdir(parents=True, exist_ok=True)
    local_skill = workspace / "skill.md"
    if not local_skill.exists():
        shutil.copyfile(ENV_DIR / "skills/initial.md", local_skill)
    link = ENV_DIR / "workspace"
    if link.resolve() != workspace:
        if link.is_symlink():
            link.unlink()
        link.symlink_to(workspace, target_is_directory=True)
    skill_path = (args.skill or local_skill).resolve(strict=True)
    return workspace, skill_path


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
    ids = [row["id"] for row in rows]
    (out / "item_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    results = run_batch(
        rows, str(out), skill_path.read_text(encoding="utf-8"), workers=args.workers,
        max_turns=config["max_turns"], max_completion_tokens=answer_budget,
        exec_timeout=config["exec_timeout"], task_timeout=config["exec_timeout"],
    )
    by_id = {row["id"]: row for row in results}
    if len(results) != len(ids) or set(by_id) != set(ids):
        raise RuntimeError(f"Incomplete or duplicate {args.split} results in {out}")
    if not any(row.get("agent_ok") for row in results):
        raise RuntimeError(f"All {len(rows)} SearchQA requests failed; see {out / 'results.jsonl'}")
    hard, soft = compute_score(results)
    summary = dict(split=args.split, skill=str(skill_path), n_items=len(rows), hard=hard, soft=soft,
                   target_model=args.target_model, seed=args.seed, workers=args.workers,
                   max_completion_tokens=answer_budget, token_usage=get_token_summary())
    (out / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Results: hard={hard:.4f}  soft={soft:.4f}  (n={len(rows)})", flush=True)
    return [by_id[i] for i in ids]


def main():
    args = parse_args()
    splits = load_splits(args.split_dir)
    if args.check_data:
        print("Fixed SearchQA IDs verified:", ", ".join(f"{s}={len(rows)}" for s, rows in splits.items()))
        return
    rows = splits[args.split]
    if args.split == "train" and 0 < args.eval_limit < len(rows):
        rows = random.Random(args.seed).sample(rows, args.eval_limit)
    workspace, skill_path = prepare_workspace(args)
    out = Path(tempfile.mkdtemp(prefix=f"{args.split}_", dir=workspace / ".skillopt/_eval_run"))
    results = run_evaluation(rows, args, skill_path, out)
    if skill_path != (workspace / "skill.md").resolve():
        shutil.copyfile(skill_path, workspace / "skill.md")
    if args.split == "train":
        # Lite's exporter expects question, context, expected answer and agent output.
        feedback_file = out / "train_feedback.jsonl"
        with feedback_file.open("w", encoding="utf-8") as feedback:
            for item, result in zip(rows, results, strict=True):
                row = dict(result, question=item["question"], context=item["context"],
                           correct_text=" | ".join(item["answers"]), predicted_text=result["response"])
                feedback.write(json.dumps(row, ensure_ascii=False) + "\n")
        subprocess.run([sys.executable, str(ENV_DIR.parent / "make_samples.py"),
                        "--results", str(feedback_file), "--workspace", str(workspace),
                        "--env", "searchqa", "--limit", str(args.limit)], check=True)
    print(f"Workspace: {workspace}\nEvaluation: {out}")


if __name__ == "__main__":
    main()
