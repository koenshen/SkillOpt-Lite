"""SearchQA dev-only evaluation for the Lite loop. Scores are keyed by fixed item ID."""
import hashlib
import json
from fractions import Fraction


def add_dev_options(parser):
    parser.add_argument("--dev-early-stop", action="store_true",
                        help="Stop candidate dev evaluation only on a certified Lite rollback")
    parser.add_argument("--dev-order", choices=("hard-first", "easy-first"),
                        default="hard-first")
    parser.add_argument("--dev-batch-size", type=int, default=4)


def write_json(path, value):
    # Readers must never see a partly written identity, accounting checkpoint or decision.
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _results(path, *, resume=False):
    if not path.exists():
        return {}
    by_id = {}
    with path.open("r+b" if resume else "rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                # Only the unterminated tail of the currently open round can
                # be an interrupted append. Completed records must still parse.
                if not resume or line.endswith(b"\n"):
                    raise
                stream.truncate(offset)
                print(f"Recovering interrupted dev append at byte {offset} in {path}", flush=True)
                break
            item_id = row["id"]
            if item_id in by_id:
                raise ValueError(f"Duplicate dev result ID {item_id} in {path}")
            by_id[item_id] = row
            if resume and not line.endswith(b"\n"):
                stream.write(b"\n")  # Keep a complete last answer separate from the next append.
    return by_id


def _full_action(correct, soft, state, n):
    if state is None:
        return "baseline"
    diff = 100 * (correct - state["current_correct"])
    if diff >= n:
        return "accept_new_best" if correct > state["best_correct"] else "accept"
    if diff <= -n:
        return "reject"
    # A one-point EM dead band retains the candidate on disk. Only the native
    # two-point F1 exception moves the recorded reference in that band.
    if soft - state["current_soft"] >= 0.02 - 1e-12:
        return "accept"
    return "flat"


def _usage_since_restart(previous, current):
    combined = {stage: dict(counts) for stage, counts in previous.items()}
    for stage, counts in current.items():
        dest = combined.setdefault(stage, {})
        for name, number in counts.items():
            dest[name] = dest.get(name, 0) + number
    return combined


def evaluate_round(rows, out, round_idx, settings, skill, run_items, get_usage):
    """Evaluate one immutable round. run_items appends unseen IDs to results.jsonl."""
    import fcntl

    ids = [r["id"] for r in rows]
    by_id = {r["id"]: r for r in rows}
    n = len(rows)
    if not n or len(by_id) != n:
        raise ValueError("Expected nonempty dev with unique item IDs")
    out.mkdir(parents=True, exist_ok=True)
    with (out.parent / ".dev-gate.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if round_idx > 0 and not (out.parent / f"val_round_{round_idx - 1:04d}" / "eval_summary.json").is_file():
            raise ValueError(f"Complete dev round {round_idx - 1} before starting round {round_idx}")
        fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        config = {**settings, "dev_fingerprint": fingerprint}
        identity = {"round": round_idx, "skill_sha256": hashlib.sha256(skill.encode()).hexdigest(),
                    "settings": config}
        identity_path = out / "gate_identity.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError(f"Dev round {round_idx} already belongs to a different candidate or configuration")
        else:
            if any(out.iterdir()):
                raise ValueError(f"Unidentified dev results in {out}")
            write_json(identity_path, identity)

        summary_path = out / "eval_summary.json"
        result_file = out / "results.jsonl"
        existing = _results(result_file, resume=not summary_path.exists())
        if not set(existing) <= set(by_id):
            raise ValueError(f"Unexpected dev item IDs in {out}")
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            if set(existing) != set(summary["evaluated_ids"]):
                raise ValueError(f"Dev round {round_idx} results differ from its completed summary")
            return summary

        history = {i: [0, 0] for i in ids}
        state = None
        for idx in range(round_idx):
            prev_dir = out.parent / f"val_round_{idx:04d}"
            prev = json.loads((prev_dir / "eval_summary.json").read_text())
            if prev["settings"] != config:
                raise ValueError(f"Incompatible dev configuration in {prev_dir}")
            previous = _results(prev_dir / "results.jsonl")
            if set(previous) != set(prev["evaluated_ids"]):
                raise ValueError(f"Incomplete dev history in {prev_dir}")
            for i, row in previous.items():
                history[i][0] += int(row["hard"])
                history[i][1] += 1
            state = prev["state_after"]

        order = ids if state is None else sorted(
            ids, key=lambda i: Fraction(history[i][0], history[i][1] or 1),
            reverse=config["order"] == "easy-first",
        )
        size = n if state is None else config["batch_size"]
        usage_path = out / "gate_progress.json"
        progress = json.loads(usage_path.read_text()) if usage_path.exists() else {}
        prior_usage = progress.get("token_usage", {})
        usage_complete = progress.get("token_usage_complete", not existing) and not progress.get("in_flight")
        batches = []
        correct = 0
        valid = 0
        for start in range(0, n, size):
            batch = order[start:start + size]
            pending = [by_id[i] for i in batch if i not in existing]
            if pending:
                write_json(usage_path, {"in_flight": True, "token_usage_complete": usage_complete,
                                        "token_usage": _usage_since_restart(prior_usage, get_usage())})
                # Lite's resume-aware runner returns existing results as well as this batch.
                result = run_items(pending)
                existing = {r["id"]: r for r in result}
                if len(result) != len(existing) or not set(existing) <= set(ids) or not set(batch) <= set(existing):
                    raise ValueError(f"Missing, duplicate or unexpected dev answers in {out}")
                write_json(usage_path, {"in_flight": False, "token_usage_complete": usage_complete,
                                        "token_usage": _usage_since_restart(prior_usage, get_usage())})
            observed = [existing[i] for i in batch]
            correct += sum(int(row["hard"]) for row in observed)
            valid += sum(bool(row["agent_ok"]) for row in observed)
            k = start + len(batch)
            upper = correct + n - k
            batches.append({"n_evaluated": k, "correct": correct, "upper_correct": upper})
            # Never turn a total transport outage into a successful partial rejection.
            if state is not None and valid and k < n and 100 * (upper - state["current_correct"]) <= -n:
                break
        if not valid:
            raise RuntimeError(f"All {n} dev calls failed; see {result_file}")
        evaluated_ids = order[:k]
        if set(existing) != set(evaluated_ids):
            raise ValueError(f"Existing dev results disagree with the expected prefix in {out}")
        stopped = k < n
        soft = None if stopped else sum(existing[i]["soft"] for i in ids) / n
        action = "reject" if stopped else _full_action(correct, soft, state, n)
        state = dict(state or {})
        if action in ("baseline", "accept", "accept_new_best"):
            state.update(current_correct=correct, current_soft=soft)
        if action in ("baseline", "accept_new_best"):
            state.update(best_correct=correct, best_soft=soft, best_round=round_idx)
        summary = {**identity, "split": "val", "n_items": k, "n_total": n, "n_evaluated": k,
                   "stopped": stopped, "hard": None if stopped else correct / n, "soft": soft,
                   "correct_observed": correct, "lower_bound": correct / n, "upper_bound": upper / n,
                   "evaluated_ids": evaluated_ids, "batches": batches, "action": action,
                   "state_after": state, "token_usage": _usage_since_restart(prior_usage, get_usage()),
                   "token_usage_complete": bool(usage_complete)}
        write_json(out / "item_ids.json", evaluated_ids)
        write_json(summary_path, summary)
        return summary
