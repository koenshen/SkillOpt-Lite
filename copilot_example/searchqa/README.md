# searchqa/

Reference implementation for **SearchQA**: single-turn open-domain question
answering grounded in retrieved passages. Lifted from `skillopt/envs/searchqa/`
for plugin debugging.

## Task

- **Input:** a natural-language question + a `[DOC]`-delimited context blob of
  retrieved passages (truncated to ~6k chars).
- **Agent:** one LLM call. System prompt = `prompts/rollout_system.md` filled in
  with the current skill; user prompt = question + context.
- **Output:** answer wrapped in `<answer>...</answer>`.
- **Scoring:** see `evaluator.py` — string/normalized match against gold.

## Files

| File | Purpose |
|---|---|
| `rollout.py` | `process_one(item, skill) -> result`; `run_batch(items, skill)` parallel runner |
| `evaluator.py` | `evaluate(predicted, gold) -> {hard, soft, fail_reason}` |
| `adapter.py` | dataset row → rollout input shape |
| `dataloader.py` | load materialized SearchQA split files |
| `reflect.py` | per-sample error analysis (old SkillOpt reflect, kept for reference) |
| `prompts/rollout_system.md` | agent system prompt template (`{skill_section}` placeholder) |
| `prompts/analyst_error.md`, `analyst_success.md` | old reflect prompts |
| `skills/initial.md` | baseline skill — good starting point for plugin demo |

## Fixed SearchQA workflow

On the CPU server the default data path is `../SkillOpt/data/searchqa_split`
relative to the Lite repository root. Set `SEARCHQA_SPLIT_DIR` or pass
`--split_dir` for another copy. Each split contains an `items.json` array
with `id`, `question`, `context` and `answers`. The loader verifies the counts
and SHA-256 fingerprints of the ordered IDs for all 400 train, 200 dev and
1,400 test items. Lite uses its own SearchQA rollout, model client and scorer.

In a coding-agent session opened in `copilot_example/searchqa/`, the native
loop can be invoked with its usual parameters, for example
`/skillopt-loop rounds=3 batch=20`. Each train call samples from the fixed
400 items using the round seed. The agent receives the training samples
exported under `workspace/.skillopt/samples/`. Each validation call scores all
200 dev items. The final call scores the selected skill on all 1,400 test items.
The evaluator uses strict EM and multi-answer token F1. Its per-item results
and summary are in `workspace/.skillopt/_eval_run/`.

`bash run.sh --check-data` verifies the fixed splits without model credentials or
workspace writes. `bash run.sh --help` describes the phase runner. The native target is
`gpt-5.4` with medium reasoning. Set `SKILLOPT_LITE_TARGET_MODEL` if the
deployment uses a different name. An existing SkillOpt OpenAI-compatible
configuration uses its `OPENAI_COMPATIBLE_BASE_URL` and
`OPENAI_COMPATIBLE_API_KEY` (or target-specific equivalents). The runner
also accepts Lite's original Azure and OpenAI authentication modes through
`env.sh`. Install Lite’s `requirements.txt` in a Python 3.10+ environment.
The runner uses the Lite repository’s `.venv/bin/python` if present, otherwise
`python3`; `SKILLOPT_LITE_PYTHON` selects a specific interpreter.

## Notes (vs the old reflect.py)

The `/skillopt-loop` does **not** use `reflect.py` / `analyst_*.md`.
Reflect, aggregate, and rewrite are all collapsed into a single Copilot
coding-agent call that reads the failed sample files directly. Those files
are kept here only as reference for what the old multi-step pipeline did.
