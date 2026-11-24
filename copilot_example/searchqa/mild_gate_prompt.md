### 3. Validate the patched skill on dev

Use the current round number in this command, starting at 1. Round 0 was
the full baseline. For example, set `r=1` in the same Bash invocation for
the first candidate, then 2 for the second, and so on.

```bash
rm -rf workspace/.skillopt/samples/failed/* workspace/.skillopt/samples/passed/*
bash run.sh --skill workspace/skill.md --split val --eval_limit 0 --limit 0 --dev-round "$r"
```

Wait for the gate to finish before proceeding. Read the emitted `Gate:`
action or `eval_summary.json` in its reported evaluation directory.
An interrupted or repeated validation uses the same round number and skill.

### 4. Apply the gate decision

The runner maintains the Lite one-point EM dead band, two-point F1 exception,
and numeric reference scores. Apply its action exactly:

- `reject`: restore this round's before snapshot to `workspace/skill.md`.
  Increment the regression streak and clear the noop streak.
- `flat`: keep the candidate on disk. Increment the noop streak and clear
  the regression streak.
- `accept` or `accept_new_best`: keep the candidate, clear both streaks,
  and for `accept_new_best` also save a best snapshot by the usual convention.

Use `state_after` for the current score, best score and best round. A
certified partial reject is a normal rollback. Its full candidate EM, F1
and delta are unknown. In progress and final reports show the evaluated
count and EM bounds, mark F1 unmeasured, and do not rerun it for a score.
Continue with the usual train rollout, including after rollback. Retain
the native stopping conditions and restore the best snapshot for full test.
