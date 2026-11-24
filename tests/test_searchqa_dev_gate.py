"""Offline regression checks for Lite's fixed SearchQA mild dev gate."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from copilot_example.searchqa import fixed_searchqa as runner, opencode_loop
from copilot_example.searchqa.dev_gate import evaluate_round


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.rows = [{"id": str(i)} for i in range(200)]

    def run_round(self, idx, wrong, *, order="hard-first", bonus=0, errors=(), crash_after=None):
        root = self.root / order / f"val_round_{idx:04d}"
        path = root / "results.jsonl"
        calls = []

        def rollout(items):
            old = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
            cached_ids = {r["id"] for r in old}
            calls.append([r["id"] for r in items])
            self.assertTrue(cached_ids.isdisjoint(calls[-1]))
            for item in items:
                ident = int(item["id"])
                hard = int(ident not in wrong)
                row = dict(id=str(ident), hard=hard, soft=hard + (0 if hard else bonus),
                           agent_ok=ident not in errors)
                with path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                old.append(row)
                if crash_after and len(old) == crash_after:
                    raise InterruptedError
            return old

        def usage():
            return {"_total": {"calls": sum(map(len, calls))}}

        s = evaluate_round(self.rows, root, idx, {"order": order, "batch_size": 4},
                           f"candidate {idx}: {wrong}", rollout, usage)
        return s, calls

    def test_bound_is_lite_reject_and_equal_scores_do_not_drop(self):
        initial, calls = self.run_round(0, range(42))
        self.assertEqual((len(calls), len(calls[0]), initial["hard"]), (1, 200, .79))
        stop, calls = self.run_round(1, range(44))
        self.assertEqual((stop["action"], stop["n_evaluated"], stop["upper_bound"]), ("reject", 44, .78))
        self.assertIsNone(stop["soft"])
        self.assertIsNone(stop["hard"])
        self.assertEqual(max(map(len, calls)), 4)
        self.assertEqual(stop["state_after"], initial["state_after"])
        with self.assertRaisesRegex(ValueError, "different candidate"):
            self.run_round(1, range(42))
        self.assertEqual(self.run_round(1, range(44)), (stop, []))
        flat, _ = self.run_round(2, range(43))
        self.assertEqual((flat["action"], flat["n_evaluated"]), ("flat", 200))
        self.assertEqual(flat["state_after"], initial["state_after"])
        soft_accept, _ = self.run_round(3, range(43), bonus=.5)
        self.assertEqual(soft_accept["action"], "accept")
        self.assertEqual(soft_accept["state_after"]["current_correct"], 157)
        self.assertEqual(soft_accept["state_after"]["best_round"], 0)
        best, _ = self.run_round(4, range(40))
        self.assertEqual((best["action"], best["state_after"]["best_round"]), ("accept_new_best", 4))

    def test_easy_first_then_resume_interrupted_batch(self):
        self.run_round(0, range(42), order="easy-first")
        with self.assertRaises(InterruptedError):
            self.run_round(1, range(44), order="easy-first", crash_after=6)
        summary, calls = self.run_round(1, range(44), order="easy-first")
        self.assertEqual(summary["evaluated_ids"][:4], ["42", "43", "44", "45"])
        self.assertEqual(calls[0], ["48", "49"])  # second batch partially survived
        self.assertEqual(summary["n_evaluated"], 200)
        self.assertFalse(summary["token_usage_complete"])

    def test_early_transport_errors_do_not_mask_total_outage(self):
        self.run_round(0, range(0))
        s, _ = self.run_round(1, range(200), errors=range(4))
        self.assertEqual(s["n_evaluated"], 8)  # do not stop on only failed API calls
        with self.assertRaisesRegex(RuntimeError, "All 200 dev calls failed"):
            self.run_round(2, range(200), errors=range(200))

    def test_open_code_and_phase_runner_share_fresh_workspace(self):
        run_dir = self.root / "run with spaces"
        root = runner.ENV_DIR
        old_cwd = Path.cwd()
        self.addCleanup(os.chdir, old_cwd)
        argv = ["opencode_loop.py", "--dev-early-stop", "--dev-run-dir", str(run_dir)]
        env = {"OPENAI_COMPATIBLE_MODEL":"mock/model", "OPENAI_COMPATIBLE_API_KEY":"fake",
               "OPENAI_COMPATIBLE_BASE_URL":"https://example.invalid/v1"}
        with patch.dict(os.environ, env), patch.object(sys, "argv", argv), patch.object(os, "execvpe") as host:
            opencode_loop.main()
        self.assertEqual(Path.cwd(), run_dir)
        prompt = host.call_args.args[1][-1]
        self.assertIn("--dev-round 0", prompt)
        self.assertIn('--dev-round "$r"', prompt)
        self.assertIn("normal rollback", prompt)
        self.assertIn("including a certified partial `Gate: reject`", prompt)
        self.assertNotIn("from the included output", prompt)
        self.assertIn('--split test --eval_limit 0', prompt)
        self.assertTrue((run_dir / ".github/prompts/skillopt-improve.prompt.md").is_file())
        import shlex
        args = shlex.split((run_dir / "run.sh").read_text().splitlines()[1])
        self.assertEqual(args[:3], ["exec", "bash", str(root/"run.sh")])
        self.assertEqual(args[args.index("--workspace")+1], str(run_dir/"workspace"))
        synthetic = [{"id":str(i), "question":f"Q{i}?", "context":"given", "answers":["ok"]} for i in range(200)]
        def answer(**kwargs):
            import re
            i = int(re.search(r"Q(\d+)\?", kwargs["user"]).group(1))
            threshold = 44 if "REVISED_SKILL" in kwargs["system"] else 42
            return "<answer>" + ("ok" if i>=threshold else "wrong") + "</answer>", {}
        with patch.object(runner,"load_splits",return_value={"val":synthetic,"test":synthetic[:2]}), \
                patch("skillopt.envs.searchqa.rollout.chat_target", side_effect=answer) as chat, \
                contextlib.redirect_stdout(io.StringIO()):
            phase_args = args[3:-1]
            for idx in (0, 1):
                if idx:
                    (run_dir/"workspace/skill.md").write_text("REVISED_SKILL")
                with patch.object(sys,"argv",["fixed_searchqa.py",*phase_args,"--split","val","--dev-round",str(idx)]):
                    runner.main()
            self.assertEqual(chat.call_count,244)
            with patch.object(sys,"argv",["fixed_searchqa.py",*phase_args,"--split","test"]):
                runner.main()
            self.assertEqual(chat.call_count,246)
            full = self.root / "original-validation"
            full.mkdir()
            with patch.object(sys,"argv",["fixed_searchqa.py","--split","val"]):
                original_args = runner.parse_args()
            self.assertFalse(original_args.dev_early_stop)
            runner.run_evaluation(synthetic, original_args, run_dir/"workspace/skill.md", full)
            self.assertEqual(chat.call_count,446)
            self.assertNotIn("stopped", json.loads((full/"eval_summary.json").read_text()))
        summary=json.loads((run_dir/"workspace/.skillopt/_eval_run/val_round_0001/eval_summary.json").read_text())
        self.assertTrue(summary["stopped"])
        self.assertIsNone(summary["hard"])


if __name__ == "__main__":
    unittest.main()
