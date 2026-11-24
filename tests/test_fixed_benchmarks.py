"""Offline fixed-task input, native optimizer host and train/dev/test contracts."""
import contextlib
import hashlib
import io
import json
import os
import shlex
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from copilot_example.searchqa import fixed_benchmarks as tasks
from copilot_example.searchqa import fixed_searchqa as runner
from copilot_example.searchqa import opencode_loop
from skillopt.envs.bird.evaluator import evaluate, extract_sql


class FixedBenchmarks(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.db_root = self.root / "databases"
        (self.db_root / "cities").mkdir(parents=True)
        self.db = self.db_root / "cities/cities.sqlite"
        with sqlite3.connect(self.db) as conn:
            conn.execute("CREATE TABLE cities(name TEXT)")
            conn.executemany("INSERT INTO cities VALUES (?)", [("Paris",), ("Paris",), ("Tokyo",)])
        self.sql = "SELECT name FROM cities WHERE name='Paris'"
        self.bird = {"id": "id1", "question": "Find the city", "db_id": "cities",
                     "evidence": "case matters", "SQL": self.sql,
                     "prompt": "schema and samples " + "Q" * 8000 + " Find the city."}
        self.qa = {"id": "id1", "question": "Name the city?", "context": "The city is Paris.",
                   "answers": ["City of Paris", "Paris"]}

    def test_manifest_order_and_db_availability(self):
        for benchmark, sample in (("triviaqa", self.qa), ("hotpotqa", self.qa), ("bird", self.bird)):
            with self.subTest(benchmark=benchmark):
                split_dir = self.root / benchmark
                manifest = {}
                for phase in ("train", "val", "test"):
                    part = split_dir / phase
                    part.mkdir(parents=True)
                    rows = [dict(sample, id=f"{phase}-{i}") for i in range(2)]
                    (part / "items.json").write_text(json.dumps(rows))
                    manifest[phase] = (2, hashlib.sha256("\n".join(r["id"] for r in rows).encode()).hexdigest())
                with patch.dict(tasks.SPLITS, {benchmark: manifest}):
                    splits = tasks.load_splits(split_dir, benchmark)
                    self.assertEqual(splits["train"][0]["id"], "train-0")
                    if benchmark == "bird":
                        tasks.check_databases(splits, self.db_root)
                        with self.assertRaises(FileNotFoundError):
                            tasks.check_databases(splits, self.root)
                    p = split_dir / "test/items.json"
                    p.write_text(json.dumps(list(reversed(splits["test"]))))
                    with self.assertRaisesRegex(ValueError, "manifest order"):
                        tasks.load_splits(split_dir, benchmark)

    def test_sql_grader(self):
        pred = "WITH q AS (SELECT DISTINCT name FROM cities) SELECT name FROM q ORDER BY name DESC;"
        ev = evaluate(f"answer\n```sql\n{pred}\n```", "SELECT name FROM cities", self.db)
        self.assertEqual((ev["pred_sql"], ev["res"]), (pred, 1))
        self.assertEqual(extract_sql("SELECT 'it''s;a'; SELECT 999;"), "SELECT 'it''s;a';")
        self.assertEqual(evaluate(self.sql.replace("Paris", "paris"), self.sql, self.db)["res"], 0)
        for bad in ("", "DELETE FROM cities", "SELECT not_found FROM cities",
                    "ATTACH DATABASE '" + str(self.root / "unwanted.sqlite") + "' AS side"):
            ev = evaluate(bad, self.sql, self.db)
            self.assertEqual((ev["res"], ev["status"]), (0, "error"))
        self.assertFalse((self.root / "unwanted.sqlite").exists())
        long_sql = ("WITH RECURSIVE a(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM a WHERE n<1000000) "
                    "SELECT SUM(n) FROM a")
        self.assertEqual(evaluate(long_sql, self.sql, self.db, timeout=0)["status"], "timeout")

    def test_metadata_queries_are_read_only(self):
        # Table-valued pragmas are legal SELECT inputs for BIRD SQL.
        meta = "SELECT name FROM pragma_table_info('cities')"
        self.assertEqual(evaluate(meta, "SELECT 'name'", self.db)["res"], 1)
        self.assertEqual(evaluate("PRAGMA compile_options", "PRAGMA compile_options", self.db)["res"], 1)
        for sql in ("PRAGMA query_only=OFF", "PRAGMA user_version=1", "SELECT load_extension('unused')"):
            with self.subTest(sql=sql):
                failure = evaluate(sql, "SELECT 1", self.db)
                self.assertEqual((failure["res"], failure["status"]), (0, "error"))

    def test_host_requires_complete_heldout_test_even_on_zero_exit(self):
        with patch.object(opencode_loop.subprocess, "run") as child:
            with self.assertRaisesRegex(RuntimeError, "without complete hotpotqa test"):
                opencode_loop.run_isolated_host(["opencode"], self.root, {}, "hotpotqa")
            child.assert_called_once_with(["opencode"], cwd=self.root, env={}, check=True)

        ids = ["a", "b"]
        output = self.root / "workspace/.skillopt/_eval_run/test_mock"
        output.mkdir(parents=True)
        (output / "eval_summary.json").write_text(json.dumps(dict(split="test", benchmark="hotpotqa", n_items=2)))
        (output / "item_ids.json").write_text(json.dumps(ids))
        digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        with patch.dict(tasks.SPLITS, {"hotpotqa": {"test": (2, digest)}}):
            results = output / "results.jsonl"
            results.write_text(json.dumps(dict(id="a", agent_ok=True)) + "\n")
            with self.assertRaisesRegex(RuntimeError, "without complete"):
                opencode_loop.verify_test(self.root, "hotpotqa")
            results.write_text("".join(json.dumps(dict(id=i, agent_ok=True)) + "\n" for i in reversed(ids)))
            opencode_loop.verify_test(self.root, "hotpotqa")

    def test_new_task_host_and_phase_commands(self):
        original_cwd = Path.cwd()
        self.addCleanup(os.chdir, original_cwd)
        for benchmark in ("triviaqa", "hotpotqa", "bird"):
            with self.subTest(benchmark=benchmark):
                run_dir = self.root / ("run " + benchmark)
                argv = ["opencode_loop.py", "--benchmark", benchmark, "--run-dir", str(run_dir)]
                env = {"OPENAI_COMPATIBLE_MODEL": "fake/model", "OPENAI_COMPATIBLE_API_KEY": "dummy",
                       "OPENAI_COMPATIBLE_BASE_URL": "https://example.invalid/v1"}
                with patch.dict(os.environ, env), patch.object(sys, "argv", argv), patch.object(opencode_loop, "run_isolated_host") as host:
                    opencode_loop.main()
                cmd, cwd, child_env, task = host.call_args.args
                self.assertEqual((cwd, child_env["PWD"], cmd[cmd.index("--dir") + 1]), (run_dir, str(run_dir), str(run_dir)))
                self.assertTrue((run_dir / "skills/initial.md").is_file())
                prompt = cmd[-1]
                self.assertIn(f"full ({tasks.SPLITS[benchmark]['test'][0]})", prompt)
                self.assertNotIn("1400", prompt)
                self.assertNotIn("--dev-round", prompt)
                template = (run_dir / ".github/prompts/skillopt-improve.prompt.md").read_text()
                self.assertIn(benchmark, template)
                if benchmark == "bird":
                    self.assertNotIn("<answer>", template)
                    self.assertIn("binary execution accuracy", (run_dir / ".github/copilot-instructions.md").read_text())
                args = shlex.split((run_dir / "run.sh").read_text().splitlines()[1])[3:-1]
                sample = self.bird if benchmark == "bird" else self.qa
                # One successful and one wrong valid response in the same split.
                rows = [dict(sample, id=f"id{i}") for i in range(2)]
                splits = {split: rows for split in ("train", "val", "test")}
                if benchmark == "bird":
                    fn = "skillopt.envs.bird.rollout.chat_target"
                    good = self.sql
                else:
                    fn = "skillopt.envs.searchqa.rollout.chat_target"
                    good = "<answer>Paris</answer>"
                bad = "SELECT nonexistent FROM cities" if benchmark == "bird" else "<answer>Tokyo</answer>"
                with patch.object(runner, "load_splits", return_value=splits), \
                        patch(fn, side_effect=[(value, {}) for value in (good, bad) * 3]) as chat:
                    with contextlib.redirect_stdout(io.StringIO()):
                        for stage in ("train", "val", "test"):
                            argv = ["fixed_searchqa.py", *args, "--split", stage, "--workers", "1",
                                    "--db-root", str(self.db_root)]
                            with patch.object(sys, "argv", argv):
                                runner.main()
                    for call in chat.call_args_list:
                        kwargs = call.kwargs
                        if benchmark == "bird":
                            self.assertEqual(kwargs["user"], self.bird["prompt"])
                            self.assertNotIn(self.sql, kwargs["user"])
                    self.assertEqual(chat.call_count, 6)
                ws = run_dir / "workspace"
                self.assertEqual((ws / ".benchmark").read_text().strip(), benchmark)
                summaries = list((ws / ".skillopt/_eval_run").glob("*/eval_summary.json"))
                self.assertEqual({json.loads(f.read_text())["split"] for f in summaries}, {"train", "val", "test"})
                self.assertTrue(all(json.loads(f.read_text())["hard"] == 0.5 for f in summaries))
                sample_md = (ws / ".skillopt/samples/passed/id0.md").read_text()
                self.assertIn(self.sql if benchmark == "bird" else "Paris", sample_md)
                if benchmark == "bird":
                    self.assertIn("Execution:", sample_md)
                    self.assertIn("Gold rows", sample_md)
                    failed = (ws / ".skillopt/samples/failed/id1.md").read_text()
                    self.assertIn("execution error", failed)
                    test_dir = next((ws / ".skillopt/_eval_run").glob("test_*"))
                    scored = [json.loads(line) for line in (test_dir / "results.jsonl").read_text().splitlines()]
                    self.assertTrue(all(r["agent_ok"] for r in scored))
                    self.assertEqual({r["exec_status"] for r in scored}, {"ok", "error"})
                with patch.object(sys, "argv", ["fixed_searchqa.py", "--benchmark", "searchqa", "--workspace", str(ws)]):
                    with self.assertRaisesRegex(ValueError, "another benchmark"):
                        runner.prepare_workspace(runner.parse_args())


if __name__ == "__main__":
    unittest.main()
