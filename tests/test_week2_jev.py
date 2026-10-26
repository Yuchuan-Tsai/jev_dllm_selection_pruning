"""Week 2 Jev interfaces: public-only requests, strict parsing, and paid-call boundaries."""
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jev_dllm.experiments import week2_jev as w
from jev_dllm.judges import pointwise as pw
from jev_dllm.judges.listwise import (build_listwise_request, nested_subsets,
                                      parse_listwise_response, presentation_order)
from jev_dllm.judges.typesafe import _NoRedirect, build_request
from jev_dllm.reproducibility import trajectory_seed
from jev_dllm.schema import Candidate, Question

SENTINEL = "PRIVATE_REFERENCE_SENTINEL"


def noul_response(request, correct=0.8, consistent=0.6, model=None):
    return {"model": model or request["model"],
            "answers": {"correct": {"type": "noul", "noul": correct},
                        "consistent": {"type": "noul", "noul": consistent}},
            "usage": {"input_tokens": 120, "output_tokens": 12}}


def choice_response(request):
    ids = list(request["questions"]["best_answer"]["criteria"])
    return {"model": request["model"], "answers": {"best_answer": {
                "type": "choice", "choice": ids[0], "confidence": 0.9,
                "probabilities": {cid: float(i == 0) for i, cid in enumerate(ids)}}},
            "usage": {"input_tokens": 900, "output_tokens": 5}}


class FakeListwise:
    def __init__(self):
        self.calls = 0

    def select(self, body):
        self.calls += 1
        raw = choice_response(body)
        return {**parse_listwise_response(raw, body), "request_seconds": 0.25,
                "api_cost_usd": None, "raw_response": raw}


class FakePointwise:
    def __init__(self, fail_at=None, raw=None):
        self.calls, self.fail_at, self.raw = 0, fail_at, raw

    def judge(self, body):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("Jev HTTP 500; no automatic retry")
        raw = self.raw(body) if self.raw else noul_response(body)
        try:
            parsed = pw.parse_pointwise_response(raw, body)
        except ValueError as exc:
            exc.raw_response, exc.request_seconds = raw, 0.1
            raise
        return {**parsed, "request_seconds": 0.1, "api_cost_usd": None, "raw_response": raw}


class PointwiseRequestTests(unittest.TestCase):
    def setUp(self):
        self.q = Question("q1", "gsm8k", "dev", "What is 2 + 3?")
        self.c = Candidate("q1", "q1/c03", 77, "2 + 3 = 5, so the answer", False, True, confidence=0.42)

    def test_only_allowlisted_fields_and_one_verbatim_candidate(self):
        body = pw.build_pointwise_request(self.q, self.c, "Is it correct?", "Is it consistent?")
        self.assertEqual(set(body), {"model", "state", "questions"})
        self.assertEqual(body["model"], "jev-1.13.0")
        self.assertEqual(body["state"], {"stage": "pointwise", "question": "What is 2 + 3?",
                                         "candidate": {"candidate_id": "q1/c03", "text": self.c.text}})
        self.assertEqual(body["questions"], {
            "correct": {"type": "noul", "instructions": "Is it correct?"},
            "consistent": {"type": "noul", "instructions": "Is it consistent?"}})
        wire = json.dumps(body)
        for private in ("confidence", "seed", "truncated", "completed", "0.42", "77", "\"answer\"", "label"):
            self.assertNotIn(private, wire)

    def test_rejects_foreign_candidate_alias_model_and_empty_instructions(self):
        other = Candidate("q2", "q2/c00", 1, "x", True, False)
        with self.assertRaises(ValueError):
            pw.build_pointwise_request(self.q, other, "a", "b")
        with self.assertRaises(ValueError):
            pw.build_pointwise_request(self.q, self.c, "a", "b", model="jev-latest")
        with self.assertRaises(ValueError):
            pw.build_pointwise_request(self.q, self.c, " ", "b")

    def test_parser_keeps_targets_separate(self):
        body = pw.build_pointwise_request(self.q, self.c, "a", "b")
        parsed = pw.parse_pointwise_response(noul_response(body, 0.9, 0.1), body)
        self.assertEqual(parsed, {"candidate_id": "q1/c03", "correct_probability": 0.9,
                                  "consistent_probability": 0.1, "model": "jev-1.13.0",
                                  "usage": {"input_tokens": 120, "output_tokens": 12}})
        self.assertEqual(pw.parse_pointwise_response(noul_response(body, 0, 1), body)["correct_probability"], 0)

    def test_parser_rejects_invalid_responses(self):
        body = pw.build_pointwise_request(self.q, self.c, "a", "b")
        bad = []
        bad.append(noul_response(body, model="jev-1.14.0"))
        bad.append(noul_response(body, model="jev-latest"))
        for value in (1.5, -0.01, float("nan"), float("inf"), True, "0.5", None):
            bad.append(noul_response(body, correct=value))
            bad.append(noul_response(body, consistent=value))
        missing = noul_response(body); del missing["answers"]["consistent"]; bad.append(missing)
        extra = noul_response(body); extra["answers"]["other"] = {"type": "noul", "noul": 0.5}; bad.append(extra)
        no_value = noul_response(body); del no_value["answers"]["correct"]["noul"]; bad.append(no_value)
        wrong_type = noul_response(body); wrong_type["answers"]["correct"]["type"] = "choice"; bad.append(wrong_type)
        for usage in ({"input_tokens": 1.0, "output_tokens": 1}, {"input_tokens": 1},
                      {"input_tokens": -1, "output_tokens": 1}, {"input_tokens": True, "output_tokens": 1}):
            r = noul_response(body); r["usage"] = usage; bad.append(r)
        no_model = noul_response(body); del no_model["model"]; bad.append(no_model)
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises((ValueError, KeyError, TypeError)):
                pw.parse_pointwise_response(raw, body)


class PointwiseTransportTests(unittest.TestCase):
    def setUp(self):
        q = Question("q1", "gsm8k", "dev", "What is 2 + 3?")
        self.body = pw.build_pointwise_request(q, Candidate("q1", "q1/c00", 1, "5", True, False), "a", "b")

    def opener(self, payload=None, error=None):
        opener = Mock()
        if error:
            opener.open.side_effect = error
        else:
            opener.open.return_value.__enter__ = lambda s: io.BytesIO(json.dumps(payload).encode())
            opener.open.return_value.__exit__ = lambda s, *a: False
        return opener

    def test_success_records_latency_and_raw_response_without_redirects(self):
        opener = self.opener(noul_response(self.body))
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-key"}), \
                patch.object(pw, "build_opener", return_value=opener) as build:
            result = pw.PointwiseJudge().judge(self.body)
        build.assert_called_once_with(_NoRedirect)
        self.assertEqual(opener.open.call_count, 1)
        self.assertEqual(result["correct_probability"], 0.8)
        self.assertGreaterEqual(result["request_seconds"], 0)
        self.assertEqual(result["raw_response"], noul_response(self.body))
        self.assertIsNone(result["api_cost_usd"])

    def test_http_error_is_not_retried_and_key_is_not_shown(self):
        opener = self.opener(error=HTTPError(pw.ENDPOINT, 500, "err", {}, None))
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "secret-test-key"}), \
                patch.object(pw, "build_opener", return_value=opener):
            with self.assertRaises(RuntimeError) as ctx:
                pw.PointwiseJudge().judge(self.body)
        self.assertEqual(opener.open.call_count, 1)
        self.assertNotIn("secret-test-key", str(ctx.exception))

    def test_invalid_response_keeps_raw_evidence(self):
        raw = noul_response(self.body, correct=2.0)
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-key"}), \
                patch.object(pw, "build_opener", return_value=self.opener(raw)):
            with self.assertRaises(ValueError) as ctx:
                pw.PointwiseJudge().judge(self.body)
        self.assertEqual(ctx.exception.raw_response, raw)

    def test_missing_key_refuses_before_network(self):
        env = {k: v for k, v in os.environ.items() if k != "TYPESAFE_API_KEY"}
        with patch.dict(os.environ, env, clear=True), patch.object(pw, "build_opener") as build:
            with self.assertRaises(ValueError):
                pw.PointwiseJudge().judge(self.body)
        build.assert_not_called()


class ListwiseTests(unittest.TestCase):
    def setUp(self):
        self.q = Question("q1", "math", "dev", "Compute 1 + 1.")
        self.pool = [Candidate("q1", f"q1/c{i:02d}", i, f"answer {i}", True, False, 0.5) for i in range(16)]

    def test_shuffle_is_deterministic_and_seeded_like_selection(self):
        ids = [c.candidate_id for c in self.pool[:8]]
        expected = list(ids)
        random.Random(trajectory_seed(w.SEED, "q1", 0)).shuffle(expected)
        self.assertEqual(presentation_order(ids, w.SEED, "q1"), expected)
        self.assertEqual(presentation_order(ids, w.SEED, "q1"), presentation_order(list(ids), w.SEED, "q1"))
        self.assertNotEqual(presentation_order(ids, w.SEED, "q1"), presentation_order(ids, w.SEED + 1, "q1"))
        self.assertEqual(sorted(presentation_order(ids, w.SEED, "q1")), sorted(ids))

    def test_matches_typesafe_build_request_for_completed_candidates(self):
        subset = self.pool[:4]
        body = build_listwise_request(self.q, subset, "Pick one.", w.SEED)
        order = presentation_order([c.candidate_id for c in subset], w.SEED, "q1")
        by_id = {c.candidate_id: c for c in subset}
        self.assertEqual(body, build_request(self.q, [by_id[i] for i in order], "Pick one."))

    def test_truncated_and_empty_answers_kept_verbatim_without_flags(self):
        subset = [Candidate("q1", "q1/c00", 1, "unfinished \\boxed{", False, True, 0.3),
                  Candidate("q1", "q1/c01", 2, "", False, True), self.pool[2]]
        body = build_listwise_request(self.q, subset, "Pick one.", w.SEED)
        self.assertEqual(set(body["state"]), {"stage", "question", "candidates"})
        sent = {c["candidate_id"]: c for c in body["state"]["candidates"]}
        self.assertEqual({k: v["text"] for k, v in sent.items()}, {c.candidate_id: c.text for c in subset})
        self.assertTrue(all(set(c) == {"candidate_id", "text"} for c in sent.values()))

    def test_nested_subsets_are_prefixes(self):
        ids = [c.candidate_id for c in self.pool]
        subsets = nested_subsets(ids, (2, 4, 8, 16))
        self.assertEqual({n: len(s) for n, s in subsets.items()}, {2: 2, 4: 4, 8: 8, 16: 16})
        for small, large in ((2, 4), (4, 8), (8, 16)):
            self.assertEqual(subsets[large][:small], subsets[small])
        for sizes in ((1, 2), (4, 2), (2, 2), (2, 32), (), (2.0,)):
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                nested_subsets(ids, sizes)
        with self.assertRaises(ValueError):
            nested_subsets(ids[:3] + ids[:1], (2,))


class Week2WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        patch("sys.stdout", new_callable=io.StringIO).start()
        self.addCleanup(patch.stopall)
        prompts ={p for t in w.TASKS for p in w.prompt_paths(t).values()}
        for name in set(w.FILES) | prompts:
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, self.root / name)
        self.questions = self.root / "data/processed/gsm8k-w2dev.json"
        self.questions.parent.mkdir(parents=True)
        rows = [asdict(Question(f"g{i}", "gsm8k", "dev", f"Add {i} and 1.")) for i in range(4)]
        self.questions.write_text(json.dumps(rows))
        self.pool = self.root / "artifacts/runs/week2-pool-gsm8k"
        self.pool.mkdir(parents=True)
        lines = []
        for i in range(4):
            for j in range(16):
                text = "" if j == 5 else f"Step {j}. Answer {i + 1}" + ("" if j != 6 else " and then")
                lines.append(json.dumps(asdict(Candidate(f"g{i}", f"g{i}/c{j:02d}", 1000 * i + j, text,
                                                         j not in (5, 6), j in (5, 6), 0.9))))
        (self.pool / "candidates.jsonl").write_text("\n".join(lines) + "\n")
        private = self.root / "data/private"
        private.mkdir(parents=True)
        (private / "gsm8k-w2dev-references.json").write_text(json.dumps([{"question_id": "g0", "answer": SENTINEL}]))
        self.ids = ["g2", "g0", "g1"]

    def prepare(self, **kwargs):
        return w.prepare(self.root, "gsm8k", self.pool, self.questions, self.ids, **kwargs)

    def test_prepare_freezes_exact_paid_call_count_offline(self):
        with patch.object(w.TypeSafeJudge, "select", side_effect=AssertionError("network")), \
                patch.object(w.PointwiseJudge, "judge", side_effect=AssertionError("network")):
            plan_dir = self.prepare()
        plan = json.loads((plan_dir / "plan.json").read_text())
        self.assertEqual(plan_dir.parent, self.root / "artifacts/runs")
        self.assertEqual(plan["planned_calls"], {"listwise": 12, "pointwise": 48, "total": 60})
        self.assertEqual(plan["calls_per_question"], {"listwise": 4, "pointwise": 16})
        self.assertEqual(plan["paid_calls_sent"], 0)
        rows = w.read_rows(plan_dir / "requests.jsonl")
        self.assertEqual([r["call_index"] for r in rows], list(range(60)))
        self.assertEqual([r["question_id"] for r in rows[::20]], self.ids)
        self.assertEqual([r.get("subset_size") for r in rows[:4]], [2, 4, 8, 16])
        wire = (plan_dir / "requests.jsonl").read_text()
        for private in (SENTINEL, "\"confidence\"", "truncated", "completed", "\"seed\"", "0.9"):
            self.assertNotIn(private, wire)
        for r in rows:
            body = r["body"]
            self.assertEqual(set(body), {"model", "state", "questions"})
            if r["kind"] == "pointwise":
                self.assertEqual(set(body["state"]), {"stage", "question", "candidate"})
                self.assertEqual(set(body["state"]["candidate"]), {"candidate_id", "text"})
            else:
                self.assertEqual(set(body["state"]), {"stage", "question", "candidates"})
                self.assertEqual(sorted(c["candidate_id"] for c in body["state"]["candidates"]),
                                 sorted(r["subset_ids"]))
        # Empty and truncated answers are sent verbatim.
        texts = {r["candidate_id"]: r["body"]["state"]["candidate"]["text"] for r in rows if r["kind"] == "pointwise"}
        self.assertEqual(texts["g0/c05"], "")
        self.assertEqual(texts["g0/c06"], "Step 6. Answer 1 and then")
        self.assertEqual(w.check(self.root, plan_dir)[1], rows)

    def test_prepare_is_deterministic_and_uses_task_prompts(self):
        a, b = self.prepare(), self.prepare()
        self.assertEqual((a / "requests.jsonl").read_bytes(), (b / "requests.jsonl").read_bytes())
        rows = w.read_rows(a / "requests.jsonl")
        self.assertEqual(rows[0]["body"]["questions"]["best_answer"]["instructions"],
                         (self.root / "prompts/jev_final.txt").read_text())
        self.assertEqual(rows[4]["body"]["questions"]["correct"]["instructions"],
                         (self.root / "prompts/week02/jev_pointwise_correct_math.txt").read_text())
        self.assertEqual(w.prompt_paths("mbpp")["listwise"], "prompts/jev_humaneval.txt")
        self.assertEqual(w.prompt_paths("mbpp")["correct"], "prompts/week02/jev_pointwise_correct_code.txt")
        with self.assertRaises(ValueError):
            w.prompt_paths("humaneval")

    def test_injected_candidate_order_defines_nested_subsets(self):
        def reverse(question_id, ids):
            return list(reversed(ids))
        plan_dir = self.prepare(candidate_order=reverse)
        plan = json.loads((plan_dir / "plan.json").read_text())
        self.assertEqual(plan["candidate_orders"]["g0"][:2], ["g0/c15", "g0/c14"])
        rows = w.read_rows(plan_dir / "requests.jsonl")
        self.assertEqual(rows[0]["subset_ids"], ["g2/c15", "g2/c14"])
        self.assertEqual([r["candidate_id"] for r in rows[4:20]], plan["candidate_orders"]["g2"])
        w.check(self.root, plan_dir)
        for bad in (lambda q, ids: ids[:1] * 16, lambda q, ids: ids + ["other"], lambda q, ids: ids[:8]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.prepare(candidate_order=bad)

    def test_rejects_heldout_and_unknown_questions(self):
        with self.assertRaises(ValueError):
            w.prepare(self.root, "gsm8k", self.pool, self.questions, ["g0", "missing"])
        with self.assertRaises(ValueError):
            w.prepare(self.root, "gsm8k", self.pool, self.questions, ["g0", "g0"])
        rows = json.loads(self.questions.read_text())
        rows[0]["split"] = "heldout"
        self.questions.write_text(json.dumps(rows))
        with self.assertRaises(ValueError):
            w.prepare(self.root, "gsm8k", self.pool, self.questions, ["g0"])

    def test_check_detects_changed_inputs(self):
        for mutate in (
                lambda d: (d / "requests.jsonl").write_text((d / "requests.jsonl").read_text().replace("Step 1.", "Step 1!")),
                lambda d: (self.pool / "candidates.jsonl").write_text((self.pool / "candidates.jsonl").read_text().replace("Step 2.", "Step 2!")),
                lambda d: (self.root / w.CONSISTENT_PROMPT).write_text("changed"),
                lambda d: (self.root / "src/jev_dllm/judges/pointwise.py").write_text("# changed\n")):
            with self.subTest(mutate=mutate):
                self.setUp()
                plan_dir = self.prepare()
                mutate(plan_dir)
                with self.assertRaises(ValueError):
                    w.check(self.root, plan_dir)

    def test_live_refuses_call_count_mismatch_before_any_request(self):
        plan_dir = self.prepare()
        listwise, pointwise = FakeListwise(), FakePointwise()
        for approved in (59, 61, 0, "60", 60.0, True):
            with self.subTest(approved=approved), self.assertRaises(ValueError):
                w.live(self.root, plan_dir, approved, listwise_judge=listwise, pointwise_judge=pointwise)
        self.assertEqual((listwise.calls, pointwise.calls), (0, 0))
        self.assertFalse((plan_dir / "live").exists())
        self.assertFalse((self.root / w.CLAIMS).exists())

    def test_live_records_latency_usage_and_refuses_duplicate_launch(self):
        plan_dir = self.prepare()
        listwise, pointwise = FakeListwise(), FakePointwise()
        out = w.live(self.root, plan_dir, 60, listwise_judge=listwise, pointwise_judge=pointwise)
        self.assertEqual((listwise.calls, pointwise.calls), (12, 48))
        m = json.loads((out / "manifest.json").read_text())
        self.assertEqual((m["status"], m["attempted_calls"], m["successful_calls"], m["automatic_retries"]),
                         ("complete", 60, 60, 0))
        self.assertEqual(m["usage"]["listwise"]["input_tokens"], 12 * 900)
        self.assertEqual(m["usage"]["pointwise"]["input_tokens"], 48 * 120)
        self.assertAlmostEqual(m["usage"]["pointwise"]["request_seconds_mean"], 0.1)
        results = w.read_rows(out / "results.jsonl")
        self.assertEqual(len(results), 60)
        self.assertTrue(all("request_seconds" in r and "wall_seconds" in r and "usage" in r for r in results))
        point = results[4]
        self.assertEqual((point["kind"], point["correct_probability"], point["consistent_probability"]),
                         ("pointwise", 0.8, 0.6))
        # Same plan again, and an identical batch prepared in a new directory.
        for again in (plan_dir, self.prepare()):
            with self.subTest(again=again), self.assertRaises(ValueError):
                w.live(self.root, again, 60, listwise_judge=FakeListwise(), pointwise_judge=FakePointwise())
        self.assertEqual(len(w.read_rows(out / "results.jsonl")), 60)

    def test_live_stops_on_first_error_without_retry(self):
        plan_dir = self.prepare()
        listwise, pointwise = FakeListwise(), FakePointwise(fail_at=3)
        with self.assertRaises(RuntimeError):
            w.live(self.root, plan_dir, 60, listwise_judge=listwise, pointwise_judge=pointwise)
        self.assertEqual((listwise.calls, pointwise.calls), (4, 3))
        m = json.loads((plan_dir / "live/manifest.json").read_text())
        self.assertEqual((m["status"], m["attempted_calls"], m["successful_calls"], m["failed_call_index"]),
                         ("failed", 7, 6, 6))
        self.assertEqual(len(w.read_rows(plan_dir / "live/results.jsonl")), 6)
        with self.assertRaises(ValueError):
            w.live(self.root, plan_dir, 60, listwise_judge=FakeListwise(), pointwise_judge=FakePointwise())

    def test_live_keeps_rejected_raw_response(self):
        plan_dir = self.prepare()
        pointwise = FakePointwise(raw=lambda body: noul_response(body, correct=float("nan")))
        with self.assertRaises(ValueError):
            w.live(self.root, plan_dir, 60, listwise_judge=FakeListwise(), pointwise_judge=pointwise)
        self.assertEqual(pointwise.calls, 1)
        failed = json.loads((plan_dir / "live/failed_response.json").read_text())
        self.assertEqual(failed["call_index"], 4)
        self.assertIn("NaN", failed["raw_response_json"])

    def test_live_needs_api_key_before_claiming(self):
        plan_dir = self.prepare()
        env = {k: v for k, v in os.environ.items() if k != "TYPESAFE_API_KEY"}
        with patch.dict(os.environ, env, clear=True), \
                patch.object(w.TypeSafeJudge, "select", side_effect=AssertionError("network")):
            with self.assertRaises(ValueError) as ctx:
                w.live(self.root, plan_dir, 60)
        self.assertFalse((self.root / w.CLAIMS).exists())
        self.assertFalse((plan_dir / "live").exists())
        env_file = self.root / ".env"
        env_file.write_text("TYPESAFE_API_KEY=secret-from-file\n")
        with patch.dict(os.environ, env, clear=True), \
                patch.object(w.TypeSafeJudge, "select", side_effect=RuntimeError("stub network")) as select, \
                patch("sys.stdout", new_callable=io.StringIO) as stdout:
            with self.assertRaises(RuntimeError):
                w.live(self.root, plan_dir, 60, env_file=env_file)
            self.assertEqual(os.environ["TYPESAFE_API_KEY"], "secret-from-file")
        self.assertEqual(select.call_count, 1)
        self.assertNotIn("secret-from-file", stdout.getvalue())
        self.assertNotIn("secret-from-file", "".join(p.read_text() for p in (plan_dir / "live").iterdir()))


if __name__ == "__main__":
    unittest.main()
