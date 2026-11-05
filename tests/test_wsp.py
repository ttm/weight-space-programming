from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from wsp.__main__ import _round_floats
from wsp.evaluation import evaluate, equivalent, permute_hidden, random_baseline, rescale_relu, stress
from wsp.network import load_candidate, validate_network
from wsp.session import start, probe, finalize
from wsp.repair import load_patch, repair_report, apply_patch
from wsp.tasks import TASKS, samples

ROOT = Path(__file__).resolve().parents[1]


class HandcraftedTests(unittest.TestCase):
    def test_every_correct_candidate_passes_all_splits(self):
        for path in (ROOT / "examples").glob("*.json"):
            if "failure" in path.name:
                continue
            with self.subTest(path=path.name):
                result = evaluate(load_candidate(path))
                self.assertTrue(result["all_passed"], result)

    def test_each_task_has_example(self):
        names = {load_candidate(path)["task"] for path in (ROOT / "examples").glob("*.json")}
        self.assertEqual(names, set(TASKS))

    def test_deliberate_failure_is_detected(self):
        result = evaluate(load_candidate(ROOT / "examples/xor2_linear_failure.json"))
        self.assertEqual(result["splits"]["test"]["value"], .75)
        self.assertFalse(result["all_passed"])

    def test_regression_judged_by_largest_error(self):
        # One test point off by 1e-5 keeps MSE under 1e-12; the largest error must still fail it.
        candidate = load_candidate(ROOT / "examples/absdiff2.json")
        xs = sorted(x[0] for x, _ in samples("absdiff2", "test"))
        threshold = (xs[-1] + xs[-2]) / 2  # only the largest first input exceeds it
        hidden, output = candidate["network"]["layers"]
        hidden["weights"].append([1, 0]); hidden["bias"].append(-threshold)
        output["weights"][0].append(1e-5 / (xs[-1] - threshold))
        split = evaluate(candidate, ("test",))["splits"]["test"]
        self.assertLess(split["value"], 1e-12)
        self.assertAlmostEqual(split["max_abs_error"], 1e-5)
        self.assertFalse(split["passed"])

    def test_invalid_shapes_and_values(self):
        self.assertRaises(ValueError, validate_network, {"layers": [{"weights": [[float("nan")]], "bias": [0], "activation": "linear"}]})
        self.assertRaises(ValueError, validate_network, {"layers": [{"weights": [[1], [1, 2]], "bias": [0, 0], "activation": "linear"}]})

    def test_equivalent_reparameterizations(self):
        for filename in ("xor2.json", "parity3.json", "absdiff2.json", "diamond2.json", "max2.json"):
            with self.subTest(filename=filename):
                candidate = load_candidate(ROOT / "examples" / filename)
                self.assertTrue(equivalent(candidate["network"], permute_hidden(candidate["network"]), candidate["task"]))
                self.assertTrue(equivalent(candidate["network"], rescale_relu(candidate["network"]), candidate["task"]))

    def test_splits_deterministic(self):
        self.assertEqual(samples("compare2", "test"), samples("compare2", "test"))
        self.assertNotEqual(samples("compare2", "test"), samples("compare2", "ood"))

    def test_seed_salt_holds_back_test_inputs(self):
        splits = ("validation", "test", "ood", "edge")
        with mock.patch.dict(os.environ):
            os.environ.pop("WSP_SEED_SALT", None)
            public = {split: samples("compare2", split) for split in splits}
        with mock.patch.dict(os.environ, {"WSP_SEED_SALT": "evaluator-secret"}), tempfile.TemporaryDirectory() as d:
            salted = {split: samples("compare2", split) for split in splits}
            self.assertEqual([salted[s] == public[s] for s in splits], [True, False, False, True])
            start(Path(d) / "s.json", "compare2", "blind", "optional")
            self.assertTrue(finalize(Path(d) / "s.json", ROOT / "examples/compare2.json")["all_passed"])
            final = json.loads((Path(d) / "s.json").read_text())["final"]
            self.assertEqual(final["seed_salt_sha256"], sha256(b"evaluator-secret").hexdigest())

    def test_reports_absorb_platform_float_noise(self):
        # absdiff2 random-baseline MSE as computed on two platforms; their libm results differ in the last bit.
        self.assertEqual(_round_floats(24.94790748556399), _round_floats(24.947907485563984))
        report = {"value": 1e-31, "passed": True, "n": [512, 0.75]}
        self.assertEqual(json.dumps(_round_floats(report)), json.dumps(report))

    def test_stress_and_random_baseline(self):
        candidate = load_candidate(ROOT / "examples/xor2.json")
        self.assertEqual(random_baseline(candidate, 3)["trials"], 3)
        summary = stress(candidate)
        self.assertTrue(summary["invariances"]["hidden_neuron_permutation"])
        self.assertIn("0.1", summary["noise"])

    def test_single_parameter_repair(self):
        source = load_candidate(ROOT / "fixtures/xor2_corrupted.json")
        patch = load_patch(ROOT / "repairs/xor2_bias_fix.json")
        report = repair_report(source, patch)
        self.assertFalse(report["before"]["all_passed"])
        self.assertTrue(report["after"]["all_passed"])
        self.assertEqual(report["edited_parameters"], 1)
        wrong_patch = {**patch, "edits": [{**patch["edits"][0], "from": 20}]}
        with self.assertRaises(ValueError): apply_patch(source, wrong_patch)

    def test_blind_no_probe_and_single_final(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "s.json"
            cand = ROOT / "examples/xor2.json"
            start(path, "xor2", "blind", "required")
            with self.assertRaises(ValueError): probe(path, cand)
            self.assertTrue(finalize(path, cand)["all_passed"])
            with self.assertRaises(ValueError): finalize(path, cand)

    def test_validation_budget_enforced(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "s.json"
            cand = ROOT / "examples/xor2.json"
            start(path, "xor2", "budgeted", "optional", budget=1)
            self.assertTrue(probe(path, cand)["all_passed"])
            with self.assertRaises(ValueError): probe(path, cand)

    def test_explanation_constraint(self):
        with tempfile.TemporaryDirectory() as d:
            cand = ROOT / "examples/xor2.json"
            start(Path(d) / "a.json", "xor2", "blind", "forbidden")
            with self.assertRaises(ValueError): finalize(Path(d) / "a.json", cand)
            data = load_candidate(cand)
            del data["explanation"]
            candidate_path = Path(d) / "candidate.json"
            candidate_path.write_text(json.dumps(data))
            start(Path(d) / "b.json", "xor2", "blind", "forbidden")
            self.assertTrue(finalize(Path(d) / "b.json", candidate_path)["all_passed"])

    def test_rejected_submissions_count(self):
        with tempfile.TemporaryDirectory() as d:
            path, bad, good = Path(d) / "s.json", Path(d) / "bad.json", ROOT / "examples/xor2.json"
            bad.write_text("{not json")
            start(path, "xor2", "budgeted", "optional", budget=2)
            with self.assertRaises(ValueError): probe(path, bad)  # uses one of the two probes
            self.assertTrue(probe(path, good)["all_passed"])
            with self.assertRaises(ValueError): probe(path, good)  # budget used up, nothing recorded
            with self.assertRaises(ValueError): finalize(path, bad)  # the failed final submission
            with self.assertRaises(ValueError): finalize(path, good)  # no second chance
            session = json.loads(path.read_text())
            self.assertEqual([p["error"] is not None for p in session["probes"]], [True, False])
            self.assertEqual(session["probes"][1]["sha256"], sha256(good.read_bytes()).hexdigest())
            self.assertTrue(session["finalized"])
            self.assertIsNone(session["final"]["result"])
            self.assertIn("JSONDecodeError", session["final"]["error"])

    def test_start_records_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            prompt = Path(d) / "prompt.txt"
            prompt.write_bytes(b"Return only weights.")
            session = start(Path(d) / "s.json", "xor2", "blind", "optional",
                            participant="pilot-01", model="example-model-1", prompt_file=prompt)
            self.assertEqual(session["provenance"]["model"], "example-model-1")
            self.assertEqual(session["provenance"]["prompt_sha256"], sha256(b"Return only weights.").hexdigest())


if __name__ == "__main__":
    unittest.main()
