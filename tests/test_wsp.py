from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

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
            path = Path(d) / "s.json"
            cand = ROOT / "examples/xor2.json"
            start(path, "xor2", "blind", "forbidden")
            with self.assertRaises(ValueError): finalize(path, cand)
            data = load_candidate(cand)
            del data["explanation"]
            candidate_path = Path(d) / "candidate.json"
            candidate_path.write_text(json.dumps(data))
            self.assertTrue(finalize(path, candidate_path)["all_passed"])


if __name__ == "__main__":
    unittest.main()
