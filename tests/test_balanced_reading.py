"""Function-preservation and nondegenerate-scoring controls for reading experiments."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

from wsp.balanced_reading import FAMILIES, construct, make_bank, evaluate_predictions, _truth_table
from wsp.network import predict
from wsp.weight_reading import fingerprint, transform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from run_balanced_reading import run, decode, prompt_for
from verify_balanced_reading import verify


class BalancedBankTests(unittest.TestCase):
    def test_constructive_balance_and_gauge_fingerprint(self):
        for family in FAMILIES:
            for seed in range(50):
                with self.subTest(family=family,seed=seed):
                    original=construct(family,seed)
                    altered=transform(original,seed=seed+99)
                    self.assertEqual(_truth_table(original).count("1"),8)
                    self.assertEqual(_truth_table(altered),_truth_table(original))
                    self.assertEqual(fingerprint(original),fingerprint(altered))

    def test_same_topology_balanced_hidden_families(self):
        bank, key=make_bank(seed=1234,per_family=3)
        self.assertEqual(len(bank["items"]),len(FAMILIES)*3*2)
        self.assertEqual(len({i["id"] for i in bank["items"]}),len(bank["items"]))
        self.assertEqual({a["ones"] for a in key["answers"].values()},{8})
        self.assertEqual({json.dumps(i["architecture"],sort_keys=True) for i in bank["items"]}.__len__(),1)
        self.assertTrue(all("family" not in i and "truth_table" not in i for i in bank["items"]))

    def test_constant_baseline_is_not_interpretation(self):
        bank,key=make_bank(per_family=2)
        guesses={item["id"]:"0"*16 for item in bank["items"]}
        summary=evaluate_predictions(key,guesses)
        self.assertEqual(summary["exact"],0)
        self.assertEqual(summary["mean_bit_accuracy"],0.5)
        self.assertEqual(summary["pairs_agree"],8)
        self.assertEqual(summary["pairs_both_exact"],0)
        self.assertEqual(summary["pairs_trivial_agreement"],8)
        self.assertEqual(summary["constant_predictions"],16)

    def test_oracle_perfect_pair_success(self):
        bank,key=make_bank(per_family=2)
        guesses={ident:answer["truth_table"] for ident,answer in key["answers"].items()}
        summary=evaluate_predictions(key,guesses)
        self.assertEqual(summary["exact"],16)
        self.assertEqual(summary["pairs_both_exact"],8)
        self.assertEqual(summary["pairs_trivial_agreement"],0)
        self.assertEqual(summary["mean_bit_accuracy"],1.0)

    def test_output_validation(self):
        self.assertEqual(decode('{"truth_table":"0101010101010101"}'),("0101010101010101",None))
        for raw in ('```json\n{"truth_table":"0"}\n```','{"truth_table":"0101"}',
                    '{"truth_table":"0000000000000000","task":"x"}','not json'):
            with self.subTest(raw=raw):
                self.assertIsNone(decode(raw)[0])

    def test_study_runs_without_model_and_replays_without_inference(self):
        for participant,expected_exact in (("oracle",8),("zero",0)):
            report=run(seed=456,per_family=1,participant=participant)
            self.assertEqual(report["summary"]["exact"],expected_exact)
            self.assertEqual(verify(report)["exact"],expected_exact)
            bad=deepcopy(report)
            bad["records"][0]["raw"] += "x"
            with self.assertRaises(ValueError):verify(bad)

    def test_no_reference_or_pair_label_in_prompt(self):
        bank,key=make_bank(per_family=1)
        for item in bank["items"]:
            prompt=prompt_for(item,"direct")
            self.assertNotIn("truth_table\":\""+key["answers"][item["id"]]["truth_table"]+"\"",prompt)
            for field in ("group", "source_seed", "family"):
                self.assertNotIn(f'"{field}"',prompt)


if __name__ == "__main__":unittest.main()
