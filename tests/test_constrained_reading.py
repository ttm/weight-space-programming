"""Controls for post-hoc constrained-output method, no downloads or API calls."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from run_constrained_reading import run,prompt_for
from verify_constrained_reading import verify
from wsp.balanced_reading import make_bank


class ConstrainedReaderTests(unittest.TestCase):
    def test_oracle_and_constant_controls(self):
        oracle=run(seed=20261105,per_family=1,participant="oracle")
        zero=run(seed=20261105,per_family=1,participant="zero")
        self.assertEqual(verify(oracle)["both_exact"] if "both_exact" in verify(oracle) else verify(oracle)["pairs_both_exact"],4)
        self.assertEqual(verify(zero)["exact"],0)
        self.assertEqual(verify(zero)["mean_bit_accuracy"],.5)
        self.assertEqual(verify(oracle)["conditional_binary_token_brier"],0)
        self.assertEqual(verify(zero)["conditional_binary_token_brier"],.5)

    def test_raw_hash_and_score_tampering_fails(self):
        report=run(seed=1,per_family=1,participant="zero")
        modified=deepcopy(report)
        modified["records"][0]["raw"]="1"*16
        with self.assertRaises(ValueError):verify(modified)
        modified=deepcopy(report)
        modified["summary"]["exact"]=16
        with self.assertRaises(ValueError):verify(modified)

    def test_no_input_truth_table_shown(self):
        bank,key=make_bank(seed=20261105,per_family=1)
        for item in bank["items"]:
            prompt=prompt_for(item)
            self.assertNotIn("0000 0001 0010 0011",prompt)
            self.assertNotIn("truth_table",prompt)
            self.assertNotIn(key["answers"][item["id"]]["truth_table"],prompt)


if __name__=="__main__":unittest.main()
