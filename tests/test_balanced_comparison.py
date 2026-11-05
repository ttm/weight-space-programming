"""Paired-arm comparison must count underlying functions, not duplicate views."""
from __future__ import annotations
from pathlib import Path
import sys
import unittest
from copy import deepcopy

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from compare_balanced_reading import compare
from run_balanced_reading import run


class ComparisonTests(unittest.TestCase):
    def test_constant_conditions_have_no_effect(self):
        a=run(seed=20261105,per_family=1,condition="direct",participant="zero")
        b=run(seed=20261105,per_family=1,condition="deliberate",participant="zero")
        result=compare(a,b)
        self.assertEqual(result["n_independent_functions"],4)
        self.assertEqual(result["n_network_presentations"],8)
        self.assertEqual(result["conditions"]["direct"]["both_exact_pairs"],0)
        self.assertEqual(result["discordant_functions"],{"only_direct":0,"only_deliberate":0})

    def test_nonpaired_comparisons_are_rejected(self):
        a=run(seed=20261105,per_family=1,condition="direct",participant="zero")
        b=run(seed=20261105,per_family=1,condition="deliberate",participant="zero")
        bad=deepcopy(b)
        bad["model"]="other-model"
        with self.assertRaises(ValueError):compare(a,bad)
        bad=deepcopy(b)
        bad["seed"]+=1
        with self.assertRaises(ValueError):compare(a,bad)


if __name__=="__main__":unittest.main()
