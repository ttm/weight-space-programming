"""Reproducibility and protocol regression tests for direct weight synthesis round 2."""
from copy import deepcopy
from pathlib import Path
import json
import unittest
from wsp.direct_challenges_v2 import TASKS,CONDITIONS,examples,oracle,evaluate,score_bytes,report
from wsp.architecture import check_network
from wsp.network import forward

BASE=Path(__file__).resolve().parents[1]
DIR=BASE/'experiments/submissions/chatgpt-gpt6-2026-11-05-round2'

class RoundTwoTests(unittest.TestCase):
    def _submission(self,task):return json.loads((DIR/(task+'.json')).read_text())

    def test_all_fourteen_are_present(self):
        self.assertEqual(len(TASKS),14)
        self.assertEqual(set(TASKS),{p.stem for p in DIR.glob('*.json') if p.stem!='manifest'})

    def test_first_submissions_are_immutable_in_reporting(self):
        result=report(DIR)
        self.assertEqual((result['tasks'],result['passed'],result['failed']),(14,13,1))
        self.assertEqual([r['task'] for r in result['rows'] if not r['passed']],['annulus_l1_2'])

    def test_non_annulus_math_behavior(self):
        for task in TASKS:
            if task=='annulus_l1_2':continue
            with self.subTest(task=task):self.assertTrue(evaluate(task,self._submission(task))['pass'])

    def test_multilabel_addition_exhaustive(self):
        for task in ('full_adder3','compare_two_2bit'):
            s=self._submission(task)
            for x,y in examples(task,'test'):
                answer=[int(v>0) for v in forward(s['network'],x)]
                self.assertEqual(answer,y,(task,x))

    def test_strict_topology_and_explanation_rules(self):
        task='max3'
        s=self._submission(task)
        bad=deepcopy(s)
        bad['network']['layers'][1]['activation']='tanh'
        self.assertFalse(score_bytes(task,json.dumps(bad).encode())['passed'])
        self.assertIn('architecture',score_bytes(task,json.dumps(bad).encode())['error'])
        missing=deepcopy(s);missing['explanation']='Forbidden'
        self.assertFalse(score_bytes(task,json.dumps(missing).encode())['passed'])
        required=deepcopy(self._submission('median3'));del required['explanation']
        self.assertFalse(score_bytes('median3',json.dumps(required).encode())['passed'])

    def test_boundary_regression_and_classification(self):
        for task in ('trapezoid1','max3','median3','rotated_quadrants4','fourlevel1','hinge2'):
            with self.subTest(task=task):
                self.assertTrue(evaluate(task,self._submission(task),splits=('edge',))['pass'])

    def test_failure_mechanism_and_labeled_posthoc_fix(self):
        first=self._submission('annulus_l1_2')
        revised=json.loads((BASE/'experiments/revisions/gpt6_round2_annulus_posthoc.json').read_text())
        self.assertFalse(evaluate('annulus_l1_2',first)['pass'])
        self.assertTrue(evaluate('annulus_l1_2',revised)['pass'])
        for seed in range(10):
            self.assertTrue(evaluate('annulus_l1_2',revised,salt=f'round2-holdout-{seed}')['pass'])
        points=[([2.7,1.3],0),([0.75,0.0],1),([0.5,0.0],0),([2.0,0.0],0)]
        for x,label in points:
            self.assertEqual(int(forward(revised['network'],x)[0]>0),label)

    def test_splits_seeded_and_exhaustive(self):
        self.assertEqual(len(examples('parity5','test')),32)
        self.assertEqual(len(examples('exactly3of6','test')),64)
        self.assertEqual(len(examples('mux6','test')),64)
        self.assertEqual(len(examples('full_adder3','test')),8)
        self.assertEqual(len(examples('compare_two_2bit','test')),16)
        self.assertEqual(examples('median3','test',salt='A'),examples('median3','test',salt='A'))
        self.assertNotEqual(examples('median3','test',salt='A'),examples('median3','test',salt='B'))
        self.assertEqual(examples('median3','validation',salt='A'),examples('median3','validation',salt='B'))

    def test_impossibility_certificate_not_confused_with_proof_verification(self):
        certificate=evaluate('parity3_affine_impossible',self._submission('parity3_affine_impossible'))
        self.assertEqual(certificate['certificate'],'submitted, proof not machine verified')

if __name__=='__main__': unittest.main()
