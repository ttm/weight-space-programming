"""Regression tests for direct synthesis and machine-checked evaluation rules."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from wsp.direct_synthesis import TASKS, CONDITIONS, examples, oracle, evaluate, report, score_bytes
from wsp.direct_sessions import start, submit

ROOT=Path(__file__).resolve().parents[1]
SUBMISSIONS=ROOT/'experiments'/'submissions'/'chatgpt-gpt6-2026-11-05'

class DirectSynthesisTests(unittest.TestCase):
    def test_all_task_prompts_omit_reference_weights(self):
        self.assertEqual(len(TASKS),10)
        self.assertEqual(set(CONDITIONS),{'B-E','B-W','P-E','P-W'})
        for task in TASKS.values():
            with self.subTest(task=task.id):
                public=task.public()
                self.assertNotIn('network',public)
                self.assertNotIn('reference_network',str(public))
                self.assertEqual(public['task'],task.id)
                self.assertEqual(public['architecture']['layers'][-1]['outputs'],1 if task.kind in ('binary','regression','unsatisfiable') else 3)

    def test_finite_oracles(self):
        self.assertEqual([oracle('parity3',list(x)) for x in ((0,0,0),(1,0,0),(1,1,0),(1,1,1))],[0,1,0,1])
        self.assertEqual([oracle('exactly2of4',[1]*i+[0]*(4-i)) for i in range(5)],[0,0,1,0,0])
        self.assertEqual([oracle('select3',[s,a,b]) for s,a,b in ((0,1,0),(1,1,0),(1,0,1),(0,0,1))],[1,0,1,0])

    def test_splits_and_salt(self):
        a=examples('clip2','test',salt='one')
        self.assertEqual(a,examples('clip2','test',salt='one'))
        self.assertNotEqual(a,examples('clip2','test',salt='two'))
        self.assertEqual(examples('clip2','validation',salt='one'),examples('clip2','validation',salt='two'))
        self.assertEqual(len(examples('parity3','test')),8)
        self.assertEqual(len(examples('exactly2of4','test')),16)
        self.assertEqual(len(examples('clip2','ood')),512)

    def test_all_first_attempts_are_replayed_without_modification(self):
        r=report(SUBMISSIONS)
        self.assertEqual(r['tasks'],10)
        self.assertEqual(r['passed'],9)
        self.assertEqual([x['task'] for x in r['records'] if not x['pass']],['xor2_impossible'])
        for x in r['records']:
            self.assertEqual(len(x['sha256']),64)

    def test_negative_control_fails_format_but_proof_is_mathematically_valid(self):
        submission=json.loads((SUBMISSIONS/'xor2_impossible.json').read_text())
        scored=score_bytes('xor2_impossible',json.dumps(submission).encode())
        self.assertFalse(scored['pass'])
        self.assertIn('Explanation is required',scored['error'])
        amended={**submission,'explanation':submission['reason']}
        self.assertTrue(evaluate('xor2_impossible',amended)['pass'])
        self.assertNotIn('network',amended)

    def test_wrong_topology_explanation_and_nonfinite_rejected(self):
        orig=json.loads((SUBMISSIONS/'parity3.json').read_text())
        net=deepcopy(orig)
        net['network']['layers'][0]['activation']='tanh'
        self.assertFalse(score_bytes('parity3',json.dumps(net).encode())['pass'])
        net=deepcopy(orig);net.pop('explanation')
        self.assertFalse(score_bytes('parity3',json.dumps(net).encode())['pass'])
        net=deepcopy(orig);net['network']['layers'][0]['weights'][0][0]=float('nan')
        self.assertFalse(score_bytes('parity3',json.dumps(net).encode())['pass'])
        net=deepcopy(orig);net['network']['layers'][0]['weights'].append([0,0,0]);net['network']['layers'][0]['bias'].append(0)
        net['network']['layers'][1]['weights'][0].append(0)
        self.assertFalse(score_bytes('parity3',json.dumps(net).encode())['pass'])
        self.assertFalse(score_bytes('parity3',b'```json\n{}\n```')['pass'])

    def test_controller_boundary_is_assessed(self):
        task='threeway_controller1'
        net=json.loads((SUBMISSIONS/(task+'.json')).read_text())
        self.assertTrue(evaluate(task,net)['pass'])
        bad=deepcopy(net)
        bad['network']['layers'][1 if len(bad['network']['layers'])>1 else 0]['bias'][0]=1
        self.assertFalse(evaluate(task,bad)['pass'])

    def test_session_budget_and_single_final_enforced(self):
        with tempfile.TemporaryDirectory() as d:
            session=Path(d)/'trial.json'
            start(session,'parity3',participant='unit-test')
            with self.assertRaises(ValueError):submit(session,SUBMISSIONS/'parity3.json',is_probe=True)
            final=submit(session,SUBMISSIONS/'parity3.json',is_probe=False)
            self.assertTrue(final['score']['pass'])
            with self.assertRaises(ValueError):submit(session,SUBMISSIONS/'parity3.json',is_probe=False)
        with tempfile.TemporaryDirectory() as d:
            session=Path(d)/'trial.json'
            start(session,'hat1',participant='unit-test')
            for _ in range(3):
                self.assertTrue(submit(session,SUBMISSIONS/'hat1.json',is_probe=True)['score']['pass'])
            with self.assertRaises(ValueError):submit(session,SUBMISSIONS/'hat1.json',is_probe=True)
            self.assertTrue(submit(session,SUBMISSIONS/'hat1.json',is_probe=False)['score']['pass'])

if __name__=='__main__':unittest.main()
