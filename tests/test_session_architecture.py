"""Ordinary session architecture lock, not just generated-pilot checks."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from wsp.architecture import from_network
from wsp.network import load_candidate
from wsp.session import start, finalize

ROOT=Path(__file__).resolve().parents[1]


class SessionArchitectureTests(unittest.TestCase):
    def test_exact_architecture_passes(self):
        correct=ROOT/'examples'/'xor2.json'
        spec=from_network(load_candidate(correct)['network'])
        with tempfile.TemporaryDirectory() as d:
            session=start(Path(d)/'s.json','xor2','blind','optional',architecture=spec)
            self.assertEqual(session['architecture'],spec)
            self.assertEqual(len(session['architecture_sha256']),64)
            self.assertTrue(finalize(Path(d)/'s.json',correct)['all_passed'])

    def test_valid_but_different_hidden_width_is_rejected(self):
        correct=load_candidate(ROOT/'examples'/'xor2.json')
        spec=from_network(correct['network'])
        altered=deepcopy(correct)
        hidden,output=altered['network']['layers']
        hidden['weights'].append([0,0]);hidden['bias'].append(0)
        output['weights'][0].append(0)
        with tempfile.TemporaryDirectory() as d:
            candidate=Path(d)/'candidate.json'
            candidate.write_text(json.dumps(altered))
            start(Path(d)/'s.json','xor2','blind','optional',architecture=spec)
            with self.assertRaises(ValueError):finalize(Path(d)/'s.json',candidate)
            state=json.loads((Path(d)/'s.json').read_text())
            self.assertTrue(state['finalized'])
            self.assertIn('topology',state['final']['error'])
            self.assertEqual(len(state['final']['sha256']),64)

    def test_architecture_task_dim_mismatch_rejected_at_start(self):
        spec=from_network(load_candidate(ROOT/'examples'/'xor2.json')['network'])
        spec['layers'][0]['inputs']=3
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):start(Path(d)/'s.json','xor2','blind','optional',architecture=spec)


if __name__=='__main__':unittest.main()
