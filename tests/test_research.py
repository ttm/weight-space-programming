"""Generated-task, topology, invariance and participant-runner regression tests."""
from __future__ import annotations

from copy import deepcopy
import json
import os
import unittest
from unittest import mock

from wsp.architecture import from_network, check_network, architecture_hash
from wsp.baselines import search, compare
from wsp.generated import FAMILIES, make_task, evaluate_generated, examples
from wsp.pilot import run_trial, run_pilot, _decode_candidate, call_openai
from wsp.weight_reading import generate_bank, transform, fingerprint, binary_table, evaluate_guesses


class ArchitectureTests(unittest.TestCase):
    def test_wrong_width_depth_activation_rejected(self):
        task=make_task('dnf4',45)
        original=task.reference_network
        check_network(original,task.architecture)
        wrong=deepcopy(original)
        wrong['layers'][0]['weights'].append([0.,0.,0.,0.]);wrong['layers'][0]['bias'].append(0.)
        wrong['layers'][1]['weights'][0].append(0.)
        with self.assertRaisesRegex(ValueError,'topology'):check_network(wrong,task.architecture)
        wrong=deepcopy(original);wrong['layers'][0]['activation']='tanh'
        with self.assertRaisesRegex(ValueError,'topology'):check_network(wrong,task.architecture)
        wrong=deepcopy(original)
        wrong['layers'].insert(1,{'weights':[[int(i==j) for i in range(3)] for j in range(3)],'bias':[0.]*3,'activation':'linear'})
        with self.assertRaisesRegex(ValueError,'topology'):check_network(wrong,task.architecture)

    def test_invalid_architecture_does_not_reach_model(self):
        from wsp.architecture import validate_architecture
        for bad in ({'layers':[]},{'layers':[{'inputs':True,'outputs':1,'activation':'relu'}]},
                    {'layers':[{'inputs':2,'outputs':1,'activation':'evil'}]}):
            with self.assertRaises(ValueError):validate_architecture(bad)

    def test_topology_hash_stable(self):
        task=make_task('hinge1',23)
        self.assertEqual(architecture_hash(task.architecture),architecture_hash(from_network(task.reference_network)))


class GeneratedTests(unittest.TestCase):
    def test_constructive_witness_for_many_seeds(self):
        for family in FAMILIES:
            for seed in range(50):
                with self.subTest(family=family,seed=seed):
                    t=make_task(family,seed)
                    r=evaluate_generated(t,t.reference())
                    self.assertTrue(r['all_passed'],r)

    def test_samples_and_seed_salt(self):
        t=make_task('hinge1',31)
        self.assertEqual(examples(t,'test',salt='A'),examples(t,'test',salt='A'))
        self.assertNotEqual(examples(t,'test',salt='A'),examples(t,'test',salt='B'))
        self.assertEqual(examples(t,'validation',salt='A'),examples(t,'validation',salt='B'))

    def test_public_description_contains_no_witness(self):
        t=make_task('dnf4',31)
        p=t.public()
        self.assertNotIn('reference_network',p)
        self.assertNotIn('seed',p)
        self.assertNotIn('source_seed',p)
        self.assertEqual(p['architecture'],from_network(t.reference_network))

    def test_baselines_budget_is_exact_and_deterministic(self):
        t=make_task('l1ball2',31)
        a=search(t,method='random',budget=3,seed=2)
        self.assertEqual(a,search(t,method='random',budget=3,seed=2))
        self.assertEqual(a['validation_calls'],3)
        self.assertEqual(search(t,method='hill_climb',budget=2,seed=2)['validation_calls'],2)
        r=compare(seeds=1,budgets=(1,))
        self.assertEqual(len(r['records']),8)


class ReadingTests(unittest.TestCase):
    def test_preserved_function_and_canonical_fingerprint(self):
        for seed in range(40):
            t=make_task('dnf4',seed)
            original=t.reference_network
            modified=transform(original,seed=seed)
            self.assertEqual(binary_table(original),binary_table(modified))
            self.assertEqual(fingerprint(original),fingerprint(modified))

    def test_bank_matched_architecture_and_scoring(self):
        bank,key=generate_bank(count=8,seed=1)
        self.assertEqual(bank,keyless:=generate_bank(count=8,seed=1)[0])
        self.assertEqual(len({json.dumps(v['architecture'],sort_keys=True) for v in bank['challenges']}),1)
        guesses={ident:answer['truth_table'] for ident,answer in key['answers'].items()}
        report=evaluate_guesses(bank,key,guesses)
        self.assertEqual(report['exact'],8)
        self.assertEqual(report['mean_bit_accuracy'],1)
        report=evaluate_guesses(bank,key,{})
        self.assertEqual(report['exact'],0)


class PilotTests(unittest.TestCase):
    def test_oracle_and_random_stub_conditions(self):
        t=make_task('dnf4',301)
        for mode,budget in (('blind',0),('budgeted',3)):
            for explanation in ('required','forbidden'):
                result=run_trial(t,participant='oracle_stub',mode=mode,explanation=explanation,budget=budget)
                self.assertTrue(result['passed'],result)
                self.assertEqual(len(result['probes']),budget)
                self.assertEqual(result['final']['score']['all_passed'],True)
                self.assertEqual(len(result['final']['candidate_sha256']),64)
        a=run_trial(t,participant='random_stub',mode='budgeted',explanation='forbidden',budget=3)
        self.assertEqual(a,run_trial(t,participant='random_stub',mode='budgeted',explanation='forbidden',budget=3))

    def test_openai_is_opt_in(self):
        t=make_task('hinge1',22)
        with mock.patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(RuntimeError,'OPENAI_API_KEY'):
                run_trial(t,participant='openai',model='user-selected-model')

    def test_bad_candidate_json_is_not_repaired(self):
        with self.assertRaises(json.JSONDecodeError):_decode_candidate('```json\n{}\n```')


if __name__ == '__main__': unittest.main()
