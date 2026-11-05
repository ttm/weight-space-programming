"""Independent finite-difference gradient checks and baseline bookkeeping."""
from __future__ import annotations
from copy import deepcopy
import math
import unittest

from wsp.generated import make_task, examples
from wsp.gradient import loss_and_gradient, train
from wsp.baselines import sample_network
import random


class GradientTests(unittest.TestCase):
    def test_scalar_derivatives_against_finite_differences(self):
        for family,x in [('hinge1',[2.35]),('l1ball2',[0.137,0.321]),('threshold4',[1.,0.,1.,1.])]:
            with self.subTest(family=family):
                task=make_task(family,4)
                net=sample_network(task.architecture,random.Random(1234),scale=.2)
                y=examples(task,'train')[0][1]
                _,derivatives=loss_and_gradient(net,x,y,task.kind)
                delta=1e-5
                for li,layer in enumerate(net['layers']):
                    for key in ('weights','bias'):
                        if key=='weights':
                            for row_idx,row in enumerate(layer[key]):
                                for col_idx in range(len(row)):
                                    original=row[col_idx]
                                    row[col_idx]=original+delta
                                    plus=loss_and_gradient(net,x,y,task.kind)[0]
                                    row[col_idx]=original-delta
                                    minus=loss_and_gradient(net,x,y,task.kind)[0]
                                    row[col_idx]=original
                                    numerical=(plus-minus)/(2*delta)
                                    analytical=derivatives['layers'][li][key][row_idx][col_idx]
                                    self.assertAlmostEqual(numerical,analytical,places=5)
                        else:
                            for i in range(len(layer[key])):
                                original=layer[key][i]
                                layer[key][i]=original+delta
                                plus=loss_and_gradient(net,x,y,task.kind)[0]
                                layer[key][i]=original-delta
                                minus=loss_and_gradient(net,x,y,task.kind)[0]
                                layer[key][i]=original
                                numerical=(plus-minus)/(2*delta)
                                analytical=derivatives['layers'][li][key][i]
                                self.assertAlmostEqual(numerical,analytical,places=5)

    def test_adam_respects_update_budget_and_is_deterministic(self):
        task=make_task('threshold4',9)
        first=train(task,steps=200,seed=4)
        self.assertEqual(first,train(task,steps=200,seed=4))
        self.assertEqual(first['forward_backward_pairs'],200)
        self.assertEqual(first['training_examples'],16)
        self.assertGreaterEqual(first['final']['splits']['test']['value'],0)
        self.assertLessEqual(first['final']['splits']['test']['value'],1)

    def test_adam_reduces_threshold_loss_from_random_init(self):
        task=make_task('threshold4',0)
        short=train(task,steps=100,seed=0)
        long=train(task,steps=3000,seed=0)
        self.assertGreaterEqual(long['final']['splits']['test']['value'],short['final']['splits']['test']['value'])

if __name__=='__main__':unittest.main()
