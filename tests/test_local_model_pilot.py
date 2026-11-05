import json
import unittest
from wsp.generated import make_task
from wsp.pilot import _prompt
from scripts.run_local_hf_pilot import score_raw,run

class LocalModelSmokeTests(unittest.TestCase):
    def test_dry_run_has_four_feasible_tasks_no_gpu_required(self):
        report=run(dry_run=True)
        self.assertEqual(len(report['tasks']),4)
        self.assertTrue(all(r['all_passed'] for r in report['reference_results']))

    def test_strict_scoring_without_repair_of_bad_json(self):
        t=make_task('threshold4',20261105)
        good=t.reference()
        result=score_raw(t,json.dumps(good))
        self.assertTrue(result['result']['all_passed'])
        bad=score_raw(t,'```json\n{}\n```')
        self.assertIsNone(bad['result'])
        self.assertIn('JSONDecodeError',bad['error'])

if __name__=='__main__':unittest.main()
