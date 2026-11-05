import json
import unittest
from scripts.run_local_hf_reading import challenges,score_raw,run,prompt_for

class ReadingSmokeTests(unittest.TestCase):
    def test_matched_pairs_and_correct_answer_scoring(self):
        self.assertTrue(run(dry_run=True)['all_matched'])
        entries=challenges()
        self.assertEqual(len(entries),6)
        self.assertEqual(len({e['id'] for e in entries}),3)
        for e in entries:
            self.assertTrue(score_raw(json.dumps({'truth_table':e['answer']}),e['answer'])['exact'])

    def test_invalid_truth_tables_fail(self):
        self.assertFalse(score_raw('"111"','0'*16)['valid'])
        self.assertFalse(score_raw(json.dumps({'truth_table':'1'*15}),'0'*16)['valid'])

if __name__=='__main__':unittest.main()
