"""Round two session mechanics: mandatory failure counting and raw byte retention."""
from pathlib import Path
import json
import tempfile
import unittest
from wsp.direct_sessions_v2 import start,submit
from wsp.direct_challenges_v2 import TASKS

ROOT=Path(__file__).resolve().parents[1]
SUBS=ROOT/'experiments/submissions/chatgpt-gpt6-2026-11-05-round2'

class SessionRoundTwoTests(unittest.TestCase):
  def test_blind_prohibits_probe_and_retains_final(self):
    with tempfile.TemporaryDirectory() as d:
      p=Path(d)/'trial.json'
      start(p,'parity5',participant='other model',model_id='model-specified')
      with self.assertRaises(ValueError):submit(p,SUBS/'parity5.json',probe=True)
      record=submit(p,SUBS/'parity5.json',probe=False)
      self.assertTrue(record['score']['passed'])
      self.assertEqual((p.parent/(p.stem+'-artifacts')/record['name']).read_bytes(),(SUBS/'parity5.json').read_bytes())
      with self.assertRaises(ValueError):submit(p,SUBS/'parity5.json',probe=False)
  def test_three_probes_invalid_count_and_final(self):
    with tempfile.TemporaryDirectory() as d:
      p=Path(d)/'trial.json'
      start(p,'xor_pairs_and',participant='other model')
      bad=Path(d)/'wrong.json';bad.write_text('{oops')
      for _ in range(3):self.assertFalse(submit(p,bad,probe=True)['score']['passed'])
      with self.assertRaises(ValueError):submit(p,bad,probe=True)
      final=submit(p,SUBS/'xor_pairs_and.json',probe=False)
      self.assertTrue(final['score']['passed'])
      state=json.loads(p.read_text());self.assertEqual(len(state['probes']),3)
  def test_explanation_is_enforced_on_final(self):
    with tempfile.TemporaryDirectory() as d:
      p=Path(d)/'trial.json';start(p,'median3',participant='other model')
      bad=json.loads((SUBS/'median3.json').read_text());del bad['explanation']
      path=Path(d)/'candidate.json';path.write_text(json.dumps(bad))
      self.assertFalse(submit(p,path,probe=False)['score']['passed'])
      with self.assertRaises(ValueError):submit(p,SUBS/'median3.json',probe=False)

if __name__=='__main__':unittest.main()
