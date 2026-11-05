import copy
import json
from pathlib import Path
import unittest

from wsp.provenance import index, validate, OWNER_REPORTED, UNKNOWN_PATCH, EXCEPTION_MINIMAL

ROOT=Path(__file__).resolve().parents[1]

class ProvenanceTests(unittest.TestCase):
    def test_existing_artifacts_have_auditable_provenance(self):
        result=index(ROOT)
        self.assertEqual(result['total'],14)
        self.assertEqual(result['structured'],13)
        paths={r['path'] for r in result['artifacts']}
        self.assertIn(EXCEPTION_MINIMAL,paths)
        for record in result['artifacts']:
            if record['path']==EXCEPTION_MINIMAL:
                self.assertIsNone(record['provenance'])
            else:
                validate(record['provenance'])
                self.assertEqual(len(record['sha256']),64)

    def test_unconfirmed_patch_remains_unknown(self):
        self.assertIsNone(UNKNOWN_PATCH['model'])
        self.assertEqual(UNKNOWN_PATCH['creator_type'],'unknown')
        self.assertIsNone(OWNER_REPORTED['generated_at'])
        self.assertIsNone(OWNER_REPORTED['prompt_reference'])

    def test_missing_provenance_fields_fail(self):
        missing=copy.deepcopy(OWNER_REPORTED)
        missing.pop('tools_used')
        with self.assertRaises(ValueError):validate(missing)
        wrong=copy.deepcopy(OWNER_REPORTED)
        wrong['tools_used']='python'
        with self.assertRaises(ValueError):validate(wrong)

if __name__=='__main__':unittest.main()
