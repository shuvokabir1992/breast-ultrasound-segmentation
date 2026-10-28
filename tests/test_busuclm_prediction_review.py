"""Small CPU checks of deterministic review selection and saved-mask validation."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from PIL import Image
import review_busuclm_predictions as review


def fixture_exports():
    rows = []
    for patient, dice in (("AAAA", .1), ("BBBB", .3), ("CCCC", .6), ("DDDD", .9)):
        rows.append({"id": patient + "_000", "patient_id": patient, "label": "benign",
                     "dice": dice, "foreground_fraction": .1})
    for patient, area in (("EEEE", .01), ("FFFF", .1), ("GGGG", .3), ("HESN", .2)):
        rows.append({"id": patient + "_000", "patient_id": patient, "label": "normal",
                     "dice": None, "foreground_fraction": area})
    return {seed: copy.deepcopy(rows) for seed in review.SEEDS}


class ReviewTests(unittest.TestCase):
    def test_selection_is_deterministic_and_contains_hesn_control(self):
        exports = fixture_exports()
        cases = review.choose_cases(exports)
        self.assertEqual(len(cases), 8)
        self.assertEqual(len({r["id"] for r in cases}), 8)
        self.assertEqual(cases[0]["id"], "AAAA_000")
        self.assertEqual(cases[-1]["id"], "HESN_000")
        for rows in exports.values():
            rows.reverse()
        self.assertEqual(cases, review.choose_cases(exports))

    def test_mismatched_seed_ids_and_labels_fail(self):
        exports = fixture_exports()
        exports[43].pop()
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            review.choose_cases(exports)
        exports = fixture_exports()
        exports[44][0]["label"] = "normal"
        with self.assertRaisesRegex(ValueError, "metadata differ"):
            review.choose_cases(exports)

    def test_native_binary_prediction_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mask.png"
            source = np.zeros((8, 12), dtype=np.uint8)
            source[2:5, 3:8] = 255
            Image.fromarray(source).save(path)
            self.assertTrue(np.array_equal(review.checked_prediction(path, (8, 12)), source > 0))
            with self.assertRaisesRegex(ValueError, "native dimensions"):
                review.checked_prediction(path, (12, 8))
            source[0, 0] = 127
            Image.fromarray(source).save(path)
            with self.assertRaisesRegex(ValueError, "not binary"):
                review.checked_prediction(path, (8, 12))

    def test_score_mismatch_is_not_silently_accepted(self):
        actual = {"dice": .5, "hd95_px": None, "predicted_pixels": 10}
        review.compare_scores(actual, dict(actual))
        saved = dict(actual, dice=.6)
        with self.assertRaisesRegex(ValueError, "pixel score differs"):
            review.compare_scores(actual, saved)


if __name__ == "__main__":
    unittest.main()
