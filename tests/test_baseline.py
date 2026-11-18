import unittest

import numpy as np
import pandas as pd
import torch

from lib import baseline


class EncodeRequestsTests(unittest.TestCase):
    def test_shifts_pads_and_truncates(self) -> None:
        encoded = baseline.encode_requests_as_ascii_codes(pd.Series(["AB", "é" + "x" * 200]), max_characters=4)
        self.assertEqual(encoded.tolist()[0], [ord("A") + 1, ord("B") + 1, 0, 0])
        self.assertEqual(encoded[1, 0], ord("?") + 1)  # Non-ASCII is replaced, not dropped.
        self.assertEqual(encoded.shape, (2, 4))


class CnnBiLstmTests(unittest.TestCase):
    def test_outputs_one_logit_per_class(self) -> None:
        classifier = baseline.CnnBiLstmClassifier(class_count=3)
        codes = torch.randint(0, baseline.CHARACTER_VOCABULARY_SIZE, (2, baseline.SEQUENCE_MAX_CHARACTERS))
        self.assertEqual(tuple(classifier(codes).shape), (2, 3))


class CompareWithPaperTests(unittest.TestCase):
    def test_gap_and_tolerance(self) -> None:
        comparison = baseline.compare_with_paper("DS_Augmented_v2.csv", {"accuracy": 97.0, "macro_f1": 50.0})
        self.assertAlmostEqual(comparison.loc["accuracy", "gap_points"], -0.42)
        self.assertTrue(comparison.loc["accuracy", "within_5_points"])
        self.assertFalse(comparison.loc["macro_f1", "within_5_points"])  # 7.61 points below.

    def test_unknown_dataset_is_rejected(self) -> None:
        with self.assertRaises(KeyError):
            baseline.compare_with_paper("other.csv", {"accuracy": 0, "macro_f1": 0})


class ClassWeightTests(unittest.TestCase):
    def test_rare_classes_weigh_more(self) -> None:
        weights = baseline.balanced_class_weights(np.array([0, 0, 0, 1]), 2)
        self.assertGreater(weights[1], weights[0])


if __name__ == "__main__":
    unittest.main()
