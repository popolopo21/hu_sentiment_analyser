import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..\..')))

from src.sentimentClassifier.data.preprocess.target_cleaner import ScoreCleaner, RANGE_3, RANGE_5, RANGE_10

class TestScoreCleaner(unittest.TestCase):

    def test_init(self):
        # Test valid initializations
        cleaner = ScoreCleaner(input_label_range=RANGE_3, output_label_range=RANGE_5)
        self.assertEqual(cleaner.input_label_range, RANGE_3)
        self.assertEqual(cleaner.output_label_range, RANGE_5)

        # Test invalid initializations
        with self.assertRaises(ValueError):
            ScoreCleaner(input_label_range=7, output_label_range=RANGE_5)

    def test_convert_from_3_to_5(self):
        cleaner = ScoreCleaner(input_label_range=RANGE_3, output_label_range=RANGE_5)
        self.assertEqual(cleaner._convert_from_3_to_5(1), 0)
        self.assertEqual(cleaner._convert_from_3_to_5(2), 2)
        self.assertEqual(cleaner._convert_from_3_to_5(3), 4)

    def test_convert_from_10_to_3(self):
        cleaner = ScoreCleaner(input_label_range=RANGE_3, output_label_range=RANGE_10)
        self.assertEqual(cleaner._convert_from_10_to_3(10), 2)
        self.assertEqual(cleaner._convert_from_10_to_3(6), 1)
        self.assertEqual(cleaner._convert_from_10_to_3(1), 0)

    def test_convert(self):
        cleaner = ScoreCleaner(input_label_range=RANGE_3, output_label_range=RANGE_5)
        self.assertEqual(cleaner._convert(2), 2)  # Test no conversion needed
        self.assertEqual(cleaner._convert(3), 4)  # Test conversion from 3 to 5

    def test_clean_rating(self):
        cleaner = ScoreCleaner(input_label_range=RANGE_3, output_label_range=RANGE_3, delimiter=",", rating_at_end=True, clean_rating=True)
        self.assertEqual(cleaner._clean_rating("Review,3"), 3)

    def test_process(self):
        cleaner = ScoreCleaner(input_label_range=RANGE_3, output_label_range=RANGE_5,rating_at_end=False, clean_rating=True, delimiter="/")
        self.assertEqual(cleaner.process(["2/Good", "3/Excellent"]), [2, 4])
