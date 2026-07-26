import unittest

from kb_pipeline.data_prepare import filter_records_by_question_type


class DataPrepareFilterTests(unittest.TestCase):
    def test_filters_other_non_mathematical_records(self):
        records = [
            {"task_id": 1, "question_type": "Algebraic Operations"},
            {"task_id": 2, "question_type": "Other / Non-Mathematical"},
            {"task_id": 3, "question_type": "other / non-mathematical"},
        ]
        kept, filtered, stats = filter_records_by_question_type(
            records,
            ["Other / Non-Mathematical"],
        )
        self.assertEqual([item["task_id"] for item in kept], [1])
        self.assertEqual([item["task_id"] for item in filtered], [2, 3])
        self.assertEqual(sum(stats.values()), 2)


if __name__ == "__main__":
    unittest.main()
