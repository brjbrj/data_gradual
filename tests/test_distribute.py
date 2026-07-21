import unittest

from kb_pipeline.distribute import distribute_mastery_records


def _records(n=8):
    return [
        {
            "task_id": idx,
            "mastery": idx / max(1, n - 1),
            "question": f"q{idx}",
            "answer": str(idx),
        }
        for idx in range(n)
    ]


def _sources(n=8):
    return {
        idx: {
            "task_id": idx,
            "question_type": "Algebraic Operations" if idx % 2 else "Word Problems",
        }
        for idx in range(n)
    }


class DistributeMasteryRecordsTests(unittest.TestCase):
    def test_legacy_policy_preserves_positive_minimum(self):
        outputs = distribute_mastery_records(
            _records(),
            _sources(),
            target_multiplier=6,
            n_min=1,
            n_max=10,
            allocation_policy="legacy",
        )
        counts = [item["target_count"] for item in outputs]
        self.assertEqual(sum(counts), 40)
        self.assertGreaterEqual(min(counts), 1)
        self.assertLessEqual(max(counts), 10)

    def test_threshold_marginal_uses_zero_or_active_threshold(self):
        outputs = distribute_mastery_records(
            _records(12),
            _sources(12),
            target_multiplier=4,
            n_min=0,
            n_max=12,
            allocation_policy="threshold_marginal",
            active_threshold=5,
            marginal_alpha=0.7,
            threshold_boost=2.0,
        )
        counts = [item["target_count"] for item in outputs]
        self.assertLessEqual(sum(counts), 36)
        self.assertTrue(all(count == 0 or count >= 5 for count in counts))
        self.assertLessEqual(max(counts), 12)
        self.assertTrue(any(count == 0 for count in counts))
        self.assertTrue(any(count >= 5 for count in counts))

    def test_threshold_marginal_default_policy_marker(self):
        outputs = distribute_mastery_records(
            _records(),
            _sources(),
            target_multiplier=5,
            n_min=0,
            n_max=20,
            allocation_policy="threshold_marginal",
            active_threshold=5,
        )
        self.assertEqual(
            {item["allocation_policy"] for item in outputs},
            {"threshold_marginal"},
        )


if __name__ == "__main__":
    unittest.main()
