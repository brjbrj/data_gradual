import unittest

from kb_pipeline.step_refine import _step_quality_issue


class StepRefineQualityTests(unittest.TestCase):
    def test_rejects_command_first_calculate_wording(self):
        steps = [
            "Step 1: First, calculate the total meals by multiplying dogs by meals per day: 40 * 2 = 80 meals.",
            "Step 2: Then, find the remaining meals: 80 - 45 = 35 meals.",
        ]
        self.assertIn("command-first", _step_quality_issue(steps))

    def test_accepts_problem_anchored_reasoning_steps(self):
        steps = [
            "Step 1: From the problem, there are 40 dogs and each eats 2 meals a day, so the total meals needed per day is 40 * 2 = 80 meals.",
            "Step 2: Since there are 3 caretakers and each prepares 15 meals daily, the total meals prepared per day is 3 * 15 = 45 meals.",
            "Step 3: Combining these quantities, the remaining meals needed per day is 80 - 45 = 35 meals.",
        ]
        self.assertEqual(_step_quality_issue(steps), "")


if __name__ == "__main__":
    unittest.main()
