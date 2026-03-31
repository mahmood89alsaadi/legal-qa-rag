import sys
sys.path.insert(0, ".")

from src.evaluation.metrics import compute_accuracy, mcq_difficulty_analysis
from src.utils.prompt_builder import build_prompt


def test_accuracy():
    preds = [
        {"predicted_answer": "A", "gold_answer": "A"},
        {"predicted_answer": "B", "gold_answer": "A"},
        {"predicted_answer": "C", "gold_answer": "C"},
    ]
    assert compute_accuracy(preds) == 2/3, "Accuracy should be 2/3"


def test_difficulty_drop():
    mcq4 = [{"predicted_answer": "A", "gold_answer": "A"}] * 9 + \
           [{"predicted_answer": "B", "gold_answer": "A"}]
    mcq20 = [{"predicted_answer": "A", "gold_answer": "A"}] * 6 + \
            [{"predicted_answer": "B", "gold_answer": "A"}] * 4
    result = mcq_difficulty_analysis(mcq4, mcq20)
    assert result["mcq4_accuracy"] == 0.9
    assert result["mcq20_accuracy"] == 0.6
    assert result["drop"] == 0.3


def test_prompt_styles():
    options = ["Option A text", "Option B text", "Option C text", "Option D text"]
    for style in ["baseline", "adversarial", "auditor"]:
        prompt = build_prompt("What is the regulation?", options, style=style)
        assert "Question:" in prompt
        assert "Options:" in prompt
        assert "Answer:" in prompt


if __name__ == "__main__":
    test_accuracy()
    test_difficulty_drop()
    test_prompt_styles()
    print("All tests passed ✓")
