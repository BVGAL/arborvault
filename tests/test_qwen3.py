import copy
import json
from pathlib import Path

import pytest

from arborvault.qwen3 import TransformersSelector, encode_completion, load_recipe


class Tokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, **kwargs):
        assert [m["role"] for m in messages] == ["system", "user"]
        assert kwargs == {"tokenize": True, "add_generation_prompt": True, "enable_thinking": False}
        return [10, 20, 30]

    def encode(self, content, add_special_tokens):
        assert content == '["0"]' and not add_special_tokens
        return [40, 50]


MESSAGES = [
    {"role": "system", "content": "Select evidence"},
    {"role": "user", "content": "Candidate zero"},
    {"role": "assistant", "content": '["0"]'},
]


def test_completion_starts_at_exact_non_thinking_inference_boundary_and_includes_eos():
    encoded = encode_completion(Tokenizer(), MESSAGES, 6)
    assert encoded == {"input_ids": [10, 20, 30, 40, 50, 99], "completion_length": 3}
    # Completion-only loss shifts the last prompt logit onto the first JSON token.
    count = encoded["completion_length"]
    assert encoded["input_ids"][-count:] == [40, 50, 99]
    assert encoded["input_ids"][-count - 1 : -1] == [30, 40, 50]
    with pytest.raises(ValueError, match="truncation"):
        encode_completion(Tokenizer(), MESSAGES, 5)
    with pytest.raises(ValueError, match="conversation"):
        encode_completion(Tokenizer(), MESSAGES[:-1], 6)


def test_recipe_rejects_unpinned_or_incompatible_training_settings(tmp_path):
    original = Path(__file__).resolve().parents[1] / "configs/qwen3-routing-v1.json"
    recipe = load_recipe(original)
    for key, value in (
        ("base_revision", "main"),
        ("thinking", True),
        ("epochs", 0),
        ("lora_dropout", 1),
    ):
        bad = copy.deepcopy(recipe)
        bad[key] = value
        file = tmp_path / "recipe.json"
        file.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            load_recipe(file)


def test_transformers_selector_rejects_invalid_budgets_without_loading_gpu_packages():
    with pytest.raises(ValueError, match="budget"):
        TransformersSelector(None, None, num_ctx=256, max_output_tokens=256)
    selector = TransformersSelector(None, None)
    assert selector.usage() == {"calls": 0, "failures": 0, "prompt_tokens": 0, "output_tokens": 0}
