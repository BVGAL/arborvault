import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from arborvault.qwen3 import TransformersSelector, completion_loss, encode_completion, load_recipe


class Tokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, **kwargs):
        assert [m["role"] for m in messages] == ["system", "user"]
        assert kwargs == {
            "tokenize": True,
            "return_dict": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
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


def test_completion_only_logits_match_full_masked_loss_and_gradients():
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    logits = torch.randn(1, 5, 8, requires_grad=True)

    class Model:
        device = "cpu"

        def __call__(self, input_ids, use_cache, logits_to_keep):
            assert not use_cache and input_ids.tolist() == [[0, 1, 2, 3, 4]]
            return SimpleNamespace(logits=logits[:, -logits_to_keep:])

    reduced, count = completion_loss(
        Model(), {"input_ids": [0, 1, 2, 3, 4], "completion_length": 2}
    )
    # Reference: ordinary full-sequence causal loss with prompt labels ignored.
    labels = torch.tensor([[-100, -100, -100, 3, 4]])
    full = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 8), labels[:, 1:].reshape(-1)
    )
    assert count == 2
    torch.testing.assert_close(reduced, full)
    reduced_gradient = torch.autograd.grad(reduced, logits, retain_graph=True)[0]
    full_gradient = torch.autograd.grad(full, logits)[0]
    torch.testing.assert_close(reduced_gradient, full_gradient)
