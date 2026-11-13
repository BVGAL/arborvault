from types import SimpleNamespace

import pytest

from arborvault.routing_attention import sequential_sdpa


def test_sequential_attention_matches_batched_prefill_decode_and_per_beam_masks():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    torch.manual_seed(8)
    module = SimpleNamespace(training=False, num_key_value_groups=2, is_causal=True)
    key, value = torch.randn(4, 2, 7, 8), torch.randn(4, 2, 7, 8)
    for length in (1, 7):
        query = torch.randn(4, 4, length, 8)
        per_beam = torch.zeros(4, 1, length, 7)
        per_beam[1, :, :, -2:] = float("-inf")
        for mask in (None, per_beam, per_beam[:1]):
            expected, _ = sdpa_attention_forward(module, query, key, value, mask, scaling=0.25)
            actual, _ = sequential_sdpa(module, query, key, value, mask, scaling=0.25)
            torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="inference"):
        sequential_sdpa(module, query, key, value, None, dropout=0.1)


def test_attention_registration_preserves_sdpa_causal_generation():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    from arborvault.routing_attention import enable_sequential_sdpa

    torch.manual_seed(17)
    model = transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=128,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            eos_token_id=127,
            pad_token_id=127,
            attn_implementation="sdpa",
        )
    ).eval()
    inputs = torch.tensor([[1, 2, 3], [1, 2, 3]])
    with torch.inference_mode():
        expected = model(input_ids=inputs).logits
        try:
            enable_sequential_sdpa()
            actual = model(input_ids=inputs).logits
        finally:
            transformers.AttentionInterface.register("sdpa", sdpa_attention_forward)
    torch.testing.assert_close(actual, expected)
