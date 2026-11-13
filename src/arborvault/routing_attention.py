"""Inference-only SDPA batch scheduling for the optional Qwen beam experiment."""


def sequential_sdpa(module, query, key, value, attention_mask, dropout=0.0, **kwargs):
    import torch
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    if module.training or dropout != 0 or kwargs.get("cache") is not None:
        raise ValueError("Sequential routing attention supports inference without a paged cache")
    batch = query.shape[0]
    if key.shape[0] != batch or value.shape[0] != batch:
        raise ValueError("Attention batch dimensions differ")
    if batch == 1:
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, dropout=0, **kwargs
        )

    def slice_batch(tensor, index):
        if tensor is not None and tensor.ndim >= 3 and tensor.shape[0] == batch:
            return tensor[index : index + 1]
        return tensor

    outputs = []
    for index in range(batch):
        options = dict(kwargs)
        if "position_bias" in options:
            options["position_bias"] = slice_batch(options["position_bias"], index)
        output, _ = sdpa_attention_forward(
            module,
            query[index : index + 1],
            key[index : index + 1],
            value[index : index + 1],
            slice_batch(attention_mask, index),
            dropout=0,
            **options,
        )
        outputs.append(output)
    return torch.cat(outputs, dim=0), None


def enable_sequential_sdpa():
    """Opt in inside the dedicated experiment process; preserve SDPA mask construction."""
    from transformers import AttentionInterface

    # Replace the implementation under the existing key so Transformers retains its standard
    # SDPA causal/padding mask builder. Weights, dtype, context and decoding policy are unchanged.
    # Do not call this in a shared training/serving process: it is intentionally inference-only.
    AttentionInterface.register("sdpa", sequential_sdpa)
