"""Small auditable LoRA adaptation for the pinned MiniLM classifier.

Research-only optional torch dependency. Base encoder matrices stay frozen;
rank-8 updates, pooler and output classifier are trained. Saved JSON contains only
trainable tensors, never a pickle or the downloaded base model.
"""

import hashlib
import math
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .support_probe import encode_batch, model_input


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank <= 0:
            raise ValueError("Invalid LoRA target")
        self.base = base
        self.base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * self.scale


def attach(model, config):
    model.requires_grad_(False)
    targets = [
        name
        for name, module in model.named_modules()
        if name.startswith("bert.encoder.layer.")
        and isinstance(module, nn.Linear)
        and any(name.endswith("." + suffix) for suffix in config["lora_modules"])
    ]
    if len(targets) != 36:
        raise ValueError("Expected exactly six linear targets per each of six BERT layers")
    for name in targets:
        parent, field = name.rsplit(".", 1)
        module = model.get_submodule(parent)
        setattr(
            module,
            field,
            LoRALinear(getattr(module, field), config["lora_rank"], config["lora_alpha"]),
        )
    model.bert.pooler.requires_grad_(True)
    model.classifier.requires_grad_(True)
    return targets


def trainable_state(model):
    return {
        name: tensor.detach().cpu().tolist()
        for name, tensor in model.named_parameters()
        if tensor.requires_grad
    }


def restore(model, state):
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(parameters) != set(state):
        raise ValueError("Adapter tensor keys differ")
    with torch.no_grad():
        for name, parameter in parameters.items():
            tensor = torch.tensor(state[name], dtype=parameter.dtype)
            if tensor.shape != parameter.shape or not torch.isfinite(tensor).all():
                raise ValueError("Adapter shape or finite-value check failed")
            parameter.copy_(tensor)


def frozen_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.named_parameters():
        if not tensor.requires_grad:
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def group_batches(rows, config):
    """Balanced stratum cycle, independent seeded sampling of complete groups."""
    if {r["role"] for r in rows} != {"fit"}:
        raise ValueError("Training sampler only accepts fit examples")
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["group_id"]].append(index)
    by_stratum = {s: [] for s in config["strata"]}
    for indices in groups.values():
        stratum = rows[indices[0]]["phenomenon"]
        if any(rows[i]["phenomenon"] != stratum for i in indices):
            raise ValueError("Group crosses strata")
        expected = 2 if stratum == "human" else 4
        if len(indices) != expected or sum(rows[i]["answerable"] for i in indices) != expected // 2:
            raise ValueError("Incomplete balanced training group")
        by_stratum[stratum].append(indices)
    if any(not v for v in by_stratum.values()):
        raise ValueError("Empty training stratum")
    rng = np.random.default_rng(config["seed"])
    for step in range(max(config["checkpoints"])):
        stratum = config["strata"][step % len(config["strata"])]
        source = by_stratum[stratum]
        count = config["batch_size"] // len(source[0])
        chosen = rng.choice(len(source), size=count, replace=False)
        yield stratum, [i for g in chosen for i in source[g]]


def ranking_pairs(rows):
    """Match opposite-label contexts for exactly the same question in a group."""
    by_question = defaultdict(list)
    for i, row in enumerate(rows):
        if row["suite"] == "contrast":
            by_question[(row["group_id"], row["question"])].append(i)
    pairs = []
    for indices in by_question.values():
        positive = [i for i in indices if rows[i]["answerable"]]
        negative = [i for i in indices if not rows[i]["answerable"]]
        if len(positive) != 1 or len(negative) != 1:
            raise ValueError("Ranking requires a complete counterfactual question pair")
        pairs.append((positive[0], negative[0]))
    return pairs


def objective(logits, rows, config):
    labels = torch.tensor([r["answerable"] for r in rows], dtype=logits.dtype, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, labels)
    pairs = ranking_pairs(rows)
    rank = logits.sum() * 0
    if pairs:
        pos, neg = zip(*pairs, strict=True)
        rank = F.softplus(config["ranking_margin"] - logits[list(pos)] + logits[list(neg)]).mean()
    return bce + config["ranking_weight"] * rank, bce, rank


def lr_multiplier(step, config):
    if step <= config["warmup_steps"]:
        return step / config["warmup_steps"]
    progress = (step - config["warmup_steps"]) / (
        max(config["checkpoints"]) - config["warmup_steps"]
    )
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))


def predict(model, tokenizer, rows, config, deadline, *, features=False, mode="context"):
    model.eval()
    logits, vectors = [], []
    with torch.inference_mode():
        for offset in range(0, len(rows), config["batch_size"]):
            deadline()
            batch = rows[offset : offset + config["batch_size"]]
            _, tensors = encode_batch(
                tokenizer, [model_input(r, mode) for r in batch], config["max_length"]
            )
            result = model(**tensors, output_hidden_states=features)
            logits.extend(result.logits[:, 0].float().tolist())
            if features:
                vectors.extend(result.hidden_states[-1][:, 0].float().tolist())
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite model output")
    return np.asarray(logits), np.round(np.asarray(vectors), 7) if features else None
