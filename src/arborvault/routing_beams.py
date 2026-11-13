"""Development-triggered v2 ablation: bounded union of constrained beam hypotheses."""

import hashlib
import json
import math

from .ollama import decode_selection
from .routing_v2 import QwenSelector, SelectionGrammar, fit_messages

POLICY = {
    "version": "routing-v2-beams-1",
    "num_beams": 4,
    "length_penalty": 1.0,
    "early_stopping": True,
    "union_width": 4,
    "abstention": "empty highest-scoring hypothesis",
}
POLICY_SHA256 = hashlib.sha256(json.dumps(POLICY, sort_keys=True).encode()).hexdigest()


def combine_hypotheses(hypotheses, labels, width):
    """Validate every finite-scored hypothesis; preserve score order and explicit abstention."""
    ranked = sorted(
        (h for h in hypotheses if math.isfinite(h["score"])),
        key=lambda h: -h["score"],
    )
    if not ranked:
        raise ValueError("No finite beam hypotheses")
    selections = [decode_selection(h["text"], labels, width) for h in ranked]
    if not selections[0]:
        return []
    selected = []
    for ids in selections:
        for identity in ids:
            if identity not in selected and len(selected) < width:
                selected.append(identity)
    return selected


def before_eos(tokens, eos_ids):
    for index, token in enumerate(tokens):
        if token in eos_ids:
            return tokens[:index]
    raise ValueError("Incomplete beam hypothesis")


class BeamQwenSelector(QwenSelector):
    def select(self, query, candidates, width):
        import torch

        from .qwen3 import end_token_ids

        if width != POLICY["union_width"] or not candidates:
            raise ValueError("This frozen beam policy requires width four and nonempty candidates")
        event = {
            "ok": False,
            "candidate_ids": [n.id for n in candidates],
            "paths": [n.path[1:] for n in candidates],
            "decoding_policy_sha256": POLICY_SHA256,
        }
        try:
            conversation, labels, tokens, budget = fit_messages(
                self.tokenizer, query, candidates, width, self.num_ctx - self.max_output_tokens
            )
            event.update(
                prompt_tokens=len(tokens),
                excerpt_character_budget=budget,
                prompt_sha256=hashlib.sha256(
                    json.dumps(conversation, sort_keys=True).encode()
                ).hexdigest(),
            )
            eos = end_token_ids(self.model, self.tokenizer)
            grammar = SelectionGrammar(self.tokenizer, len(candidates), width, eos)
            ids = torch.tensor([tokens], device=self.model.device)
            # A one-candidate frontier has just two distinct legal arrays.
            beams = min(POLICY["num_beams"], len(candidates) + 1)

            def allowed(batch, sequence):
                prefix = sequence[len(tokens) :].tolist()
                # Generation may retain finished beams while other hypotheses finish.
                if any(token in eos for token in prefix):
                    return sorted(eos)
                return grammar.next_tokens(prefix)

            event["stage"] = "generation"
            with torch.inference_mode():
                output = self.model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    do_sample=False,
                    max_new_tokens=self.max_output_tokens,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_beams=beams,
                    num_return_sequences=beams,
                    length_penalty=POLICY["length_penalty"],
                    early_stopping=POLICY["early_stopping"],
                    return_dict_in_generate=True,
                    output_scores=True,
                    prefix_allowed_tokens_fn=allowed,
                )
            hypotheses = []
            for sequence, score in zip(output.sequences, output.sequences_scores, strict=True):
                value = float(score)
                if not math.isfinite(value):
                    continue
                suffix = before_eos(sequence[len(tokens) :].tolist(), eos)
                hypotheses.append(
                    {
                        "text": self.tokenizer.decode(suffix, skip_special_tokens=False),
                        "score": value,
                        "output_tokens": len(suffix) + 1,
                    }
                )
            event.update(
                stage="validation",
                hypotheses=hypotheses,
                num_beams=beams,
                output_tokens=sum(h["output_tokens"] for h in hypotheses),
                output_token_accounting="sum of returned hypothesis lengths, including EOS",
            )
            selected = combine_hypotheses(hypotheses, labels, width)
            event.update(ok=True, selected=selected)
            return selected
        except (ValueError, RuntimeError) as exc:
            event["error_type"] = type(exc).__name__
            raise
        finally:
            self.events.append(event)
