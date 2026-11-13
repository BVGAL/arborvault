"""Post-hoc diagnostics on fixed oracle states; no training or policy selection."""

import hashlib
import json
import statistics

from .qwen3 import encode_completion, end_token_ids
from .routing_v2 import SelectionGrammar, fit_messages


def oracle_states(tree, query):
    """Follow the first annotated evidence set, preserving runtime candidate order.

    Alternative annotations are retained as metadata. They never supply future
    tokens to the freely generated decision. Unknown questions have one empty state.
    """
    gold = set(query["gold_sets"][0]) if query["gold_sets"] else set()
    ancestors = tree.ancestors(gold)
    union = tree.ancestors({i for group in query["gold_sets"] for i in group})
    frontier = tree.children[tree.root.id]
    if tree.root.content:
        frontier = [tree.root]
    depth = 0
    while frontier:
        targets = [n.id for n in frontier if n.id in ancestors]
        yield {
            "id": query["id"] + "/" + str(depth),
            "query_id": query["id"],
            "document": query["document"],
            "depth": depth,
            "answerable": query["answerable"],
            "candidate_ids": [n.id for n in frontier],
            "target_ids": targets,
            "union_target_ids": [n.id for n in frontier if n.id in union],
        }
        frontier = [child for i in targets for child in tree.children[i]]
        depth += 1


def set_metrics(predicted, target):
    predicted, target = set(predicted), set(target)
    true = len(predicted & target)
    precision = true / len(predicted) if predicted else float(not target)
    recall = true / len(target) if target else float(not predicted)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * true / (len(predicted) + len(target)) if predicted or target else 1.0,
        "exact": predicted == target,
        "selected_count": len(predicted),
        "target_count": len(target),
        "empty": not predicted,
    }


def token_diagnostics(logits, target, grammar, tokenizer):
    """Full-vocabulary NLL plus token decisions under the original grammar.

    These are conditional token metrics, NOT a decoded action or set accuracy.
    Grammar positions with one legal token are excluded from choice accuracy.
    """
    import torch

    if logits.ndim != 2 or logits.shape[0] != len(target) or not target:
        raise ValueError("Completion logits/target mismatch")
    tokens = []
    target_array = json.loads(tokenizer.decode(target[:-1], skip_special_tokens=True))
    for index, wanted in enumerate(target):
        row = logits[index].float()
        legal = grammar.next_tokens(target[:index])
        if wanted not in legal:
            raise ValueError("Canonical target is outside the inference grammar")
        allowed = row[legal]
        best = legal[int(allowed.argmax().item())]
        closed_early = False
        if best != wanted and best not in grammar.eos_ids:
            try:
                value = json.loads(
                    tokenizer.decode(target[:index] + [best], skip_special_tokens=True)
                )
                closed_early = isinstance(value, list) and len(value) < len(target_array)
            except (ValueError, UnicodeError):
                pass
        tokens.append(
            {
                "target": wanted,
                "raw_correct": int(row.argmax().item()) == wanted,
                "constrained_correct": best == wanted,
                "legal_count": len(legal),
                "nll": float((torch.logsumexp(row, 0) - row[wanted]).item()),
                "constrained_nll": float((torch.logsumexp(allowed, 0) - row[wanted]).item()),
                "premature_close": closed_early,
            }
        )
    return tokens


def diagnose_state(model, tokenizer, selector, tree, query, state, *, num_ctx=4096):
    import torch

    candidates = tree.views([tree.nodes[i] for i in state["candidate_ids"]])
    conversation, labels, _, budget = fit_messages(tokenizer, query, candidates, 4, num_ctx - 64)
    target_labels = [label for label, node in labels.items() if node in state["target_ids"]]
    completion = {"role": "assistant", "content": json.dumps(target_labels, separators=(",", ":"))}
    encoded = encode_completion(tokenizer, conversation + [completion], num_ctx)
    count = encoded["completion_length"]
    ids = torch.tensor([encoded["input_ids"]], device=model.device)
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False, logits_to_keep=count + 1).logits
        if logits.shape[1] != count + 1:
            raise ValueError("Runtime did not honor completion-only logits")
        tokens = token_diagnostics(
            logits[0, :-1],
            encoded["input_ids"][-count:],
            SelectionGrammar(tokenizer, len(candidates), 4, end_token_ids(model, tokenizer)),
            tokenizer,
        )
    del logits, ids
    # A fresh generation sees only the question and candidates, never the target.
    predicted = selector.select(query, candidates, 4)
    event = selector.events[-1]
    prompt_hash = hashlib.sha256(json.dumps(conversation, sort_keys=True).encode()).hexdigest()
    if event["prompt_sha256"] != prompt_hash:
        raise ValueError("Teacher-forced and free decision prompts differ")
    return {
        **state,
        "predicted": predicted,
        "metrics": set_metrics(predicted, state["target_ids"]),
        "union_metrics": set_metrics(predicted, state["union_target_ids"]),
        "teacher_forced": tokens,
        "prompt_sha256": prompt_hash,
        "prompt_tokens": event["prompt_tokens"],
        "excerpt_character_budget": budget,
    }


def summarize_states(rows):
    def subset(items):
        tokens = [t for r in items for t in r["teacher_forced"]]
        choices = [t for t in tokens if t["legal_count"] > 1]
        # Qwen accepts multiple EOS IDs. Choosing between equivalent terminators
        # is not a branch/cardinality decision, so exclude the final target token.
        content_choices = [
            t for r in items for t in r["teacher_forced"][:-1] if t["legal_count"] > 1
        ]
        positive = [r for r in items if r["target_ids"]]
        empty = [r for r in items if not r["target_ids"]]

        def mean(values):
            return statistics.mean(values) if values else None

        return {
            "states": len(items),
            "positive_states": len(positive),
            "empty_target_states": len(empty),
            "completion_tokens": len(tokens),
            "choice_tokens": len(choices),
            "token_nll": mean([t["nll"] for t in tokens]),
            "constrained_token_nll": mean([t["constrained_nll"] for t in tokens]),
            "raw_token_accuracy": mean([t["raw_correct"] for t in tokens]),
            "choice_token_accuracy": mean([t["constrained_correct"] for t in choices]),
            "nonterminal_choice_tokens": len(content_choices),
            "nonterminal_choice_accuracy": mean(
                [t["constrained_correct"] for t in content_choices]
            ),
            "premature_close_tokens": sum(t["premature_close"] for t in tokens),
            "positive_precision": mean([r["metrics"]["precision"] for r in positive]),
            "positive_precision_any_annotation": mean(
                [r["union_metrics"]["precision"] for r in positive]
            ),
            "positive_recall": mean([r["metrics"]["recall"] for r in positive]),
            "positive_f1": mean([r["metrics"]["f1"] for r in positive]),
            "positive_exact": mean([r["metrics"]["exact"] for r in positive]),
            "positive_selected_mean": mean([r["metrics"]["selected_count"] for r in positive]),
            "positive_empty": sum(r["metrics"]["empty"] for r in positive),
            "empty_target_abstentions": sum(r["metrics"]["empty"] for r in empty),
        }

    return {
        "all": subset(rows),
        "first": subset([r for r in rows if r["depth"] == 0]),
        "deeper": subset([r for r in rows if r["depth"] > 0]),
    }
