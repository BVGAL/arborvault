"""Lazy, budgeted long-document retrieval. No evidence labels enter traversal."""

import math

from .candidate_scoring import BM25, candidate_text, views_by_id
from .routing_eval import context, score


class DocumentIndex:
    def __init__(self, tree):
        self.tree = tree
        self.views = views_by_id(tree)
        self.ids = list(self.views)
        self.texts = {i: candidate_text(v) for i, v in self.views.items()}
        self.lexical = BM25(list(self.texts.values()))
        self.leaves = [i for i in self.ids if not tree.children[i]]

    def scores(self, question):
        return dict(zip(self.ids, self.lexical.scores(question), strict=True))


def traverse(index, lexical, get_scores, *, mode, budget, width=4):
    """Charge each candidate once within a policy; reuse caches across policies.

    An oversized frontier is BM25-prefiltered to the remaining pair budget.
    Every evaluated leaf remains eligible, even if it is outside the branch beam.
    get_scores only receives candidate IDs, never labels or reference answers.
    """
    if mode not in {"flat", "tree_scored"} or budget < 1 or width < 1:
        raise ValueError("Invalid traversal policy")
    tree = index.tree
    values, reached, used, trace = {}, [], [], []

    def prefilter(ids, limit):
        return sorted(ids, key=lambda i: (-lexical[i], tree.nodes[i].ordinal))[:limit]

    def ranked(ids):
        return sorted(ids, key=lambda i: (-values[i], tree.nodes[i].ordinal))

    frontier = index.leaves if mode == "flat" else [n.id for n in tree.children[tree.root.id]]
    while frontier and len(used) < budget:
        ids = prefilter(frontier, budget - len(used))
        scores = get_scores(ids)
        if set(scores) != set(ids) or any(not math.isfinite(v) for v in scores.values()):
            raise ValueError("Missing or non-finite scores")
        values.update(scores)
        used.extend(ids)
        leaves = [i for i in ids if not tree.children[i]]
        branches = ranked([i for i in ids if tree.children[i]])[:width]
        reached.extend(leaves)
        trace.append({"candidates": ids, "expanded": branches, "retained_leaves": leaves})
        frontier = [n.id for i in branches for n in tree.children[i]]
    if len(used) != len(set(used)) or len(used) > budget:
        raise AssertionError("Traversal exceeded its unique-pair budget")
    return {
        "scored_ids": used,
        "reached_ids": ranked(reached),
        "trace": trace,
        "max_passage_score": max((values[i] for i in reached), default=None),
    }


def result_row(tree, query, policy):
    selected = context([tree.nodes[i] for i in policy["reached_ids"]])
    return {
        "id": query["id"],
        "document": query["document"],
        "answerable": query["answerable"],
        "category": query["category"],
        "predicted": [n.id for n, _ in selected],
        "abstained": not selected,
        "returned_words": sum(len(text.split()) for _, text in selected),
        "pairs": len(policy["scored_ids"]),
        **score(selected, query),
    }
