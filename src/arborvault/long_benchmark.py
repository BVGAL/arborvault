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


def verify_recorded_traversal(
    index, lexical, scores, recorded, *, mode, budget, width=4, lexical_tolerance=0.0
):
    """Independently validate exact candidate order, expansion and final ranking.

    The frozen GPU source summed a hash-ordered set of query terms. Some equal
    BM25 values therefore differ at machine precision between Python processes.
    Canonical summation reproduces this experiment exactly. An optional tolerance
    up to 1e-12 is available for auditing legacy traces, but is NOT enabled in
    this benchmark's analysis. Model decisions must always reproduce exactly.
    No labels enter verification; substantive ranking differences fail closed.
    """
    if (
        mode not in {"flat", "tree_scored"}
        or budget < 1
        or width < 1
        or not 0 <= lexical_tolerance <= 1e-12
    ):
        raise ValueError("Invalid traversal policy")
    tree = index.tree
    frontier = index.leaves if mode == "flat" else [n.id for n in tree.children[tree.root.id]]
    used, reached, ties = [], [], []
    for step_number, step in enumerate(recorded["trace"]):
        if not frontier or len(used) >= budget:
            raise ValueError("Unexpected traversal step")
        ids = step["candidates"]
        expected = sorted(frontier, key=lambda i: (-lexical[i], tree.nodes[i].ordinal))[
            : budget - len(used)
        ]
        if len(ids) != len(expected) or len(set(ids)) != len(ids) or set(ids) - set(frontier):
            raise ValueError("Invalid recorded frontier membership")
        if ids != expected:
            if lexical_tolerance == 0:
                raise ValueError("Exact lexical replay mismatch")
            ranked_frontier = sorted(frontier, key=lambda i: (-lexical[i], tree.nodes[i].ordinal))
            differences = [abs(lexical[a] - lexical[b]) for a, b in zip(ids, expected, strict=True)]
            if max(differences, default=0) > lexical_tolerance:
                raise ValueError("Substantive lexical replay mismatch")
            excluded = set(ranked_frontier) - set(ids)
            if excluded and min(lexical[i] for i in ids) + lexical_tolerance < max(
                lexical[i] for i in excluded
            ):
                raise ValueError("Invalid lexical cutoff")
            ties.append(
                {
                    "step": step_number,
                    "max_lexical_difference": max(differences, default=0),
                    "candidate_set_changed_vs_canonical": set(ids) != set(expected),
                }
            )
        if any(i not in scores or not math.isfinite(scores[i]) for i in ids):
            raise ValueError("Missing model scores")
        branches = sorted(
            [i for i in ids if tree.children[i]], key=lambda i: (-scores[i], tree.nodes[i].ordinal)
        )[:width]
        leaves = [i for i in ids if not tree.children[i]]
        if branches != step["expanded"] or leaves != step["retained_leaves"]:
            raise ValueError("Model expansion replay mismatch")
        used.extend(ids)
        reached.extend(leaves)
        frontier = [n.id for i in branches for n in tree.children[i]]
    if frontier and len(used) < budget:
        raise ValueError("Incomplete recorded traversal")
    if len(used) != len(set(used)) or len(used) > budget:
        raise ValueError("Invalid pair budget")
    reached = sorted(reached, key=lambda i: (-scores[i], tree.nodes[i].ordinal))
    if (
        recorded["scored_ids"] != used
        or recorded["reached_ids"] != reached
        or recorded["max_passage_score"] != max((scores[i] for i in reached), default=None)
    ):
        raise ValueError("Final ranking replay mismatch")
    return ties
