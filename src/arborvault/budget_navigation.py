"""Experimental incremental tree search with an explicit branch allowance.

No answer labels enter this module. ContentGuidance is a lexical-assisted tree
policy: it searches all leaf text to build branch hints, and is NOT pure navigation.
"""

import json
import math


class ContentGuidance:
    def __init__(self, index, lexical, representatives=3):
        tree = index.tree
        self.texts = dict(index.texts)
        self.priorities = dict(lexical)
        descendants = {}

        def visit(identity):
            children = tree.children[identity]
            if not children:
                descendants[identity] = [identity]
                return [identity]
            leaves = [leaf for child in children for leaf in visit(child.id)]
            descendants[identity] = leaves
            if identity in index.texts:
                ranked = sorted(leaves, key=lambda i: (-lexical[i], tree.nodes[i].ordinal))
                self.priorities[identity] = max(lexical[i] for i in leaves)
                value = json.loads(index.texts[identity])
                # Equal space per representative prevents the first from hiding the rest.
                snippets = [
                    " / ".join(tree.nodes[i].path[-2:])[:180] + ": " + tree.nodes[i].content[:240]
                    for i in ranked[:representatives]
                ]
                value["excerpt"] = ("Query-matched descendants: " + " | ".join(snippets))[:1400]
                self.texts[identity] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            return leaves

        visit(tree.root.id)


def budgeted_traverse(index, priorities, get_scores, *, budget, branch_batch=2, passage_batch=4):
    """Retain all alternatives; expand one scored branch between small batches.

    At most floor(3 * budget / 8) neural evaluations go to branches. Evaluated
    branches can still be expanded after that allowance is spent. Unscored branches
    are never expanded for free. Leaves remain eligible regardless of their parent
    score. The allowance is a cap, not a guarantee of sufficient reachable leaves.
    """
    if type(budget) is not int or budget < 1 or branch_batch < 1 or passage_batch < 1:
        raise ValueError("Invalid navigation budget")
    tree = index.tree
    branch_limit = (3 * budget) // 8
    available = {n.id for n in tree.children[tree.root.id]}
    pending, values, used, reached, trace = set(), {}, [], [], []
    branch_count = 0

    def ordered(ids, scores):
        return sorted(ids, key=lambda i: (-scores[i], tree.nodes[i].ordinal))

    def evaluate(ids):
        if not ids:
            return
        found = get_scores(ids)
        if set(found) != set(ids) or any(not math.isfinite(v) for v in found.values()):
            raise ValueError("Missing or non-finite navigation score")
        values.update(found)
        used.extend(ids)
        available.difference_update(ids)

    while len(used) < budget:
        branches = ordered([i for i in available if tree.children[i]], priorities)[
            : min(branch_batch, branch_limit - branch_count, budget - len(used))
        ]
        evaluate(branches)
        branch_count += len(branches)
        pending.update(branches)
        expanded = ordered(pending, values)[:1]
        for identity in expanded:
            pending.remove(identity)
            available.update(n.id for n in tree.children[identity])
        leaves = ordered([i for i in available if not tree.children[i]], priorities)[
            : min(passage_batch, budget - len(used))
        ]
        evaluate(leaves)
        reached.extend(leaves)
        if not branches and not expanded and not leaves:
            break
        trace.append({"branches_scored": branches, "expanded": expanded, "leaves_scored": leaves})
    if len(set(used)) != len(used) or len(used) > budget or branch_count > branch_limit:
        raise AssertionError("Navigation exceeded its budget")
    return {
        "scored_ids": used,
        "reached_ids": ordered(reached, values),
        "trace": trace,
        "branch_limit": branch_limit,
        "branch_pairs": branch_count,
        "unexpanded_scored_branches": ordered(pending, values),
        "unscored_frontier": ordered(available, priorities),
        "stop_reason": "pair_budget"
        if len(used) == budget
        else ("branch_allowance" if available else "tree_exhausted"),
        "max_passage_score": max((values[i] for i in reached), default=None),
    }
