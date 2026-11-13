"""Validate and combine oracle-state diagnostics, historical traces and audit judgments."""

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from summarize_routing_v2 import clustered_delta

from arborvault.qasper import load_split
from arborvault.qwen3 import file_hash
from arborvault.routing_diagnostics import oracle_states, summarize_states
from arborvault.routing_eval import context

NAMES = {"base": "base-dev", "step-88": "seed-42-step-88-dev", "step-175": "seed-42-step-175-dev"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(root):
    manifest = read(root / "manifest.json")
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p != root / "manifest.json"
    }
    if actual != set(manifest):
        raise ValueError("Diagnostic inventory mismatch")
    for name, expected in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != expected:
            raise ValueError("Diagnostic checksum mismatch")
    if read(root / "complete.json").get("complete") is not True:
        raise ValueError("Incomplete diagnostics")


def query_metric(rows, metric, *, positives=True):
    groups = defaultdict(list)
    for row in rows:
        if positives and not row["target_ids"]:
            continue
        groups[row["query_id"]].append(row)
    return [
        {
            "id": identity,
            "document": items[0]["document"],
            "answerable": True,
            "recall": statistics.mean(r["metrics"][metric] for r in items),
        }
        for identity, items in groups.items()
    ]


def trace_diagnostics(tree, query, row, states):
    by_depth = {s["depth"]: s for s in states}
    seen, first_prune, first_all_lost = set(), None, None
    changed, off_evidence = 0, 0
    primary = tree.ancestors(query["gold_sets"][0]) if query["gold_sets"] else set()
    for depth, step in enumerate(row["trace"]):
        oracle = by_depth.get(depth)
        if oracle is None or set(step["candidates"]) != set(oracle["candidate_ids"]):
            changed += 1
        if query["answerable"] and not set(step["candidates"]) & primary:
            off_evidence += 1
        seen.update(i for i in step["selected"] if tree.nodes[i].content)
        if not query["gold_sets"]:
            continue
        possible = []
        for gold in query["gold_sets"]:
            surviving = {
                g for g in gold if g in seen or tree.ancestors([g]) & set(step["selected"])
            }
            possible.append(len(surviving) / len(gold))
        best = max(possible)
        if best < 1 and first_prune is None:
            first_prune = depth
        if best == 0 and first_all_lost is None:
            first_all_lost = depth
    reached = set(row["reached_before_topk_and_word_budget"])
    pre_budget = max((len(reached & set(g)) / len(g) for g in query["gold_sets"]), default=None)
    return {
        "id": query["id"],
        "answerable": query["answerable"],
        "first_irreversible_prune_depth": first_prune,
        "first_all_evidence_lost_depth": first_all_lost,
        "frontiers_different_from_canonical_oracle": changed,
        "frontiers_without_primary_gold_candidate": off_evidence,
        "pre_budget_recall": pre_budget,
        "final_recall": row["recall"],
    }


def audit_summary(dataset, reports, audit_dir, judgments):
    _, trees, queries = load_split(dataset, "dev")
    queries = {q["id"]: q for q in queries}
    packet, key, protocol = (
        read(audit_dir / n) for n in ("packet.json", "selection.json", "protocol.json")
    )
    if (
        hashlib.sha256(json.dumps(packet, sort_keys=True).encode()).hexdigest()
        != protocol["packet_sha256"]
    ):
        raise ValueError("Frozen audit packet changed")
    if len(judgments) != 55 or len({r["case"] for r in judgments}) != 55:
        raise ValueError("Expected 55 unique completed reviews")
    by_case = {r["case"]: r for r in judgments}
    if set(by_case) != {r["case"] for r in key}:
        raise ValueError("Judgments do not match frozen sample")
    indexed = {
        name: {r["id"]: r for r in report["methods"]["navigation"]["queries"]}
        for name, report in reports.items()
    }
    rows = []
    for item in key:
        judgment = by_case[item["case"]]
        if not judgment["rationale"] or judgment["confidence"] not in {"high", "medium", "low"}:
            raise ValueError("Incomplete audit judgment")
        query = queries[item["query_id"]]
        tree = trees[query["document"]]
        alternative = set(judgment["alternative_ids"])
        gold = {i for g in query["gold_sets"] for i in g}
        if alternative and (not alternative <= tree.nodes.keys() or not alternative - gold):
            raise ValueError("Alternative must contain valid unannotated evidence")
        coverage = {}
        for variant, index in indexed.items():
            original = index[query["id"]]
            rendered = context(
                [tree.nodes[i] for i in original["reached_before_topk_and_word_budget"]]
            )
            if [n.id for n, _ in rendered] != original["predicted"]:
                raise ValueError("Cannot reconstruct archived context")
            full = {n.id for n, text in rendered if text == " ".join(n.content.split())}
            coverage[variant] = {
                "original_recall": original["recall"],
                "alternative_ids_present": bool(alternative)
                and alternative <= set(original["predicted"]),
                "alternative_paragraphs_fully_visible": bool(alternative) and alternative <= full,
            }
        rows.append(
            {
                **item,
                **judgment,
                "question": query["question"],
                "gold_sets": query["gold_sets"],
                "source": "https://arxiv.org/abs/" + query["document"],
                "context_check": coverage,
            }
        )
    failures = [r for r in rows if r["stratum"] != "control"]
    controls = [r for r in rows if r["stratum"] == "control"]
    if len(failures) != 50 or len(controls) != 5:
        raise ValueError("Audit quotas changed")
    return {
        "reviewer": "AI-assisted qualitative review; not independently human-validated",
        "scores_changed": False,
        "failure_cases": len(failures),
        "control_cases": len(controls),
        "mapping_checks_passed": len(packet),
        "failure_gold_assessment": dict(Counter(r["gold_assessment"] for r in failures)),
        "control_gold_assessment": dict(Counter(r["gold_assessment"] for r in controls)),
        "failure_cases_with_proposed_alternatives": sum(
            bool(r["alternative_ids"]) for r in failures
        ),
        "failure_cases_with_high_confidence_alternatives": sum(
            bool(r["alternative_ids"]) and r["confidence"] == "high" for r in failures
        ),
        "high_confidence_alternative_fully_visible_in_a_failing_variant": [
            r["case"]
            for r in failures
            if r["confidence"] == "high"
            and any(
                c["original_recall"] < 1 and c["alternative_paragraphs_fully_visible"]
                for c in r["context_check"].values()
            )
        ],
        "protocol": protocol,
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/qasper-routing-v2"))
    parser.add_argument("--reports", type=Path, default=Path("docs/benchmarks/routing-v2"))
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = {variant: read(args.reports / (name + ".json")) for variant, name in NAMES.items()}
    protocol = read(args.audit / "protocol.json")
    for name, expected in protocol["input_report_hashes"].items():
        if file_hash(args.reports / (name + ".json")) != expected:
            raise ValueError("Archived navigation report changed")
    result = {"audit": audit_summary(args.dataset, reports, args.audit, read(args.judgments))}
    if args.diagnostics:
        verify(args.diagnostics)
        plan = read(args.diagnostics / "plan.json")
        if file_hash(args.dataset / "manifest.json") != plan["dataset_sha256"]:
            raise ValueError("Diagnostic dataset mismatch")
        _, trees, queries = load_split(args.dataset, "dev")
        states = [s for q in queries for s in oracle_states(trees[q["document"]], q)]
        if (
            hashlib.sha256(json.dumps(states, sort_keys=True).encode()).hexdigest()
            != plan["oracle_states_sha256"]
        ):
            raise ValueError("Oracle state inventory mismatch")
        result.update(
            plan=plan,
            environment=read(args.diagnostics / "environment.json"),
            runtime=read(args.diagnostics / "complete.json"),
            models={},
        )
        if args.replay:
            verify(args.replay)
            replay_plan = read(args.replay / "plan.json")
            for key in ("recipe_sha256", "dataset_sha256", "base_revision", "adapter_hashes"):
                if replay_plan[key] != plan[key]:
                    raise ValueError("Navigation replay inputs differ: " + key)
            if read(args.replay / "environment.json") != result["environment"]:
                raise ValueError("Navigation replay environment differs")
            result["replay_plan"] = replay_plan
        all_rows = {}
        for variant, report in reports.items():
            historical = report
            if args.replay:
                report = read(args.replay / (variant + ".json"))
            rows = [
                json.loads(s)
                for s in (args.diagnostics / (variant + "-states.jsonl")).read_text().splitlines()
            ]
            if len(rows) != len(states) or any(
                any(row[k] != state[k] for k in state)
                for row, state in zip(rows, states, strict=True)
            ):
                raise ValueError("Unequal diagnostic state sets")
            all_rows[variant] = rows
            nav = {r["id"]: r for r in report["methods"]["navigation"]["queries"]}
            traces = [
                trace_diagnostics(
                    trees[q["document"]],
                    q,
                    nav[q["id"]],
                    [s for s in states if s["query_id"] == q["id"]],
                )
                for q in queries
            ]
            first = [r for r in rows if r["depth"] == 0]
            agreements = sum(
                set(r["predicted"]) == set(nav[r["query_id"]]["trace"][0]["selected"])
                for r in first
            )
            result["models"][variant] = {
                "oracle": summarize_states(rows),
                "query_macro_oracle_recall": statistics.mean(
                    r["recall"] for r in query_metric(rows, "recall")
                ),
                "navigation": {
                    k: v for k, v in report["methods"]["navigation"].items() if k != "queries"
                },
                "archived_navigation": {
                    k: v for k, v in historical["methods"]["navigation"].items() if k != "queries"
                },
                "initial_state_agreement_with_navigation": {
                    "matches": agreements,
                    "total": len(first),
                },
                "first_irreversible_prune_histogram": dict(
                    Counter(
                        str(t["first_irreversible_prune_depth"]) for t in traces if t["answerable"]
                    )
                ),
                "first_all_lost_histogram": dict(
                    Counter(
                        str(t["first_all_evidence_lost_depth"]) for t in traces if t["answerable"]
                    )
                ),
                "traces": traces,
            }
        result["paired_oracle_decisions"] = {
            variant: {
                metric: clustered_delta(
                    query_metric(all_rows["base"], metric), query_metric(rows, metric)
                )
                for metric in ("recall", "precision", "f1")
            }
            for variant, rows in all_rows.items()
            if variant != "base"
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "written": True,
                "diagnostics": bool(args.diagnostics),
                "audit": {
                    k: v for k, v in result["audit"].items() if k not in {"cases", "protocol"}
                },
            }
        )
    )


if __name__ == "__main__":
    main()
