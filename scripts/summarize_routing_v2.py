"""Verify and summarize completed v2 artifacts, without selecting checkpoints on test."""

import argparse
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def clustered_delta(base, tuned, repetitions=2000):
    b = {r["id"]: r for r in base if r["answerable"]}
    t = {r["id"]: r for r in tuned if r["answerable"]}
    if b.keys() != t.keys():
        raise ValueError("Paired question sets differ")
    groups = defaultdict(list)
    for identity, row in b.items():
        if row["document"] != t[identity]["document"]:
            raise ValueError("Paired document scopes differ")
        groups[row["document"]].append(t[identity]["recall"] - row["recall"])
    keys = sorted(groups)
    rng, samples = random.Random(20261113), []
    for _ in range(repetitions):
        values = [delta for key in rng.choices(keys, k=len(keys)) for delta in groups[key]]
        samples.append(statistics.mean(values))
    samples.sort()
    values = [delta for group in groups.values() for delta in group]
    return {
        "mean_delta": statistics.mean(values),
        "bootstrap_95_percent_interval": [
            samples[int(repetitions * 0.025)],
            samples[int(repetitions * 0.975)],
        ],
        "resampling_unit": "document",
        "documents": len(keys),
        "replicates": repetitions,
        "seed": 20261113,
        "queries_improved": sum(v > 0 for v in values),
        "queries_regressed": sum(v < 0 for v in values),
        "queries_unchanged": sum(v == 0 for v in values),
    }


def verify_artifacts(root):
    manifest = read(root / "artifact-manifest.json")
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p != root / "artifact-manifest.json"
    }
    if actual != set(manifest):
        raise ValueError("Artifact inventory mismatch")
    for name, expected in manifest.items():
        path = (root / name).resolve()
        if (
            not path.is_relative_to(root.resolve())
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("Artifact checksum mismatch")
    return manifest


def diagnostics(report):
    rows = report["methods"]["navigation"]["queries"]
    events = [event for row in rows for event in row["model_events"]]
    latencies = sorted(row["latency_ms"] for row in rows)
    return {
        "selection_width_histogram": dict(
            sorted(Counter(len(e.get("selected", [])) for e in events).items())
        ),
        "mean_returned_paragraphs": statistics.mean(len(r["predicted"]) for r in rows),
        "answerable_first_route_gold_coverage": sum(
            bool(r["trace"] and any(r["trace"][0]["gold_ancestor_selected"]))
            for r in rows
            if r["answerable"]
        ),
        "first_route_empty_answerable": sum(
            bool(r["trace"] and not r["trace"][0]["selected"]) for r in rows if r["answerable"]
        ),
        "reasons": dict(Counter(r["reason"] or "completed" for r in rows)),
        "latency_median_ms": statistics.median(latencies),
        "latency_p95_ms": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
        "latency_note": "Single sequential pass; not a repeated serving benchmark",
    }


def check_pairs(reference, other):
    def keys(rows):
        values = {r["id"]: (r["document"], r["answerable"]) for r in rows}
        if len(values) != len(rows):
            raise ValueError("Duplicate question IDs")
        return values

    if keys(reference) != keys(other):
        raise ValueError("Paired question metadata differ")


def summarize(root):
    manifest = verify_artifacts(root)
    provenance = read(root / "training-provenance.json")
    selection = read(root / "selection-before-test.json")
    if selection != provenance["selection"] or not selection["test_not_yet_evaluated"]:
        raise ValueError("Locked selection and final provenance disagree")
    reports = {
        name: read(root / "evaluation" / (name + ".json"))
        for name in ("base-test", "tuned-test", "baselines-test")
    }
    base, tuned = reports["base-test"], reports["tuned-test"]
    for key in (
        "dataset_manifest_sha256",
        "split",
        "k",
        "word_budget",
        "base_revision",
        "prompt_sha256",
        "recipe_sha256",
        "code_revision",
    ):
        if base[key] != tuned[key]:
            raise ValueError("Mismatched comparison metadata: " + key)
    methods = {}
    for variant, report in reports.items():
        for method, result in report["methods"].items():
            check_pairs(base["methods"]["navigation"]["queries"], result["queries"])
            methods[variant + "/" + method] = {k: v for k, v in result.items() if k != "queries"}
    return {
        "scope": base["scope"],
        "dataset_manifest_sha256": base["dataset_manifest_sha256"],
        "artifact_files_verified": len(manifest),
        "methods": methods,
        "paired_navigation": clustered_delta(
            base["methods"]["navigation"]["queries"], tuned["methods"]["navigation"]["queries"]
        ),
        "paired_navigation_lexical": clustered_delta(
            base["methods"]["navigation_lexical"]["queries"],
            tuned["methods"]["navigation_lexical"]["queries"],
        ),
        "base_navigation_minus_dense": clustered_delta(
            reports["baselines-test"]["methods"]["dense_minilm"]["queries"],
            base["methods"]["navigation"]["queries"],
        ),
        "base_lexical_minus_dense": clustered_delta(
            reports["baselines-test"]["methods"]["dense_minilm"]["queries"],
            base["methods"]["navigation_lexical"]["queries"],
        ),
        "selection": selection,
        "diagnostics": {"base": diagnostics(base), "tuned": diagnostics(tuned)},
        "test_complete": base["complete"] and tuned["complete"],
        "limitations": [
            "Public, filtered, English-only QASPER subset; model pretraining exposure unknown.",
            "Document-scoped paragraph retrieval, not global corpus search or generated-answer quality.",
            "Two training seeds; bootstrap intervals are descriptive, conditional on this pilot.",
        ],
    }


def summarize_beams(root, original):
    manifest = verify_artifacts(root)
    provenance = read(root / "beam-provenance.json")
    selection = read(root / "selection-before-test.json")
    if selection != provenance["selection"] or not selection["test_not_yet_evaluated"]:
        raise ValueError("Locked beam selection and final provenance disagree")
    base, tuned = [read(root / "evaluation" / (tag + "-test.json")) for tag in ("base", "tuned")]
    greedy = read(original / "evaluation/base-test.json")
    dense = read(original / "evaluation/baselines-test.json")["methods"]["dense_minilm"]
    for key in (
        "dataset_manifest_sha256",
        "split",
        "k",
        "word_budget",
        "base_revision",
        "prompt_sha256",
        "recipe_sha256",
    ):
        if not base[key] == tuned[key] == greedy[key]:
            raise ValueError("Mismatched ablation metadata: " + key)
    if base["decoding_policy_sha256"] != tuned["decoding_policy_sha256"]:
        raise ValueError("Mismatched decoding policy")
    methods = {}
    for tag, report in (("base-test", base), ("tuned-test", tuned)):
        for method, result in report["methods"].items():
            check_pairs(greedy["methods"]["navigation"]["queries"], result["queries"])
            methods[tag + "/" + method] = {k: v for k, v in result.items() if k != "queries"}
    check_pairs(base["methods"]["navigation"]["queries"], dense["queries"])
    return {
        "artifact_files_verified": len(manifest),
        "methods": methods,
        "selection": selection,
        "test_complete": base["complete"] and tuned["complete"],
        "paired_navigation": clustered_delta(
            base["methods"]["navigation"]["queries"], tuned["methods"]["navigation"]["queries"]
        ),
        "base_beams_minus_base_greedy": clustered_delta(
            greedy["methods"]["navigation"]["queries"], base["methods"]["navigation"]["queries"]
        ),
        "tuned_beams_minus_dense": clustered_delta(
            dense["queries"], tuned["methods"]["navigation"]["queries"]
        ),
        "base_lexical_minus_dense": clustered_delta(
            dense["queries"], base["methods"]["navigation_lexical"]["queries"]
        ),
        "tuned_lexical_minus_dense": clustered_delta(
            dense["queries"], tuned["methods"]["navigation_lexical"]["queries"]
        ),
        "diagnostics": {"base": diagnostics(base), "tuned": diagnostics(tuned)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--beams", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.artifacts)
    if args.beams:
        report["beams"] = summarize_beams(args.beams, args.artifacts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
