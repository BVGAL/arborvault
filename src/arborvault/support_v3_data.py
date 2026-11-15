"""Training replay plus new article- and template-separated evaluation data."""

import json
from collections import Counter
from pathlib import Path

from .support_data import digest, existing_questions, file_hash, normalized, valid_answer_spans
from .support_probe import write_json
from .support_v2_data import validate
from .support_v3_contrasts import generate


def build(config, source, output, tokenizer):
    if output.exists():
        raise ValueError("Refusing to overwrite dataset")
    if file_hash(source) != config["human_source_sha256"]:
        raise ValueError("Human source hash mismatch")
    old, locks = [], {}
    for name, expected in config["prior_datasets"].items():
        path = Path(name) / "examples.jsonl"
        if file_hash(path) != expected:
            raise ValueError("Previous data changed")
        locks[path.as_posix()] = expected
        old.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    latest = [
        json.loads(line)
        for line in Path("datasets/support-v2/examples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    rows = [
        dict(r)
        for r in latest
        if r["role"] == "fit" or (r["role"] == "selection" and r["suite"] == "human")
    ]
    used_docs = {r["document"] for r in old}
    used_contexts = {normalized(r["context"]) for r in old}
    used_questions = {normalized(r["question"]) for r in old}
    history, history_locks = existing_questions(Path("datasets"))
    eligible = {}
    for article in json.loads(source.read_text(encoding="utf-8"))["data"]:
        if article["title"] in used_docs:
            continue
        pairs = []
        for pi, paragraph in enumerate(article["paragraphs"]):
            context = paragraph["context"]
            if normalized(context) in used_contexts:
                continue
            ct = len(tokenizer.encode(context, add_special_tokens=False))
            if ct + 3 >= config["max_length"]:
                continue
            choices = {True: [], False: []}
            for q in paragraph["qas"]:
                y = not q["is_impossible"]
                if (
                    normalized(q["question"]) in used_questions
                    or digest(normalized(q["question"])) in history
                ):
                    continue
                if y and not valid_answer_spans(context, q["answers"]):
                    continue
                if not y and q["answers"]:
                    raise ValueError("Negative source has answers")
                length = ct + len(tokenizer.encode(q["question"], add_special_tokens=False)) + 3
                if length <= config["max_length"]:
                    choices[y].append((q, length))
            if all(choices.values()):
                chosen = [
                    min(choices[y], key=lambda x: digest("support-v3/question/" + x[0]["id"]))
                    for y in (True, False)
                ]
                pairs.append((context, pi, chosen))
        if len(pairs) >= config["human_pairs_per_article"]:
            eligible[article["title"]] = pairs
    selected = []
    target = config["human_calibration_articles"] + config["human_test_articles"]
    for title in sorted(eligible, key=lambda d: digest("support-v3/article/" + d)):
        role = "calibration" if len(selected) < config["human_calibration_articles"] else "test"
        new_rows = []
        for context, pi, pair in sorted(
            eligible[title], key=lambda p: digest("support-v3/paragraph/" + p[0])
        ):
            if normalized(context) in used_contexts or any(
                normalized(q["question"]) in used_questions for q, _ in pair
            ):
                continue
            if len({normalized(q["question"]) for q, _ in pair}) != 2:
                continue
            for q, length in pair:
                new_rows.append(
                    dict(
                        id="human/" + q["id"],
                        group_id="human/" + digest(normalized(context)),
                        document=title,
                        role=role,
                        suite="human",
                        phenomenon="human",
                        question=q["question"],
                        context=context,
                        answerable=not q["is_impossible"],
                        answers=q["answers"],
                        plausible_answers=q.get("plausible_answers", []),
                        input_tokens=length,
                        source_split="train",
                        source_paragraph_index=pi,
                        license="CC-BY-SA-4.0",
                        annotation="Original unmodified SQuAD 2.0 label, question and paragraph",
                    )
                )
            used_contexts.add(normalized(context))
            used_questions.update(normalized(q["question"]) for q, _ in pair)
            if len(new_rows) == 2 * config["human_pairs_per_article"]:
                break
        if len(new_rows) != 2 * config["human_pairs_per_article"]:
            raise ValueError("Insufficient distinct pairs in selected article")
        selected.append(title)
        rows.extend(new_rows)
        if len(selected) == target:
            break
    if len(selected) != target:
        raise ValueError(f"Insufficient fresh human articles: {len(selected)}/{target}")
    rows.extend(generate(config["contrast_groups_per_family"]))
    for row in rows:
        actual = len(tokenizer(row["question"], row["context"], truncation=False)["input_ids"])
        if "input_tokens" in row and row["input_tokens"] != actual:
            raise ValueError("Token accounting changed")
        row["input_tokens"] = actual
    validate(rows)
    output.mkdir(parents=True)
    path = output / "examples.jsonl"
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
        newline="\n",
    )
    manifest = dict(
        id=config["id"],
        examples_sha256=file_hash(path),
        source_sha256=file_hash(source),
        config_sha256=file_hash(Path(config["config_path"])),
        prior_datasets=locks,
        previous_question_files=history_locks,
        novel_human_articles=selected,
        eligible_unused_articles=len(eligible),
        counts={
            role: dict(Counter(r["phenomenon"] for r in rows if r["role"] == role))
            for role in ("fit", "selection", "calibration", "test")
        },
        licenses={"human": "CC-BY-SA-4.0", "contrast": "Apache-2.0"},
        novelty="New calibration/test human articles excluded from every v1/v2 role; new contrast families/entities excluded from training. Public source may occur in model pretraining.",
        synthetic_scope="Authored and reviewed by the same AI-assisted author; no independent adjudication; 15 test families, 5 per phenomenon.",
    )
    write_json(output / "manifest.json", manifest)
    return manifest
