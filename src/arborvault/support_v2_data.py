"""Fresh human transfer articles plus balanced, family-separated semantic contrasts."""

import json
from collections import Counter, defaultdict
from pathlib import Path

from .support_contrasts import generate
from .support_data import digest, existing_questions, file_hash, normalized, valid_answer_spans
from .support_probe import write_json


def validate(rows):
    ids, documents, questions, contexts, groups, families, entities = (
        set(),
        {},
        {},
        {},
        defaultdict(list),
        {},
        {},
    )
    for row in rows:
        if row["id"] in ids:
            raise ValueError("Duplicate example ID")
        ids.add(row["id"])
        for mapping, key in (
            (documents, row["document"]),
            (questions, normalized(row["question"])),
            (contexts, normalized(row["context"])),
        ):
            if mapping.setdefault(key, row["role"]) != row["role"]:
                raise ValueError("Content or document crosses partitions")
        if not 1 <= row["input_tokens"] <= 512 or type(row["answerable"]) is not bool:
            raise ValueError("Invalid token limit or label")
        if row["answerable"] and not valid_answer_spans(row["context"], row["answers"]):
            raise ValueError("Unsupported positive annotation")
        if not row["answerable"] and row["answers"]:
            raise ValueError("Negative has an answer span")
        if row["suite"] == "contrast":
            if families.setdefault(row["template_family"], row["role"]) != row["role"]:
                raise ValueError("Template family crosses partitions")
            for entity in row["entities"]:
                if entities.setdefault(entity, row["group_id"]) != row["group_id"]:
                    raise ValueError("Fictional entity reused in another scenario")
            if row["answerable"] != (row["oracle"]["state"] == row["oracle"]["requested_state"]):
                raise ValueError("Counterfactual label contradicts oracle")
        groups[row["group_id"]].append(row)
    for group in groups.values():
        if any((r["role"], r["suite"]) != (group[0]["role"], group[0]["suite"]) for r in group):
            raise ValueError("Group crosses role or suite")
        if group[0]["suite"] == "contrast":
            if len(group) != 4:
                raise ValueError("Contrast group requires four crossed cases")
            for field in ("question", "context"):
                counts = defaultdict(list)
                for row in group:
                    counts[row[field]].append(row["answerable"])
                if len(counts) != 2 or any(sorted(v) != [False, True] for v in counts.values()):
                    raise ValueError("Single-input label shortcut in contrast group")
        elif (
            len(group) != 2
            or {r["answerable"] for r in group} != {True, False}
            or len({r["context"] for r in group}) != 1
        ):
            raise ValueError("Human paragraph pair is incomplete")
    if set(documents.values()) != {"fit", "selection", "calibration", "test"}:
        raise ValueError("Missing partition")


def build(config, source, output, tokenizer):
    if output.exists():
        raise ValueError("Refusing to overwrite a dataset")
    if file_hash(source) != config["human_source_sha256"]:
        raise ValueError("Human source hash mismatch")
    old_path = Path(config["old_dataset"]) / "examples.jsonl"
    if file_hash(old_path) != config["old_examples_sha256"]:
        raise ValueError("Previous dataset hash mismatch")
    old = [json.loads(line) for line in old_path.read_text(encoding="utf-8").splitlines()]
    used_docs = {r["document"] for r in old}
    used_questions = {normalized(r["question"]) for r in old}
    used_contexts = {normalized(r["context"]) for r in old}
    previous_questions, history_locks = existing_questions(Path("datasets"))
    rows = []
    for row in old:
        if row["role"] in {"fit", "selection"}:
            rows.append(
                dict(
                    row,
                    id="human/" + row["id"],
                    group_id="human/" + row["pair_id"],
                    suite="human",
                    phenomenon="human",
                    license="CC-BY-SA-4.0",
                )
            )
    articles = json.loads(source.read_text(encoding="utf-8"))["data"]
    selected_docs, eligible, exclusions = [], {}, Counter()
    for article in articles:
        if article["title"] in used_docs:
            continue
        pairs = []
        for index, paragraph in enumerate(article["paragraphs"]):
            context = paragraph["context"]
            if normalized(context) in used_contexts:
                exclusions["prior_context"] += 1
                continue
            cids = tokenizer.encode(context, add_special_tokens=False)
            if len(cids) + 3 >= config["max_length"]:
                continue
            candidates = {True: [], False: []}
            for q in paragraph["qas"]:
                label = not q["is_impossible"]
                if (
                    normalized(q["question"]) in used_questions
                    or digest(normalized(q["question"])) in previous_questions
                ):
                    continue
                if label and not valid_answer_spans(context, q["answers"]):
                    continue
                if not label and q["answers"]:
                    raise ValueError("Human negative has positive spans")
                length = (
                    len(cids) + len(tokenizer.encode(q["question"], add_special_tokens=False)) + 3
                )
                if length <= config["max_length"]:
                    candidates[label].append((q, length))
            if all(candidates.values()):
                chosen = [
                    min(
                        candidates[label], key=lambda q: digest("support-v2/question/" + q[0]["id"])
                    )
                    for label in (True, False)
                ]
                pairs.append((context, index, chosen))
        if len(pairs) >= config["human_pairs_per_article"]:
            eligible[article["title"]] = pairs
    target = config["human_calibration_articles"] + config["human_test_articles"]
    for title in sorted(eligible, key=lambda d: digest("support-v2/article/" + d)):
        role = (
            "calibration" if len(selected_docs) < config["human_calibration_articles"] else "test"
        )
        new_rows = []
        for context, index, pair in sorted(
            eligible[title], key=lambda p: digest("support-v2/paragraph/" + p[0])
        ):
            if normalized(context) in used_contexts or any(
                normalized(q["question"]) in used_questions for q, _ in pair
            ):
                continue
            if len({normalized(q["question"]) for q, _ in pair}) != 2:
                continue
            for q, length in pair:
                new_rows.append(
                    {
                        "id": "human/" + q["id"],
                        "group_id": "human/" + digest(normalized(context)),
                        "document": title,
                        "role": role,
                        "suite": "human",
                        "phenomenon": "human",
                        "context": context,
                        "question": q["question"],
                        "answerable": not q["is_impossible"],
                        "answers": q["answers"],
                        "plausible_answers": q.get("plausible_answers", []),
                        "source_split": "train",
                        "source_paragraph_index": index,
                        "input_tokens": length,
                        "license": "CC-BY-SA-4.0",
                        "annotation": "Original SQuAD 2.0 human labels; unmodified question and paragraph",
                    }
                )
            used_contexts.add(normalized(context))
            used_questions.update(normalized(q["question"]) for q, _ in pair)
            if len(new_rows) == 2 * config["human_pairs_per_article"]:
                break
        if len(new_rows) < 2 * config["human_pairs_per_article"]:
            raise ValueError("Selected article has insufficient distinct pairs")
        selected_docs.append(title)
        rows.extend(new_rows)
        if len(selected_docs) == target:
            break
    if len(selected_docs) != target:
        raise ValueError("Insufficient unused human articles")
    rows.extend(generate(config["contrast_groups_per_family"]))
    for row in rows:
        actual = len(tokenizer(row["question"], row["context"], truncation=False)["input_ids"])
        if "input_tokens" in row and row["input_tokens"] != actual:
            raise ValueError("Whole-pair token accounting mismatch")
        row["input_tokens"] = actual
    validate(rows)
    output.mkdir(parents=True)
    path = output / "examples.jsonl"
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "id": config["id"],
        "examples_sha256": file_hash(path),
        "source_sha256": file_hash(source),
        "old_examples_sha256": file_hash(old_path),
        "config_sha256": file_hash(Path(config["config_path"])),
        "previous_question_files": history_locks,
        "licenses": {"human": "CC-BY-SA-4.0", "contrast": "Apache-2.0"},
        "counts": {
            role: {
                suite: {
                    "examples": len(group),
                    "supported": sum(r["answerable"] for r in group),
                    "documents": len({r["document"] for r in group}),
                    "template_families": len(
                        {r["template_family"] for r in group if r["suite"] == "contrast"}
                    ),
                }
                for suite in ("human", "contrast")
                if (group := [r for r in rows if r["role"] == role and r["suite"] == suite])
            }
            for role in ("fit", "selection", "calibration", "test")
        },
        "novel_human_articles": selected_docs,
        "excluded": dict(exclusions),
        "novelty_scope": "New human calibration/test articles absent from all support-v1 roles; original public source/pretraining exposure is not excluded",
        "synthetic_scope": "Fictional records with role-disjoint template families and scenario entities; oracle is not independent human adjudication",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def load(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    path = directory / "examples.jsonl"
    if file_hash(path) != manifest["examples_sha256"]:
        raise ValueError("Dataset hash mismatch")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    validate(rows)
    return manifest, rows
