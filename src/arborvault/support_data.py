"""Human-labeled context support pairs, grouped by source article.

Dataset derivatives use CC BY-SA 4.0; this implementation uses Apache-2.0.
Unanswerable labels come from SQuAD 2.0 annotators, never absence of a gold ID.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalized(text):
    return " ".join(text.casefold().split())


def existing_questions(root):
    hashes, files = set(), {}
    paths = list(root.glob("*/questions.json")) + list(root.glob("*/*questions.jsonl"))
    paths += list(root.glob("*/queries.jsonl"))
    for path in sorted(paths):
        if path.parent.name == "support-v1":
            continue
        files[path.relative_to(root).as_posix()] = file_hash(path)
        text = path.read_text(encoding="utf-8")
        rows = json.loads(text) if path.suffix == ".json" else map(json.loads, text.splitlines())
        for row in rows:
            question = row.get("question", row.get("query"))
            if question:
                hashes.add(digest(normalized(question)))
    return hashes, files


def valid_answer_spans(context, answers):
    return bool(answers) and all(
        isinstance(a.get("answer_start"), int)
        and a["answer_start"] >= 0
        and isinstance(a.get("text"), str)
        and bool(a["text"].strip())
        and context[a["answer_start"] : a["answer_start"] + len(a["text"])] == a["text"]
        for a in answers
    )


def build(config, source_dir, output, tokenizer, previous_datasets):
    if output.exists():
        raise ValueError("Refusing to overwrite a frozen dataset")
    for name, spec in config["sources"].items():
        if file_hash(source_dir / name) != spec["sha256"]:
            raise ValueError("Source hash mismatch: " + name)
    previous, history_locks = existing_questions(previous_datasets)
    counters, articles = Counter(), {"train": {}, "dev": {}}
    specials = tokenizer.num_special_tokens_to_add(pair=True)
    all_context_roles = {}
    for split in articles:
        original = json.loads((source_dir / f"{split}-v2.0.json").read_text(encoding="utf-8"))
        for article in original["data"]:
            title = article["title"]
            pairs = []
            for number, paragraph in enumerate(article["paragraphs"]):
                context = paragraph["context"]
                context_hash = digest(normalized(context))
                context_ids = tokenizer.encode(context, add_special_tokens=False)
                if len(context_ids) + specials >= config["max_length"]:
                    counters["paragraph_too_long"] += 1
                    continue
                candidates = {True: [], False: []}
                for q in paragraph["qas"]:
                    answerable = not q["is_impossible"]
                    if answerable and not valid_answer_spans(context, q.get("answers", [])):
                        counters["invalid_positive_span"] += 1
                        continue
                    if not answerable and q.get("answers"):
                        raise ValueError("Unanswerable source item has positive spans")
                    question_hash = digest(normalized(q["question"]))
                    if question_hash in previous:
                        counters["previous_question_overlap"] += 1
                        continue
                    length = len(tokenizer.encode(q["question"], add_special_tokens=False))
                    length += len(context_ids) + specials
                    if length > config["max_length"]:
                        counters["question_context_too_long"] += 1
                        continue
                    candidates[answerable].append((q, length, question_hash))
                if not all(candidates.values()):
                    counters["paragraph_without_both_labels"] += 1
                    continue
                chosen = [
                    min(
                        candidates[label], key=lambda x: digest("support-v1/question/" + x[0]["id"])
                    )
                    for label in (True, False)
                ]
                pairs.append((context_hash, number, context, chosen))
                all_context_roles.setdefault(context_hash, set()).add((split, title))
            articles[split][title] = pairs
    if set(articles["train"]) & set(articles["dev"]):
        raise ValueError("Source training and dev article titles overlap")
    # Exclude repeated passages across articles, including repetitions within one role.
    for split, documents in articles.items():
        for title, pairs in documents.items():
            articles[split][title] = [p for p in pairs if len(all_context_roles[p[0]]) == 1]
    eligible = sorted(
        [
            d
            for d, pairs in articles["train"].items()
            if len(pairs) >= config["minimum_training_article_pairs"]
        ],
        key=lambda d: digest("support-v1/article/" + d),
    )
    n_fit, n_select, n_cal = [
        config[key + "_articles"] for key in ("training", "selection", "calibration")
    ]
    if len(eligible) < n_fit + n_select + n_cal:
        raise ValueError("Insufficient eligible source articles")
    roles = {
        "fit": ("train", eligible[:n_fit]),
        "selection": ("train", eligible[n_fit : n_fit + n_select]),
        "calibration": ("train", eligible[n_fit + n_select : n_fit + n_select + n_cal]),
        "test": ("dev", sorted(articles["dev"])),
    }
    rows, used_questions, used_contexts = [], set(), set()
    for role, (split, titles) in roles.items():
        for title in titles:
            selected = 0
            for chash, number, context, chosen in sorted(
                articles[split][title], key=lambda p: digest("support-v1/paragraph/" + p[0])
            ):
                hashes = [qhash for _, _, qhash in chosen]
                if chash in used_contexts or set(hashes) & used_questions or len(set(hashes)) != 2:
                    counters["duplicate_selected_context_or_question"] += 1
                    continue
                used_contexts.add(chash)
                used_questions.update(hashes)
                for q, length, qhash in chosen:
                    actual = tokenizer(q["question"], context, truncation=False)["input_ids"]
                    if len(actual) != length:
                        raise ValueError("Tokenizer length accounting mismatch")
                    rows.append(
                        {
                            "id": q["id"],
                            "pair_id": chash,
                            "document": title,
                            "role": role,
                            "question": q["question"],
                            "context": context,
                            "answerable": not q["is_impossible"],
                            "input_tokens": length,
                            "source_split": split,
                            "source_paragraph_index": number,
                            "answers": q.get("answers", []),
                            "plausible_answers": q.get("plausible_answers", []),
                            "question_hash": qhash,
                            "annotation": "Original SQuAD 2.0 human labels; texts and spans unchanged",
                        }
                    )
                selected += 1
                if selected == config["pairs_per_article"][role]:
                    break
            if not selected:
                raise ValueError("Article has no selected pairs: " + title)
    excluded = set(config.get("excluded_fit_pair_ids", []))
    if any(r["role"] != "fit" and r["pair_id"] in excluded for r in rows):
        raise ValueError("Pre-outcome audit exclusions must belong to fit only")
    counters["pre_outcome_audit_excluded_fit_examples"] = sum(
        r["pair_id"] in excluded for r in rows
    )
    rows = [r for r in rows if r["pair_id"] not in excluded]
    validate(rows)
    output.mkdir(parents=True)
    data_path = output / "examples.jsonl"
    data_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    (output / "LICENSE-DATA.txt").write_bytes((source_dir / "cc-by-sa-4.0.txt").read_bytes())
    manifest = {
        "id": config["id"],
        "license": "CC-BY-SA-4.0",
        "sources": config["sources"],
        "selection": "Hash-order articles and paragraphs; one human-positive and one human-negative question per identical paragraph; no model scores used",
        "tokenizer": {
            "model": config["model"],
            "revision": config["revision"],
            "max_length": config["max_length"],
            "truncation": False,
        },
        "counts": {
            role: {
                "articles": len({r["document"] for r in rows if r["role"] == role}),
                "paragraphs": len({r["pair_id"] for r in rows if r["role"] == role}),
                "answerable": sum(r["answerable"] for r in rows if r["role"] == role),
                "unanswerable": sum(not r["answerable"] for r in rows if r["role"] == role),
            }
            for role in roles
        },
        "excluded_or_ineligible": dict(counters),
        "pre_outcome_audit_excluded_fit_pairs": sorted(excluded),
        "previous_question_files": history_locks,
        "novelty_scope": "No exact normalized question reuse or repeated normalized paragraph across roles/articles; pretraining and semantic overlap are unknown",
        "files": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def validate(rows):
    ids, questions, document_roles, pairs = set(), set(), {}, {}
    for row in rows:
        if row["id"] in ids or normalized(row["question"]) in questions:
            raise ValueError("Duplicate support question")
        ids.add(row["id"])
        questions.add(normalized(row["question"]))
        if row["question_hash"] != digest(normalized(row["question"])):
            raise ValueError("Question hash mismatch")
        if row["pair_id"] != digest(normalized(row["context"])):
            raise ValueError("Context hash mismatch")
        if type(row["answerable"]) is not bool or (not row["answerable"] and row["answers"]):
            raise ValueError("Invalid support label")
        if document_roles.setdefault(row["document"], row["role"]) != row["role"]:
            raise ValueError("Article crosses partitions")
        if not 1 <= row["input_tokens"] <= 512:
            raise ValueError("Invalid full-context token length")
        if row["answerable"] and not valid_answer_spans(row["context"], row["answers"]):
            raise ValueError("Invalid positive evidence span")
        if row["source_split"] != ("dev" if row["role"] == "test" else "train"):
            raise ValueError("Invalid original source split")
        pairs.setdefault(row["pair_id"], []).append(row)
    for pair in pairs.values():
        if len(pair) != 2 or {r["answerable"] for r in pair} != {True, False}:
            raise ValueError("Each paragraph requires opposite labels")
        if any(pair[0][k] != pair[1][k] for k in ("context", "document", "role")):
            raise ValueError("Paired questions must use the identical context and partition")
    if set(document_roles.values()) != {"fit", "selection", "calibration", "test"}:
        raise ValueError("Missing support partition")


def load(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if file_hash(directory / name) != expected:
            raise ValueError("Support corpus hash mismatch")
    rows = [
        json.loads(line)
        for line in (directory / "examples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    validate(rows)
    return manifest, rows
