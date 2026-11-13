"""Reference control: standard passage text instead of the scorers' JSON representation."""

import argparse
import json
import os
import time
from pathlib import Path

from arborvault.candidate_scoring import BM25
from arborvault.qasper import load_split
from arborvault.qwen3 import file_hash
from arborvault.routing_eval import context, score, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing reference results")
    import torch
    from transformers import AutoModel, AutoTokenizer

    config = json.loads(Path("configs/candidate-scoring-v1.json").read_text())
    dataset = Path("datasets/qasper-routing-v2")
    _, trees, queries = load_split(dataset, "dev")
    tokenizer = AutoTokenizer.from_pretrained(
        config["embedding_model"], revision=config["embedding_revision"], trust_remote_code=False
    )
    model = (
        AutoModel.from_pretrained(
            config["embedding_model"],
            revision=config["embedding_revision"],
            trust_remote_code=False,
        )
        .to("cuda")
        .eval()
    )
    torch.set_num_threads(4)
    started = time.monotonic()

    def embed(texts):
        values = []
        with torch.inference_mode():
            for offset in range(0, len(texts), 32):
                inputs = tokenizer(
                    texts[offset : offset + 32],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to("cuda")
                hidden = model(**inputs).last_hidden_state.masked_fill(
                    ~inputs["attention_mask"][..., None].bool(), 0
                )
                values.append(
                    torch.nn.functional.normalize(
                        hidden.sum(1) / inputs["attention_mask"].sum(1)[..., None], dim=1
                    )
                )
        return torch.cat(values)

    indexed, rows = {}, {"dense_e5_text": [], "bm25_text": [], "hybrid_rrf_text": []}
    for query in queries:
        document = query["document"]
        if document not in indexed:
            tree = trees[document]
            leaves = [n for n in tree.nodes.values() if not tree.children[n.id]]
            texts = [" / ".join(n.path[1:]) + "\n" + n.content for n in leaves]
            indexed[document] = (leaves, embed(["passage: " + t for t in texts]), BM25(texts))
        leaves, vectors, lexical = indexed[document]
        dense = (vectors @ embed(["query: " + query["question"]])[0]).tolist()
        lex = lexical.scores(query["question"])
        orders = {
            "dense_e5_text": sorted(range(len(leaves)), key=lambda i: (-dense[i], i)),
            "bm25_text": sorted(range(len(leaves)), key=lambda i: (-lex[i], i)),
        }
        fusion = [0.0] * len(leaves)
        for ranking in orders.values():
            for rank, i in enumerate(ranking, 1):
                fusion[i] += 1 / (60 + rank)
        orders["hybrid_rrf_text"] = sorted(range(len(leaves)), key=lambda i: (-fusion[i], i))
        for method, ranking in orders.items():
            selected = context([leaves[i] for i in ranking])
            rows[method].append(
                {
                    "id": query["id"],
                    "document": document,
                    "answerable": query["answerable"],
                    "predicted": [n.id for n, _ in selected],
                    **score(selected, query),
                }
            )
    result = {
        "methods": {name: summarize(values) for name, values in rows.items()},
        "scope": "Additional development reference, no training or tuning",
        "representation": "Section/paragraph path followed by full original passage text; standard E5 query/passage prefixes; 512-token model limit",
        "reason": "Avoid attributing a neural-scorer advantage to an embedding baseline's JSON input format",
        "model": config["embedding_model"],
        "revision": config["embedding_revision"],
        "dataset_sha256": file_hash(dataset / "manifest.json"),
        "code_revision": os.getenv("ARBORVAULT_CODE_REVISION", "unknown"),
        "elapsed_seconds": time.monotonic() - started,
        "k": 5,
        "word_budget": 512,
        "unknown_threshold": None,
        "gpu": torch.cuda.get_device_name(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {"complete": True, "methods": len(rows), "elapsed_seconds": result["elapsed_seconds"]}
        )
    )


if __name__ == "__main__":
    main()
