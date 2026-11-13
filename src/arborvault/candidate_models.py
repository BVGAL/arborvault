"""Optional GPU scorers; both consume the exact same fitted candidate text."""

import json

from .candidate_scoring import SYSTEM
from .qwen3 import prompt_tokens


def conversation(question, text):
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": json.dumps(
                {"question": question, "candidate": json.loads(text)}, ensure_ascii=False
            ),
        },
    ]


def fit_text(question, text, qwen_tokenizer, cross_tokenizer, max_length=512):
    """Shorten only excerpts, equally for both scorers; retain full question/path."""
    value = json.loads(text)
    original = len(value["excerpt"])
    while True:
        result = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        qlen = len(prompt_tokens(qwen_tokenizer, conversation(question, result)))
        clen = len(cross_tokenizer(question, result, add_special_tokens=True)["input_ids"])
        if max(qlen, clen) <= max_length:
            return result, {
                "qwen_tokens": qlen,
                "cross_encoder_tokens": clen,
                "excerpt_characters_removed": original - len(value["excerpt"]),
            }
        if not value["excerpt"]:
            raise ValueError("Question/path alone exceeds the shared candidate budget")
        value["excerpt"] = value["excerpt"][
            : max(0, len(value["excerpt"]) - max(32, len(value["excerpt"]) // 10))
        ]


class QwenScorer:
    def __init__(self, model, tokenizer, batch_size=8):
        self.model, self.tokenizer, self.batch_size = model, tokenizer, batch_size
        labels = [tokenizer.encode(label, add_special_tokens=False) for label in ("no", "yes")]
        if any(len(ids) != 1 for ids in labels) or labels[0] == labels[1]:
            raise ValueError("Binary verbalizers must be distinct single tokens")
        self.no, self.yes = (ids[0] for ids in labels)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    def logits(self, questions, texts):
        rows = [
            prompt_tokens(self.tokenizer, conversation(q, t))
            for q, t in zip(questions, texts, strict=True)
        ]
        if any(len(row) > 512 for row in rows):
            raise ValueError("Unfitted scoring input")
        features = self.tokenizer.pad({"input_ids": rows}, padding=True, return_tensors="pt").to(
            self.model.device
        )
        # Left padding requires explicit positions to match individual unpadded forwards.
        positions = features["attention_mask"].long().cumsum(-1) - 1
        positions.masked_fill_(features["attention_mask"] == 0, 0)
        result = (
            self.model(**features, position_ids=positions, use_cache=False, logits_to_keep=1)
            .logits[:, -1]
            .float()
        )
        return result[:, self.yes] - result[:, self.no], [len(row) for row in rows]

    def score(self, questions, texts):
        import torch

        self.model.eval()
        scores, counts = [], []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                logits, length = self.logits(
                    questions[start : start + self.batch_size],
                    texts[start : start + self.batch_size],
                )
                scores.extend(logits.cpu().tolist())
                counts.extend(length)
        return scores, counts


class CrossEncoderScorer:
    def __init__(self, model, tokenizer, batch_size=32):
        self.model, self.tokenizer, self.batch_size = model, tokenizer, batch_size

    def logits(self, questions, texts):
        features = self.tokenizer(
            questions, texts, padding=True, truncation=False, return_tensors="pt"
        ).to(self.model.device)
        if features["input_ids"].shape[1] > 512:
            raise ValueError("Unfitted scoring input")
        values = self.model(**features).logits
        if values.shape[-1] != 1:
            raise ValueError("Expected a scalar cross-encoder")
        return values[:, 0].float(), features["attention_mask"].sum(-1).tolist()

    def score(self, questions, texts):
        import torch

        self.model.eval()
        scores, counts = [], []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                logits, length = self.logits(
                    questions[start : start + self.batch_size],
                    texts[start : start + self.batch_size],
                )
                scores.extend(logits.cpu().tolist())
                counts.extend(length)
        return scores, counts
