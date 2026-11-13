"""Optional Qwen3 training/evaluation helpers; GPU dependencies are imported on demand."""

import hashlib
import json
import re
import time
from importlib.metadata import version
from pathlib import Path

from .ollama import decode_selection, routing_messages


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_recipe(path):
    recipe = json.loads(Path(path).read_text(encoding="utf-8"))
    if recipe["base_model"] != "Qwen/Qwen3-4B" or not re.fullmatch(
        r"[0-9a-f]{40}", recipe["base_revision"]
    ):
        raise ValueError("This recipe requires the original Qwen3-4B and an immutable revision")
    for field in ("epochs", "gradient_accumulation_steps", "max_length", "lora_r", "lora_alpha"):
        if type(recipe[field]) is not int or recipe[field] < 1:
            raise ValueError("Expected positive integer: " + field)
    if recipe["learning_rate"] <= 0 or not 0 <= recipe["lora_dropout"] < 1:
        raise ValueError("Invalid learning rate or dropout")
    if recipe["thinking"] is not False or recipe["quantization"] != "nf4-double-bf16":
        raise ValueError("Only the fixed non-thinking NF4 protocol is supported")
    if recipe["evaluation_num_ctx"] <= recipe["evaluation_max_output_tokens"]:
        raise ValueError("Context must exceed output budget")
    return recipe


def prompt_tokens(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def encode_completion(tokenizer, messages, max_length):
    """Train the exact inference prefix followed by JSON and EOS; never truncate targets."""
    if [m["role"] for m in messages] != ["system", "user", "assistant"]:
        raise ValueError("Expected one routing conversation")
    prefix = prompt_tokens(tokenizer, messages[:-1])
    completion = tokenizer.encode(messages[-1]["content"], add_special_tokens=False)
    if not completion or tokenizer.eos_token_id is None:
        raise ValueError("Missing completion or EOS token")
    completion += [tokenizer.eos_token_id]
    if len(prefix) + len(completion) > max_length:
        raise ValueError("Example exceeds max_length; refusing silent truncation")
    return {"input_ids": prefix + completion, "completion_length": len(completion)}


def packages():
    return {
        name: version(name)
        for name in ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "safetensors")
    }


def load_model(recipe):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("A CUDA GPU supporting BF16 is required")
    tokenizer = AutoTokenizer.from_pretrained(
        recipe["base_model"], revision=recipe["base_revision"], trust_remote_code=False
    )
    if not tokenizer.chat_template:
        raise ValueError("The original model chat template is required")
    model = AutoModelForCausalLM.from_pretrained(
        recipe["base_model"],
        revision=recipe["base_revision"],
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
    if model.config.model_type != "qwen3":
        raise ValueError("Unexpected architecture")
    return model, tokenizer


def completion_loss(model, example):
    """Equivalent to masked causal-LM loss, without materializing vocabulary logits for prompts."""
    import torch
    from torch.nn import functional as F

    ids = torch.tensor([example["input_ids"]], device=model.device)
    count = example["completion_length"]
    # The logit at the last prompt token predicts the first completion token.
    output = model(input_ids=ids, use_cache=False, logits_to_keep=count + 1)
    if output.logits.shape[1] != count + 1:
        raise ValueError("Runtime did not honor completion-only logits")
    return F.cross_entropy(output.logits[0, :-1].float(), ids[0, -count:]), count


class TransformersSelector:
    """Sequential experiment selector, sharing the production routing prompt and decoder."""

    def __init__(self, model, tokenizer, *, num_ctx=8192, max_output_tokens=256, max_calls=1056):
        if min(max_output_tokens, max_calls) < 1 or num_ctx <= max_output_tokens:
            raise ValueError("Invalid inference budget")
        self.model, self.tokenizer = model, tokenizer
        self.num_ctx, self.max_output_tokens, self.max_calls = num_ctx, max_output_tokens, max_calls
        self.calls, self.events = 0, []

    def select(self, query, candidates, width):
        import torch

        if self.calls >= self.max_calls:
            raise ValueError("Selector call budget exhausted")
        self.calls += 1
        event = {"ok": False, "prompt_tokens": 0, "output_tokens": 0}
        start = time.perf_counter()
        try:
            messages, labels = routing_messages(query, candidates, width)
            tokens = prompt_tokens(self.tokenizer, messages)
            event["prompt_tokens"] = len(tokens)
            if len(tokens) + self.max_output_tokens > self.num_ctx:
                raise ValueError("Prompt exceeds context budget; refusing truncation")
            ids = torch.tensor([tokens], device=self.model.device)
            with torch.inference_mode():
                result = self.model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    do_sample=False,
                    max_new_tokens=self.max_output_tokens,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )[0, len(tokens) :].tolist()
            event["output_tokens"] = len(result)
            if not result or result[-1] != self.tokenizer.eos_token_id:
                raise ValueError("Incomplete selector result")
            answer = self.tokenizer.decode(result, skip_special_tokens=True)
            selected = decode_selection(answer.strip(), labels, width)
            event["ok"] = True
            return selected
        finally:
            event["latency_ms"] = (time.perf_counter() - start) * 1000
            self.events.append(event)

    def usage(self):
        return {
            "calls": self.calls,
            "failures": sum(not event["ok"] for event in self.events),
            "prompt_tokens": sum(event["prompt_tokens"] for event in self.events),
            "output_tokens": sum(event["output_tokens"] for event in self.events),
        }
