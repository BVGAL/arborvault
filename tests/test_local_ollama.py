import json

import httpx
import pytest

from arborvault.domain import Node
from arborvault.llm_eval import local_model_provenance, thinking_value
from arborvault.ollama import OllamaSelector


def test_local_context_and_non_thinking_are_explicit():
    captured = []

    def respond(request):
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "done": True,
                "message": {"content": '["0"]'},
                "prompt_eval_count": 100,
                "eval_count": 5,
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        selector = OllamaSelector(
            "http://127.0.0.1:11434/api",
            "qwen3:4b",
            think=False,
            num_ctx=8192,
            max_output_tokens=256,
            client=client,
        )
        assert selector.select(
            "question", [Node("id", None, "Title", "Text", ["Title"], 0)], 4
        ) == ["id"]
    assert captured[0]["think"] is False
    assert captured[0]["options"]["num_ctx"] == 8192
    assert captured[0]["options"]["num_predict"] == 256
    assert thinking_value("off") is False and thinking_value("default") is None


def test_possible_context_truncation_is_reported_as_failure():
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                json={
                    "done": True,
                    "message": {"content": "[]"},
                    "prompt_eval_count": 8000,
                },
            )
        )
    ) as client:
        selector = OllamaSelector(
            "http://localhost:11434/api",
            "model",
            num_ctx=8192,
            max_output_tokens=256,
            client=client,
        )
        with pytest.raises(ValueError, match="context budget"):
            selector.select("question", [], 4)
        assert selector.usage()["failures"] == 1


def test_provenance_excludes_modelfile_paths_and_requires_installed_model():
    def respond(request):
        data = {
            "/api/version": {"version": "test-runtime"},
            "/api/tags": {"models": [{"name": "model", "digest": "sha256:test", "size": 123}]},
            "/api/show": {
                "modelfile": "private-machine-path",
                "template": "template",
                "parameters": "parameters",
                "details": {"quantization_level": "Q4_K_M"},
            },
        }
        return httpx.Response(200, json=data[request.url.path])

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = local_model_provenance("http://localhost:11434/api", "model", client)
        assert result["model_digest"] == "sha256:test"
        assert "private-machine-path" not in json.dumps(result)
        with pytest.raises(ValueError, match="installed"):
            local_model_provenance("http://localhost:11434/api", "missing", client)
