import httpx
import pytest


class TestZeroEntropyEmbeddingModel:
    def test_payload_defaults_to_document_embeddings(self):
        from open_notebook.ai.zeroentropy import ZeroEntropyEmbeddingModel

        model = ZeroEntropyEmbeddingModel(
            model_name="zembed-1",
            config={"api_key": "ze-test", "dimensions": 1280, "latency": "fast"},
        )

        payload = model._payload(["hello"])
        model.close()

        assert payload == {
            "model": "zembed-1",
            "input_type": "document",
            "input": ["hello"],
            "encoding_format": "float",
            "dimensions": 1280,
            "latency": "fast",
        }

    def test_credential_base_url_is_ignored(self):
        from open_notebook.ai.zeroentropy import (
            ZEROENTROPY_API_BASE_URL,
            ZeroEntropyEmbeddingModel,
        )

        model = ZeroEntropyEmbeddingModel(
            model_name="zembed-1",
            config={"api_key": "ze-test", "base_url": "http://attacker.example/v1"},
        )

        try:
            assert model.base_url == ZEROENTROPY_API_BASE_URL
        finally:
            model.close()

    def test_payload_allows_query_override(self):
        from open_notebook.ai.zeroentropy import ZeroEntropyEmbeddingModel

        model = ZeroEntropyEmbeddingModel(
            model_name="zembed-1",
            config={"api_key": "ze-test"},
        )

        payload = model._payload(["search text"], input_type="query", dimensions=640)
        model.close()

        assert payload["input_type"] == "query"
        assert payload["dimensions"] == 640

    def test_parse_response_returns_embeddings_in_order(self):
        from open_notebook.ai.zeroentropy import ZeroEntropyEmbeddingModel

        model = ZeroEntropyEmbeddingModel(
            model_name="zembed-1",
            config={"api_key": "ze-test"},
        )
        response = httpx.Response(
            200,
            json={
                "results": [
                    {"embedding": [0.1, 0.2]},
                    {"embedding": [0.3, 0.4]},
                ],
                "usage": {"total_bytes": 1, "total_tokens": 1},
            },
            request=httpx.Request("POST", "https://api.zeroentropy.dev/v1/models/embed"),
        )

        result = model._parse_response(response)
        model.close()

        assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_parse_response_rejects_missing_results(self):
        from open_notebook.ai.zeroentropy import ZeroEntropyEmbeddingModel

        model = ZeroEntropyEmbeddingModel(
            model_name="zembed-1",
            config={"api_key": "ze-test"},
        )
        response = httpx.Response(
            200,
            json={"usage": {"total_bytes": 1}},
            request=httpx.Request("POST", "https://api.zeroentropy.dev/v1/models/embed"),
        )

        try:
            with pytest.raises(RuntimeError, match="missing results"):
                model._parse_response(response, expected_count=1)
        finally:
            model.close()

    def test_parse_response_rejects_count_mismatch(self):
        from open_notebook.ai.zeroentropy import ZeroEntropyEmbeddingModel

        model = ZeroEntropyEmbeddingModel(
            model_name="zembed-1",
            config={"api_key": "ze-test"},
        )
        response = httpx.Response(
            200,
            json={"results": []},
            request=httpx.Request("POST", "https://api.zeroentropy.dev/v1/models/embed"),
        )

        try:
            with pytest.raises(RuntimeError, match="0 embeddings for 1 inputs"):
                model._parse_response(response, expected_count=1)
        finally:
            model.close()

    def test_error_message_is_sanitized(self):
        from open_notebook.ai.zeroentropy import ZeroEntropyEmbeddingModel

        model = ZeroEntropyEmbeddingModel(
            model_name="zembed-1",
            config={"api_key": "ze-test"},
        )
        response = httpx.Response(
            500,
            text="secret notebook content",
            request=httpx.Request("POST", "https://api.zeroentropy.dev/v1/models/embed"),
        )

        try:
            with pytest.raises(RuntimeError) as exc:
                model._handle_error(response)
            assert "secret notebook content" not in str(exc.value)
            assert "HTTP 500" in str(exc.value)
        finally:
            model.close()

    def test_invalid_dimensions_are_rejected(self):
        from open_notebook.ai.zeroentropy import ZeroEntropyEmbeddingModel

        with pytest.raises(ValueError, match="dimensions"):
            ZeroEntropyEmbeddingModel(
                model_name="zembed-1",
                config={"api_key": "ze-test", "dimensions": 123},
            )

    def test_invalid_payload_input_type_is_rejected(self):
        from open_notebook.ai.zeroentropy import ZeroEntropyEmbeddingModel

        model = ZeroEntropyEmbeddingModel(
            model_name="zembed-1",
            config={"api_key": "ze-test"},
        )

        try:
            with pytest.raises(ValueError, match="input_type"):
                model._payload(["test"], input_type="invalid")
        finally:
            model.close()

    def test_missing_api_key_is_clear(self, monkeypatch):
        from open_notebook.ai.zeroentropy import ZeroEntropyEmbeddingModel

        monkeypatch.delenv("ZEROENTROPY_API_KEY", raising=False)

        with pytest.raises(ValueError, match="ZeroEntropy API key not found"):
            ZeroEntropyEmbeddingModel(model_name="zembed-1")
