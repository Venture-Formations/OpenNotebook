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

    def test_missing_api_key_is_clear(self, monkeypatch):
        from open_notebook.ai.zeroentropy import ZeroEntropyEmbeddingModel

        monkeypatch.delenv("ZEROENTROPY_API_KEY", raising=False)

        with pytest.raises(ValueError, match="ZeroEntropy API key not found"):
            ZeroEntropyEmbeddingModel(model_name="zembed-1")
