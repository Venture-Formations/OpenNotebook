"""ZeroEntropy embedding provider for Open Notebook."""

import os
from typing import Any, Dict, List, Optional

import httpx
from esperanto.providers.embedding.base import EmbeddingModel, Model
from esperanto.utils import validate_and_decode_embedding


class ZeroEntropyEmbeddingModel(EmbeddingModel):
    """Embedding adapter for ZeroEntropy's native models/embed API."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        config = config or {}
        config.update(kwargs)

        self.api_key = api_key or config.get("api_key") or os.getenv("ZEROENTROPY_API_KEY")
        if not self.api_key:
            raise ValueError("ZeroEntropy API key not found")

        self.base_url = (
            base_url
            or config.get("base_url")
            or os.getenv("ZEROENTROPY_BASE_URL")
            or "https://api.zeroentropy.dev/v1"
        ).rstrip("/")
        self.input_type = (
            config.get("input_type")
            or os.getenv("ZEROENTROPY_INPUT_TYPE")
            or "document"
        )
        self.dimensions = config.get("dimensions") or os.getenv("ZEROENTROPY_DIMENSIONS")
        self.latency = config.get("latency") or os.getenv("ZEROENTROPY_LATENCY")
        timeout = float(config.get("timeout", os.getenv("ZEROENTROPY_TIMEOUT", 120.0)))

        if self.input_type not in {"query", "document"}:
            raise ValueError("ZeroEntropy input_type must be either 'query' or 'document'")

        clean_config = {
            k: v
            for k, v in config.items()
            if k
            not in {
                "api_key",
                "base_url",
                "dimensions",
                "input_type",
                "latency",
                "timeout",
            }
        }
        clean_config["timeout"] = timeout

        super().__init__(
            model_name=model_name or self._get_default_model(),
            api_key=self.api_key,
            base_url=self.base_url,
            config=clean_config,
        )
        self._create_http_clients()

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _handle_error(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return

        try:
            error_data = response.json()
            detail = error_data.get("detail") or error_data.get("message")
            if isinstance(detail, list):
                detail = "; ".join(str(item) for item in detail)
            elif isinstance(detail, dict):
                detail = detail.get("message") or str(detail)
            error_message = detail or f"HTTP {response.status_code}"
        except Exception:
            error_message = f"HTTP {response.status_code}: {response.text}"

        raise RuntimeError(f"ZeroEntropy API error: {error_message}")

    def _payload(self, texts: List[str], **kwargs) -> Dict[str, Any]:
        input_type = kwargs.pop("input_type", self.input_type)
        dimensions = kwargs.pop("dimensions", self.dimensions)
        latency = kwargs.pop("latency", self.latency)
        payload: Dict[str, Any] = {
            "model": self.get_model_name(),
            "input_type": input_type,
            "input": texts,
            "encoding_format": "float",
            **kwargs,
        }
        if dimensions:
            payload["dimensions"] = int(dimensions)
        if latency:
            payload["latency"] = latency
        return payload

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        texts = [self._clean_text(text) for text in texts]
        response = self.client.post(
            f"{self.base_url}/models/embed",
            headers=self._get_headers(),
            json=self._payload(texts, **kwargs),
        )
        self._handle_error(response)
        return self._parse_response(response)

    async def aembed(self, texts: List[str], **kwargs) -> List[List[float]]:
        texts = [self._clean_text(text) for text in texts]
        response = await self.async_client.post(
            f"{self.base_url}/models/embed",
            headers=self._get_headers(),
            json=self._payload(texts, **kwargs),
        )
        self._handle_error(response)
        return self._parse_response(response)

    def _parse_response(self, response: httpx.Response) -> List[List[float]]:
        response_data = response.json()
        results = []
        for idx, data in enumerate(response_data.get("results", [])):
            raw = data.get("embedding")
            results.append(validate_and_decode_embedding(idx, raw))
        return results

    def _get_default_model(self) -> str:
        return "zembed-1"

    @property
    def provider(self) -> str:
        return "zeroentropy"

    def _get_models(self) -> List[Model]:
        return [Model(id="zembed-1", owned_by="zeroentropy", context_window=None)]
