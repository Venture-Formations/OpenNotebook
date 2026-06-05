"""ZeroEntropy embedding provider for Open Notebook."""

import os
from typing import Any, Dict, List, Optional

import httpx
from esperanto.providers.embedding.base import EmbeddingModel, Model
from esperanto.utils import validate_and_decode_embedding

ZEROENTROPY_API_BASE_URL = "https://api.zeroentropy.dev/v1"
ALLOWED_DIMENSIONS = {2560, 1280, 640, 320, 160, 80, 40}


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

        # Do not accept credential-provided base_url for hosted ZeroEntropy.
        # It would allow any app user with credential access to redirect future
        # source/note embeddings plus the provider bearer token to an arbitrary URL.
        # Admin-controlled env override remains available for regional endpoints.
        self.base_url = (
            base_url
            or os.getenv("ZEROENTROPY_BASE_URL")
            or ZEROENTROPY_API_BASE_URL
        ).rstrip("/")
        self.input_type = (
            config.get("input_type")
            or "document"
        )
        self.dimensions = int(config.get("dimensions") or os.getenv("ZEROENTROPY_DIMENSIONS") or 2560)
        self.latency = config.get("latency") or os.getenv("ZEROENTROPY_LATENCY")
        timeout = float(config.get("timeout", os.getenv("ZEROENTROPY_TIMEOUT", 120.0)))

        if self.input_type not in {"query", "document"}:
            raise ValueError("ZeroEntropy input_type must be either 'query' or 'document'")
        if self.dimensions not in ALLOWED_DIMENSIONS:
            allowed = ", ".join(str(d) for d in sorted(ALLOWED_DIMENSIONS, reverse=True))
            raise ValueError(f"ZeroEntropy dimensions must be one of: {allowed}")

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
            code = error_data.get("code") or error_data.get("error")
            suffix = f" ({code})" if isinstance(code, str) and code else ""
        except Exception:
            suffix = ""

        if response.status_code == 401:
            error_message = "Invalid API key"
        elif response.status_code == 403:
            error_message = "API key lacks required permissions"
        elif response.status_code == 429:
            error_message = "Rate limit exceeded"
        else:
            error_message = f"HTTP {response.status_code}{suffix}"

        raise RuntimeError(f"ZeroEntropy API error: {error_message}")

    def _payload(self, texts: List[str], **kwargs) -> Dict[str, Any]:
        input_type = kwargs.pop("input_type", self.input_type)
        dimensions = int(kwargs.pop("dimensions", self.dimensions))
        latency = kwargs.pop("latency", self.latency)
        if input_type not in {"query", "document"}:
            raise ValueError("ZeroEntropy input_type must be either 'query' or 'document'")
        if dimensions not in ALLOWED_DIMENSIONS:
            allowed = ", ".join(str(d) for d in sorted(ALLOWED_DIMENSIONS, reverse=True))
            raise ValueError(f"ZeroEntropy dimensions must be one of: {allowed}")
        payload: Dict[str, Any] = {
            "model": self.get_model_name(),
            "input_type": input_type,
            "input": texts,
            "encoding_format": "float",
            **kwargs,
        }
        payload["dimensions"] = dimensions
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
        return self._parse_response(response, expected_count=len(texts))

    async def aembed(self, texts: List[str], **kwargs) -> List[List[float]]:
        texts = [self._clean_text(text) for text in texts]
        response = await self.async_client.post(
            f"{self.base_url}/models/embed",
            headers=self._get_headers(),
            json=self._payload(texts, **kwargs),
        )
        self._handle_error(response)
        return self._parse_response(response, expected_count=len(texts))

    def _parse_response(
        self, response: httpx.Response, expected_count: Optional[int] = None
    ) -> List[List[float]]:
        response_data = response.json()
        results_data = response_data.get("results")
        if not isinstance(results_data, list):
            raise RuntimeError("ZeroEntropy API response missing results")
        if expected_count is not None and len(results_data) != expected_count:
            raise RuntimeError(
                "ZeroEntropy API returned "
                f"{len(results_data)} embeddings for {expected_count} inputs"
            )
        results = []
        for idx, data in enumerate(results_data):
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
