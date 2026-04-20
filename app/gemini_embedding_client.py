from __future__ import annotations

import json
from typing import Iterable
from urllib import error, request


class GeminiEmbeddingClient:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        output_dimensionality: int = 768,
        timeout_seconds: int = 30,
        batch_size: int = 32,
    ) -> None:
        """HTTP client tối giản gọi Gemini Embedding API (batchEmbedContents).

        - Chia batch theo `batch_size` để tránh payload quá lớn.
        - Thử gửi `outputDimensionality` (nếu API reject → retry không gửi field này).
        """
        self.api_key = api_key
        self.model = model
        self.output_dimensionality = output_dimensionality
        self.timeout_seconds = timeout_seconds
        self.batch_size = max(1, batch_size)
        self.endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:batchEmbedContents"
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed danh sách text thành danh sách vector float (giữ thứ tự input)."""
        if not texts:
            return []

        output: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            output.extend(self._embed_batch(batch))
        return output

    def _embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed một batch (internal)."""
        requests_payload = []
        for text in texts:
            body = {
                "model": f"models/{self.model}",
                "content": {"parts": [{"text": text}]},
            }
            # outputDimensionality is optional; some API versions may ignore/deny it.
            if self.output_dimensionality > 0:
                body["outputDimensionality"] = self.output_dimensionality
            requests_payload.append(body)

        payload = {"requests": requests_payload}
        try:
            data = self._post_json(payload)
        except RuntimeError as exc:
            # Retry without outputDimensionality for API compatibility.
            if "outputDimensionality" in str(exc):
                for item in requests_payload:
                    item.pop("outputDimensionality", None)
                data = self._post_json({"requests": requests_payload})
            else:
                raise

        embeddings = data.get("embeddings") or []
        vectors: list[list[float]] = []
        for emb in embeddings:
            values = emb.get("values") or []
            vectors.append([float(v) for v in values])
        return vectors

    def _post_json(self, payload: dict) -> dict:
        """POST JSON đến Gemini endpoint; raise RuntimeError với message giàu ngữ cảnh."""
        req = request.Request(
            self.endpoint,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            data=json.dumps(payload).encode("utf-8"),
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Gemini embedding HTTP {exc.code}: {body}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Gemini embedding network error: {exc}") from exc

