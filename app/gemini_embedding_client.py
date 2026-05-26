from __future__ import annotations

from typing import Iterable

from google import genai
from google.genai import types


class GeminiEmbeddingClient:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        output_dimensionality: int = 768,
        timeout_seconds: int = 30,
        batch_size: int = 32,
        use_vertexai: bool = False,
        google_cloud_project: str = "",
        google_cloud_location: str = "us-central1",
    ) -> None:
        """Embedding client for Gemini API key mode or Vertex AI mode."""
        self.api_key = api_key
        self.model = model
        self.output_dimensionality = output_dimensionality
        self.timeout_seconds = timeout_seconds
        self.batch_size = max(1, batch_size)
        self.use_vertexai = use_vertexai
        self.google_cloud_project = google_cloud_project
        self.google_cloud_location = google_cloud_location
        self._client = self._build_client()

    def _build_client(self) -> genai.Client:
        if self.use_vertexai:
            if not self.google_cloud_project:
                raise RuntimeError("GOOGLE_CLOUD_PROJECT is required for Vertex AI.")
            if not self.google_cloud_location:
                raise RuntimeError("GOOGLE_CLOUD_LOCATION is required for Vertex AI.")
            return genai.Client(
                vertexai=True,
                project=self.google_cloud_project,
                location=self.google_cloud_location,
                http_options=types.HttpOptions(api_version="v1"),
            )

        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is required when GOOGLE_GENAI_USE_VERTEXAI=false.",
            )
        return genai.Client(api_key=self.api_key)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches and keep output order the same as input."""
        if not texts:
            return []

        output: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            output.extend(self._embed_batch(batch))
        return output

    def _embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        contents = list(texts)
        if not contents:
            return []

        config = types.EmbedContentConfig()
        if self.output_dimensionality > 0:
            config = types.EmbedContentConfig(
                output_dimensionality=self.output_dimensionality,
            )

        try:
            response = self._client.models.embed_content(
                model=self.model,
                contents=contents,
                config=config,
            )
        except Exception as exc:  # noqa: BLE001
            mode = "Vertex AI" if self.use_vertexai else "Gemini API key"
            raise RuntimeError(f"{mode} embedding request failed: {exc}") from exc

        vectors: list[list[float]] = []
        for emb in list(response.embeddings or []):
            values = [float(v) for v in list(emb.values or [])]
            vectors.append(values)
        return vectors

