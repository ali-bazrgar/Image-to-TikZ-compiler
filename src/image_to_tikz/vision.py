from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

try:
    import requests
except Exception:  # optional dependency
    requests = None


VISION_INSTRUCTION = """Analyze this diagram for a downstream program that must reproduce it in TikZ.
Do NOT write TikZ. Return JSON only with this shape:
{
  \"scene_type\": string,
  \"elements\": [
    {\"id\": string, \"kind\": string, \"role\": string, \"description\": string,
     \"text\": string|null, \"approx_bbox\": [x,y,w,h], \"relations\": [string], \"confidence\": number}
  ],
  \"global_relations\": [string],
  \"semantic_summary\": string,
  \"uncertainties\": [string]
}
Use normalized coordinates from 0 to 1 when possible. Be explicit about arrows, connectors,
labels, mathematical symbols, nodes, axes, plots, containers, circles, polygons and repeated structure."""


class VisionEnricher:
    """Optional semantic layer for any OpenAI-compatible multimodal endpoint."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: int = 120):
        if requests is None:
            raise RuntimeError("Install the optional 'llm' dependency to use vision enrichment.")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def analyze(self, image_path: str | Path) -> dict[str, Any]:
        path = Path(image_path)
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(path.suffix.lower(), "image/png")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_INSTRUCTION},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                ],
            }],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(self.base_url + "/v1/chat/completions", json=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        content = data["choices"]["message"]["content"] if isinstance(data["choices"], dict) else data["choices"][0]["message"]["content"]
        return _parse_json(content)


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Vision model returned JSON but not an object")
    return parsed
