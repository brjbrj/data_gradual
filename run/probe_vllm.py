from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Optional


def _extract_model_name(payload: Any) -> Optional[str]:
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                model = first.get("id") or first.get("name")
                if model:
                    return str(model)
        model = payload.get("model") or payload.get("name")
        if model:
            return str(model)
    return None


def main() -> int:
    base_url = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8911/v1").rstrip("/")
    api_key = (
        os.environ.get("VLLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "EMPTY"
    )
    request = urllib.request.Request(
        base_url + "/models",
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        return 1

    model = _extract_model_name(payload)
    if model:
        print(model)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
