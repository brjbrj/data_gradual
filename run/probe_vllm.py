from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
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


def _load_pipeline_env_if_needed() -> None:
    if os.environ.get("PIPELINE_CONFIG_LOADED") == "1":
        return
    root = Path(__file__).resolve().parents[1]
    config_file = root / "config" / "pipeline.env"
    if not config_file.exists():
        return
    for raw_line in config_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip("'\"")
    os.environ["PIPELINE_CONFIG_LOADED"] = "1"


def main() -> int:
    _load_pipeline_env_if_needed()
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
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        print(
            f"HTTPError status={exc.code} url={base_url}/models body={body}",
            file=sys.stderr,
        )
        return 1
    except urllib.error.URLError as exc:
        print(f"URLError url={base_url}/models reason={exc}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"JSONDecodeError url={base_url}/models error={exc}", file=sys.stderr)
        return 1

    model = _extract_model_name(payload)
    if model:
        print(model)
        return 0
    print(f"No model id in /models payload: {payload}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
