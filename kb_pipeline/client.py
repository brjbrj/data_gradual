from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ChatMessage:
    role: str
    content: str


class VLLMClient:
    """Minimal OpenAI-compatible client for a local vLLM server."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("VLLM_BASE_URL") or "http://localhost:8911/v1").rstrip("/")
        self.model = model or os.environ.get("VLLM_MODEL") or os.environ.get("OPENAI_MODEL") or "local-model"
        self.api_key = api_key or os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        if timeout is None:
            env_timeout = os.environ.get("VLLM_TIMEOUT") or os.environ.get("OPENAI_TIMEOUT")
            if env_timeout is None or env_timeout == "":
                self.timeout = 600
            else:
                try:
                    self.timeout = max(1, int(env_timeout))
                except Exception:
                    self.timeout = 600
        else:
            self.timeout = max(1, int(timeout))
        if max_retries is None:
            env_value = os.environ.get("VLLM_MAX_RETRIES") or os.environ.get("OPENAI_MAX_RETRIES")
            if env_value is None or env_value == "":
                self.max_retries = 2
            else:
                try:
                    self.max_retries = int(env_value)
                except Exception:
                    self.max_retries = 2
        else:
            self.max_retries = int(max_retries)

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")

        last_error: Optional[BaseException] = None
        infinite_retries = self.max_retries < 0
        attempt = 0
        while infinite_retries or attempt <= self.max_retries:
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = response.read().decode("utf-8")
                payload = json.loads(body)
                return payload["choices"][0]["message"]["content"]
            except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                if not infinite_retries and attempt >= self.max_retries:
                    break
                time.sleep(min(3.0 * (attempt + 1), 15.0))
                attempt += 1
        raise RuntimeError(f"vLLM request failed: {last_error}")
