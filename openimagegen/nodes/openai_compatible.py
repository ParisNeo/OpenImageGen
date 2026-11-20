
"""
@file        openai_compatible.py
@author      ParisNeo
@description Prompt expansion/generation through an OpenAI-compatible chat endpoint.
"""

from __future__ import annotations

import ipaddress
import json
import os
import socket
import ssl
from typing import Any, Dict
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .base import BaseWorkflowNode, NodeExecutionContext, NodeField, NodePort, NodeSpec
from .registry import get_node_registry

MAX_RESPONSE_BYTES = 2 * 1024 * 1024

class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

def _resolve_api_key(params: Dict[str, Any]) -> str | None:
    mode = params.get("auth_mode", "environment")
    if mode == "none":
        return None
    if mode == "api_key":
        key = str(params.get("api_key", "")).strip()
        if not key:
            raise ValueError("API key authentication is selected, but api_key is empty.")
        return key
    variable = str(params.get("env_var", "OPENAI_API_KEY")).strip()
    if not variable or not variable.replace("_", "").isalnum():
        raise ValueError("env_var must be a valid environment variable name.")
    key = os.getenv(variable)
    if not key:
        raise ValueError(
            f"Environment variable '{variable}' is not set on the OpenImageGen server."
        )
    return key

def _validated_base_url(base_url: str) -> tuple[str, Any]:
    value = base_url.strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Server URL must be an absolute http:// or https:// URL.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Credentials, query strings, and fragments are not allowed in the URL.")
    if parsed.hostname in {"169.254.169.254", "metadata.google.internal"}:
        raise ValueError("Cloud metadata endpoints are blocked.")
    return value, parsed


def _endpoint(base_url: str) -> tuple[str, bool]:
    value, parsed = _validated_base_url(base_url)
    if not value.endswith("/chat/completions"):
        value += "/chat/completions"
    return value, parsed.scheme == "https"


def list_remote_models(params: Dict[str, Any]) -> list[str]:
    """Read model identifiers from an OpenAI-compatible ``GET /models`` endpoint."""
    value, parsed = _validated_base_url(str(params.get("base_url", "")))
    endpoint = value if value.endswith("/models") else f"{value}/models"
    api_key = _resolve_api_key(params)
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(endpoint, headers=headers, method="GET")

    handlers: list[Any] = [_NoRedirects()]
    if parsed.scheme == "https":
        from urllib.request import HTTPSHandler
        verify = params.get("verify_ssl", True)
        context = ssl.create_default_context() if verify else ssl._create_unverified_context()
        handlers.append(HTTPSHandler(context=context))
    opener = build_opener(*handlers)
    timeout = max(1, min(60, int(params.get("timeout_seconds", 30))))
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as error:
        detail = error.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"Remote server returned HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach OpenAI-compatible server: {error.reason}") from error
    except (TimeoutError, socket.timeout) as error:
        raise RuntimeError(f"Remote server timed out after {timeout} seconds.") from error
    if len(raw) > MAX_RESPONSE_BYTES:
        raise RuntimeError("Remote response exceeded the 2 MiB safety limit.")
    try:
        data = json.loads(raw)
        models = sorted({
            str(item["id"]).strip()
            for item in data["data"]
            if isinstance(item, dict) and str(item.get("id", "")).strip()
        })
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("Remote response is not a valid OpenAI model-list response.") from error
    if not models:
        raise ValueError("The remote server returned no models.")
    return models


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "output_text"}
        )
    raise ValueError("The server response did not contain textual message content.")

@get_node_registry().register
class OpenAICompatibleNode(BaseWorkflowNode):
    """Produce text without exposing remote credentials to the browser response or job history."""

    spec = NodeSpec(
        type="openai_compatible", title="OpenAI Compatible", category="AI text", icon="✦",
        color="#2563eb",
        description=(
            "Generate or rewrite a prompt using /chat/completions on vLLM, LM Studio, "
            "llama.cpp, OpenAI, or another compatible server."
        ),
        inputs=[NodePort(slot="prompt", label="prompt (optional)", kind="text")],
        outputs=[NodePort(slot="text", label="text", kind="text")],
        fields=[
            NodeField(key="base_url", label="Server URL / base /v1", type="text", required=True,
                      default="http://127.0.0.1:11434/v1", placeholder="http://127.0.0.1:1234/v1"),
            NodeField(key="verify_ssl", label="Verify SSL certificates", type="boolean", default=True,
                      help="Keep enabled for production. URL scheme controls whether TLS is used."),
            NodeField(key="model", label="Remote model", type="text", required=True,
                      default="", placeholder="model-name"),
            NodeField(key="auth_mode", label="Authentication", type="select", default="environment",
                      options=[{"value": "environment", "label": "Server environment variable"},
                               {"value": "api_key", "label": "User API key"},
                               {"value": "none", "label": "No authentication"}]),
            NodeField(key="env_var", label="Environment variable", type="text", default="OPENAI_API_KEY"),
            NodeField(key="api_key", label="User API key", type="password", secret=True, default="",
                      help="Used once for this run; never saved, exported, logged, or included in job details."),
            NodeField(key="system_prompt", label="System prompt", type="textarea",
                      default="Write one concise, detailed image-generation prompt. Return only the prompt."),
            NodeField(key="prompt", label="Prompt (or connect text)", type="textarea", default=""),
            NodeField(key="temperature", label="Temperature", type="number", default=.7, min=0, max=2, step=.1),
            NodeField(key="max_tokens", label="Maximum tokens", type="number", default=512, min=1, max=32768, step=1),
            NodeField(key="timeout_seconds", label="Timeout (seconds)", type="number", default=60, min=1, max=600, step=1),
        ],
    )

    def execute(self, params: Dict[str, Any], inputs: Dict[str, Any], context: NodeExecutionContext) -> Dict[str, Any]:
        prompt = str(inputs.get("prompt") or params.get("prompt") or "").strip()
        model = str(params.get("model", "")).strip()
        if not prompt:
            raise ValueError("Connect a text prompt or enter one in the OpenAI node.")
        if not model:
            raise ValueError("Remote model is required.")
        endpoint, uses_tls = _endpoint(str(params.get("base_url", "")))
        api_key = _resolve_api_key(params)
        context.checkpoint()
        context.report(5, f"Contacting OpenAI-compatible model '{model}'…")

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": str(params.get("system_prompt", ""))},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(params.get("temperature", .7)),
            "max_tokens": int(params.get("max_tokens", 512)),
            "stream": False,
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")

        ssl_context = None
        if uses_tls:
            verify = params.get("verify_ssl", True)
            ssl_context = ssl.create_default_context() if verify else ssl._create_unverified_context()
        opener = build_opener(_NoRedirects())
        timeout = max(1, min(600, int(params.get("timeout_seconds", 60))))
        from urllib.request import HTTPSHandler
        if ssl_context is not None:
            opener = build_opener(_NoRedirects(), HTTPSHandler(context=ssl_context))
        try:
            with opener.open(request, timeout=timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except HTTPError as error:
            detail = error.read(4096).decode("utf-8", errors="replace")
            raise RuntimeError(f"Remote server returned HTTP {error.code}: {detail}") from error
        except URLError as error:
            raise RuntimeError(f"Could not reach OpenAI-compatible server: {error.reason}") from error
        except (TimeoutError, socket.timeout) as error:
            raise RuntimeError(f"Remote server timed out after {timeout} seconds.") from error
        if len(raw) > MAX_RESPONSE_BYTES:
            raise RuntimeError("Remote response exceeded the 2 MiB safety limit.")
        context.checkpoint("Workflow cancelled while receiving the remote response.")
        try:
            data = json.loads(raw)
            content = data["choices"][0]["message"]["content"]
            text = _content_text(content).strip()
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
            raise ValueError("Remote response is not a valid OpenAI chat-completion response.") from error
        if not text:
            raise ValueError("Remote model returned empty text.")
        context.report(90, f"Remote model returned {len(text)} characters.")
        return {"text": text}

class _null_context:
    def __enter__(self): return None
    def __exit__(self, *args): return False
