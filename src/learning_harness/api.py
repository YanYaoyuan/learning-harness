from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from urllib.parse import urlsplit


class ChatClient:
    """Stateless OpenAI-compatible chat endpoint; never logs keys or retries bills."""

    def __init__(self, settings):
        self.base_url = os.environ.get(settings["base_url_env"], "").rstrip("/")
        self.model = os.environ.get(settings["model_env"], "")
        self.key = os.environ.get(settings["api_key_env"], "")
        self.extra_body = settings.get('extra_body', {})
        if not isinstance(self.extra_body, dict) or set(self.extra_body) - {'enable_thinking', 'response_format'}:
            raise ValueError('Unsupported extra request fields')
        if not self.base_url.startswith(("http://", "https://")) or not self.model:
            raise ValueError("Configure model endpoint and model name in environment variables")
        parsed = urlsplit(self.base_url)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Put credentials in the API-key environment variable, not in the endpoint URL")

    def complete(self, messages, max_tokens=128, temperature=0.0, seed=None):
        body = dict(model=self.model, messages=messages, max_tokens=max_tokens, temperature=temperature)
        body.update(self.extra_body)
        if seed is not None:
            body["seed"] = seed
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        request = urllib.request.Request(self.base_url + "/chat/completions",
                                         data=json.dumps(body).encode(), headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            # Response bodies may contain request details. Keep them out of logs.
            raise RuntimeError(f"Model endpoint returned HTTP {exc.code}") from None
        usage = result.get("usage", {})
        if not all(isinstance(usage.get(k), int) and usage[k] >= 0
                   for k in ("prompt_tokens", "completion_tokens")):
            raise ValueError("Endpoint must return actual prompt/completion token usage")
        text = result["choices"][0]["message"]["content"]
        if not isinstance(text, str):
            raise ValueError("Endpoint did not return text")
        return dict(text=text, input_tokens=usage["prompt_tokens"],
                    output_tokens=usage["completion_tokens"], response_model=result.get("model"),
                    finish_reason=result['choices'][0].get('finish_reason'),
                    response_id=result.get('id'))
