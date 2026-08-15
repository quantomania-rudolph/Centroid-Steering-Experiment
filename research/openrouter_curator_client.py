"""OpenRouter client for calm-tail curation — free models for generation, paid QA gated."""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from load_openrouter_env import load_openrouter_api_key
from openrouter_budget import BudgetExhausted, QaBudgetLedger

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "curator_models.yaml"

ErrorClass = Literal[
    "transient_network",
    "rate_limit",
    "token_limit",
    "empty_response",
    "auth",
    "client_error",
    "paid_model_blocked",
    "unknown",
]


@dataclass(frozen=True)
class ModelSpec:
    id: str
    tier: str
    role: str = "generation"
    weight: float = 1.0


@dataclass
class OpenRouterResponse:
    ok: bool
    model: str
    content: str
    error_class: ErrorClass | None = None
    detail: str = ""
    status_code: int | None = None
    usage: dict[str, Any] | None = None


def load_curator_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or DEFAULT_CONFIG
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def generation_models(cfg: dict[str, Any]) -> list[ModelSpec]:
    """Return generation roster in order (paid primary allowed)."""
    out: list[ModelSpec] = []
    for row in cfg.get("generation_models") or []:
        model_id = str(row["id"])
        tier = str(row.get("tier") or "free").lower()
        if tier == "free":
            if model_id != "openrouter/free" and ":free" not in model_id and not model_id.endswith(":free"):
                raise ValueError(f"free generation model must be free slug or openrouter/free: {model_id}")
        elif tier != "paid":
            raise ValueError(f"generation tier must be free or paid: {tier}")
        out.append(
            ModelSpec(
                id=model_id,
                tier=tier,
                role=str(row.get("role") or "generation"),
                weight=float(row.get("weight") or 1.0),
            )
        )
    if not out:
        raise ValueError("no generation models configured")
    return out


def qa_model(cfg: dict[str, Any]) -> ModelSpec:
    qa = cfg.get("qa") or {}
    return ModelSpec(
        id=str(qa["model"]),
        tier=str(qa.get("tier") or "paid"),
        role="qa",
    )


class OpenRouterCuratorClient:
    def __init__(
        self,
        *,
        config: dict[str, Any] | None = None,
        api_key: str | None = None,
        allow_paid: bool = False,
        ledger: QaBudgetLedger | None = None,
    ) -> None:
        self.config = config or load_curator_config()
        self.allow_paid = allow_paid
        self.ledger = ledger or QaBudgetLedger.from_config(self.config)
        or_cfg = self.config.get("openrouter") or {}
        self.base_url = str(or_cfg.get("base_url") or "https://openrouter.ai/api/v1").rstrip("/")
        self.api_key = api_key or load_openrouter_api_key()
        self.http_referer = str(or_cfg.get("http_referer") or "")
        self.app_title = str(or_cfg.get("app_title") or "UNI_CAA01_calm_curator")

    def _headers(self) -> dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            h["HTTP-Referer"] = self.http_referer
        if self.app_title:
            h["X-Title"] = self.app_title
        return h

    def _classify_error(self, exc: Exception, body: str = "") -> ErrorClass:
        text = f"{exc} {body}".lower()
        if isinstance(exc, urllib.error.HTTPError):
            if exc.code == 401:
                return "auth"
            if exc.code == 429:
                return "rate_limit"
            if exc.code in (502, 503, 504):
                return "transient_network"
            if exc.code == 404 and "nvidia" in text:
                return "transient_network"  # Nemotron intermittent provider 404 — retry, never demote
            if exc.code == 400 and ("context" in text or "token" in text):
                return "token_limit"
            if 400 <= exc.code < 500:
                return "client_error"
        if isinstance(exc, (urllib.error.URLError, TimeoutError)):
            return "transient_network"
        if "rate limit" in text or "overloaded" in text:
            return "rate_limit"
        return "unknown"

    def _request_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.2,
        json_object: bool = False,
    ) -> OpenRouterResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_object:
            payload["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            body = ""
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    body = ""
            return OpenRouterResponse(
                ok=False,
                model=model,
                content="",
                error_class=self._classify_error(exc, body),
                detail=body[:500] or str(exc),
                status_code=getattr(exc, "code", None),
            )

        try:
            message = raw["choices"][0]["message"]
            content = message.get("content")
            if content is None:
                content = message.get("reasoning_content") or message.get("refusal") or ""
        except (KeyError, IndexError, TypeError) as exc:
            return OpenRouterResponse(
                ok=False,
                model=model,
                content="",
                error_class="empty_response",
                detail=str(exc),
            )

        if content is None or not str(content).strip() or str(content).strip() == "None":
            return OpenRouterResponse(
                ok=False,
                model=model,
                content="",
                error_class="empty_response",
                detail=f"zero-length completion raw_msg_keys={list(message.keys()) if isinstance(message, dict) else type(message)}",
            )

        return OpenRouterResponse(
            ok=True,
            model=model,
            content=str(content),
            usage=raw.get("usage"),
        )

    def paid_generation_fallback_model(self) -> ModelSpec | None:
        fb = self.config.get("paid_generation_fallback") or {}
        if not fb.get("enabled"):
            return None
        return ModelSpec(
            id=str(fb["model"]),
            tier=str(fb.get("tier") or "paid"),
            role=str(fb.get("role") or "generation_last_resort"),
        )

    def _assert_generation_model(self, model: str, *, last_resort: bool = False) -> None:
        allowed = {m.id for m in generation_models(self.config)}
        if model in allowed:
            return
        paid_fb = (self.config.get("paid_generation_fallback") or {}).get("model")
        fb_enabled = bool((self.config.get("paid_generation_fallback") or {}).get("enabled"))
        if model == paid_fb and (last_resort and fb_enabled or self.allow_paid):
            return
        raise PermissionError(
            f"paid_model_blocked: generation model {model} not in roster "
            f"{sorted(allowed)} (last_resort={last_resort})."
        )

    def complete_generation(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 2048,
        last_resort: bool = False,
    ) -> OpenRouterResponse:
        self._assert_generation_model(model, last_resort=last_resort)
        return self._request_chat(model=model, messages=messages, max_tokens=max_tokens)

    def complete_generation_last_resort(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int = 2048,
    ) -> OpenRouterResponse:
        fb = self.paid_generation_fallback_model()
        if fb is None:
            raise PermissionError("paid_generation_fallback disabled in config")
        self.ledger.record_gen_fallback()
        return self.complete_generation(
            model=fb.id, messages=messages, max_tokens=max_tokens, last_resort=True
        )

    def complete_qa(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
    ) -> OpenRouterResponse:
        qa = qa_model(self.config)
        mt = max_tokens or int((self.config.get("qa") or {}).get("max_output_tokens") or 256)
        # DeepSeek V4 Flash intermittently returns content=null with response_format=json_object
        return self._request_chat(
            model=qa.id, messages=messages, max_tokens=mt, temperature=0.0, json_object=False
        )

    def ping_model(self, model: str, *, role: Literal["generation", "qa"] = "generation") -> OpenRouterResponse:
        if role == "generation":
            self._assert_generation_model(model)
        messages = [{"role": "user", "content": "Reply with exactly: ok"}]
        return self._request_chat(model=model, messages=messages, max_tokens=16, temperature=0.0)

    def ping_all_free_generation(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for spec in generation_models(self.config):
            resp = self.ping_model(spec.id, role="generation")
            results.append(
                {
                    "model": spec.id,
                    "tier": spec.tier,
                    "ok": resp.ok,
                    "error_class": resp.error_class,
                    "detail": resp.detail[:200] if resp.detail else "",
                    "preview": resp.content[:80] if resp.ok else "",
                }
            )
        return results

    def complete_with_retry(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 2048,
        role: Literal["generation", "qa"] = "generation",
        brief_id: str | None = None,
    ) -> OpenRouterResponse:
        retry_cfg = self.config.get("retry") or {}
        max_attempts = int(retry_cfg.get("max_attempts_per_request") or 5)
        base_ms = int(retry_cfg.get("backoff_base_ms") or 2000)
        max_ms = int(retry_cfg.get("backoff_max_ms") or 300000)

        last: OpenRouterResponse | None = None
        for attempt in range(1, max_attempts + 1):
            if role == "qa":
                last = self.complete_qa(messages=messages, max_tokens=max_tokens)
            else:
                last = self.complete_generation(model=model, messages=messages, max_tokens=max_tokens)

            if last.ok:
                return last

            if last.error_class in ("auth", "client_error", "paid_model_blocked"):
                return last

            if attempt >= max_attempts:
                break

            if last.error_class == "rate_limit":
                rl = retry_cfg.get("rate_limit_backoff_ms") or [30000, 60000, 120000]
                delay = rl[min(attempt - 1, len(rl) - 1)] / 1000.0
            else:
                delay = min(max_ms, base_ms * (2 ** (attempt - 1))) / 1000.0
                delay += random.uniform(0, 0.5)

            time.sleep(delay)

        assert last is not None
        return last
