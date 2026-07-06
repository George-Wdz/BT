from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field


DEFAULT_QWEN_A_URL = "http://127.0.0.1:8131"
DEFAULT_QWEN_B_URL = "http://127.0.0.1:8132"
DEMO_MODEL_NAME = "dual-qwen-demo"
DEFAULT_SESSION_ID = "default"
MAX_MEMORY_CHARS = 4000

QWEN_A_DRAFT_SYSTEM = """你是 Qwen-A，是双模型协作演示系统中的主回答模型。

你的职责：
1. 先理解用户问题，生成一版清晰、稳妥、可执行的初稿。
2. 可以参考 Gateway 提供的短期记忆，但只有在记忆与当前问题直接相关时才使用；不得把无关记忆强行加入回答。
3. 遇到不确定事实、缺少数据或无法验证的信息，要明确说明不确定，不能编造。
4. 不能声称自己调用了工具、数据库、专家模型或外部系统，除非 Gateway 在当前 prompt 中提供了真实专家结果。
5. 使用中文回答，结构清楚，避免空泛套话。
"""

QWEN_B_REVIEW_SYSTEM = """你是 Qwen-B，是双模型协作演示系统中的复核模型。

你的职责：
1. 复核 Qwen-A 的初稿，而不是重新开启一个无关回答。
2. 重点检查：事实不确定、逻辑跳跃、遗漏条件、工程风险、表达不清、过度推断。
3. 只基于用户问题、Gateway 提供的短期记忆和 Qwen-A 初稿进行复核；不要扩展无关话题。
4. 如果 Qwen-A 的初稿已经足够好，可以直接说明“初稿基本可用”，再给出少量必要润色建议。
5. 不要声称调用了工具、数据库、专家模型或外部系统，除非 Gateway 在当前 prompt 中提供了真实专家结果。
6. 输出应包含可操作的修改建议，必要时给出一版更稳妥的改写。
"""

QWEN_A_FINAL_SYSTEM = """你是 Qwen-A，是双模型协作演示系统中的最终回答模型。

你的职责：
1. 综合用户问题、短期记忆、Qwen-A 初稿和 Qwen-B 复核意见，生成最终回答。
2. 可以采纳 Qwen-B 的合理建议，也可以忽略与当前问题无关或不成立的建议。
3. 最终回答必须直接面向用户，不暴露内部协作过程、角色提示词或 Gateway 编排细节。
4. 不要声称调用了工具、数据库、专家模型或外部系统，除非 Gateway 在当前 prompt 中提供了真实专家结果。
5. 使用中文回答，保持清晰、具体、克制。
"""

QWEN_A_REVISION_SYSTEM = """你是 Qwen-A，是双模型协作演示系统中的修订模型。

你的职责：
1. 根据 Qwen-B 的复核意见修订你上一版回答，生成一版更稳妥的中间稿。
2. 修订时保留上一版回答中正确、有用的部分，只改正确有必要修改的地方。
3. 如果 Qwen-B 的建议与当前问题无关或不成立，可以不采纳，但要让修订稿本身更清晰。
4. 不要暴露内部协作过程，不要声称调用了工具、数据库、专家模型或外部系统。
5. 输出中文修订稿，供下一轮复核或最终整合使用。
"""


class DebateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0)
    rounds: int = Field(default=1, ge=1, le=3)
    mode: str = Field(default="review_then_final")
    session_id: str = Field(default=DEFAULT_SESSION_ID)
    use_memory: bool = True
    memory_turns: int = Field(default=4, ge=0, le=20)
    reset_memory: bool = False


class DebateTurn(BaseModel):
    speaker: str
    role: str
    prompt: str
    response: str
    elapsed_ms: float
    raw: dict[str, Any]


class DebateResponse(BaseModel):
    prompt: str
    mode: str
    rounds: int
    session_id: str
    memory_used: bool
    memory_size: int
    final_answer: str
    turns: list[DebateTurn]
    elapsed_ms: float
    qwen_a_url: str
    qwen_b_url: str


class OllamaGenerateRequest(BaseModel):
    model: str = DEMO_MODEL_NAME
    prompt: str = ""
    system: Optional[str] = None
    stream: bool = False
    options: dict[str, Any] = Field(default_factory=dict)
    keep_alive: Optional[Any] = None


class OllamaChatMessage(BaseModel):
    role: str
    content: str


class OllamaChatRequest(BaseModel):
    model: str = DEMO_MODEL_NAME
    messages: list[OllamaChatMessage] = Field(default_factory=list)
    stream: bool = False
    options: dict[str, Any] = Field(default_factory=dict)
    keep_alive: Optional[Any] = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_request(method: str, url: str, payload: Optional[dict[str, Any]], timeout_s: float) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=exc.code, detail=detail) from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"qwen service unreachable: {url}; {exc}") from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=f"qwen service timeout: {url}") from exc


class DualQwenGateway:
    def __init__(
        self,
        *,
        qwen_a_url: str,
        qwen_b_url: str,
        request_timeout_s: float,
    ) -> None:
        self.qwen_a_url = qwen_a_url.rstrip("/")
        self.qwen_b_url = qwen_b_url.rstrip("/")
        self.request_timeout_s = float(request_timeout_s)
        self.memory: dict[str, list[dict[str, str]]] = {}
        self.memory_lock = threading.Lock()

    def _normalize_session_id(self, value: str) -> str:
        session_id = (value or DEFAULT_SESSION_ID).strip()
        return session_id or DEFAULT_SESSION_ID

    def _memory_block(self, session_id: str, limit: int) -> str:
        if limit <= 0:
            return ""
        with self.memory_lock:
            items = list(self.memory.get(session_id, []))[-limit:]
        if not items:
            return ""
        lines = ["以下是本会话可参考的短期记忆，只用于保持上下文一致；如果与当前问题无关，可以忽略："]
        for idx, item in enumerate(items, start=1):
            user = item.get("user", "").strip()
            final = item.get("final", "").strip()
            if user or final:
                lines.append(f"{idx}. 用户：{user}\n   上次最终回答摘要：{final}")
        block = "\n".join(lines)
        if len(block) > MAX_MEMORY_CHARS:
            block = block[-MAX_MEMORY_CHARS:]
        return block

    def _append_memory(self, session_id: str, user_prompt: str, final_answer: str) -> int:
        compact_final = final_answer.strip()
        if len(compact_final) > 600:
            compact_final = compact_final[:600] + "..."
        with self.memory_lock:
            history = self.memory.setdefault(session_id, [])
            history.append({"user": user_prompt.strip(), "final": compact_final})
            del history[:-20]
            return len(history)

    def _reset_memory(self, session_id: str) -> None:
        with self.memory_lock:
            self.memory.pop(session_id, None)

    def _normalize_mode(self, value: str) -> str:
        mode = (value or "review_then_final").strip().lower()
        if mode in {"interactive", "ab_interactive", "a_b_interactive", "dialogue"}:
            return "interactive"
        return "review_then_final"

    def _call_qwen(
        self,
        *,
        base_url: str,
        prompt: str,
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> tuple[str, float, dict[str, Any]]:
        payload = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
        started = time.perf_counter()
        raw = _json_request("POST", f"{base_url}/generate", payload, self.request_timeout_s)
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        return str(raw.get("generated_text", "")), elapsed_ms, raw

    def _stream_qwen(
        self,
        *,
        base_url: str,
        speaker: str,
        role: str,
        prompt: str,
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
    ):
        payload = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
        req = urllib.request.Request(
            url=f"{base_url}/generate/stream",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        generated_parts: list[str] = []
        raw_done: dict[str, Any] = {}
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout_s) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    event = json.loads(line)
                    event_type = event.get("type")
                    if event_type == "chunk":
                        text = str(event.get("text", ""))
                        generated_parts.append(text)
                        yield {
                            "type": "token",
                            "speaker": speaker,
                            "role": role,
                            "text": text,
                        }
                    elif event_type == "done":
                        raw_done = event
                    elif event_type == "error":
                        yield {
                            "type": "error",
                            "speaker": speaker,
                            "role": role,
                            "detail": event.get("detail", "backend stream error"),
                        }
                        return
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            yield {"type": "error", "speaker": speaker, "role": role, "detail": detail}
            return
        except (urllib.error.URLError, TimeoutError) as exc:
            yield {"type": "error", "speaker": speaker, "role": role, "detail": str(exc)}
            return

        response = "".join(generated_parts).strip()
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        yield {
            "type": "turn_done",
            "speaker": speaker,
            "role": role,
            "prompt": prompt,
            "response": response,
            "elapsed_ms": elapsed_ms,
            "raw": raw_done,
        }

    def debate_stream(self, request: DebateRequest):
        started = time.perf_counter()
        turns: list[dict[str, Any]] = []
        original_prompt = request.prompt.strip()
        if not original_prompt:
            yield {"type": "error", "detail": "prompt is required"}
            return
        session_id = self._normalize_session_id(request.session_id)
        if request.reset_memory:
            self._reset_memory(session_id)
        memory_context = self._memory_block(session_id, request.memory_turns) if request.use_memory else ""
        mode = self._normalize_mode(request.mode)
        effective_rounds = request.rounds if mode == "interactive" else 1
        yield {
            "type": "start",
            "session_id": session_id,
            "memory_used": bool(memory_context),
            "mode": mode,
            "rounds": effective_rounds,
        }

        draft_parts = ["请先独立回答用户问题。要求：中文、结构清晰、不要编造不确定事实。"]
        if memory_context:
            draft_parts.append(memory_context)
        draft_parts.append(f"用户问题：{original_prompt}")
        draft_prompt = "\n\n".join(draft_parts)
        draft_chunks: list[str] = []
        for event in self._stream_qwen(
            base_url=self.qwen_a_url,
            speaker="Qwen-A",
            role="draft",
            prompt=draft_prompt,
            system_prompt=QWEN_A_DRAFT_SYSTEM,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        ):
            if event.get("type") == "token":
                draft_chunks.append(str(event.get("text", "")))
            elif event.get("type") == "turn_done":
                turns.append(event)
            yield event
            if event.get("type") == "error":
                return
        draft = "".join(draft_chunks).strip()

        current_answer = draft
        reviewer_output = ""
        for round_idx in range(effective_rounds):
            review_prompt = (
                "你将看到用户问题和 Qwen-A 的回答。"
                "请指出回答中需要修正、补充或压缩的地方，并给出一版更稳妥的改写。"
                "如果原回答已经足够好，也请明确说明。\n\n"
                f"用户问题：{original_prompt}\n\n"
                f"Qwen-A 当前回答：\n{current_answer}"
            )
            review_chunks: list[str] = []
            for event in self._stream_qwen(
                base_url=self.qwen_b_url,
                speaker="Qwen-B",
                role=f"review_round_{round_idx + 1}",
                prompt=review_prompt,
                system_prompt=QWEN_B_REVIEW_SYSTEM,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
            ):
                if event.get("type") == "token":
                    review_chunks.append(str(event.get("text", "")))
                elif event.get("type") == "turn_done":
                    turns.append(event)
                yield event
                if event.get("type") == "error":
                    return
            reviewer_output = "".join(review_chunks).strip()

            if mode == "interactive" and round_idx < effective_rounds - 1:
                revision_prompt = (
                    "请根据用户问题、你上一版回答和 Qwen-B 的复核意见，生成一版修订稿。"
                    "这不是最终回答，而是供下一轮复核使用的中间稿。"
                    "请直接输出修订稿，不要描述内部协作过程。\n\n"
                    f"用户问题：{original_prompt}\n\n"
                    f"Qwen-A 上一版回答：\n{current_answer}\n\n"
                    f"Qwen-B 复核意见：\n{reviewer_output}"
                )
                revision_chunks: list[str] = []
                for event in self._stream_qwen(
                    base_url=self.qwen_a_url,
                    speaker="Qwen-A",
                    role=f"revision_round_{round_idx + 1}",
                    prompt=revision_prompt,
                    system_prompt=QWEN_A_REVISION_SYSTEM,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                ):
                    if event.get("type") == "token":
                        revision_chunks.append(str(event.get("text", "")))
                    elif event.get("type") == "turn_done":
                        turns.append(event)
                    yield event
                    if event.get("type") == "error":
                        return
                current_answer = "".join(revision_chunks).strip()

        final_parts = [
            "请根据用户问题、Qwen-A 初稿和 Qwen-B 复核结果，生成最终回答。只输出最终回答，不要描述内部协作过程。"
        ]
        if memory_context:
            final_parts.append(memory_context)
        final_parts.append(
            f"用户问题：{original_prompt}\n\n"
            f"Qwen-A 初稿：\n{draft}\n\n"
            f"Qwen-A 最新修订稿：\n{current_answer}\n\n"
            f"Qwen-B 复核结果：\n{reviewer_output}"
        )
        final_prompt = "\n\n".join(final_parts)
        final_chunks: list[str] = []
        for event in self._stream_qwen(
            base_url=self.qwen_a_url,
            speaker="Qwen-A",
            role="final",
            prompt=final_prompt,
            system_prompt=QWEN_A_FINAL_SYSTEM,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        ):
            if event.get("type") == "token":
                final_chunks.append(str(event.get("text", "")))
            elif event.get("type") == "turn_done":
                turns.append(event)
            yield event
            if event.get("type") == "error":
                return
        final_answer = "".join(final_chunks).strip()
        memory_size = self._append_memory(session_id, original_prompt, final_answer) if request.use_memory else 0
        yield {
            "type": "complete",
            "prompt": original_prompt,
            "mode": mode,
            "rounds": effective_rounds,
            "session_id": session_id,
            "memory_used": bool(memory_context),
            "memory_size": memory_size,
            "final_answer": final_answer,
            "turns": turns,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "qwen_a_url": self.qwen_a_url,
            "qwen_b_url": self.qwen_b_url,
        }

    def debate(self, request: DebateRequest) -> DebateResponse:
        started = time.perf_counter()
        turns: list[DebateTurn] = []
        original_prompt = request.prompt.strip()
        if not original_prompt:
            raise HTTPException(status_code=400, detail="prompt is required")
        session_id = self._normalize_session_id(request.session_id)
        if request.reset_memory:
            self._reset_memory(session_id)
        memory_context = self._memory_block(session_id, request.memory_turns) if request.use_memory else ""
        mode = self._normalize_mode(request.mode)
        effective_rounds = request.rounds if mode == "interactive" else 1

        draft_parts = ["请先独立回答用户问题。要求：中文、结构清晰、不要编造不确定事实。"]
        if memory_context:
            draft_parts.append(memory_context)
        draft_parts.append(f"用户问题：{original_prompt}")
        draft_prompt = "\n\n".join(draft_parts)
        draft, elapsed, raw = self._call_qwen(
            base_url=self.qwen_a_url,
            prompt=draft_prompt,
            system_prompt=QWEN_A_DRAFT_SYSTEM,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )
        turns.append(
            DebateTurn(
                speaker="Qwen-A",
                role="draft",
                prompt=draft_prompt,
                response=draft,
                elapsed_ms=elapsed,
                raw=raw,
            )
        )

        current_answer = draft
        reviewer_output = ""
        for round_idx in range(effective_rounds):
            review_prompt = (
                "你将看到用户问题和 Qwen-A 的回答。"
                "请指出回答中需要修正、补充或压缩的地方，并给出一版更稳妥的改写。"
                "如果原回答已经足够好，也请明确说明。\n\n"
                f"用户问题：{original_prompt}\n\n"
                f"Qwen-A 当前回答：\n{current_answer}"
            )
            reviewer_output, elapsed, raw = self._call_qwen(
                base_url=self.qwen_b_url,
                prompt=review_prompt,
                system_prompt=QWEN_B_REVIEW_SYSTEM,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
            )
            turns.append(
                DebateTurn(
                    speaker="Qwen-B",
                    role=f"review_round_{round_idx + 1}",
                    prompt=review_prompt,
                    response=reviewer_output,
                    elapsed_ms=elapsed,
                    raw=raw,
                )
            )

            if mode == "interactive" and round_idx < effective_rounds - 1:
                revision_prompt = (
                    "请根据用户问题、你上一版回答和 Qwen-B 的复核意见，生成一版修订稿。"
                    "这不是最终回答，而是供下一轮复核使用的中间稿。"
                    "请直接输出修订稿，不要描述内部协作过程。\n\n"
                    f"用户问题：{original_prompt}\n\n"
                    f"Qwen-A 上一版回答：\n{current_answer}\n\n"
                    f"Qwen-B 复核意见：\n{reviewer_output}"
                )
                current_answer, elapsed, raw = self._call_qwen(
                    base_url=self.qwen_a_url,
                    prompt=revision_prompt,
                    system_prompt=QWEN_A_REVISION_SYSTEM,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                )
                turns.append(
                    DebateTurn(
                        speaker="Qwen-A",
                        role=f"revision_round_{round_idx + 1}",
                        prompt=revision_prompt,
                        response=current_answer,
                        elapsed_ms=elapsed,
                        raw=raw,
                    )
                )

        final_parts = [
            "请根据用户问题、Qwen-A 初稿和 Qwen-B 复核结果，生成最终回答。只输出最终回答，不要描述内部协作过程。"
        ]
        if memory_context:
            final_parts.append(memory_context)
        final_parts.append(
            f"用户问题：{original_prompt}\n\n"
            f"Qwen-A 初稿：\n{draft}\n\n"
            f"Qwen-A 最新修订稿：\n{current_answer}\n\n"
            f"Qwen-B 复核结果：\n{reviewer_output}"
        )
        final_prompt = "\n\n".join(final_parts)
        final_answer, elapsed, raw = self._call_qwen(
            base_url=self.qwen_a_url,
            prompt=final_prompt,
            system_prompt=QWEN_A_FINAL_SYSTEM,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )
        turns.append(
            DebateTurn(
                speaker="Qwen-A",
                role="final",
                prompt=final_prompt,
                response=final_answer,
                elapsed_ms=elapsed,
                raw=raw,
            )
        )
        memory_size = self._append_memory(session_id, original_prompt, final_answer) if request.use_memory else 0

        return DebateResponse(
            prompt=original_prompt,
            mode=mode,
            rounds=effective_rounds,
            session_id=session_id,
            memory_used=bool(memory_context),
            memory_size=memory_size,
            final_answer=final_answer,
            turns=turns,
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
            qwen_a_url=self.qwen_a_url,
            qwen_b_url=self.qwen_b_url,
        )

    def health(self) -> dict[str, Any]:
        backends = {}
        for name, url in (("qwen_a", self.qwen_a_url), ("qwen_b", self.qwen_b_url)):
            try:
                backends[name] = _json_request("GET", f"{url}/health", None, min(self.request_timeout_s, 10.0))
            except HTTPException as exc:
                backends[name] = {"status": "error", "detail": exc.detail}
        return {
            "status": "ok",
            "service": "dual_qwen_gateway",
            "model": DEMO_MODEL_NAME,
            "qwen_a_url": self.qwen_a_url,
            "qwen_b_url": self.qwen_b_url,
            "backends": backends,
        }

    def ollama_generate_payload(self, request: OllamaGenerateRequest) -> dict[str, Any]:
        options = request.options or {}
        prompt = request.prompt.strip()
        if request.system:
            prompt = f"{request.system.strip()}\n\n{prompt}"
        result = self.debate(
            DebateRequest(
                prompt=prompt,
                max_new_tokens=int(options.get("num_predict") or options.get("max_new_tokens") or 256),
                temperature=float(options.get("temperature", 0.0)),
                rounds=int(options.get("rounds", 1)),
                mode=str(options.get("mode", "review_then_final")),
                session_id=str(options.get("session_id", DEFAULT_SESSION_ID)),
                use_memory=bool(options.get("use_memory", True)),
            )
        )
        return {
            "model": request.model or DEMO_MODEL_NAME,
            "created_at": _utc_now_iso(),
            "response": result.final_answer,
            "done": True,
            "done_reason": "stop",
            "total_duration": int(result.elapsed_ms * 1_000_000),
            "prompt_eval_count": 0,
            "eval_count": 0,
            "dual_qwen": result.dict(),
        }

    def ollama_chat_payload(self, request: OllamaChatRequest) -> dict[str, Any]:
        options = request.options or {}
        prompt = "\n".join(
            f"{message.role}: {message.content}" for message in request.messages if message.content.strip()
        ).strip()
        result = self.debate(
            DebateRequest(
                prompt=prompt,
                max_new_tokens=int(options.get("num_predict") or options.get("max_new_tokens") or 256),
                temperature=float(options.get("temperature", 0.0)),
                rounds=int(options.get("rounds", 1)),
                mode=str(options.get("mode", "review_then_final")),
                session_id=str(options.get("session_id", DEFAULT_SESSION_ID)),
                use_memory=bool(options.get("use_memory", True)),
            )
        )
        return {
            "model": request.model or DEMO_MODEL_NAME,
            "created_at": _utc_now_iso(),
            "message": {"role": "assistant", "content": result.final_answer},
            "done": True,
            "done_reason": "stop",
            "total_duration": int(result.elapsed_ms * 1_000_000),
            "prompt_eval_count": 0,
            "eval_count": 0,
            "dual_qwen": result.dict(),
        }


def _single_line_stream(payload: dict[str, Any]):
    yield (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Dual Qwen Demo</title>
  <style>
    :root {
      --bg:#f5f6f8; --panel:#fff; --text:#1f2933; --muted:#667085; --line:#d8dee8;
      --a:#0f766e; --a-soft:#e6f4f1; --b:#475467; --b-soft:#eef2f7; --final:#175cd3; --danger:#b42318;
    }
    * { box-sizing:border-box; }
    body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--text); }
    header { width:min(1280px,calc(100vw - 32px)); margin:18px auto 10px; display:flex; justify-content:space-between; align-items:flex-end; gap:12px; }
    h1 { margin:0; font-size:22px; font-weight:650; }
    .status { color:var(--muted); font-size:13px; min-height:20px; }
    main { width:min(1280px,calc(100vw - 32px)); margin:0 auto 18px; display:grid; grid-template-rows:minmax(360px,1fr) auto; gap:14px; }
    .dialog-grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; min-height:420px; }
    section { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }
    .agent-panel { display:flex; flex-direction:column; min-height:420px; }
    .panel-head { display:flex; justify-content:space-between; align-items:center; gap:10px; padding-bottom:10px; border-bottom:1px solid var(--line); }
    .agent-name { font-weight:650; }
    .agent-role { color:var(--muted); font-size:13px; }
    .messages { flex:1; overflow:auto; padding:12px 2px 2px; display:flex; flex-direction:column; gap:10px; }
    .message { border:1px solid var(--line); border-radius:8px; padding:10px 12px; background:#fbfcfd; }
    .message.a { border-color:#a8d8cf; background:var(--a-soft); }
    .message.b { border-color:#cfd7e3; background:var(--b-soft); }
    .message.final { border-color:#b2ccff; background:#eff4ff; }
    .message-title { font-size:12px; color:var(--muted); margin-bottom:6px; display:flex; justify-content:space-between; gap:10px; }
    .message-body { white-space:pre-wrap; line-height:1.58; font-size:14px; }
    .composer { display:grid; grid-template-columns:1fr 330px; gap:14px; }
    label { display:block; font-size:13px; color:var(--muted); margin-bottom:6px; }
    textarea,input,select { width:100%; border:1px solid var(--line); border-radius:6px; padding:10px; font:inherit; background:#fff; }
    textarea { min-height:118px; resize:vertical; line-height:1.5; }
    .row { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:10px; }
    .check-row { display:flex; align-items:center; justify-content:space-between; gap:10px; margin-top:10px; color:var(--muted); font-size:13px; }
    .check-row input { width:auto; }
    button { height:42px; border:0; border-radius:6px; background:var(--a); color:#fff; font:inherit; font-weight:650; cursor:pointer; padding:0 14px; }
    button:hover { filter:brightness(.94); }
    button:disabled { cursor:wait; opacity:.65; }
    .secondary { background:#344054; }
    .final-box { min-height:110px; white-space:pre-wrap; line-height:1.6; border:1px solid #b2ccff; border-radius:8px; padding:12px; background:#eff4ff; }
    details { margin-top:10px; }
    summary { cursor:pointer; color:var(--muted); font-size:13px; }
    pre { white-space:pre-wrap; word-break:break-word; background:#f8fafc; border:1px solid var(--line); border-radius:6px; padding:12px; max-height:320px; overflow:auto; font-size:12px; }
    @media (max-width: 980px) {
      header { align-items:flex-start; flex-direction:column; }
      .dialog-grid, .composer { grid-template-columns:1fr; }
      .agent-panel { min-height:300px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Dual Qwen Demo</h1>
      <div class="status">左侧 Qwen-A 负责初稿和终稿，右侧 Qwen-B 负责复核。Gateway 负责提示词、短期记忆和调用顺序。</div>
    </div>
    <div class="status" id="status">模型名：dual-qwen-demo</div>
  </header>
  <main>
    <div class="dialog-grid">
      <section class="agent-panel">
        <div class="panel-head">
          <div>
            <div class="agent-name">Qwen-A</div>
            <div class="agent-role">初稿 / 最终整合</div>
          </div>
          <div class="agent-role">GPU 0,1</div>
        </div>
        <div class="messages" id="aMessages"></div>
      </section>
      <section class="agent-panel">
        <div class="panel-head">
          <div>
            <div class="agent-name">Qwen-B</div>
            <div class="agent-role">复核 / 质疑 / 改写建议</div>
          </div>
          <div class="agent-role">GPU 2,3</div>
        </div>
        <div class="messages" id="bMessages"></div>
      </section>
    </div>
    <div class="composer">
      <section>
        <label for="prompt">用户输入</label>
        <textarea id="prompt">请分析：服务层多智能体和模型内部 MoE 有什么区别？</textarea>
        <button id="send">发送</button>
      </section>
      <section>
        <div class="row" style="margin-top:0">
          <div><label for="maxNewTokens">输出 tokens</label><input id="maxNewTokens" type="number" min="1" max="2048" value="256" /></div>
          <div><label for="temperature">temperature</label><input id="temperature" type="number" min="0" step="0.1" value="0.0" /></div>
        </div>
        <div class="row">
          <div>
            <label for="mode">协作模式</label>
            <select id="mode">
              <option value="review_then_final" selected>B复核</option>
              <option value="interactive">A/B交互</option>
            </select>
          </div>
          <div><label for="rounds">交互轮数</label><input id="rounds" type="number" min="1" max="3" value="1" /></div>
        </div>
        <div class="row">
          <div><label for="sessionId">会话 ID</label><input id="sessionId" value="default" /></div>
          <div><label>模式说明</label><input id="modeHint" value="A初稿 -> B复核 -> A终稿" disabled /></div>
        </div>
        <div class="check-row">
          <label><input id="useMemory" type="checkbox" checked /> 使用短期记忆</label>
          <label><input id="resetMemory" type="checkbox" /> 本轮清空记忆</label>
        </div>
        <label style="margin-top:12px">最终回答</label>
        <div class="final-box" id="answer"></div>
        <details>
          <summary>原始调用记录</summary>
          <pre id="trace"></pre>
        </details>
      </section>
    </div>
  </main>
  <script>
    const send = document.getElementById("send");
    const aMessages = document.getElementById("aMessages");
    const bMessages = document.getElementById("bMessages");
    const answer = document.getElementById("answer");
    const trace = document.getElementById("trace");
    const statusEl = document.getElementById("status");
    const streamNodes = {};
    const modeSelect = document.getElementById("mode");
    const modeHint = document.getElementById("modeHint");

    modeSelect.addEventListener("change", () => {
      modeHint.value = modeSelect.value === "interactive"
        ? "A初稿 -> B复核 -> A修订 ... -> A终稿"
        : "A初稿 -> B复核 -> A终稿";
    });

    function addMessage(container, cssClass, title, body, elapsedMs) {
      const item = document.createElement("div");
      item.className = `message ${cssClass}`;
      const head = document.createElement("div");
      head.className = "message-title";
      head.innerHTML = `<span>${title}</span><span>${elapsedMs ? elapsedMs + " ms" : ""}</span>`;
      const content = document.createElement("div");
      content.className = "message-body";
      content.textContent = body || "";
      item.appendChild(head);
      item.appendChild(content);
      container.appendChild(item);
      container.scrollTop = container.scrollHeight;
      return { item, head, content };
    }

    function renderTurns(data) {
      aMessages.innerHTML = "";
      bMessages.innerHTML = "";
      for (const turn of data.turns || []) {
        const title = `${turn.speaker} · ${turn.role}`;
        if (turn.speaker === "Qwen-A") {
          addMessage(aMessages, turn.role === "final" ? "final" : "a", title, turn.response, turn.elapsed_ms);
        } else {
          addMessage(bMessages, "b", title, turn.response, turn.elapsed_ms);
        }
      }
      if (!bMessages.children.length) {
        addMessage(bMessages, "b", "Qwen-B", "等待复核任务", "");
      }
    }

    function streamKey(event) {
      return `${event.speaker}:${event.role}`;
    }

    function ensureStreamNode(event) {
      const key = streamKey(event);
      if (streamNodes[key]) return streamNodes[key];
      const container = event.speaker === "Qwen-A" ? aMessages : bMessages;
      const cssClass = event.speaker === "Qwen-A" ? (event.role === "final" ? "final" : "a") : "b";
      const node = addMessage(container, cssClass, `${event.speaker} · ${event.role}`, "", "");
      streamNodes[key] = node;
      return node;
    }

    function handleStreamEvent(event) {
      if (event.type === "start") {
        statusEl.textContent = `流式交互中：session=${event.session_id}，memory_used=${event.memory_used}`;
        return;
      }
      if (event.type === "token") {
        const node = ensureStreamNode(event);
        node.content.textContent += event.text || "";
        node.content.parentElement.parentElement.scrollTop = node.content.parentElement.parentElement.scrollHeight;
        if (event.speaker === "Qwen-A" && event.role === "final") {
          answer.textContent += event.text || "";
        }
        return;
      }
      if (event.type === "turn_done") {
        const node = ensureStreamNode(event);
        const title = node.head.querySelector("span:first-child");
        const elapsed = node.head.querySelector("span:last-child");
        if (title) title.textContent = `${event.speaker} · ${event.role}`;
        if (elapsed) elapsed.textContent = `${event.elapsed_ms || ""} ms`;
        return;
      }
      if (event.type === "complete") {
        answer.textContent = event.final_answer || answer.textContent;
        trace.textContent = JSON.stringify(event, null, 2);
        statusEl.textContent = `完成：session=${event.session_id}，memory_size=${event.memory_size}，elapsed=${event.elapsed_ms} ms`;
        document.getElementById("resetMemory").checked = false;
        return;
      }
      if (event.type === "error") {
        statusEl.textContent = "流式生成失败";
        answer.textContent = event.detail || JSON.stringify(event);
      }
    }

    send.addEventListener("click", async () => {
      send.disabled = true;
      statusEl.textContent = "连接流式生成接口...";
      aMessages.innerHTML = "";
      bMessages.innerHTML = "";
      answer.textContent = "";
      trace.textContent = "";
      for (const key of Object.keys(streamNodes)) delete streamNodes[key];
      addMessage(aMessages, "a", "Qwen-A", "等待生成初稿...", "");
      addMessage(bMessages, "b", "Qwen-B", "等待 Qwen-A 初稿...", "");
      try {
        const payload = {
          prompt: document.getElementById("prompt").value,
          max_new_tokens: Number(document.getElementById("maxNewTokens").value),
          temperature: Number(document.getElementById("temperature").value),
          rounds: Number(document.getElementById("rounds").value),
          mode: document.getElementById("mode").value,
          session_id: document.getElementById("sessionId").value || "default",
          use_memory: document.getElementById("useMemory").checked,
          reset_memory: document.getElementById("resetMemory").checked
        };
        const res = await fetch("/debate/stream", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
        if (!res.ok || !res.body) {
          const data = await res.json().catch(() => ({}));
          throw new Error(data.detail || JSON.stringify(data) || `HTTP ${res.status}`);
        }
        aMessages.innerHTML = "";
        bMessages.innerHTML = "";
        const reader = res.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, {stream:true});
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";
          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;
            handleStreamEvent(JSON.parse(trimmed));
          }
        }
        if (buffer.trim()) handleStreamEvent(JSON.parse(buffer.trim()));
      } catch (err) {
        aMessages.innerHTML = "";
        bMessages.innerHTML = "";
        answer.textContent = String(err);
        statusEl.textContent = "请求失败";
      } finally {
        send.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


def create_app(runner: DualQwenGateway) -> FastAPI:
    app = FastAPI(title="Dual Qwen Gateway", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_HTML

    @app.get("/health")
    def health():
        return runner.health()

    @app.get("/api/tags")
    def ollama_tags():
        return {
            "models": [
                {
                    "name": DEMO_MODEL_NAME,
                    "model": DEMO_MODEL_NAME,
                    "modified_at": _utc_now_iso(),
                    "size": 0,
                    "digest": "dual-qwen-service-layer-demo",
                    "details": {
                        "format": "fastapi",
                        "family": "qwen",
                        "families": ["qwen"],
                        "parameter_size": "14B x 2",
                        "quantization_level": "bf16",
                    },
                }
            ]
        }

    @app.get("/api/version")
    def ollama_version():
        return {"version": "dual-qwen-fastapi-0.1.0"}

    @app.get("/api/ps")
    def ollama_ps():
        return {"models": []}

    @app.post("/api/show")
    def ollama_show(payload: dict[str, Any]):
        return {
            "modelfile": f"FROM {payload.get('model') or DEMO_MODEL_NAME}\n",
            "parameters": "temperature 0\n",
            "template": "{{ .Prompt }}",
            "details": {
                "format": "fastapi",
                "family": "qwen",
                "families": ["qwen"],
                "parameter_size": "14B x 2",
                "quantization_level": "bf16",
            },
        }

    @app.post("/debate", response_model=DebateResponse)
    def debate(request: DebateRequest):
        return runner.debate(request)

    @app.post("/debate/stream")
    def debate_stream(request: DebateRequest):
        def _events():
            for event in runner.debate_stream(request):
                yield (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")

        return StreamingResponse(_events(), media_type="application/x-ndjson")

    @app.post("/generate", response_model=DebateResponse)
    def generate(request: DebateRequest):
        return runner.debate(request)

    @app.post("/api/generate")
    def ollama_generate(request: OllamaGenerateRequest):
        payload = runner.ollama_generate_payload(request)
        if request.stream:
            return StreamingResponse(_single_line_stream(payload), media_type="application/x-ndjson")
        return payload

    @app.post("/api/chat")
    def ollama_chat(request: OllamaChatRequest):
        payload = runner.ollama_chat_payload(request)
        if request.stream:
            return StreamingResponse(_single_line_stream(payload), media_type="application/x-ndjson")
        return payload

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8030)
    parser.add_argument("--qwen-a-url", default=DEFAULT_QWEN_A_URL)
    parser.add_argument("--qwen-b-url", default=DEFAULT_QWEN_B_URL)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    runner = DualQwenGateway(
        qwen_a_url=args.qwen_a_url,
        qwen_b_url=args.qwen_b_url,
        request_timeout_s=args.request_timeout_s,
    )
    app = create_app(runner)
    print(f"Serving Dual Qwen Gateway on http://{args.host}:{args.port}", flush=True)
    print(f"Qwen-A -> {args.qwen_a_url}", flush=True)
    print(f"Qwen-B -> {args.qwen_b_url}", flush=True)
    print(f"Ollama-compatible model name: {DEMO_MODEL_NAME}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
