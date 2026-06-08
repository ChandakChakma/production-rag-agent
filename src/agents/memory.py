"""
Conversation memory with two strategies:
  1. SlidingWindowMemory — keeps last N message pairs (fast, simple)
  2. CompressedMemory    — when window fills, summarizes older messages via LLM

Interview talking point: Compression lets agents handle long conversations
without hitting the context window limit, while retaining semantic continuity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import tiktoken
from openai import OpenAI

from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Message:
    role: str  # "user" | "assistant" | "system" | "tool"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ConversationMemory:
    """
    Sliding window with LLM-based compression.

    When the total token count exceeds max_tokens:
      1. Keep the system prompt.
      2. Compress the oldest 50% of messages into a summary using GPT.
      3. Insert the summary as a synthetic assistant message.
      4. Retain the most recent messages verbatim.
    """

    _COMPRESS_PROMPT = (
        "Summarize the following conversation history concisely. "
        "Preserve all factual information, decisions made, and context "
        "needed to continue the conversation. Output only the summary.\n\n"
        "{history}"
    )

    def __init__(
        self,
        max_tokens: int = 2000,
        window_size: int = 10,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.max_tokens = max_tokens
        self.window_size = window_size  # message pairs
        self._model = model
        self._messages: list[Message] = []
        self._system_prompt: str | None = None
        self._compressed_summary: str | None = None

        settings = get_settings()
        self._client = OpenAI(api_key=settings.openai_api_key)
        try:
            self._enc = tiktoken.encoding_for_model(model)
        except KeyError:
            self._enc = tiktoken.get_encoding("cl100k_base")

    def set_system(self, prompt: str) -> None:
        self._system_prompt = prompt

    def add_user(self, content: str, **metadata: Any) -> None:
        self._messages.append(Message(role="user", content=content, metadata=metadata))
        self._maybe_compress()

    def add_assistant(self, content: str, **metadata: Any) -> None:
        self._messages.append(Message(role="assistant", content=content, metadata=metadata))

    def add_tool_result(self, tool_name: str, content: str) -> None:
        self._messages.append(
            Message(role="assistant", content=f"[{tool_name} result]: {content}")
        )

    def get_messages(self) -> list[dict[str, str]]:
        """Return messages in OpenAI format, respecting window size."""
        result: list[dict[str, str]] = []

        if self._system_prompt:
            result.append({"role": "system", "content": self._system_prompt})

        if self._compressed_summary:
            result.append({
                "role": "assistant",
                "content": f"[Conversation summary]: {self._compressed_summary}",
            })

        # Only return last window_size * 2 messages (user + assistant pairs)
        windowed = self._messages[-(self.window_size * 2):]
        result.extend({"role": m.role, "content": m.content} for m in windowed)
        return result

    def clear(self) -> None:
        self._messages.clear()
        self._compressed_summary = None

    def token_count(self) -> int:
        text = " ".join(m.content for m in self._messages)
        return len(self._enc.encode(text))

    # ── Private ───────────────────────────────────────────────────────

    def _maybe_compress(self) -> None:
        if self.token_count() <= self.max_tokens:
            return
        if len(self._messages) < 4:
            return

        # Compress the oldest half
        split = len(self._messages) // 2
        old_messages = self._messages[:split]
        self._messages = self._messages[split:]

        history_text = "\n".join(
            f"{m.role.upper()}: {m.content}" for m in old_messages
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": self._COMPRESS_PROMPT.format(history=history_text),
                    }
                ],
                temperature=0.0,
                max_tokens=400,
            )
            summary = response.choices[0].message.content or ""
            if self._compressed_summary:
                self._compressed_summary = f"{self._compressed_summary}\n{summary}"
            else:
                self._compressed_summary = summary
            logger.info(
                "memory_compressed",
                messages_compressed=len(old_messages),
                summary_len=len(summary),
            )
        except Exception as exc:
            # On compression failure, just truncate (degraded gracefully)
            logger.warning("memory_compression_failed", error=str(exc))
            self._compressed_summary = f"[{split} earlier messages truncated]"
