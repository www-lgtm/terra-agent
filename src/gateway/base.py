"""Base adapter interface for messaging platforms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class MessageEvent:
    platform: str
    chat_id: str
    user_id: str
    text: str
    message_id: str = ""
    media_urls: list[str] | None = None


class BasePlatformAdapter(ABC):
    """Abstract base for messaging platform adapters."""

    @abstractmethod
    async def start(self) -> None:
        """Start listening for messages."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop listening."""

    @abstractmethod
    async def send_message(self, chat_id: str, text: str) -> bool:
        """Send a text message."""

    @abstractmethod
    async def send_image(self, chat_id: str, image_path: str) -> bool:
        """Send an image."""
