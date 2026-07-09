"""Bounded workflow session state with TTL eviction."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float:
        ...


class MonotonicClock:
    def now(self) -> float:
        return time.monotonic()


class ManualClock:
    def __init__(self, initial: float = 0.0) -> None:
        self._now = initial

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


@dataclass(frozen=True, slots=True)
class SessionAction:
    action_id: str
    updated_at: float


class WorkflowSessionStore:
    def __init__(
        self,
        *,
        ttl_seconds: float = 1800.0,
        max_sessions: int = 100_000,
        clock: Clock | None = None,
    ) -> None:
        if ttl_seconds <= 0.0:
            raise ValueError("ttl_seconds must be positive")
        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        self._ttl_seconds = ttl_seconds
        self._max_sessions = max_sessions
        self._clock = clock or MonotonicClock()
        self._actions: dict[str, SessionAction] = {}

    @property
    def ttl_seconds(self) -> float:
        return self._ttl_seconds

    @property
    def max_sessions(self) -> int:
        return self._max_sessions

    @property
    def size(self) -> int:
        self.evict_expired()
        return len(self._actions)

    def get(self, session_id: str) -> str | None:
        action = self._actions.get(session_id)
        if action is None:
            return None
        if self._is_expired(action):
            self._actions.pop(session_id, None)
            return None
        return action.action_id

    def set(self, session_id: str, action_id: str) -> None:
        self.evict_expired()
        self._actions.pop(session_id, None)
        self._actions[session_id] = SessionAction(action_id, self._clock.now())
        self._evict_over_capacity()

    def pop(self, session_id: str) -> None:
        self._actions.pop(session_id, None)

    def clear(self) -> None:
        self._actions.clear()

    def evict_expired(self) -> int:
        expired = [
            session_id
            for session_id, action in self._actions.items()
            if self._is_expired(action)
        ]
        for session_id in expired:
            self._actions.pop(session_id, None)
        return len(expired)

    def _is_expired(self, action: SessionAction) -> bool:
        return self._clock.now() - action.updated_at > self._ttl_seconds

    def _evict_over_capacity(self) -> int:
        evicted = 0
        while len(self._actions) > self._max_sessions:
            oldest = next(iter(self._actions))
            self._actions.pop(oldest, None)
            evicted += 1
        return evicted
