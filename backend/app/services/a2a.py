"""A small Agent-to-Agent hand-off.

MCP connects an agent to tools. A2A connects an agent to another agent.
This is the subset we actually use: an Agent Card (who can do what) and a
task that moves submitted → working → completed, failed, or canceled.

It is not a full A2A JSON-RPC server. The same task object is what the
manager passes in-process and what POST /api/multi-agent/tasks returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


TASK_STATES = ("submitted", "working", "completed", "failed", "canceled")

# Terminal states have no legal next step.
_ALLOWED: dict[str, set[str]] = {
    "submitted": {"working", "canceled"},
    "working": {"completed", "failed", "canceled"},
    "completed": set(),
    "failed": set(),
    "canceled": set(),
}


class A2AStateError(ValueError):
    """Raised when a task jumps to a state the lifecycle does not allow."""


@dataclass
class AgentCard:
    """Published description of one agent. Other agents read this before handing work over."""

    id: str
    name: str
    description: str
    skills: list[str]
    url: str
    version: str = "0.1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "skills": list(self.skills),
            "url": self.url,
            "version": self.version,
        }


@dataclass
class A2ATask:
    id: str
    specialist_id: str
    instruction: str
    state: str = "submitted"
    history: list[str] = field(default_factory=lambda: ["submitted"])
    messages: list[dict[str, str]] = field(default_factory=list)
    artifact: str = ""
    error: str = ""

    def transition(self, state: str) -> None:
        if state not in TASK_STATES:
            raise A2AStateError(f"unknown task state {state!r}")
        if state not in _ALLOWED[self.state]:
            raise A2AStateError(f"cannot move a task from {self.state} to {state}")
        self.state = state
        self.history.append(state)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "specialistId": self.specialist_id,
            "instruction": self.instruction,
            "state": self.state,
            "history": list(self.history),
            "messages": list(self.messages),
            "artifact": self.artifact,
            "error": self.error,
        }
