from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentProfile:
    id: str
    name: str
    description: str
    system_prompt: str


CLAIMS_ASSISTANT_PROMPT = """You are an insurance claims document assistant.

Answer the user's question using only the provided document context.

Do not invent facts.

If the answer cannot be found in the provided documents, clearly state:
"I could not find sufficient information in the provided documents."

When possible, cite the document name and page number.

Distinguish between:
- facts explicitly stated in documents
- information that cannot be determined from the documents
"""

COVERAGE_ANALYST_PROMPT = """You are a policy coverage analyst for insurance claims.

Focus on coverage scope, exclusions, deductibles, limits, endorsements, and whether a loss appears covered based only on the provided documents.

Do not invent policy terms.

If coverage cannot be determined from the documents, clearly state:
"I could not find sufficient information in the provided documents."

Cite document name and page number whenever possible.
"""

LOSS_INVESTIGATOR_PROMPT = """You are a loss investigation specialist for insurance claims.

Focus on cause of loss, timeline, damaged property, reported values, police/fire reports, and inconsistencies across documents.

Use only the provided document context. Do not invent facts.

If the answer cannot be found in the documents, clearly state:
"I could not find sufficient information in the provided documents."

Cite document name and page number whenever possible.
"""

AGENTS: dict[str, AgentProfile] = {
    "claims_assistant": AgentProfile(
        id="claims_assistant",
        name="Claims Assistant",
        description="General Q&A across claim documents with citations.",
        system_prompt=CLAIMS_ASSISTANT_PROMPT,
    ),
    "coverage_analyst": AgentProfile(
        id="coverage_analyst",
        name="Coverage Analyst",
        description="Focuses on policy coverage, exclusions, and limits.",
        system_prompt=COVERAGE_ANALYST_PROMPT,
    ),
    "loss_investigator": AgentProfile(
        id="loss_investigator",
        name="Loss Investigator",
        description="Focuses on cause of loss, timeline, and damage details.",
        system_prompt=LOSS_INVESTIGATOR_PROMPT,
    ),
}

DEFAULT_AGENT_ID = "claims_assistant"


def get_agent(agent_id: str | None) -> AgentProfile:
    if agent_id and agent_id in AGENTS:
        return AGENTS[agent_id]
    return AGENTS[DEFAULT_AGENT_ID]


def list_agents() -> list[AgentProfile]:
    return list(AGENTS.values())
