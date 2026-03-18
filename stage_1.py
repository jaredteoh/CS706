"""
AutoEmpirical MAS - Stage I: Research Definition
=================================================
Replicates AutoEmpirical's Stage I using the CAMEL AI framework.

Two agents:
  1. RepoSelectorAgent  - selects representative open-source repositories
                          given a research theme and resource constraints.
  2. RQFormulatorAgent  - formulates concrete research questions given the
                          selected repositories and broader study context.
"""

import json
import os
import sys
from dataclasses import dataclass
from typing import Optional

from camel.agents import ChatAgent
from camel.messages import BaseMessage

sys.path.insert(0, os.path.dirname(__file__))
from tools.models import ModelConfig, build_model, parse_json


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Stage1Config:
    """Top-level configuration for Stage I."""

    # Research theme provided by the human researcher (required)
    research_theme: str

    # Optional guidance on resource / scope constraints
    scope_constraints: str = (
        "Focus on actively maintained open-source projects with at least "
        "100 GitHub issues and meaningful community activity. "
        "Limit to projects with issues spanning 2020 onwards."
    )


    num_repos: int = 5
    num_rqs: int = 3

    output_path: str = "outputs/stage1_output.json"

    model: ModelConfig = None

    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class Stage1Output:
    """Structured output produced by Stage I."""
    research_theme: str
    selected_repos: list[dict]          # [{name, url, rationale}]
    research_questions: list[str]
    raw_repo_response: str = ""
    raw_rq_response: str = ""


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

def build_repo_selector(model) -> ChatAgent:
    """
    RepoSelectorAgent

    Role: Software engineering research assistant specialising in
    open-source repository analysis.

    Task: Given a research theme and scope constraints, identify the most
    representative repositories for an empirical fault study.
    """
    system_prompt = (
        "You are a software engineering research assistant specialising in "
        "open-source repository analysis. Your job is to identify the most "
        "representative open-source projects for empirical software fault studies.\n\n"
        "When selecting repositories you must:\n"
        "1. Prioritise projects with active issue trackers and high fault report volume.\n"
        "2. Ensure diversity — include both core libraries and third-party/application projects.\n"
        "3. Avoid overlooking lesser-known but relevant projects; do not focus only on "
        "the most prominent names.\n"
        "4. Justify each selection with a concise rationale.\n\n"
        "Always respond with valid JSON matching the schema:\n"
        '{"repositories": [{"name": str, "url": str, "rationale": str}]}'
    )
    return ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="Repo Selector",
            content=system_prompt,
        ),
        model=model,
    )


def build_rq_formulator(model) -> ChatAgent:
    """
    RQFormulatorAgent

    Role: Empirical software engineering researcher.

    Task: Given selected repositories and a research theme, formulate
    concrete, answerable research questions suitable for a fault study.
    """
    system_prompt = (
        "You are an experienced empirical software engineering researcher. "
        "Your job is to formulate concrete, specific, and answerable research "
        "questions for an empirical fault study.\n\n"
        "Good research questions for fault studies:\n"
        "- Focus on fault characteristics (symptoms, root causes, prevalence).\n"
        "- Are scoped to the selected repositories and domain.\n"
        "- Are answerable through systematic issue/commit analysis.\n"
        "- Avoid being too broad (e.g. 'what are all bugs?') or too narrow "
        "(e.g. single-function level).\n\n"
        "Always respond with valid JSON matching the schema:\n"
        '{"research_questions": [str]}'
    )
    return ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="RQ Formulator",
            content=system_prompt,
        ),
        model=model,
    )


# ---------------------------------------------------------------------------
# Stage I pipeline
# ---------------------------------------------------------------------------

class Stage1Pipeline:
    """
    Orchestrates the two Stage I agents sequentially:
      RepoSelectorAgent → RQFormulatorAgent → Stage1Output
    """

    def __init__(self, config: Optional[Stage1Config] = None):
        self.config = config or Stage1Config()
        model = build_model(self.config.model)
        self.repo_selector = build_repo_selector(model)
        self.rq_formulator = build_rq_formulator(model)

    # ------------------------------------------------------------------
    # Step 1: Repository selection
    # ------------------------------------------------------------------

    def _select_repos(self) -> tuple[list[dict], str]:
        prompt = (
            f"Research theme: {self.config.research_theme}\n\n"
            f"Scope constraints: {self.config.scope_constraints}\n\n"
            f"Please select {self.config.num_repos} representative open-source "
            "repositories for an empirical fault study on this theme. "
            "Include a mix of core libraries and third-party/application projects — "
            "do not focus only on the most prominent names. "
            "Return your answer as JSON."
        )

        response = self.repo_selector.step(
            BaseMessage.make_user_message(role_name="Researcher", content=prompt)
        )
        raw = response.msg.content
        repos = parse_json(raw, field="repositories", default=[])
        return repos, raw

    # ------------------------------------------------------------------
    # Step 2: Research question formulation
    # ------------------------------------------------------------------

    def _formulate_rqs(self, repos: list) -> tuple:
        repo_summary = "\n".join(
            f"- {r.get('name', 'Unknown')} ({r.get('url', '')}): "
            f"{r.get('rationale', '')}"
            for r in repos
        )
        prompt = (
            f"Research theme: {self.config.research_theme}\n\n"
            f"Selected repositories:\n{repo_summary}\n\n"
            f"Please formulate {self.config.num_rqs} concrete and answerable "
            "research questions for an empirical fault study covering these "
            "repositories. Focus on fault symptoms, root causes, and "
            "prevalence patterns. Return your answer as JSON."
        )
        response = self.rq_formulator.step(
            BaseMessage.make_user_message(role_name="Researcher", content=prompt)
        )
        raw = response.msg.content
        rqs = parse_json(raw, field="research_questions", default=[])
        return rqs, raw

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> Stage1Output:
        print(f"[Stage I] Research theme: {self.config.research_theme}")
 
        print("[Stage I] Step 1/2 — Selecting repositories...")
        repos, raw_repos = self._select_repos()
        print(f"          Selected {len(repos)} repositories.")
        for r in repos:
            print(f"          • {r.get('name', '?')} — {r.get('rationale', '')[:80]}...")
 
        print("[Stage I] Step 2/2 — Formulating research questions...")
        rqs, raw_rqs = self._formulate_rqs(repos)
        print(f"          Formulated {len(rqs)} research questions.")
        for i, rq in enumerate(rqs, 1):
            print(f"          RQ{i}: {rq}")
 
        output = Stage1Output(
            research_theme=self.config.research_theme,
            selected_repos=repos,
            research_questions=rqs,
            raw_repo_response=raw_repos,
            raw_rq_response=raw_rqs,
        )
 
        self._save(output)
        return output

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save(self, output: Stage1Output) -> None:
        """Persist Stage I output to JSON for consumption by Stage II."""
        payload = {
            "research_theme": output.research_theme,
            "selected_repos": output.selected_repos,
            "research_questions": output.research_questions,
        }
        os.makedirs(os.path.dirname(self.config.output_path), exist_ok=True)
        with open(self.config.output_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[Stage I] Output saved to {self.config.output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = Stage1Config(
        research_theme="JavaScript-based deep learning system faults",
        scope_constraints=(
            "Focus on actively maintained open-source projects with at least "
            "100 GitHub issues. Limit to projects with issues spanning 2020 onwards."
        ),
        num_repos=5,
        num_rqs=3,
        output_path="outputs/stage1_output.json",
        model=ModelConfig(
            model_type="llama3",
        ),
    )
 
    pipeline = Stage1Pipeline(config)
    pipeline.run()
