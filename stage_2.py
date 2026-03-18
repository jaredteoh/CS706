"""
AutoEmpirical MAS - Stage II: Fault-Related Issue Filtering
============================================================
Reads:  outputs/stage1_output.json
Writes: outputs/stage2_output.json

Three CAMEL ChatAgents:
  1. ContextRetrieverAgent  - fetches linked PR diff + changed files
                              from GitHub API (our improvement over baseline)
  2. FilterAgent            - binary fault/non-fault decision using
                              issue text + code context
  3. ConfidenceScorerAgent  - reviews FilterAgent's reasoning, assigns
                              adjusted confidence, flags low-confidence cases

Run standalone:
    python stage2_issue_filtering.py

Or via orchestrator:
    python main.py
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.types import ModelPlatformType
from dotenv import load_dotenv

load_dotenv()

# Allow running as standalone or as part of the package
sys.path.insert(0, os.path.dirname(__file__))
from tools.models import ModelConfig, build_model, parse_json


# ---------------------------------------------------------------------------
# Stage II configuration
# ---------------------------------------------------------------------------

@dataclass
class Stage2Config:
    # I/O paths
    stage1_output_path: str = "outputs/stage1_output.json"
    output_path: str = "outputs/stage2_output.json"

    # Filtering criteria (mirrors AutoEmpirical's DL fault criteria)
    filtering_criteria: str = (
        "An issue is fault-related if ALL of the following hold:\n"
        "1. DL-Relevance: contains at least one deep learning keyword "
        "(e.g. tensor, model, layer, gradient, loss, inference, training, "
        "WebGL, dispose, op, kernel, backend).\n"
        "2. Fault Reporting: describes an observable problem, error, or "
        "system failure — NOT a feature request, question, or discussion.\n"
        "3. Technical Clarity: provides enough technical detail to enable "
        "fault analysis.\n"
        "4. Not excluded: does not carry exclusion labels like "
        "'awaiting response', is not about deprecated versions (pre-2020), "
        "and has at least one substantive response."
    )

    # Confidence threshold — below this the issue is flagged for human review
    confidence_threshold: float = 0.75

    # GitHub API
    github_token: Optional[str] = field(
        default_factory=lambda: os.getenv("GITHUB_TOKEN")
    )
    github_api_base: str = "https://api.github.com"
    github_rate_limit_delay: float = 0.5

    # Cap for testing — set to None to process all issues
    max_issues: Optional[int] = 20

    # LLM backend (shared config from shared/models.py)
    model: ModelConfig = field(default_factory=ModelConfig)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IssueContext:
    issue_number: int
    repo: str
    title: str
    body: str
    labels: list
    created_at: str
    linked_pr: Optional[dict] = None
    diff_summary: Optional[str] = None
    changed_files: Optional[list] = None
    fetch_error: Optional[str] = None


@dataclass
class FilterDecision:
    issue_number: int
    repo: str
    is_fault_related: bool
    reasoning: str
    confidence: float
    flagged_for_review: bool = False


# ---------------------------------------------------------------------------
# GitHub client (used by ContextRetrieverAgent)
# ---------------------------------------------------------------------------

class GitHubClient:
    """Thin GitHub REST API wrapper. No database — pure on-demand retrieval."""

    def __init__(self, config: Stage2Config):
        headers = {"Accept": "application/vnd.github+json",
                   "X-GitHub-Api-Version": "2022-11-28"}
        if config.github_token:
            headers["Authorization"] = f"Bearer {config.github_token}"
        self.client = httpx.Client(
            base_url=config.github_api_base,
            headers=headers,
            timeout=15.0,
        )

    def get_issue(self, owner, repo, number):
        r = self.client.get(f"/repos/{owner}/{repo}/issues/{number}")
        r.raise_for_status()
        return r.json()

    def get_timeline(self, owner, repo, number):
        r = self.client.get(
            f"/repos/{owner}/{repo}/issues/{number}/timeline",
            headers={"Accept": "application/vnd.github.mockingbird-preview+json"},
        )
        r.raise_for_status()
        return r.json()

    def get_pr_diff(self, owner, repo, pr_number):
        r = self.client.get(
            f"/repos/{owner}/{repo}/pulls/{pr_number}",
            headers={"Accept": "application/vnd.github.diff"},
        )
        r.raise_for_status()
        return r.text[:2000]

    def get_pr_files(self, owner, repo, pr_number):
        r = self.client.get(f"/repos/{owner}/{repo}/pulls/{pr_number}/files")
        r.raise_for_status()
        return [f["filename"] for f in r.json()]

    def list_issues(self, owner, repo, state="all", per_page=100, max_pages=10):
        """Fetch issues (excluding pull requests) for a repo, newest first."""
        issues = []
        for page in range(1, max_pages + 1):
            r = self.client.get(
                f"/repos/{owner}/{repo}/issues",
                params={"state": state, "per_page": per_page,
                        "page": page, "sort": "created", "direction": "desc"},
            )
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            # GitHub issues endpoint also returns PRs — filter them out
            issues.extend(i for i in batch if "pull_request" not in i)
        return issues

    def close(self):
        self.client.close()


# ---------------------------------------------------------------------------
# Agent 1: ContextRetrieverAgent
# ---------------------------------------------------------------------------

class ContextRetrieverAgent:
    """
    Fetches code context (PR diff, changed files) from GitHub for each issue.

    KEY IMPROVEMENT over AutoEmpirical: baseline only uses issue text.
    By retrieving the linked PR diff, FilterAgent has direct evidence of
    whether a real code fix exists — substantially stronger signal for
    root cause classification.

    Degrades gracefully: if GitHub fetch fails, sets fetch_error and
    downstream agents fall back to text-only classification.

    Note: This agent does not use a CAMEL ChatAgent — it's a pure API
    retrieval step with no LLM call needed. The LLM agents start at Step 2.
    """

    def __init__(self, config: Stage2Config):
        self.config = config
        self.github = GitHubClient(config)

    def retrieve(self, owner: str, repo: str,
                 issue_number: int) -> IssueContext:
        # Fetch issue metadata
        try:
            issue = self.github.get_issue(owner, repo, issue_number)
        except Exception as e:
            return IssueContext(
                issue_number=issue_number, repo=f"{owner}/{repo}",
                title="", body="", labels=[], created_at="",
                fetch_error=f"Issue fetch failed: {e}",
            )

        ctx = IssueContext(
            issue_number=issue_number,
            repo=f"{owner}/{repo}",
            title=issue.get("title", ""),
            body=(issue.get("body") or "")[:3000],
            labels=[l["name"] for l in issue.get("labels", [])],
            created_at=issue.get("created_at", ""),
        )

        # Try to find and fetch the linked PR
        try:
            time.sleep(self.config.github_rate_limit_delay)
            timeline = self.github.get_timeline(owner, repo, issue_number)
            pr_number = self._find_linked_pr(timeline)
            if pr_number:
                time.sleep(self.config.github_rate_limit_delay)
                ctx.diff_summary = self.github.get_pr_diff(
                    owner, repo, pr_number)
                ctx.changed_files = self.github.get_pr_files(
                    owner, repo, pr_number)
                ctx.linked_pr = {"number": pr_number}
        except Exception as e:
            ctx.fetch_error = f"PR fetch failed: {e}"

        return ctx

    @staticmethod
    def _find_linked_pr(timeline: list) -> Optional[int]:
        for event in timeline:
            if event.get("event") == "cross-referenced":
                pr = event.get("source", {}).get("issue", {}).get(
                    "pull_request")
                if pr:
                    parts = pr.get("url", "").rstrip("/").split("/")
                    try:
                        return int(parts[-1])
                    except (ValueError, IndexError):
                        pass
        return None

    def close(self):
        self.github.close()


# ---------------------------------------------------------------------------
# Agent 2: FilterAgent
# ---------------------------------------------------------------------------

class FilterAgent:
    """
    CAMEL ChatAgent that makes a binary fault/non-fault decision.

    Uses BOTH issue text and retrieved code context (diff, changed files).
    Returns JSON: {is_fault_related, reasoning, confidence}
    """

    SYSTEM_PROMPT = (
        "You are a software engineering researcher specialising in empirical "
        "fault studies. Determine whether a GitHub issue is fault-related "
        "based on the issue text and any available code context.\n\n"
        "Respond with valid JSON only — no prose, no markdown fences.\n"
        'Schema: {"is_fault_related": bool, "reasoning": str, '
        '"confidence": float}\n'
        "confidence is a float between 0.0 and 1.0."
    )

    def __init__(self, config: Stage2Config, model):
        self.config = config
        self.agent = ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="Filter Agent",
                content=self.SYSTEM_PROMPT,
            ),
            model=model,
            token_limit=config.model.token_limit,
        )

    def classify(self, ctx: IssueContext) -> dict:
        prompt = self._build_prompt(ctx)
        response = self.agent.step(
            BaseMessage.make_user_message(
                role_name="Researcher", content=prompt)
        )
        # Reset memory between issues so context doesn't bleed across
        self.agent.reset()
        return parse_json(response.msg.content,
                          default={"is_fault_related": False,
                                   "reasoning": "parse failed",
                                   "confidence": 0.0})

    def _build_prompt(self, ctx: IssueContext) -> str:
        parts = [
            f"Repository: {ctx.repo}",
            f"Issue #{ctx.issue_number}: {ctx.title}",
            f"Created: {ctx.created_at}",
            f"Labels: {', '.join(ctx.labels) if ctx.labels else 'none'}",
            "", "Issue body:", ctx.body or "(empty)",
        ]
        if ctx.diff_summary:
            parts += ["", "Linked PR diff (first 2000 chars):",
                      ctx.diff_summary]
        if ctx.changed_files:
            parts += ["", "Changed files:",
                      "\n".join(f"  - {f}" for f in ctx.changed_files[:20])]
        if ctx.fetch_error:
            parts += ["", f"Note: code context unavailable ({ctx.fetch_error})."
                      " Base decision on issue text only."]
        parts += ["", "Filtering criteria:", self.config.filtering_criteria,
                  "", "Is this issue fault-related? Respond with JSON only."]
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Agent 3: ConfidenceScorerAgent
# ---------------------------------------------------------------------------

class ConfidenceScorerAgent:
    """
    CAMEL ChatAgent that reviews FilterAgent's decision.

    Checks reasoning for logical gaps, adjusts confidence, and flags
    borderline cases for human review.
    Returns JSON: {adjusted_confidence, review_notes, flag_for_human_review}
    """

    SYSTEM_PROMPT = (
        "You are a senior software engineering researcher reviewing a "
        "fault classification decision.\n\n"
        "Check whether the reasoning is sound and consistent with the "
        "evidence. Identify contradictions, missing evidence, or borderline "
        "cases. Output an adjusted confidence score.\n\n"
        "Respond with valid JSON only — no prose, no markdown fences.\n"
        'Schema: {"adjusted_confidence": float, "review_notes": str, '
        '"flag_for_human_review": bool}\n'
        "adjusted_confidence is a float between 0.0 and 1.0."
    )

    def __init__(self, config: Stage2Config, model):
        self.config = config
        self.agent = ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="Confidence Scorer",
                content=self.SYSTEM_PROMPT,
            ),
            model=model,
            token_limit=config.model.token_limit,
        )

    def review(self, ctx: IssueContext, filter_result: dict) -> dict:
        verdict = "FAULT-RELATED" if filter_result.get(
            "is_fault_related") else "NOT FAULT-RELATED"
        context_note = (
            "Issue text + PR diff + changed files"
            if ctx.diff_summary else "Issue text only"
        )
        prompt = (
            f"Issue: {ctx.repo}#{ctx.issue_number} — {ctx.title}\n\n"
            f"Classification: {verdict}\n"
            f"Reasoning: {filter_result.get('reasoning', '')}\n"
            f"Initial confidence: {filter_result.get('confidence', 0.0)}\n"
            f"Context available: {context_note}\n"
            f"Fetch errors: {ctx.fetch_error or 'none'}\n\n"
            f"Review this decision. Flag for human review if "
            f"adjusted_confidence < {self.config.confidence_threshold}. "
            f"Respond with JSON only."
        )
        response = self.agent.step(
            BaseMessage.make_user_message(
                role_name="Researcher", content=prompt)
        )
        self.agent.reset()
        return parse_json(response.msg.content,
                          default={"adjusted_confidence": filter_result.get(
                              "confidence", 0.5),
                              "review_notes": "parse failed",
                              "flag_for_human_review": True})


# ---------------------------------------------------------------------------
# Stage II pipeline
# ---------------------------------------------------------------------------

class Stage2Pipeline:
    """
    Orchestrates the three Stage II agents per issue:
      ContextRetrieverAgent → FilterAgent → ConfidenceScorerAgent
    """

    def __init__(self, config: Optional[Stage2Config] = None):
        self.config = config or Stage2Config()
        model = build_model(self.config.model)
        self.retriever = ContextRetrieverAgent(self.config)
        self.filter_agent = FilterAgent(self.config, model)
        self.scorer = ConfidenceScorerAgent(self.config, model)

    def fetch_issues_from_stage1(self) -> list:
        """Read Stage I output and fetch candidate issues for each repo."""
        with open(self.config.stage1_output_path) as f:
            stage1 = json.load(f)

        repos = stage1.get("selected_repos", [])
        print(f"[Stage II] Fetching issues for {len(repos)} repos from Stage I...")

        all_issues = []
        for repo_entry in repos:
            repo_full = repo_entry.get("name", "")  # e.g. "tensorflow/tfjs"
            if "/" not in repo_full:
                print(f"           Skipping '{repo_full}' — not in owner/repo format")
                continue
            owner, repo_name = repo_full.split("/", 1)
            try:
                raw = self.retriever.github.list_issues(owner, repo_name)
                for issue in raw:
                    all_issues.append({
                        "repo": repo_full,
                        "issue_number": issue["number"],
                        "title": issue.get("title", ""),
                        "body": (issue.get("body") or "")[:500],
                        "labels": [l["name"] for l in issue.get("labels", [])],
                    })
                print(f"           {repo_full}: {len(raw)} issues fetched")
                time.sleep(self.config.github_rate_limit_delay)
            except Exception as e:
                print(f"           {repo_full}: fetch failed — {e}")

        print(f"[Stage II] Total candidate issues: {len(all_issues)}")
        return all_issues

    def run(self, issues: list) -> dict:
        if self.config.max_issues:
            issues = issues[:self.config.max_issues]

        fault_related, non_fault, flagged = [], [], []
        fetch_failures = 0
        total = len(issues)

        for i, issue in enumerate(issues):
            owner, repo_name = issue["repo"].split("/", 1)
            num = issue["issue_number"]
            print(f"[Stage II] {i+1}/{total} — {issue['repo']}#{num}")

            # Agent 1: retrieve context
            ctx = self.retriever.retrieve(owner, repo_name, num)
            if ctx.fetch_error and not ctx.title:
                ctx.title = issue.get("title", "")
                ctx.body = issue.get("body", "")
                ctx.labels = issue.get("labels", [])
                fetch_failures += 1
                print(f"           fetch failed: {ctx.fetch_error}")
            elif ctx.linked_pr:
                print(f"           PR #{ctx.linked_pr['number']} retrieved")
            else:
                print(f"           no linked PR — text-only")

            # Agent 2: filter
            filter_result = self.filter_agent.classify(ctx)
            is_fault = filter_result.get("is_fault_related", False)

            # Agent 3: confidence scoring
            review = self.scorer.review(ctx, filter_result)
            confidence = float(review.get("adjusted_confidence",
                                          filter_result.get("confidence", 0.5)))
            flag = review.get("flag_for_human_review", False) or (
                confidence < self.config.confidence_threshold)

            decision = FilterDecision(
                issue_number=num, repo=issue["repo"],
                is_fault_related=is_fault,
                reasoning=filter_result.get("reasoning", ""),
                confidence=confidence,
                flagged_for_review=flag,
            )

            (fault_related if is_fault else non_fault).append(decision)
            if flag:
                flagged.append(decision)

            status = "FAULT" if is_fault else "non-fault"
            print(f"           → {status} "
                  f"(conf: {confidence:.2f})"
                  f"{' [FLAGGED]' if flag else ''}")

            time.sleep(self.config.github_rate_limit_delay)

        output = {
            "total_processed": total,
            "fault_related_count": len(fault_related),
            "non_fault_count": len(non_fault),
            "flagged_for_review_count": len(flagged),
            "context_fetch_failures": fetch_failures,
            "fault_related": [self._decision_dict(d) for d in fault_related],
            "non_fault": [self._decision_dict(d) for d in non_fault],
            "flagged_for_review": [self._decision_dict(d) for d in flagged],
        }

        os.makedirs(os.path.dirname(self.config.output_path), exist_ok=True)
        with open(self.config.output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n[Stage II] Saved → {self.config.output_path}")
        self._print_summary(output)
        return output

    @staticmethod
    def _decision_dict(d: FilterDecision) -> dict:
        return {
            "issue_number": d.issue_number, "repo": d.repo,
            "is_fault_related": d.is_fault_related,
            "reasoning": d.reasoning, "confidence": d.confidence,
            "flagged_for_review": d.flagged_for_review,
        }

    @staticmethod
    def _print_summary(output: dict) -> None:
        t = output["total_processed"]
        print(f"[Stage II] Total: {t} | "
              f"Fault: {output['fault_related_count']} | "
              f"Non-fault: {output['non_fault_count']} | "
              f"Flagged: {output['flagged_for_review_count']} | "
              f"Fetch failures: {output['context_fetch_failures']}")

    def close(self):
        self.retriever.close()


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = Stage2Config(
        max_issues=20,  # cap per run — set to None for full run
        model=ModelConfig(
            platform=ModelPlatformType.OLLAMA,
            model_type="llama3",
        ),
    )

    pipeline = Stage2Pipeline(config)
    try:
        issues = pipeline.fetch_issues_from_stage1()
        pipeline.run(issues)
    finally:
        pipeline.close()
