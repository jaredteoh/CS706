"""
AutoEmpirical MAS - Stage III: Fault Taxonomy Classification
=============================================================
Reads:  outputs/stage2_output.json  (fault_related list)
Writes: outputs/stage3_output.json

Four CAMEL ChatAgents:
  1. ClassifierAgent  - proposes symptom + root cause label with reasoning
  2. CriticAgent      - challenges the label, forces justification
  3. ClassifierAgent  - rebuts the challenge (same agent, new turn)
  4. ResolverAgent    - reads full debate, makes final classification

Debate loop is dynamic — continues until either:
  - Resolver confidence >= confidence_threshold, OR
  - max_rounds is reached (safety cap)

This directly targets AutoEmpirical's ~50% root cause accuracy by:
  - Forcing the classifier to justify its reasoning under scrutiny
  - Having a separate resolver that is not anchored to the first label
  - Using the full debate transcript as context for the final decision
  - SPLIT DEBATE: symptom and root cause debated separately so that
    root cause reasoning is grounded in the already-settled symptom

Taxonomy source: Quan et al. ASE 2022
  Symptoms   : 5 primary, 15 subcategories, 15 leaf types
  Root causes : 5 primary, 17 subcategories
"""

import csv
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.types import ModelPlatformType
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))
from tools.models import ModelConfig, build_model, parse_json


# ---------------------------------------------------------------------------
# Taxonomy (Quan et al. ASE 2022)
# ---------------------------------------------------------------------------

SYMPTOM_TAXONOMY = """
Bug Symptom Taxonomy (AutoEmpirical — Quan et al. 2022):

[A] Crash — functionality terminated unexpectedly with error messages
  [A.1] Reference Error
        A.1.1 DL Operator Exception: DL-related function exceptions
        A.1.2 Function Inaccessible: Traditional function exceptions
        A.1.3 Tensor Disposed: Disposed tensors accessed by program
        A.1.4 Attribute/Return Value Undefined: Undefined variable properties or function return values
        A.1.5 Training Argument Exception: Issues with training arguments
  [A.2] Data & Model Error
        A.2.1 Tensor Shape/Type/Value Error: Incorrect data types, shapes, or values for DL tensors
        A.2.2 JS Variable Shape/Type/Value Error: Incorrect data types, shapes, or values for JS variables
        A.2.3 Model Usage/Design Error: Failure in model usage or structure construction
  [A.3] Fetch Failure — crashes during web API requests for model files or data (e.g. same-origin policy)
  [A.4] Browser & Device Error — crashes showing browser or device problems (e.g. WebGL not supported)

[B] Poor Performance — slows execution, consumes excessive resources, bad user experience
  [B.1] Time
        B.1.1 Slow Execution: Systems work but are extremely slow during DL tasks
        B.1.2 Browser Hangs: Systems cease to respond; browser becomes unresponsive
  [B.2] Memory
        B.2.1 Memory Leak: Gradual increase in memory usage over time
        B.2.2 Out of Memory: System terminates due to insufficient memory
        B.2.3 Abnormal GPU Memory/Utilization: Unexpectedly high or low GPU memory usage
  [B.3] Others
        B.3.1 Regression: Performance issues occurring after TensorFlow.js upgrades
        B.3.2 Unstable: Inference results are inconsistent or unstable

[C] Build & Initialization Failure — failures during build, install, or initialization of DL environments
  C.1 TF.js/JS Application Compile Failure
  C.2 npm Package Installation Failure
  C.3 Multi-backend Initialization Failure

[D] Incorrect Functionality — systems run without crashes but produce incorrect results
  D.1 Inconsistency between Backends/Platforms/Devices
  D.2 Poor Accuracy
  D.3 Inf/None/Null Results
  D.4 Others

[E] Document Error — invalid links, incorrect instructions, or missing tutorials in official docs

Output the most specific ID that matches (e.g. A.1 for Reference Error crash, B.2.1 for Memory Leak).
Use top-level IDs (C, D, E) when the issue fits the category but no subcategory is clearly applicable.
"""

ROOT_CAUSE_TAXONOMY = """
Root Cause Taxonomy (AutoEmpirical — Quan et al. 2022):

[A] Incorrect Programming — faults caused by code implementation issues
  [A.1] Unimplemented Operator — DL operators not yet supported/implemented by TF.js
  [A.2] Inconsistent Modules in TF.js — inconsistent implementations between TF.js modules
  [A.3] API Misuse — misunderstanding of APIs: missing/redundant calls, wrong names, invalid params
  [A.4] Incorrect Code Logic — faulty implementation in DL algorithms, memory management, or env adaptability
  [A.5] Incompatibility between 3rd-party DL Library and TF.js — version mismatches with 3rd-party DL libs
  [A.6] Import Error — missing/incorrect import of TF.js or importing multiple versions simultaneously
  [A.7] Improper Exception Handling — missing exceptions, suspicious exceptions, or confusing error messages

[B] Configuration & Dependency Error — faults caused by incorrect configuration and dependencies
  [B.1] Multi-environment Misconfiguration — incorrect bundler configs for heterogeneous environments
  [B.2] Dependency Error — missing/redundant deps, version mismatches, or security vulnerabilities
  [B.3] Untimely Update — issues from not updating tensorflow.so or npm packages in time
  [B.4] Confused Document — problems caused by unclear or incorrect TF.js documentation

[C] Data/Model Error — faults introduced by DL models and data
  [C.1] Data/Model Inaccessibility — data/models cannot be accessed (browser limits, wrong paths)
  [C.2] Improper Model/Tensor Attribute — poor model design, improper parameters, or inappropriate model size

[D] Execution Environment Error — faults from imperfect support for hardware/software environments
  [D.1] Device Incompatibility — issues on specific hardware and operating systems
  [D.2] Browser Incompatibility — compatibility issues with PC or mobile browsers
  [D.3] Cross-platform App Framework Incompatibility — incompatibility with React Native, Electron, etc.
  [D.4] WebGL Limits — faults caused by inherited limitations of WebGL

[E] Unknown — root cause is difficult to analyze or unclear from available information

Output the subcategory ID (e.g. A.4, B.2, D.2). Use E only when the root cause truly cannot be determined.
"""

# Lookup table so Phase 2 agents receive the symptom description,
# not just the bare ID (e.g. "A.1.1 — DL Operator Exception")
SYMPTOM_LABELS = {
    "A":     "Crash",
    "A.1":   "Reference Error",
    "A.1.1": "DL Operator Exception",
    "A.1.2": "Function Inaccessible",
    "A.1.3": "Tensor Disposed",
    "A.1.4": "Attribute/Return Value Undefined",
    "A.1.5": "Training Argument Exception",
    "A.2":   "Data & Model Error",
    "A.2.1": "Tensor Shape/Type/Value Error",
    "A.2.2": "JS Variable Shape/Type/Value Error",
    "A.2.3": "Model Usage/Design Error",
    "A.3":   "Fetch Failure",
    "A.4":   "Browser & Device Error",
    "B":     "Poor Performance",
    "B.1":   "Time",
    "B.1.1": "Slow Execution",
    "B.1.2": "Browser Hangs",
    "B.2":   "Memory",
    "B.2.1": "Memory Leak",
    "B.2.2": "Out of Memory",
    "B.2.3": "Abnormal GPU Memory/Utilization",
    "B.3":   "Others",
    "B.3.1": "Regression",
    "B.3.2": "Unstable",
    "C":     "Build & Initialization Failure",
    "C.1":   "TF.js/JS Application Compile Failure",
    "C.2":   "npm Package Installation Failure",
    "C.3":   "Multi-backend Initialization Failure",
    "D":     "Incorrect Functionality",
    "D.1":   "Inconsistency between Backends/Platforms/Devices",
    "D.2":   "Poor Accuracy",
    "D.3":   "Inf/None/Null Results",
    "D.4":   "Others",
    "E":     "Document Error",
}


def _symptom_context(symptom_id: str) -> str:
    """Returns 'A.1.1 - DL Operator Exception' for use in Phase 2 prompts."""
    label = SYMPTOM_LABELS.get(symptom_id, "")
    return f"{symptom_id} - {label}" if label else symptom_id


# ---------------------------------------------------------------------------
# Stage III configuration
# ---------------------------------------------------------------------------

@dataclass
class Stage3Config:
    # I/O paths
    stage2_output_path: str = "outputs/stage2_output.json"
    output_path: str = "outputs/stage3_output.json"

    # Debate settings
    max_rounds: int = 3
    confidence_threshold: float = 0.80

    # Cap for testing — None means process all
    max_issues: Optional[int] = None

    # Optional: load directly from clean_CollectedIssues.csv (skips stage2)
    csv_path: Optional[str] = None

    # LLM backend
    model: ModelConfig = field(default_factory=ModelConfig)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DebateRound:
    round_number: int
    classifier_label: dict
    critic_challenge: str
    classifier_rebuttal: str


@dataclass
class TaxonomyDecision:
    issue_number: int
    repo: str
    symptom_id: str
    root_cause_id: str
    reasoning: str
    confidence: float
    debate_rounds: int
    debate_transcript: dict     # {"symptom": [...], "root_cause": [...]}
    ground_truth_symptom: str = ""
    ground_truth_root_cause: str = ""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _format_issue_context(issue: dict) -> str:
    """Build the issue context block passed to every Stage 3 agent prompt."""
    parts = []
    if issue.get("title"):
        parts.append(f"Title: {issue['title']}")
    if issue.get("state"):
        parts.append(f"State: {issue['state']}")
    if issue.get("body"):
        parts.append(f"\nBody:\n{issue['body']}")
    if issue.get("comments_content"):
        parts.append(f"\nComments:\n{issue['comments_content']}")
    if issue.get("reasoning") and not issue.get("body"):
        parts.append(f"\nStage 2 analysis:\n{issue['reasoning']}")
    return "\n".join(parts) if parts else "(no issue content available)"


def _format_transcript(transcript: list) -> str:
    lines = []
    for turn in transcript:
        r = turn["round"]
        lines += [
            f"--- Round {r} ---",
            f"Classifier: {turn.get('label_id', '')}",
            f"  Reasoning: {turn.get('classifier_reasoning', '')}",
            f"Critic: {turn.get('challenge', '')}",
            f"Classifier rebuttal: {turn.get('rebuttal', '')}",
            "",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent 1a: SymptomClassifierAgent
# ---------------------------------------------------------------------------

class SymptomClassifierAgent:
    """
    Classifies ONLY the symptom.
    Unchanged from original ClassifierAgent except scope is symptom-only.
    Returns JSON: {symptom_id, reasoning, confidence}
    """

    SYSTEM_PROMPT = (
        "You are an expert in analyzing JavaScript-based deep learning systems bugs.\n\n"
        "Given a GitHub issue, classify ONLY the bug symptom ID from the taxonomy.\n\n"
        "Rules:\n"
        "- Use ONLY the provided taxonomy IDs — do not invent new ones.\n"
        "- Choose the most specific ID that fits the evidence.\n"
        "- Use top-level IDs (C, D, E) only when no subcategory clearly applies.\n\n"
        "Respond with valid JSON only — no prose, no markdown fences.\n"
        "Schema: {\"symptom_id\": str, \"reasoning\": str, \"confidence\": float}\n"
        "confidence is a float between 0.0 and 1.0."
    )

    def __init__(self, config: Stage3Config, model):
        self.config = config
        self.agent = ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="Symptom Classifier",
                content=self.SYSTEM_PROMPT,
            ),
            model=model,
            token_limit=config.model.token_limit,
        )

    def classify(self, issue: dict) -> dict:
        prompt = (
            f"Issue: {issue['repo']}#{issue['issue_number']}\n"
            f"{_format_issue_context(issue)}\n\n"
            f"{SYMPTOM_TAXONOMY}\n"
            "Classify the symptom only. Respond with JSON only."
        )
        response = self.agent.step(
            BaseMessage.make_user_message(role_name="Researcher", content=prompt)
        )
        self.agent.reset()
        return parse_json(response.msg.content,
                          default={"symptom_id": "A",
                                   "reasoning": "parse failed",
                                   "confidence": 0.0})

    def rebut(self, issue: dict, current: dict, challenge: str) -> dict:
        prompt = (
            f"Issue: {issue['repo']}#{issue['issue_number']}\n"
            f"{_format_issue_context(issue)}\n\n"
            f"Your current symptom classification:\n"
            f"  symptom_id: {current.get('symptom_id')}\n"
            f"  Reasoning: {current.get('reasoning')}\n\n"
            f"Critic's challenge:\n{challenge}\n\n"
            f"{SYMPTOM_TAXONOMY}\n"
            "Respond to the challenge. Revise if the critic raises a valid point, "
            "or defend with stronger evidence. Respond with JSON only."
        )
        response = self.agent.step(
            BaseMessage.make_user_message(role_name="Researcher", content=prompt)
        )
        self.agent.reset()
        return parse_json(response.msg.content, default=current)


# ---------------------------------------------------------------------------
# Agent 1b: RootCauseClassifierAgent
# ---------------------------------------------------------------------------

class RootCauseClassifierAgent:
    """
    Classifies ONLY the root cause, receiving the settled symptom as context.
    Unchanged from original ClassifierAgent except scope is root-cause-only.
    Returns JSON: {root_cause_id, reasoning, confidence}
    """

    SYSTEM_PROMPT = (
        "You are an expert in analyzing JavaScript-based deep learning systems bugs.\n\n"
        "Given a GitHub issue and its confirmed symptom, classify ONLY the root cause ID.\n\n"
        "Rules:\n"
        "- Use ONLY the provided taxonomy IDs — do not invent new ones.\n"
        "- Always use a subcategory ID (e.g. A.4 not just A).\n"
        "- Use E only when the root cause truly cannot be determined.\n\n"
        "Respond with valid JSON only — no prose, no markdown fences.\n"
        "Schema: {\"root_cause_id\": str, \"reasoning\": str, \"confidence\": float}\n"
        "confidence is a float between 0.0 and 1.0."
    )

    def __init__(self, config: Stage3Config, model):
        self.config = config
        self.agent = ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="Root Cause Classifier",
                content=self.SYSTEM_PROMPT,
            ),
            model=model,
            token_limit=config.model.token_limit,
        )

    def classify(self, issue: dict, symptom_id: str) -> dict:
        prompt = (
            f"Issue: {issue['repo']}#{issue['issue_number']}\n"
            f"{_format_issue_context(issue)}\n\n"
            f"Confirmed symptom: {_symptom_context(symptom_id)}\n\n"
            f"{ROOT_CAUSE_TAXONOMY}\n"
            "Classify the root cause only. Respond with JSON only."
        )
        response = self.agent.step(
            BaseMessage.make_user_message(role_name="Researcher", content=prompt)
        )
        self.agent.reset()
        return parse_json(response.msg.content,
                          default={"root_cause_id": "E",
                                   "reasoning": "parse failed",
                                   "confidence": 0.0})

    def rebut(self, issue: dict, symptom_id: str,
              current: dict, challenge: str) -> dict:
        prompt = (
            f"Issue: {issue['repo']}#{issue['issue_number']}\n"
            f"{_format_issue_context(issue)}\n\n"
            f"Confirmed symptom: {_symptom_context(symptom_id)}\n\n"
            f"Your current root cause classification:\n"
            f"  root_cause_id: {current.get('root_cause_id')}\n"
            f"  Reasoning: {current.get('reasoning')}\n\n"
            f"Critic's challenge:\n{challenge}\n\n"
            f"{ROOT_CAUSE_TAXONOMY}\n"
            "Respond to the challenge. Revise if the critic raises a valid point, "
            "or defend with stronger evidence. Respond with JSON only."
        )
        response = self.agent.step(
            BaseMessage.make_user_message(role_name="Researcher", content=prompt)
        )
        self.agent.reset()
        return parse_json(response.msg.content, default=current)


# ---------------------------------------------------------------------------
# Agent 2a: SymptomCriticAgent
# ---------------------------------------------------------------------------

class SymptomCriticAgent:
    """
    Challenges the symptom classification.
    Same logic as original CriticAgent, scoped to symptom only.
    Returns plain text (not JSON).
    """

    SYSTEM_PROMPT = (
        "You are a senior software engineering researcher critically "
        "reviewing a bug SYMPTOM classification.\n\n"
        "Your job is to challenge the classification if you see weaknesses:\n"
        "1. Is there a more specific label that fits better?\n"
        "2. Is the reasoning consistent with the issue evidence?\n"
        "3. Is the symptom being confused with the root cause?\n"
        "4. Is a top-level ID used when a subcategory is determinable?\n\n"
        "Be direct and specific. Reference the taxonomy labels by name.\n"
        "If the classification is genuinely strong, say so briefly and "
        "explain why — do not challenge for the sake of it.\n\n"
        "Respond in plain text (not JSON)."
    )

    def __init__(self, config: Stage3Config, model):
        self.config = config
        self.agent = ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="Symptom Critic",
                content=self.SYSTEM_PROMPT,
            ),
            model=model,
            token_limit=config.model.token_limit,
        )

    def challenge(self, issue: dict, label: dict) -> str:
        prompt = (
            f"Issue: {issue['repo']}#{issue['issue_number']}\n"
            f"{_format_issue_context(issue)}\n\n"
            f"Proposed symptom classification:\n"
            f"  symptom_id : {label.get('symptom_id')}\n"
            f"  Reasoning  : {label.get('reasoning')}\n"
            f"  Confidence : {label.get('confidence')}\n\n"
            f"{SYMPTOM_TAXONOMY}\n"
            "Challenge this symptom classification if warranted. Be specific."
        )
        response = self.agent.step(
            BaseMessage.make_user_message(role_name="Researcher", content=prompt)
        )
        self.agent.reset()
        return response.msg.content.strip()


# ---------------------------------------------------------------------------
# Agent 2b: RootCauseCriticAgent
# ---------------------------------------------------------------------------

class RootCauseCriticAgent:
    """
    Challenges the root cause classification.
    Same logic as original CriticAgent, scoped to root cause only.
    Returns plain text (not JSON).
    """

    SYSTEM_PROMPT = (
        "You are a senior software engineering researcher critically "
        "reviewing a bug ROOT CAUSE classification.\n\n"
        "Your job is to challenge the classification if you see weaknesses:\n"
        "1. Is there a more specific label that fits better?\n"
        "2. Is the reasoning consistent with the issue evidence?\n"
        "3. Is the root cause being confused with the symptom?\n"
        "4. Is 'Unknown' being used too hastily when a label is determinable?\n\n"
        "Be direct and specific. Reference the taxonomy labels by name.\n"
        "If the classification is genuinely strong, say so briefly and "
        "explain why — do not challenge for the sake of it.\n\n"
        "Respond in plain text (not JSON)."
    )

    def __init__(self, config: Stage3Config, model):
        self.config = config
        self.agent = ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="Root Cause Critic",
                content=self.SYSTEM_PROMPT,
            ),
            model=model,
            token_limit=config.model.token_limit,
        )

    def challenge(self, issue: dict, symptom_id: str, label: dict) -> str:
        prompt = (
            f"Issue: {issue['repo']}#{issue['issue_number']}\n"
            f"{_format_issue_context(issue)}\n\n"
            f"Confirmed symptom: {_symptom_context(symptom_id)}\n\n"
            f"Proposed root cause classification:\n"
            f"  root_cause_id : {label.get('root_cause_id')}\n"
            f"  Reasoning     : {label.get('reasoning')}\n"
            f"  Confidence    : {label.get('confidence')}\n\n"
            f"{ROOT_CAUSE_TAXONOMY}\n"
            "Challenge this root cause classification if warranted. Be specific."
        )
        response = self.agent.step(
            BaseMessage.make_user_message(role_name="Researcher", content=prompt)
        )
        self.agent.reset()
        return response.msg.content.strip()


# ---------------------------------------------------------------------------
# Agent 3a: SymptomResolverAgent
# ---------------------------------------------------------------------------

class SymptomResolverAgent:
    """
    Reads the symptom debate transcript and makes the final symptom decision.
    Same logic as original ResolverAgent, scoped to symptom only.
    Returns JSON: {symptom_id, reasoning, confidence}
    """

    SYSTEM_PROMPT = (
        "You are a principal software engineering researcher making a "
        "final BUG SYMPTOM classification decision.\n\n"
        "You will receive the full debate between a classifier and a critic. "
        "Your job is to:\n"
        "1. Weigh the arguments from both sides objectively.\n"
        "2. Select the most accurate symptom_id from the taxonomy.\n"
        "3. Assign a confidence score reflecting how certain you are.\n\n"
        "Use ONLY the provided taxonomy IDs.\n"
        "A high confidence (>= 0.8) means the debate can stop.\n"
        "A low confidence (< 0.8) means another round may help.\n\n"
        "Respond with valid JSON only — no prose, no markdown fences.\n"
        "Schema: {\"symptom_id\": str, \"reasoning\": str, \"confidence\": float}"
    )

    def __init__(self, config: Stage3Config, model):
        self.config = config
        self.agent = ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="Symptom Resolver",
                content=self.SYSTEM_PROMPT,
            ),
            model=model,
            token_limit=config.model.token_limit,
        )

    def resolve(self, issue: dict, transcript: list) -> dict:
        transcript_text = _format_transcript(transcript)
        prompt = (
            f"Issue: {issue['repo']}#{issue['issue_number']}\n"
            f"{_format_issue_context(issue)}\n\n"
            f"{SYMPTOM_TAXONOMY}\n"
            f"Debate transcript:\n{transcript_text}\n\n"
            "Based on the full debate above, make the final symptom classification. "
            "Respond with JSON only."
        )
        response = self.agent.step(
            BaseMessage.make_user_message(role_name="Researcher", content=prompt)
        )
        self.agent.reset()
        return parse_json(response.msg.content,
                          default={"symptom_id": "A",
                                   "reasoning": "parse failed",
                                   "confidence": 0.0})


# ---------------------------------------------------------------------------
# Agent 3b: RootCauseResolverAgent
# ---------------------------------------------------------------------------

class RootCauseResolverAgent:
    """
    Reads the root cause debate transcript and makes the final root cause decision.
    Same logic as original ResolverAgent, scoped to root cause only.
    Returns JSON: {root_cause_id, reasoning, confidence}
    """

    SYSTEM_PROMPT = (
        "You are a principal software engineering researcher making a "
        "final ROOT CAUSE classification decision.\n\n"
        "You will receive the full debate between a classifier and a critic. "
        "Your job is to:\n"
        "1. Weigh the arguments from both sides objectively.\n"
        "2. Select the most accurate root_cause_id from the taxonomy.\n"
        "3. Assign a confidence score reflecting how certain you are.\n\n"
        "Use ONLY the provided taxonomy IDs.\n"
        "A high confidence (>= 0.8) means the debate can stop.\n"
        "A low confidence (< 0.8) means another round may help.\n\n"
        "Respond with valid JSON only — no prose, no markdown fences.\n"
        "Schema: {\"root_cause_id\": str, \"reasoning\": str, \"confidence\": float}"
    )

    def __init__(self, config: Stage3Config, model):
        self.config = config
        self.agent = ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="Root Cause Resolver",
                content=self.SYSTEM_PROMPT,
            ),
            model=model,
            token_limit=config.model.token_limit,
        )

    def resolve(self, issue: dict, symptom_id: str, transcript: list) -> dict:
        transcript_text = _format_transcript(transcript)
        prompt = (
            f"Issue: {issue['repo']}#{issue['issue_number']}\n"
            f"{_format_issue_context(issue)}\n\n"
            f"Confirmed symptom: {_symptom_context(symptom_id)}\n\n"
            f"{ROOT_CAUSE_TAXONOMY}\n"
            f"Debate transcript:\n{transcript_text}\n\n"
            "Based on the full debate above, make the final root cause classification. "
            "Respond with JSON only."
        )
        response = self.agent.step(
            BaseMessage.make_user_message(role_name="Researcher", content=prompt)
        )
        self.agent.reset()
        return parse_json(response.msg.content,
                          default={"root_cause_id": "E",
                                   "reasoning": "parse failed",
                                   "confidence": 0.0})


# ---------------------------------------------------------------------------
# Stage III pipeline
# ---------------------------------------------------------------------------

class Stage3Pipeline:
    """
    Two-phase debate pipeline per issue:

    Phase 1 — Symptom debate:
      SymptomClassifier → SymptomCritic → SymptomClassifier → SymptomResolver
      (repeat until confidence >= threshold or max_rounds)
            ↓
      settled symptom_id passed as context to Phase 2

    Phase 2 — Root cause debate:
      RootCauseClassifier → RootCauseCritic → RootCauseClassifier → RootCauseResolver
      (repeat until confidence >= threshold or max_rounds)
    """

    def __init__(self, config: Optional[Stage3Config] = None):
        self.config = config or Stage3Config()
        model = build_model(self.config.model)
        self.symptom_classifier = SymptomClassifierAgent(self.config, model)
        self.symptom_critic = SymptomCriticAgent(self.config, model)
        self.symptom_resolver = SymptomResolverAgent(self.config, model)
        self.rc_classifier = RootCauseClassifierAgent(self.config, model)
        self.rc_critic = RootCauseCriticAgent(self.config, model)
        self.rc_resolver = RootCauseResolverAgent(self.config, model)

    def load_issues_from_csv(self, csv_path: str) -> list:
        """
        Load fault issues directly from clean_CollectedIssues.csv.
        """
        issues = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row.get("Faults", "").strip()
                if not url:
                    continue
                parts = url.rstrip("/").split("/")
                try:
                    issues_idx = parts.index("issues")
                    repo = "/".join(parts[issues_idx - 2: issues_idx])
                    issue_number = int(parts[issues_idx + 1])
                except (ValueError, IndexError):
                    continue

                issues.append({
                    "repo": repo,
                    "issue_number": issue_number,
                    "title": row.get("title", ""),
                    "body": row.get("body", ""),
                    "state": row.get("state", ""),
                    "created_at": row.get("created_at", ""),
                    "comments_content": row.get("comments_content", ""),
                    "ground_truth_symptom": row.get("symptom_id", ""),
                    "ground_truth_root_cause": row.get("root_causes_id", ""),
                })

        if self.config.max_issues:
            issues = issues[:self.config.max_issues]

        print(f"[Stage III] Loaded {len(issues)} issues from CSV: {csv_path}")
        return issues

    def run(self, issues: list) -> dict:
        if self.config.max_issues:
            issues = issues[:self.config.max_issues]

        decisions = []
        total = len(issues)

        for i, issue in enumerate(issues):
            print(f"\n[Stage III] {i+1}/{total} — "
                  f"{issue['repo']}#{issue['issue_number']}")

            decision = self._classify_with_debate(issue)
            decisions.append(decision)

            print(f"           Symptom ID   : {decision.symptom_id}")
            print(f"           Root cause ID: {decision.root_cause_id}")
            print(f"           Confidence: {decision.confidence:.2f} "
                  f"({decision.debate_rounds} round(s))")

        output = self._build_output(decisions)
        os.makedirs(os.path.dirname(self.config.output_path), exist_ok=True)
        with open(self.config.output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n[Stage III] Saved → {self.config.output_path}")
        self._print_summary(output)
        return output

    def _classify_with_debate(self, issue: dict) -> TaxonomyDecision:
        sym_transcript = []
        rc_transcript = []

        # ── Phase 1: Symptom debate ──────────────────────────────────────
        print(f"           [Symptom debate]")
        sym_label = self.symptom_classifier.classify(issue)
        sym_final = sym_label
        sym_rounds = 0

        for round_num in range(1, self.config.max_rounds + 1):
            print(f"             Round {round_num}...", end=" ")

            challenge = self.symptom_critic.challenge(issue, sym_label)
            rebuttal = self.symptom_classifier.rebut(issue, sym_label, challenge)

            sym_transcript.append({
                "round": round_num,
                "label_id": sym_label.get("symptom_id", ""),
                "classifier_reasoning": sym_label.get("reasoning", ""),
                "challenge": challenge,
                "rebuttal": rebuttal.get("reasoning", ""),
            })

            sym_label = rebuttal
            sym_rounds = round_num

            resolver_result = self.symptom_resolver.resolve(issue, sym_transcript)
            sym_conf = float(resolver_result.get("confidence", 0.0))
            sym_final = resolver_result

            print(f"conf={sym_conf:.2f}")

            if sym_conf >= self.config.confidence_threshold:
                print(f"             Confident — stopping symptom debate")
                break
            elif round_num < self.config.max_rounds:
                print(f"             Low confidence — continuing...")

        settled_symptom = sym_final.get("symptom_id", "A")
        print(f"           Settled symptom: {settled_symptom}")

        # ── Phase 2: Root cause debate (uses settled symptom) ────────────
        print(f"           [Root cause debate]")
        rc_label = self.rc_classifier.classify(issue, settled_symptom)
        rc_final = rc_label
        rc_rounds = 0

        for round_num in range(1, self.config.max_rounds + 1):
            print(f"             Round {round_num}...", end=" ")

            challenge = self.rc_critic.challenge(issue, settled_symptom, rc_label)
            rebuttal = self.rc_classifier.rebut(
                issue, settled_symptom, rc_label, challenge)

            rc_transcript.append({
                "round": round_num,
                "label_id": rc_label.get("root_cause_id", ""),
                "classifier_reasoning": rc_label.get("reasoning", ""),
                "challenge": challenge,
                "rebuttal": rebuttal.get("reasoning", ""),
            })

            rc_label = rebuttal
            rc_rounds = round_num

            resolver_result = self.rc_resolver.resolve(
                issue, settled_symptom, rc_transcript)
            rc_conf = float(resolver_result.get("confidence", 0.0))
            rc_final = resolver_result

            print(f"conf={rc_conf:.2f}")

            if rc_conf >= self.config.confidence_threshold:
                print(f"             Confident — stopping root cause debate")
                break
            elif round_num < self.config.max_rounds:
                print(f"             Low confidence — continuing...")

        return TaxonomyDecision(
            issue_number=issue["issue_number"],
            repo=issue["repo"],
            symptom_id=sym_final.get("symptom_id", "A"),
            root_cause_id=rc_final.get("root_cause_id", "E"),
            reasoning=(
                f"Symptom: {sym_final.get('reasoning', '')}\n"
                f"Root cause: {rc_final.get('reasoning', '')}"
            ),
            confidence=round(
                (float(sym_final.get("confidence", 0.0)) +
                 float(rc_final.get("confidence", 0.0))) / 2, 4
            ),
            debate_rounds=sym_rounds + rc_rounds,
            debate_transcript={
                "symptom": sym_transcript,
                "root_cause": rc_transcript,
            },
            ground_truth_symptom=issue.get("ground_truth_symptom", ""),
            ground_truth_root_cause=issue.get("ground_truth_root_cause", ""),
        )

    @staticmethod
    def _build_output(decisions: list) -> dict:
        return {
            "total_classified": len(decisions),
            "avg_confidence": (
                sum(d.confidence for d in decisions) / len(decisions)
                if decisions else 0.0
            ),
            "avg_debate_rounds": (
                sum(d.debate_rounds for d in decisions) / len(decisions)
                if decisions else 0.0
            ),
            "classifications": [
                {
                    "issue_number": d.issue_number,
                    "repo": d.repo,
                    "symptom_id": d.symptom_id,
                    "root_cause_id": d.root_cause_id,
                    "reasoning": d.reasoning,
                    "confidence": d.confidence,
                    "debate_rounds": d.debate_rounds,
                    "debate_transcript": d.debate_transcript,
                    "ground_truth_symptom": d.ground_truth_symptom,
                    "ground_truth_root_cause": d.ground_truth_root_cause,
                }
                for d in decisions
            ],
        }

    @staticmethod
    def _print_summary(output: dict) -> None:
        print(f"[Stage III] Total classified : {output['total_classified']}")
        print(f"            Avg confidence   : "
              f"{output['avg_confidence']:.2f}")
        print(f"            Avg debate rounds: "
              f"{output['avg_debate_rounds']:.1f}")


# ---------------------------------------------------------------------------
# Standalone entry point  (unchanged from your original)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from tools.models import model_config_from_name

    parser = argparse.ArgumentParser(description="Stage III: Fault Taxonomy Classification")
    parser.add_argument("--csv-path", default=None,
                        help="Path to clean_CollectedIssues.csv for direct evaluation")
    parser.add_argument("--model", default="llama3",
                        help="Model name (e.g. gpt-4o, claude-3-7-sonnet, gemini-2.5-flash)")
    parser.add_argument("--max-issues", type=int, default=None,
                        help="Cap on number of issues to process (default: all)")
    parser.add_argument("--max-rounds", type=int, default=3,
                        help="Max debate rounds per issue per phase (default: 3)")
    parser.add_argument("--confidence-threshold", type=float, default=0.80,
                        help="Resolver confidence threshold to stop debate (default: 0.80)")
    args = parser.parse_args()

    model_cfg = model_config_from_name(args.model)
    config = Stage3Config(
        max_issues=args.max_issues,
        max_rounds=args.max_rounds,
        confidence_threshold=args.confidence_threshold,
        csv_path=args.csv_path,
        model=model_cfg,
    )

    pipeline = Stage3Pipeline(config)

    if args.csv_path:
        issues = pipeline.load_issues_from_csv(args.csv_path)
    else:
        stage2_path = "outputs/stage2_output.json"
        if os.path.exists(stage2_path):
            with open(stage2_path) as f:
                stage2 = json.load(f)
            issues = stage2.get("fault_related", [])
            print(f"[Stage III] Loaded {len(issues)} fault-related issues "
                  f"from Stage II output.")
        else:
            print("[Stage III] No Stage II output found and no --csv-path given.")
            sys.exit(1)

    pipeline.run(issues)
