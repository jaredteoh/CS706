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

Taxonomy source: Quan et al. ASE 2022
  Symptoms   : 5 primary, 15 subcategories, 15 leaf types
  Root causes : 5 primary, 17 subcategories
"""

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
Symptom Taxonomy (from Quan et al. 2022):

1. Crash
   1.1 System Crash       — application or runtime crashes entirely
   1.2 Silent Crash       — crashes without error message

2. Poor Performance
   2.1 Memory Leak        — memory grows unboundedly
   2.2 Out of Memory      — OOM error during execution
   2.3 Slow Execution     — noticeably degraded speed

3. Build & Initialization Failure
   3.1 Build Failure      — fails during compilation or build
   3.2 Load Failure       — model or module fails to load
   3.3 Initialization Failure — fails during setup/init

4. Incorrect Functionality
   4.1 Incorrect Output   — wrong computation result
   4.2 Reference Error
       4.2.1 DL Operator Exception  — error in DL op execution
       4.2.2 Function Inaccessible  — function cannot be called
   4.3 Hanging            — program hangs or deadlocks

5. Document Error
   5.1 Missing Documentation  — docs absent or incomplete
   5.2 Incorrect Documentation — docs contradict actual behaviour
"""

ROOT_CAUSE_TAXONOMY = """
Root Cause Taxonomy (from Quan et al. 2022):

1. Incorrect Programming
   1.1 Algorithm Error          — wrong algorithm or logic
   1.2 Unimplemented Operator   — op not implemented for backend
   1.3 Inconsistent Modules in TF.js — mismatch between modules
   1.4 Incorrect Type Conversion — wrong dtype handling
   1.5 Incorrect Shape Handling  — tensor shape errors

2. Configuration & Dependency Error
   2.1 Version Incompatibility  — version mismatch between deps
   2.2 Missing Dependency       — required dep not present
   2.3 Incorrect Configuration  — wrong config values

3. Data/Model Error
   3.1 Unsupported Model        — model not supported by backend
   3.2 Incorrect Input Data     — wrong input format or values
   3.3 Numerical Instability    — NaN, Inf, precision issues

4. Execution Environment Error
   4.1 Platform Incompatibility — fails on specific platform/OS
   4.2 WebGL Error              — WebGL backend specific error
   4.3 Memory Management Error  — incorrect memory handling

5. Unknown
   5.1 Unknown                  — cannot determine root cause
"""


# ---------------------------------------------------------------------------
# Stage III configuration
# ---------------------------------------------------------------------------

@dataclass
class Stage3Config:
    # I/O paths
    stage2_output_path: str = "outputs/stage2_output.json"
    output_path: str = "outputs/stage3_output.json"

    # Debate settings
    max_rounds: int = 3           # safety cap on debate rounds
    confidence_threshold: float = 0.80  # resolver stops debating above this

    # Cap for testing — None means process all
    max_issues: Optional[int] = None

    # LLM backend
    model: ModelConfig = field(default_factory=ModelConfig)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DebateRound:
    round_number: int
    classifier_label: dict      # {symptom, root_cause, reasoning}
    critic_challenge: str
    classifier_rebuttal: str


@dataclass
class TaxonomyDecision:
    issue_number: int
    repo: str
    symptom: str                # leaf-level symptom label
    root_cause: str             # subcategory-level root cause label
    reasoning: str
    confidence: float
    debate_rounds: int
    debate_transcript: list     # list of DebateRound dicts


# ---------------------------------------------------------------------------
# Agent 1: ClassifierAgent
# ---------------------------------------------------------------------------

class ClassifierAgent:
    """
    CAMEL ChatAgent that proposes symptom + root cause labels.

    On first call: classifies from scratch given issue text.
    On subsequent calls: rebuts the critic's challenge, potentially
    revising or defending the original label.

    Returns JSON: {symptom, root_cause, reasoning, confidence}
    """

    SYSTEM_PROMPT = (
        "You are a software engineering researcher specialising in "
        "empirical fault taxonomy classification.\n\n"
        "You will be given a GitHub issue and must classify it into:\n"
        "  1. A symptom category (what went wrong externally)\n"
        "  2. A root cause category (why it went wrong internally)\n\n"
        "Use ONLY the provided taxonomy labels — do not invent new ones.\n"
        "Choose the most specific (leaf-level) label that fits.\n\n"
        "Respond with valid JSON only — no prose, no markdown fences.\n"
        "Schema: {\"symptom\": str, \"root_cause\": str, "
        "\"reasoning\": str, \"confidence\": float}\n"
        "confidence is a float between 0.0 and 1.0."
    )

    def __init__(self, config: Stage3Config, model):
        self.config = config
        self.agent = ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="Classifier",
                content=self.SYSTEM_PROMPT,
            ),
            model=model,
            token_limit=config.model.token_limit,
        )

    def classify(self, issue: dict) -> dict:
        """Initial classification from issue text."""
        prompt = (
            f"Issue: {issue['repo']}#{issue['issue_number']}\n"
            f"Reasoning: {issue.get('reasoning', '')}\n\n"
            f"{SYMPTOM_TAXONOMY}\n"
            f"{ROOT_CAUSE_TAXONOMY}\n"
            "Classify the symptom and root cause of this issue. "
            "Respond with JSON only."
        )
        response = self.agent.step(
            BaseMessage.make_user_message(
                role_name="Researcher", content=prompt)
        )
        self.agent.reset()
        return parse_json(response.msg.content,
                          default={"symptom": "Unknown",
                                   "root_cause": "5.1 Unknown",
                                   "reasoning": "parse failed",
                                   "confidence": 0.0})

    def rebut(self, issue: dict, current_label: dict,
              challenge: str) -> dict:
        """
        Responds to the critic's challenge.
        May revise the label or defend the original with stronger reasoning.
        """
        prompt = (
            f"Issue: {issue['repo']}#{issue['issue_number']}\n"
            f"Reasoning from issue: {issue.get('reasoning', '')}\n\n"
            f"Your current classification:\n"
            f"  Symptom: {current_label.get('symptom')}\n"
            f"  Root cause: {current_label.get('root_cause')}\n"
            f"  Reasoning: {current_label.get('reasoning')}\n\n"
            f"Critic's challenge:\n{challenge}\n\n"
            f"{SYMPTOM_TAXONOMY}\n"
            f"{ROOT_CAUSE_TAXONOMY}\n"
            "Respond to the challenge. You may revise your classification "
            "if the critic raises a valid point, or defend your original "
            "label with stronger evidence. Respond with JSON only."
        )
        response = self.agent.step(
            BaseMessage.make_user_message(
                role_name="Researcher", content=prompt)
        )
        self.agent.reset()
        return parse_json(response.msg.content,
                          default=current_label)


# ---------------------------------------------------------------------------
# Agent 2: CriticAgent
# ---------------------------------------------------------------------------

class CriticAgent:
    """
    CAMEL ChatAgent that challenges the ClassifierAgent's label.

    Looks for:
      - Alternative labels that fit better
      - Weak or missing evidence in the reasoning
      - Conflation of symptom with root cause
      - Over-generalisation to Unknown when a specific label is possible

    Returns a plain text challenge (not JSON — deliberate, natural debate).
    """

    SYSTEM_PROMPT = (
        "You are a senior software engineering researcher critically "
        "reviewing a fault taxonomy classification.\n\n"
        "Your job is to challenge the classification if you see weaknesses:\n"
        "1. Is there a more specific label that fits better?\n"
        "2. Is the reasoning consistent with the issue evidence?\n"
        "3. Is the symptom being confused with the root cause?\n"
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
                role_name="Critic",
                content=self.SYSTEM_PROMPT,
            ),
            model=model,
            token_limit=config.model.token_limit,
        )

    def challenge(self, issue: dict, label: dict) -> str:
        prompt = (
            f"Issue: {issue['repo']}#{issue['issue_number']}\n"
            f"Issue reasoning: {issue.get('reasoning', '')}\n\n"
            f"Proposed classification:\n"
            f"  Symptom   : {label.get('symptom')}\n"
            f"  Root cause: {label.get('root_cause')}\n"
            f"  Reasoning : {label.get('reasoning')}\n"
            f"  Confidence: {label.get('confidence')}\n\n"
            f"{SYMPTOM_TAXONOMY}\n"
            f"{ROOT_CAUSE_TAXONOMY}\n"
            "Challenge this classification if warranted. Be specific."
        )
        response = self.agent.step(
            BaseMessage.make_user_message(
                role_name="Researcher", content=prompt)
        )
        self.agent.reset()
        return response.msg.content.strip()


# ---------------------------------------------------------------------------
# Agent 3: ResolverAgent
# ---------------------------------------------------------------------------

class ResolverAgent:
    """
    CAMEL ChatAgent that reads the full debate transcript and makes
    the final taxonomy decision.

    Key design choice: the Resolver is NOT the Classifier — it has not
    been anchored to the first label and can freely choose any taxonomy
    entry based on the full evidence and debate.

    Returns JSON: {symptom, root_cause, reasoning, confidence}
    confidence >= threshold means the debate stops.
    """

    SYSTEM_PROMPT = (
        "You are a principal software engineering researcher making a "
        "final fault taxonomy classification decision.\n\n"
        "You will receive the full debate between a classifier and a critic. "
        "Your job is to:\n"
        "1. Weigh the arguments from both sides objectively.\n"
        "2. Select the most accurate symptom and root cause labels.\n"
        "3. Assign a confidence score reflecting how certain you are.\n\n"
        "Use ONLY the provided taxonomy labels.\n"
        "A high confidence (>= 0.8) means the debate can stop.\n"
        "A low confidence (< 0.8) means another round may help.\n\n"
        "Respond with valid JSON only — no prose, no markdown fences.\n"
        "Schema: {\"symptom\": str, \"root_cause\": str, "
        "\"reasoning\": str, \"confidence\": float}"
    )

    def __init__(self, config: Stage3Config, model):
        self.config = config
        self.agent = ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="Resolver",
                content=self.SYSTEM_PROMPT,
            ),
            model=model,
            token_limit=config.model.token_limit,
        )

    def resolve(self, issue: dict, debate_transcript: list) -> dict:
        # Format the full debate for the resolver
        transcript_text = self._format_transcript(debate_transcript)
        prompt = (
            f"Issue: {issue['repo']}#{issue['issue_number']}\n"
            f"Issue reasoning: {issue.get('reasoning', '')}\n\n"
            f"{SYMPTOM_TAXONOMY}\n"
            f"{ROOT_CAUSE_TAXONOMY}\n"
            f"Debate transcript:\n{transcript_text}\n\n"
            "Based on the full debate above, make the final classification. "
            "Respond with JSON only."
        )
        response = self.agent.step(
            BaseMessage.make_user_message(
                role_name="Researcher", content=prompt)
        )
        self.agent.reset()
        return parse_json(response.msg.content,
                          default={"symptom": "Unknown",
                                   "root_cause": "5.1 Unknown",
                                   "reasoning": "parse failed",
                                   "confidence": 0.0})

    @staticmethod
    def _format_transcript(transcript: list) -> str:
        lines = []
        for turn in transcript:
            r = turn["round"]
            lines += [
                f"--- Round {r} ---",
                f"Classifier: symptom={turn['symptom']}, "
                f"root_cause={turn['root_cause']}",
                f"  Reasoning: {turn['classifier_reasoning']}",
                f"Critic: {turn['challenge']}",
                f"Classifier rebuttal: {turn['rebuttal']}",
                "",
            ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage III pipeline
# ---------------------------------------------------------------------------

class Stage3Pipeline:
    """
    Orchestrates the debate loop per issue:

      ClassifierAgent (initial label)
            ↓
      CriticAgent (challenge)
            ↓
      ClassifierAgent (rebuttal)
            ↓
      ResolverAgent (interim decision + confidence)
            ↓
      if confidence < threshold AND rounds < max_rounds → repeat
      else → final decision
    """

    def __init__(self, config: Optional[Stage3Config] = None):
        self.config = config or Stage3Config()
        model = build_model(self.config.model)
        self.classifier = ClassifierAgent(self.config, model)
        self.critic = CriticAgent(self.config, model)
        self.resolver = ResolverAgent(self.config, model)

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

            print(f"           Symptom   : {decision.symptom}")
            print(f"           Root cause: {decision.root_cause}")
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
        transcript = []
        current_label = self.classifier.classify(issue)
        final_decision = current_label
        rounds_completed = 0

        for round_num in range(1, self.config.max_rounds + 1):
            print(f"           Round {round_num}: classifying...", end=" ")

            # Critic challenges
            challenge = self.critic.challenge(issue, current_label)

            # Classifier rebuts
            rebuttal = self.classifier.rebut(issue, current_label, challenge)

            # Log this round
            transcript.append({
                "round": round_num,
                "symptom": current_label.get("symptom", ""),
                "root_cause": current_label.get("root_cause", ""),
                "classifier_reasoning": current_label.get("reasoning", ""),
                "challenge": challenge,
                "rebuttal": rebuttal.get("reasoning", ""),
            })

            # Update current label to rebuttal
            current_label = rebuttal
            rounds_completed = round_num

            # Resolver makes interim decision
            resolver_result = self.resolver.resolve(issue, transcript)
            confidence = float(resolver_result.get("confidence", 0.0))
            final_decision = resolver_result

            print(f"conf={confidence:.2f}")

            # Stop if resolver is confident enough
            if confidence >= self.config.confidence_threshold:
                print(f"           Confident — stopping after round {round_num}")
                break
            elif round_num < self.config.max_rounds:
                print(f"           Low confidence — continuing debate...")
            else:
                print(f"           Max rounds reached — accepting best decision")

        return TaxonomyDecision(
            issue_number=issue["issue_number"],
            repo=issue["repo"],
            symptom=final_decision.get("symptom", "Unknown"),
            root_cause=final_decision.get("root_cause", "5.1 Unknown"),
            reasoning=final_decision.get("reasoning", ""),
            confidence=float(final_decision.get("confidence", 0.0)),
            debate_rounds=rounds_completed,
            debate_transcript=transcript,
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
                    "symptom": d.symptom,
                    "root_cause": d.root_cause,
                    "reasoning": d.reasoning,
                    "confidence": d.confidence,
                    "debate_rounds": d.debate_rounds,
                    "debate_transcript": d.debate_transcript,
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
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Load fault_related issues from Stage II output
    stage2_path = "outputs/stage2_output.json"

    if os.path.exists(stage2_path):
        with open(stage2_path) as f:
            stage2 = json.load(f)
        issues = stage2.get("fault_related", [])
        print(f"[Stage III] Loaded {len(issues)} fault-related issues "
              f"from Stage II output.")
    else:
        # Fallback sample for testing without Stage II
        print("[Stage III] No Stage II output found — using sample issues.")
        issues = [
            {
                "repo": "tensorflow/tfjs",
                "issue_number": 1234,
                "reasoning": (
                    "Issue describes a crash when calling tf.matMul with "
                    "mismatched tensor shapes on the WebGL backend."
                ),
            }
        ]

    config = Stage3Config(
        max_issues=None,             # set to an int to cap for testing
        max_rounds=3,
        confidence_threshold=0.80,
        model=ModelConfig(
            model_type="llama3",     # change to your pulled model
        ),
    )

    pipeline = Stage3Pipeline(config)
    pipeline.run(issues)
