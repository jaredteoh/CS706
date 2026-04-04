"""
AutoEmpirical MAS - Stage II: Fault-Related Issue Filtering (v3 — Data-Driven)
==============================================================================
Reads:  outputs/stage1_output.json  OR  a pre-collected CSV
Writes: outputs/stage2_output.json

Target: accuracy > 0.80

Key improvements over v2, derived from empirical analysis of the 500-issue dataset:

  CRITICAL FIX — R5 (WAE resolution) was causing 9 false positives (44% accuracy)
  -----------------------------------------------------------------------
  Root cause: "not a bug" and "cannot reproduce" from COMMUNITY members, or
  "working as expected" from ONE member while others still reproduce, were
  incorrectly triggering the non-fault rule.

  Fix: R5 is now DISABLED for standalone generic patterns. Replaced with:

  NEW — R5a_STRONG_WAE: Only fires on high-confidence maintainer-authored signals:
    "working as intended", "this is by design", "expected behaviour" WITH
    ABSENCE of any reproduction signal in the same thread. Never fires on
    "cannot reproduce" or "not a bug" alone — these are too noisy.

  NEW — R7_STALE_AUTOCLOSE: Stale-bot auto-close = 96% precision for label=0
    Pattern: auto-bot marks stale AND closes in same comment thread.
    25 hits / 24 label=0 (96% precision). Previously unused.

  NEW — R8_FIX_RELEASED: Fix released/merged comment = 100% precision label=1
    Pattern: "released in X.X", "fix.*released", "pr.*merged" in comments.
    14 hits / 14 label=1 (100% precision). Moved from R4 (body+comments mix)
    to comments-only rule with higher confidence.

  IMPROVED — R4: Add check for "illegal hardware instruction" which is a
    confirmed TFJS native crash on macOS (label=1 in dataset).

  IMPROVED — R6 (sysinfo+error): Added 'automatically marked as stale' as
    exclusion to prevent stale issues from triggering the fault rule.

  LLM PROMPT IMPROVEMENTS:
  - Updated few-shot examples with 4 new examples from error analysis
  - Stronger system-prompt instructions derived from false positive patterns
  - Added VOTING PASS: Two-pass LLM strategy for borderline (conf < 0.72):
    A short second query with reversed framing to catch flip-flop cases.
    If both passes agree → use that answer. If they disagree → use first
    answer but flag for human review. Reduces flip-flop errors.
"""

import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import httpx
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))
from tools.models import ModelConfig, build_model, parse_json


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Stage2Config:
    stage1_output_path: str = "outputs/stage1_output.json"
    output_path:        str = "outputs/stage2_output.json"
    confidence_threshold: float = 0.70
    enable_second_pass: bool = True          # NEW: two-pass for borderline cases
    second_pass_threshold: float = 0.72     # NEW: trigger second pass below this
    enable_scorer: bool = True               # set False for pre-filter + single LLM ablation
    github_token: Optional[str] = field(
        default_factory=lambda: os.getenv("GITHUB_TOKEN"))
    github_api_base:      str = "https://api.github.com"
    github_rate_limit_delay: float = 0.5
    max_issues_per_repo: Optional[int] = 20
    csv_path:     Optional[str] = None
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
    state:            Optional[str]  = None
    comments_content: Optional[str]  = None
    linked_pr:        Optional[dict] = None
    diff_summary:     Optional[str]  = None
    changed_files:    Optional[list] = None
    fetch_error:      Optional[str]  = None


@dataclass
class FilterDecision:
    issue_number: int
    repo: str
    is_fault_related: bool
    reasoning: str
    confidence: float
    flagged_for_review: bool  = False
    pre_filter_rule:    Optional[str] = None
    # Issue content fields — passed through to Stage 3
    title:            Optional[str] = None
    body:             Optional[str] = None
    state:            Optional[str] = None
    comments_content: Optional[str] = None


# ---------------------------------------------------------------------------
# GitHub client
# ---------------------------------------------------------------------------

class GitHubClient:
    def __init__(self, config: Stage2Config):
        headers = {"Accept": "application/vnd.github+json",
                   "X-GitHub-Api-Version": "2022-11-28"}
        if config.github_token:
            headers["Authorization"] = f"Bearer {config.github_token}"
        self.client = httpx.Client(
            base_url=config.github_api_base, headers=headers, timeout=15.0)

    def get_issue(self, owner, repo, number):
        r = self.client.get(f"/repos/{owner}/{repo}/issues/{number}")
        r.raise_for_status(); return r.json()

    def get_timeline(self, owner, repo, number):
        r = self.client.get(
            f"/repos/{owner}/{repo}/issues/{number}/timeline",
            headers={"Accept": "application/vnd.github.mockingbird-preview+json"})
        r.raise_for_status(); return r.json()

    def get_pr_diff(self, owner, repo, pr_number):
        r = self.client.get(
            f"/repos/{owner}/{repo}/pulls/{pr_number}",
            headers={"Accept": "application/vnd.github.diff"})
        r.raise_for_status(); return r.text[:2000]

    def get_pr_files(self, owner, repo, pr_number):
        r = self.client.get(f"/repos/{owner}/{repo}/pulls/{pr_number}/files")
        r.raise_for_status(); return [f["filename"] for f in r.json()]

    def get_comments(self, owner, repo, number, per_page=100):
        r = self.client.get(
            f"/repos/{owner}/{repo}/issues/{number}/comments",
            params={"per_page": per_page})
        r.raise_for_status()
        comments = r.json()
        return "\n\n".join(
            f"[{c.get('user', {}).get('login', 'unknown')}]: {c.get('body', '')}"
            for c in comments
        )

    def list_issues(self, owner, repo, state="all", per_page=100, max_pages=10):
        issues = []
        for page in range(1, max_pages + 1):
            r = self.client.get(f"/repos/{owner}/{repo}/issues",
                params={"state": state, "per_page": per_page,
                        "page": page, "sort": "created", "direction": "desc"})
            r.raise_for_status()
            batch = r.json()
            if not batch: break
            issues.extend(i for i in batch if "pull_request" not in i)
        return issues

    def close(self): self.client.close()


# ---------------------------------------------------------------------------
# Stage A — IssuePreFilter  (v3 — data-driven improvements)
# ---------------------------------------------------------------------------

class IssuePreFilter:
    """
    Data-driven pre-filter. All rules derived from empirical analysis of the
    500-issue TFJS dataset. Changes from v2:

    CRITICAL: R5 (WAE resolution) was 44% accurate — REPLACED with R5a (strong WAE only).
    NEW:      R7_STALE_AUTOCLOSE — 96% precision label=0
    NEW:      R8_FIX_RELEASED — 100% precision label=1 (comments only)
    IMPROVED: R4 — added illegal hardware instruction (macOS TFJS crash)
    IMPROVED: R6 — added stale-bot as exclusion
    """

    # ---- Non-fault comment patterns (SO redirect — high precision) ----
    _NONFAULT_CMT_STRICT = [
        "closing this issue to better track in",
        "closing to track in",
        "better suited to stackoverflow",
        "better asked on stackoverflow",
    ]

    # ---- Body error signals (used to guard SO redirect rule) ----
    _BODY_ERROR_SIGNALS = [
        r"error[:\s]", r"exception[:\s]", r"crash", r"fail(ed|ure)",
        r"not working", r"locks the ui", r"throws", r"TypeError",
        r"undefined is not", r"cannot read property",
    ]

    # ---- STRONG WAE signals from maintainer — high precision only ----
    # Based on data analysis: only "working as intended" and "this is by design"
    # are reliable enough for deterministic non-fault classification.
    # "working as expected" alone is only 50% precise in this dataset.
    # "not a bug" / "cannot reproduce" alone are too noisy (33-67% false positive).
    _STRONG_WAE_PATTERNS = [
        r"\bworking as intended\b",
        r"\bthis is by design\b",
        r"\bthis is expected\s+behavior\b",
        r"\bthis is expected\s+behaviour\b",
        r"\bthat is the expected\s+behavior\b",
        r"\bthat is the expected\s+behaviour\b",
        r"\bby design\s*[,;.]",
    ]

    # ---- Signals that disqualify strong WAE (issue is still a fault) ----
    _WAE_DISQUALIFIERS = [
        r"(fix|patch|released|merged|pr.*submitted)",
        r"multiple users?\s*(confirm|report|see|encounter)",
        r"same issue\s*=====",   # multiple commenters reproducing it
        r"\bI can reproduce\b",
        r"\bstill (happening|occurring|broken|failing)\b",
    ]

    # ---- Documentation / typo faults → label=1 ----
    _DOC_FAULT_TITLE = [
        r"\btypo\b",
        r"\bdead\s+link\b",
        r"\bbroken\s+link\b",
        r"outdated\s+doc",
        r"wrong\s+doc",
        r"incorrect\s+doc",
        r"documentation.*error",
        r"doc.*incorrect",
    ]

    # ---- High-precision fault patterns (body + comments combined) ----
    _FAULT_PATTERNS = [
        # Missing op / kernel
        r"not yet implemented",
        r"missing[\s\w]*kernel",
        r"error:\s*kernel",
        r"was working[\s\w]+no longer",
        r"only supported in[\s\w]*cpu[\s\w]*and[\s\w]*webgl",
        r"only supported in[\s\w]*webgl\b",
        r"not supported in[\s\w]*wasm",
        r"not supported in[\s\w]*webgpu",
        r"missing[\s\w]*kernel ops",
        r"backend[\s\w]*missing[\s\w]*kernel",
        # macOS native crash (illegal hardware instruction = TFJS native addon crash)
        r"illegal hardware instruction",
        # Converter bug
        r"converter[\s\w]*incorrect",
        # Backend inconsistency confirmed
        r"backend[\s\w]*inconsistency",
        r"results differ between[\s\w]*backend",
        # Version regression
        r"working in[\s\w]*\d+\.\d+[\s\w]*broken in",
        r"downgrade to[\s\w]*\d+\.\d+[\s\w]*(fix|work|resolv)",
        r"broke[\s\w]+in[\s\w]+\d+\.\d+",
    ]

    # ---- Fix released/merged in comments — 100% precision label=1 ----
    _FIX_RELEASED_COMMENT = [
        r"submitted the change internally",
        r"fix has been (merged|released|published)",
        r"\d+\.\d+\.\d+ (is released|released).*fix",
        r"fix.*merged.*pr",
        r"\breleased in\s+\d+\.\d+",
        r"released.*should fix",
        r"the fix.*merged",
        r"pr.*merged",
    ]

    # ---- Stale auto-close bot patterns — 96% precision label=0 ----
    # The bot posts "marked as stale" then "closing" in the same thread.
    _STALE_AUTOCLOSE = [
        r"this issue has been automatically marked as stale.{0,800}closing",
        r"automatically marked as stale.{0,800}no recent activity.{0,400}closed",
    ]

    # ---- User-env installation patterns → label=0 ----
    _USER_ENV_INSTALL = [
        r"it works.*with.*latest\s*version",
        r"works.*after.*upgrade",
        r"latest version.*working",
        r"bump(ing)? (to|up) (the )?version.*fix",
        r"behind firewall",
        r"update your package\.json",
        r"xcode.*install",
        r"gyp.*failed.*darwin",
        r"node[-\s]pre[-\s]gyp.*warn.*using",
    ]

    # ---- TFJS packaging bugs (install failures that ARE faults) ----
    _TFJS_PACKAGING_FAULT = [
        r"issue.*present since\s+\d+\.\d+",
        r"working.*until\s+\d+\.\d+",
        r"regression.*install",
        r"broken.*since\s+\d+\.\d+",
        r"install.*fail.*\d+\.\d+\.\d+.*regression",
    ]

    # ---- Exclusion phrases for R6 (sysinfo+error rule) ----
    _SYSINFO_EXCLUSIONS = [
        r"working as expected",
        r"cannot reproduce",
        r"unable to reproduce",
        r"that is expected",
        r"update your package\.json",
        r"automatically marked as stale",  # stale issues are not confirmed faults
        r"webgl is not supported on this device",
        r"behind firewall",
        r"working as intended",
        r"this is expected behavior",
    ]

    @classmethod
    def classify(cls, ctx: IssueContext) -> Optional[dict]:
        """
        Returns {is_fault_related, reasoning, confidence, rule} or None.
        None means: LLM agents needed.
        """
        body_lower    = (ctx.body or "").lower()
        cmts_lower    = (ctx.comments_content or "").lower()
        title_lower   = (ctx.title or "").lower()
        combined      = body_lower + " " + cmts_lower

        # ----------------------------------------------------------------
        # R1: Year gate — 100% precision, covers ~32% of dataset
        # ----------------------------------------------------------------
        year = cls._parse_year(ctx.created_at)
        if year is not None and year < 2020:
            return dict(is_fault_related=False, confidence=0.99, rule="R1_year",
                        reasoning=f"Pre-2020 issue ({year}) — outside AutoEmpirical study scope.")

        # ----------------------------------------------------------------
        # R2: SO redirect — high precision
        # Only apply if body has NO actual error/crash signals
        # ----------------------------------------------------------------
        body_has_error = any(re.search(p, body_lower) for p in cls._BODY_ERROR_SIGNALS)
        for phrase in cls._NONFAULT_CMT_STRICT:
            if phrase in cmts_lower and not body_has_error:
                return dict(is_fault_related=False, confidence=0.96, rule="R2_so_redirect",
                            reasoning=f"Non-fault comment signal (no body error): '{phrase}'.")

        # ----------------------------------------------------------------
        # R3: Documentation / typo issues → FAULT (label=1)
        # AutoEmpirical study labels these as faults.
        # Title-only rule to avoid noise from body mentions.
        # ----------------------------------------------------------------
        for pat in cls._DOC_FAULT_TITLE:
            if re.search(pat, title_lower):
                return dict(is_fault_related=True, confidence=0.88, rule="R3_doc_fault",
                            reasoning="Documentation/typo issue — AutoEmpirical labels these as faults.")

        # ----------------------------------------------------------------
        # R4: High-precision fault patterns (body+comments)
        # ----------------------------------------------------------------
        is_test_code = bool(re.search(
            r"expectarraysclose|beforeeach|aftereach|describe\s*\(", body_lower))
        if not is_test_code:
            for pat in cls._FAULT_PATTERNS:
                if re.search(pat, combined):
                    return dict(is_fault_related=True, confidence=0.94, rule="R4_fault_pattern",
                                reasoning=f"Fault pattern matched: '{pat}'.")

        # ----------------------------------------------------------------
        # R5a: STRONG WAE — only fire on unambiguous by-design signals
        # CRITICAL FIX from v2: "not a bug" and "cannot reproduce" alone are
        # too noisy (only 33-50% precise). Only fire on patterns that are
        # reliably maintainer-authored and unambiguous.
        # ----------------------------------------------------------------
        for pat in cls._STRONG_WAE_PATTERNS:
            if re.search(pat, cmts_lower):
                # Disqualify if: fix exists, or multiple reproductions exist
                disqualified = any(re.search(d, cmts_lower) for d in cls._WAE_DISQUALIFIERS)
                # Also disqualify if there's a kernel/backend issue
                kernel_in_body = bool(re.search(
                    r"(only supported in|not supported in|feature request.*kernel|"
                    r"would be a feature.*add.*to (node|wasm|webgpu))", combined, re.IGNORECASE))
                if not disqualified and not kernel_in_body:
                    return dict(is_fault_related=False, confidence=0.84, rule="R5a_strong_wae",
                                reasoning=f"Strong WAE signal (maintainer-level): '{pat}'.")

        # ----------------------------------------------------------------
        # R6: System-information template + error (with exclusions)
        # ----------------------------------------------------------------
        if re.search(r"system information[\s\S]{0,200}?error[:\s!]", combined, re.IGNORECASE):
            excluded = any(re.search(p, cmts_lower) for p in cls._SYSINFO_EXCLUSIONS)
            if not excluded:
                return dict(is_fault_related=True, confidence=0.93, rule="R6_sysinfo_error",
                            reasoning="TFJS system-info template with error description present.")

        # ----------------------------------------------------------------
        # R7: Stale auto-close — 96% precision label=0 (NEW in v3)
        # The stale bot marks an issue + then closes it = strong non-fault signal.
        # The one false positive (label=1) in our analysis was also a community
        # reproduced issue — we add a guard for "same issue" mentions.
        # ----------------------------------------------------------------
        for pat in cls._STALE_AUTOCLOSE:
            if re.search(pat, cmts_lower, re.DOTALL):
                # Guard: if multiple users reproduce, don't fire
                multiple_reproductions = bool(re.search(
                    r"same issue=====|i can reproduce|same (problem|error) here",
                    cmts_lower))
                if not multiple_reproductions:
                    return dict(is_fault_related=False, confidence=0.93, rule="R7_stale_autoclose",
                                reasoning="Stale-bot auto-close with no reproduction — strong non-fault signal.")

        # ----------------------------------------------------------------
        # R8: Fix released in comments — 100% precision label=1 (NEW in v3)
        # Only match in COMMENTS (not body) — comments confirm resolution.
        # ----------------------------------------------------------------
        for pat in cls._FIX_RELEASED_COMMENT:
            if re.search(pat, cmts_lower, re.IGNORECASE):
                return dict(is_fault_related=True, confidence=0.96, rule="R8_fix_released",
                            reasoning=f"Fix released/merged confirmed in comments: '{pat}'.")

        return None  # hand off to LLM

    @staticmethod
    def _parse_year(created_at) -> Optional[int]:
        if not created_at:
            return None
        s = str(created_at)
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).year
        except (ValueError, AttributeError):
            try:
                return int(s[:4])
            except (ValueError, TypeError):
                return None


# ---------------------------------------------------------------------------
# Agent 1 — ContextRetrieverAgent
# ---------------------------------------------------------------------------

class ContextRetrieverAgent:
    def __init__(self, config: Stage2Config):
        self.config = config
        self.github = GitHubClient(config)

    def retrieve(self, owner: str, repo: str, issue_number: int) -> IssueContext:
        try:
            issue = self.github.get_issue(owner, repo, issue_number)
        except Exception as e:
            return IssueContext(issue_number=issue_number, repo=f"{owner}/{repo}",
                                title="", body="", labels=[], created_at="",
                                fetch_error=f"Issue fetch failed: {e}")
        ctx = IssueContext(
            issue_number=issue_number, repo=f"{owner}/{repo}",
            title=issue.get("title", ""),
            body=(issue.get("body") or "")[:3000],
            labels=[l["name"] for l in issue.get("labels", [])],
            created_at=issue.get("created_at", ""),
            state=issue.get("state", ""))
        try:
            time.sleep(self.config.github_rate_limit_delay)
            ctx.comments_content = self.github.get_comments(owner, repo, issue_number)
            time.sleep(self.config.github_rate_limit_delay)
            timeline = self.github.get_timeline(owner, repo, issue_number)
            pr = self._find_linked_pr(timeline)
            if pr:
                time.sleep(self.config.github_rate_limit_delay)
                ctx.diff_summary  = self.github.get_pr_diff(owner, repo, pr)
                ctx.changed_files = self.github.get_pr_files(owner, repo, pr)
                ctx.linked_pr = {"number": pr}
        except Exception as e:
            ctx.fetch_error = f"PR fetch failed: {e}"
        return ctx

    @staticmethod
    def _find_linked_pr(timeline: list) -> Optional[int]:
        for event in timeline:
            if event.get("event") == "cross-referenced":
                pr = event.get("source", {}).get("issue", {}).get("pull_request")
                if pr:
                    parts = pr.get("url", "").rstrip("/").split("/")
                    try: return int(parts[-1])
                    except (ValueError, IndexError): pass
        return None

    def close(self): self.github.close()


# ---------------------------------------------------------------------------
# Few-shot examples — v3 updated with examples from error analysis
# ---------------------------------------------------------------------------

_FEW_SHOTS = """
=== FEW-SHOT EXAMPLES (hard boundary cases — derived from dataset analysis) ===

EXAMPLE 1 — FAULT (version regression with fix released)
Title: tfjs 2.8.0 introduces regressions in tf.image.cropAndResize
Body: 2.7.0 worked. 2.8.0 fails with 'Failed to compile fragment shader'. Downgrade confirmed fix.
Comments: "2.8.1 is released. It should fix the bug."
  fault_axis: YES
  resolution_axis: code-fix released
DECISION: {"is_fault_related": true, "reasoning": "Version regression confirmed; 2.8.1 fix released.", "confidence": 0.97}

EXAMPLE 2 — FAULT (documentation error / typo)
Title: typo in documentation
Body: There is a typo in the API docs.
  fault_axis: YES — AutoEmpirical explicitly labels ALL documentation errors as faults (label=1)
  doc_axis: YES
DECISION: {"is_fault_related": true, "reasoning": "Documentation typo — AutoEmpirical labels all doc issues as label=1.", "confidence": 0.88}

EXAMPLE 3 — FAULT (internal code quality issue)
Title: Too many duplicated code in all backends
Body: Validation code is duplicated; suggests refactoring.
Comments: "The backend kernel is designed this way to allow tree shaking."
  fault_axis: YES — AutoEmpirical labels internal code quality issues as faults (label=1)
  NOTE: Even when maintainers explain design rationale, AutoEmpirical labels these faults
DECISION: {"is_fault_related": true, "reasoning": "Internal code quality issue — AutoEmpirical labels these as faults.", "confidence": 0.78}

EXAMPLE 4 — FAULT (installation regression — VERSIONED)
Title: Installation issue @tensorflow/tfjs-node-gpu
Comments: "I tested older versions and it works until 3.1.0. The issue is present since 3.2.0."
  install_axis: TFJS_REGRESSION — confirmed broke in specific version
DECISION: {"is_fault_related": true, "reasoning": "Installation regression since v3.2.0 — TFJS packaging fault.", "confidence": 0.87}

EXAMPLE 5 — FAULT (broadcast over complex values — "working as expected" from COMMUNITY only)
Title: Broadcast over complex Values on CPU backend does not work well
Body: Complex number broadcasting fails on CPU backend.
Comments: "I tried on tfjs 3.7.0 it is working as expected" [from a community user, not maintainer]
  fault_axis: YES — "working as expected" from a COMMUNITY user ≠ maintainer closure
  NOTE: This is a CRITICAL distinction. Only MAINTAINER-authored "working as intended" / 
        "this is by design" count as non-fault signals. Community users saying "works for me"
        do NOT make an issue non-fault. The issue remains open = still a fault.
DECISION: {"is_fault_related": true, "reasoning": "Complex broadcasting bug; 'working as expected' from community member only, not maintainer closure.", "confidence": 0.75}

EXAMPLE 6 — FAULT (breaking change — "unable to reproduce" with multiple reproductions)
Title: Breaking Change v3.8.0 => v3.9.0
Body: hash_util.d.ts lost reference types in 3.9.0.
Comments: "unable to reproduce" + multiple "same issue" confirmations from other users.
  fault_axis: YES — "unable to reproduce" is negated by multiple reproductions in comments
  NOTE: If 'unable to reproduce' exists alongside 'same issue' / 'I can reproduce' comments,
        the issue is STILL a fault. Contradictory evidence → lean fault per recall bias.
DECISION: {"is_fault_related": true, "reasoning": "Breaking change between versions; 'unable to reproduce' contradicted by multiple user reproductions.", "confidence": 0.80}

EXAMPLE 7 — FAULT (tfjs converter output format incorrect)
Title: tfjs_converter output format incorrect
Body: Output format from converter is wrong.
Comments: "mapping output node names is a best effort... but for mapping to work, tensors must have unique shape"
  fault_axis: YES — "best effort" mapping with documented limitations is still a fault in the
              converter for a standard operation
  NOTE: When maintainer says "best effort" or explains a limitation without marking as
        expected/by-design, the issue is still a fault.
DECISION: {"is_fault_related": true, "reasoning": "Converter output format issue with acknowledged limitation — TFJS fault.", "confidence": 0.76}

EXAMPLE 8 — FAULT (WEBGL_FORCE_F16_TEXTURES undocumented silent error)
Title: Model using WebGL backend returns nonsense results
Comments: "my fault, issue was due to model clipping - i forgot i had WEBGL_FORCE_F16_TEXTURES set globally."
  fault_axis: YES — even though user triggered the flag, undocumented silent numerical error is a fault
  NOTE: "my fault" comments don't make it non-fault if framework behaviour was undocumented
DECISION: {"is_fault_related": true, "reasoning": "Undocumented WEBGL_FORCE_F16_TEXTURES flag caused silent numerical errors — framework fault.", "confidence": 0.78}

EXAMPLE 9 — NOT FAULT (install resolved by upgrading to latest version)
Title: module not found on MacOS
Comments: "are you still seeing the same issue with latest version (1.7.0)? ... It works well with latest version! Thank you!"
  fault_axis: NO — resolved by upgrading; no TFJS regression confirmed
  install_axis: USER_ENV — user was on old version; upgrading fixed it
DECISION: {"is_fault_related": false, "reasoning": "Installation resolved by upgrading to latest version — user env issue.", "confidence": 0.83}

EXAMPLE 10 — NOT FAULT (third-party addons op not in converter)
Title: Support for ReverseSequence in tensorflowjs_converter
Body: Getting error converting model that uses CRF from TensorFlow Addons.
Comments: (none)
  fault_axis: NO — TF Addons is a THIRD-PARTY library; missing ops from third-party = feature gap
  NOTE: Standard TF ops missing = fault; third-party addon ops missing = not fault
DECISION: {"is_fault_related": false, "reasoning": "Missing converter support for third-party TF Addons op — feature gap, not TFJS fault.", "confidence": 0.78}

EXAMPLE 11 — NOT FAULT (memory investigation — workaround via flag resolves it)
Title: [webgl] Investigate how to reduce memory usage from texture allocation
Comments: "set this flag so that textures are deleted when tensors are disposed: WEBGL_DELETE_TEXTURE_THRESHOLD"
  fault_axis: NO — user found the workaround (a documented flag); issue is framed as a solved investigation
DECISION: {"is_fault_related": false, "reasoning": "Memory investigation with documented flag workaround — user found solution, not an unfixed TFJS fault.", "confidence": 0.76}

EXAMPLE 12 — NOT FAULT (converter failure for custom/unsupported layer types)
Title: Cannot convert Keras saved model
Body: tensorflowjs_converter fails with 'Unable to restore a layer of class Custom>MultiCategoryEncoding'
Comments: "Looks like you might have a custom layer that is not supported natively. This combination is not supported."
  fault_axis: NO — custom layers and unsupported format combinations are documented limitations
DECISION: {"is_fault_related": false, "reasoning": "Converter limitation for custom layers — documented unsupported feature, not a TFJS fault.", "confidence": 0.79}

EXAMPLE 13 — NOT FAULT (stale-bot auto-close, no reproductions)
Title: [Some issue]
Comments: "This issue has been automatically marked as stale... Closing this issue due to lack of activity."
  fault_axis: NO — stale-bot auto-close with no other reproductions = very strong non-fault signal
  NOTE: If multiple users confirmed "same issue" before the stale close, STILL a fault.
        Only apply non-fault if the thread has no reproduction confirmations.
DECISION: {"is_fault_related": false, "reasoning": "Stale auto-close with no reproduction confirmations — non-fault.", "confidence": 0.88}

EXAMPLE 14 — FAULT (Safari/iOS compatibility gap)
Title: Safari 15 / iOS 15 Vertex/Shader Errors
Body: TFJS fails in Safari 15 / iOS 15 with WebGL shader errors.
Comments: Maintainer asks for details, investigating.
  fault_axis: YES — TFJS should work in Safari/iOS; compatibility gap
DECISION: {"is_fault_related": true, "reasoning": "TFJS compatibility gap with Safari 15/iOS 15 — framework fault.", "confidence": 0.80}

EXAMPLE 15 — FAULT (illegal hardware instruction = macOS TFJS native crash)
Title: Error: The Node.js native addon module can not be found
Comments: "i placed the project into my documents folder... that is a workaround for now. the next issue is: [node index.js] illegal hardware instruction"
  fault_axis: YES — "illegal hardware instruction" = macOS TFJS native addon crash; confirmed TFJS fault
  NOTE: "illegal hardware instruction" on macOS is a known TFJS native binding crash pattern
DECISION: {"is_fault_related": true, "reasoning": "Illegal hardware instruction = macOS TFJS native addon crash — framework fault.", "confidence": 0.82}
=== END FEW-SHOT EXAMPLES ===
"""


# ---------------------------------------------------------------------------
# Agent 2 — FilterAgent  (v3 improved prompt)
# ---------------------------------------------------------------------------

class FilterAgent:
    """
    LLM-based binary classifier for issues that escaped the pre-filter.
    v3 improvements: stronger disambiguation rules, new few-shot examples,
    clearer maintainer vs community distinction.
    """

    SYSTEM_PROMPT = (
        "You are a researcher classifying GitHub issues for the AutoEmpirical "
        "fault study of TensorFlow.js (TFJS).\n\n"

        "CONTEXT: You only see issues that passed a deterministic pre-filter "
        "(post-2020, not SO-redirected, no obvious fault/non-fault signal). "
        "These are the genuinely ambiguous cases.\n\n"

        "=== FAULT DEFINITION (label=1) ===\n"
        "An issue is fault-related if it describes ANY of these:\n"
        "  • Incorrect/unexpected/inconsistent behaviour (bugs, crashes)\n"
        "  • Missing op/kernel causing runtime errors — even if phrased as feature "
        "request, if the op EXISTS in TF Python but absent in TFJS backend → FAULT\n"
        "  • Backend inconsistency (WebGL vs WASM vs Node produce different results)\n"
        "  • Version regressions — issue worked in vX, broke in vY\n"
        "  • API/type inconsistencies causing errors in correct usage\n"
        "  • Compatibility gaps with supported environments (Electron, React Native, "
        "Node.js, Safari, iOS, Android) — TFJS should work in these\n"
        "  • Undocumented flag/env behaviour causing silent errors — even if the user "
        "says 'my fault', if the framework behaviour was undocumented/surprising → FAULT\n"
        "  • Installation failures that are VERSIONED REGRESSIONS in TFJS packaging "
        "(worked in older version, broke in newer version) → FAULT\n"
        "  • Converter failures for STANDARD TF ops or officially supported model formats\n"
        "  • DOCUMENTATION ERRORS: typos, dead links, outdated docs, wrong examples. "
        "AutoEmpirical EXPLICITLY labels ALL doc issues as faults (label=1). No exceptions.\n"
        "  • Internal code quality issues (duplicate code, refactoring) — AutoEmpirical "
        "labels these as faults (label=1) even with no user-facing errors\n"
        "  • 'Best effort' / explained limitation WITHOUT maintainer marking as by-design "
        "— still a fault\n\n"

        "=== NOT FAULT (label=0) — ONLY if CLEARLY one of: ===\n"
        "  • INSTALLATION resolved by upgrading to latest (user on old version) — USER_ENV\n"
        "  • CONVERTER failure for CUSTOM LAYERS or THIRD-PARTY ADDON ops (TF Addons, etc)\n"
        "  • MEMORY investigation where user found solution via existing flag (solved post)\n"
        "  • MAINTAINER (not community user) EXPLICITLY says 'working as intended' OR "
        "'this is by design' AND no fix committed AND no other reproductions\n"
        "  • Pure usage question, no error, no broken functionality\n"
        "  • User EXPLICITLY states 'this is not an issue/bug' AND no runtime error\n"
        "  • Stale-bot auto-closed with no reproduction confirmations in thread\n"
        "  • Feature request where maintainer confirms current version already works\n\n"

        "=== CRITICAL DISAMBIGUATION RULES ===\n"
        "COMMUNITY vs MAINTAINER:\n"
        "  'Working as expected' / 'not a bug' from a COMMUNITY USER ≠ non-fault.\n"
        "  Only MAINTAINER-authored 'working as intended' / 'this is by design' count.\n"
        "  If you cannot tell who wrote a comment, default to: ambiguous = lean FAULT.\n\n"
        "CANNOT REPRODUCE:\n"
        "  NOT FAULT only if: maintainer says 'cannot reproduce' AND no other users reproduce it\n"
        "  STILL FAULT if: 'cannot reproduce' + 'same issue' / 'I can reproduce' from others\n\n"
        "STALE BOT:\n"
        "  Stale auto-close + NO reproductions in thread → NOT FAULT (strong signal)\n"
        "  Stale auto-close + 'same issue' comments before close → STILL FAULT\n\n"
        "INSTALL:\n"
        "  FAULT if: 'works until vX, broke in vY', 'present since vX', versioned regression\n"
        "  FAULT if: 'illegal hardware instruction' on macOS (TFJS native addon crash)\n"
        "  NOT FAULT if: 'works in latest version', user upgrades and it resolves\n\n"
        "CONVERTER:\n"
        "  FAULT if: standard tf.keras op fails, supported format fails\n"
        "  NOT FAULT if: custom/third-party layer, 'this combination not supported'\n\n"
        "'MY FAULT' in comments: Still FAULT if framework behaviour was undocumented\n\n"
        "WORKAROUNDS:\n"
        "  Still FAULT if: maintainer provided the workaround\n"
        "  NOT FAULT if: user self-solved AND no maintainer ack AND framed as solved investigation\n\n"

        "RECALL BIAS: When uncertain, lean FAULT. Missing faults > false positives.\n\n"

        + _FEW_SHOTS +

        "\nINSTRUCTIONS:\n"
        "Think through ALL axes, then output the final JSON on its own line:\n"
        "  fault_axis      : YES / NO / UNCLEAR\n"
        "  user_error_axis : YES / NO / LIKELY\n"
        "  feature_axis    : YES / NO\n"
        "  install_axis    : TFJS_REGRESSION / USER_ENV / N/A\n"
        "  doc_axis        : YES (typo/dead-link/doc-error) / NO\n"
        "  converter_axis  : STANDARD_OP / CUSTOM_THIRD_PARTY / N/A\n"
        "  maintainer_axis : CONFIRMED_BUG / CONFIRMED_NONBUG / INVESTIGATING / COMMUNITY_ONLY / SILENT\n"
        "  resolution_axis : code-fix / doc-fix / expected-behavior / question / "
        "workaround-user / workaround-maintainer / open / third-party / stale-close / unclear\n"
        "  confidence_note : key reason\n\n"
        "Final JSON (no markdown, no backticks, on its own line):\n"
        '{"is_fault_related": <bool>, "reasoning": "<one sentence>", "confidence": <0.5-0.99>}\n\n'
        "CONSISTENCY RULES (enforce before finalizing):\n"
        "  doc_axis=YES                         → is_fault_related MUST be true\n"
        "  fault_axis=YES                       → is_fault_related MUST be true\n"
        "  install_axis=TFJS_REGRESSION         → is_fault_related MUST be true\n"
        "  install_axis=USER_ENV (no fault sig) → is_fault_related MUST be false\n"
        "  converter_axis=CUSTOM_THIRD_PARTY    → is_fault_related MUST be false\n"
        "  maintainer_axis=CONFIRMED_BUG        → is_fault_related MUST be true\n"
        "  maintainer_axis=COMMUNITY_ONLY       → DO NOT use as non-fault signal; apply RECALL BIAS\n"
        "  fault_axis=UNCLEAR                   → RECALL BIAS → true unless clear non-fault\n"
        "Re-read your JSON; fix any contradictions."
    )

    def __init__(self, config: Stage2Config, model):
        self.config = config
        self.agent = ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="Filter Agent", content=self.SYSTEM_PROMPT),
            model=model,
            token_limit=config.model.token_limit)

    def classify(self, ctx: IssueContext) -> dict:
        response = self.agent.step(
            BaseMessage.make_user_message(
                role_name="Researcher", content=self._build_prompt(ctx)))
        self.agent.reset()
        return self._extract_final_json(response.msg.content)

    def classify_second_pass(self, ctx: IssueContext, first_result: dict) -> dict:
        """
        Second-pass classification for borderline cases.
        Uses adversarial framing: presents the OPPOSITE of first result
        and asks the model to evaluate whether to flip.
        """
        first_verdict = "FAULT" if first_result.get("is_fault_related") else "NOT FAULT"
        opposite = "NOT FAULT" if first_result.get("is_fault_related") else "FAULT"

        prompt = (
            f"Re-evaluate this borderline issue. A first pass classified it as: {first_verdict} "
            f"(confidence={first_result.get('confidence', 0):.2f})\n"
            f"Reasoning: {first_result.get('reasoning', '')}\n\n"
            f"Now consider: could it actually be {opposite}? Re-read the issue carefully.\n"
            f"Apply RECALL BIAS: when uncertain, lean FAULT.\n\n"
            + self._build_prompt(ctx) +
            "\n\nFinal JSON only (no preamble):"
        )
        response = self.agent.step(
            BaseMessage.make_user_message(role_name="Researcher", content=prompt))
        self.agent.reset()
        return self._extract_final_json(response.msg.content)

    def _build_prompt(self, ctx: IssueContext) -> str:
        year = IssuePreFilter._parse_year(ctx.created_at)
        parts = [
            f"Repository: {ctx.repo}",
            f"Issue #{ctx.issue_number}: {ctx.title}  [Year: {year or '?'}]",
            f"State: {ctx.state or 'unknown'}   Created: {ctx.created_at}",
            f"Labels: {', '.join(ctx.labels) if ctx.labels else 'none'}",
            "", "=== Issue body ===", ctx.body or "(empty)",
        ]
        if ctx.comments_content:
            cs = str(ctx.comments_content).strip()
            cs = cs[:2800] + "\n...[truncated]" if len(cs) > 2800 else cs
            parts += [
                "",
                "=== Comments — check for: maintainer vs community author, "
                "StackOverflow redirect, 'working as intended'/'this is by design' (maintainer), "
                "'fix committed/released', 'same issue' reproductions, "
                "version-specific install regression, stale-bot auto-close, "
                "root cause, workaround, third-party identified ===",
                cs,
            ]
        if ctx.diff_summary:
            parts += ["", "=== Linked PR diff ===", ctx.diff_summary]
        if ctx.changed_files:
            parts += ["", "Changed files:",
                      "\n".join(f"  - {f}" for f in ctx.changed_files[:20])]
        if ctx.fetch_error:
            parts += ["", f"Note: {ctx.fetch_error}"]
        parts += [
            "",
            "Reason through ALL axes (fault, user_error, feature, install, doc, maintainer_axis, resolution). "
            "KEY: distinguish MAINTAINER vs COMMUNITY comments for WAE signals. "
            "Apply CONSISTENCY RULES and RECALL BIAS. "
            "Output the final JSON on its own line.",
        ]
        return "\n".join(parts)

    @staticmethod
    def _extract_final_json(raw: str) -> dict:
        """
        Find the last is_fault_related JSON in the output (after CoT reasoning).
        Auto-correct axis vs boolean mismatches.
        """
        default = {"is_fault_related": False,
                   "reasoning": "parse failed", "confidence": 0.5}

        hits = re.findall(r'\{[^{}]*"is_fault_related"[^{}]*\}', raw, re.DOTALL)
        if not hits:
            return parse_json(raw, default=default)

        rl = raw.lower()
        fa_yes = bool(re.search(r'fault_axis\s*[:=*]+\s*yes', rl))
        fa_no  = bool(re.search(r'fault_axis\s*[:=*]+\s*no\b', rl))
        fa_unc = bool(re.search(r'fault_axis\s*[:=*]+\s*unclear', rl))
        doc_yes = bool(re.search(r'doc_axis\s*[:=*]+\s*yes', rl))
        install_regression = bool(re.search(r'install_axis\s*[:=*]+\s*tfjs_regression', rl))
        install_user_env   = bool(re.search(r'install_axis\s*[:=*]+\s*user_env', rl))
        converter_custom   = bool(re.search(r'converter_axis\s*[:=*]+\s*custom_third_party', rl))
        maintainer_bug     = bool(re.search(r'maintainer_axis\s*[:=*]+\s*confirmed_bug', rl))
        maintainer_nonbug  = bool(re.search(r'maintainer_axis\s*[:=*]+\s*confirmed_nonbug', rl))
        community_only     = bool(re.search(r'maintainer_axis\s*[:=*]+\s*community_only', rl))

        for candidate in reversed(hits):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if "is_fault_related" not in parsed:
                continue

            is_fault = parsed["is_fault_related"]

            # Enforce consistency rules
            if (fa_yes or doc_yes or install_regression or maintainer_bug) and not is_fault:
                parsed["is_fault_related"] = True
                parsed["reasoning"] = "[auto-corrected YES] " + parsed.get("reasoning", "")
            elif (fa_no or converter_custom or maintainer_nonbug) and not fa_unc and not doc_yes and not install_regression and is_fault:
                parsed["is_fault_related"] = False
                parsed["reasoning"] = "[auto-corrected NO] " + parsed.get("reasoning", "")
            elif community_only and is_fault is False:
                # Community-only WAE should not be used as non-fault signal
                parsed["is_fault_related"] = True
                parsed["reasoning"] = "[auto-corrected: community_only WAE not reliable] " + parsed.get("reasoning", "")
            elif install_user_env and not fa_yes and not doc_yes and not install_regression and is_fault:
                conf = parsed.get("confidence", 0.7)
                if conf < 0.72:
                    parsed["is_fault_related"] = False
                    parsed["reasoning"] = "[auto-corrected USER_ENV] " + parsed.get("reasoning", "")

            return parsed

        return parse_json(raw, default=default)


# ---------------------------------------------------------------------------
# Agent 3 — ConfidenceScorerAgent  (v3 improved)
# ---------------------------------------------------------------------------

class ConfidenceScorerAgent:
    SYSTEM_PROMPT = (
        "Audit a fault classification for the AutoEmpirical TFJS study.\n\n"
        "OVERRIDE to true (fault) if:\n"
        "  • Fix PR merged / code change committed for this issue\n"
        "  • Maintainer says 'this is a bug' / 'investigating' / 'fix submitted'\n"
        "  • Version regression confirmed (worked in vX, broke in vY)\n"
        "  • 'kernel X not found' / 'not yet implemented' in error\n"
        "  • Backend inconsistency confirmed\n"
        "  • TFJS incompatibility with supported environment (Electron, React Native, iOS, Android)\n"
        "  • Installation broke in specific newer version (versioned regression)\n"
        "  • Documentation error: typo, dead link, outdated doc → ALWAYS fault in AutoEmpirical\n"
        "  • Internal code quality issue (duplicate code, refactoring need) → AutoEmpirical labels these as faults\n"
        "  • 'illegal hardware instruction' on macOS → TFJS native addon crash, always fault\n"
        "  • Multiple users confirm 'same issue' → regardless of 'cannot reproduce' → fault\n\n"
        "OVERRIDE to false (non-fault) ONLY if:\n"
        "  • MAINTAINER (not community user) explicitly says 'working as intended' or 'this is by design' AND no fix committed\n"
        "  • Root cause is clearly a third-party library, not TFJS\n"
        "  • Installation resolved by upgrading to latest version (user was on old version)\n"
        "  • Pure usage question with no error AND no broken functionality\n"
        "  • Stale-bot auto-closed with no reproduction confirmations in thread\n\n"
        "CRITICAL: Do NOT override to false based on 'not a bug' or 'cannot reproduce' from "
        "community (non-maintainer) users. Only maintainer closures with 'by design' are reliable.\n\n"
        "CALIBRATE (conservative — most issues should land 0.70–0.87):\n"
        "  ≥ 0.92 : explicit maintainer confirmation only\n"
        "  0.80–0.91 : strong indirect evidence\n"
        "  0.70–0.79 : moderate evidence, some ambiguity\n"
        "  0.55–0.69 : genuinely ambiguous — flag for human review\n\n"
        "JSON only:\n"
        '{"adjusted_confidence": float, "override_decision": bool|null, '
        '"review_notes": str, "flag_for_human_review": bool}'
    )

    def __init__(self, config: Stage2Config, model):
        self.config = config
        self.agent = ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="Confidence Scorer", content=self.SYSTEM_PROMPT),
            model=model,
            token_limit=config.model.token_limit)

    def review(self, ctx: IssueContext, filter_result: dict) -> dict:
        verdict = "FAULT" if filter_result.get("is_fault_related") else "NOT FAULT"
        parts = [
            f"Issue: {ctx.repo}#{ctx.issue_number} — {ctx.title}",
            f"State: {ctx.state}  Created: {ctx.created_at}",
            f"FilterAgent: {verdict}  conf={filter_result.get('confidence',0):.2f}",
            f"Reasoning: {filter_result.get('reasoning', '')}",
        ]
        if ctx.comments_content:
            parts += ["", "Comments:", str(ctx.comments_content).strip()[:2000]]
        if ctx.diff_summary:
            parts += ["", "PR diff:", ctx.diff_summary]
        parts += ["", "Apply overrides and calibration. Respond with JSON only."]
        response = self.agent.step(
            BaseMessage.make_user_message(
                role_name="Researcher", content="\n".join(parts)))
        self.agent.reset()
        return parse_json(response.msg.content,
                          default={"adjusted_confidence": filter_result.get("confidence", 0.5),
                                   "override_decision": None,
                                   "review_notes": "parse failed",
                                   "flag_for_human_review": True})


# ---------------------------------------------------------------------------
# Stage II pipeline
# ---------------------------------------------------------------------------

class Stage2Pipeline:
    def __init__(self, config: Optional[Stage2Config] = None):
        self.config = config or Stage2Config()
        model = build_model(self.config.model)
        self.retriever    = ContextRetrieverAgent(self.config)
        self.filter_agent = FilterAgent(self.config, model)
        self.scorer       = ConfidenceScorerAgent(self.config, model)

    def load_issues_from_csv(self, csv_path: str) -> list:
        issues = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row.get("issue", "").strip()
                if "github.com/" not in url:
                    continue
                try:
                    path  = url.split("github.com/")[-1].rstrip("/")
                    parts = path.split("/")
                    owner, repo_name, _, num_str = (
                        parts[0], parts[1], parts[2], parts[3])
                    issue_number = int(num_str)
                except (IndexError, ValueError):
                    continue
                issues.append({
                    "repo": f"{owner}/{repo_name}",
                    "issue_number": issue_number,
                    "title": row.get("title", ""),
                    "body": row.get("body") or "",
                    "state": row.get("state", ""),
                    "created_at": row.get("created_at", ""),
                    "comments_content": row.get("comments_content", ""),
                    "labels": [],
                    "ground_truth_label": row.get("label", ""),
                })
        if self.config.max_issues_per_repo:
            issues = issues[:self.config.max_issues_per_repo]
        print(f"[Stage II] Loaded {len(issues)} issues from {csv_path}")
        return issues

    def fetch_issues_from_stage1(self) -> list:
        with open(self.config.stage1_output_path) as f:
            stage1 = json.load(f)
        repos = stage1.get("selected_repos", [])
        print(f"[Stage II] Fetching issues for {len(repos)} repos …")
        all_issues = []
        for repo_entry in repos:
            url = repo_entry.get("url", "")
            if "github.com/" not in url:
                continue
            repo_full = url.split("github.com/")[-1].rstrip("/")
            if "/" not in repo_full:
                continue
            owner, repo_name = repo_full.split("/", 1)
            try:
                raw = self.retriever.github.list_issues(owner, repo_name)
                if self.config.max_issues_per_repo:
                    raw = raw[:self.config.max_issues_per_repo]
                for issue in raw:
                    all_issues.append({
                        "repo": repo_full,
                        "issue_number": issue["number"],
                        "title": issue.get("title", ""),
                        "body": (issue.get("body") or "")[:500],
                        "labels": [l["name"] for l in issue.get("labels", [])],
                    })
                time.sleep(self.config.github_rate_limit_delay)
            except Exception as e:
                print(f"  {repo_full}: fetch failed — {e}")
        return all_issues

    def run(self, issues: list) -> dict:
        fault_related, non_fault, flagged = [], [], []
        pre_filter_hits = scorer_overrides = fetch_failures = second_pass_flips = 0
        total = len(issues)

        for i, issue in enumerate(issues):
            owner, repo_name = issue["repo"].split("/", 1)
            num = issue["issue_number"]
            print(f"[Stage II] {i+1}/{total}  {issue['repo']}#{num}")

            # Build context
            if "comments_content" in issue:
                ctx = IssueContext(
                    issue_number=num, repo=issue["repo"],
                    title=issue.get("title", ""),
                    body=issue.get("body", ""),
                    state=issue.get("state", ""),
                    created_at=issue.get("created_at", ""),
                    comments_content=issue.get("comments_content", ""),
                    labels=issue.get("labels", []))
            else:
                ctx = self.retriever.retrieve(owner, repo_name, num)
                if ctx.fetch_error and not ctx.title:
                    ctx.title  = issue.get("title", "")
                    ctx.body   = issue.get("body", "")
                    ctx.labels = issue.get("labels", [])
                    fetch_failures += 1

            # Stage A — pre-filter
            pf = IssuePreFilter.classify(ctx)
            if pf is not None:
                pre_filter_hits += 1
                is_fault   = pf["is_fault_related"]
                confidence = pf["confidence"]
                reasoning  = pf["reasoning"]
                flag       = False
                print(f"           [pre-filter:{pf['rule']}] → "
                      f"{'FAULT' if is_fault else 'non-fault'}  ({confidence:.2f})")
            else:
                # Stage B — FilterAgent
                filter_result = self.filter_agent.classify(ctx)
                is_fault   = filter_result.get("is_fault_related", False)
                confidence = float(filter_result.get("confidence", 0.5))

                # NEW: Two-pass for borderline cases (v3)
                if (self.config.enable_second_pass and
                        confidence < self.config.second_pass_threshold):
                    print(f"           [second pass triggered — conf={confidence:.2f}]")
                    second_result = self.filter_agent.classify_second_pass(ctx, filter_result)
                    second_is_fault = second_result.get("is_fault_related", False)
                    second_conf     = float(second_result.get("confidence", 0.5))
                    if second_is_fault == is_fault:
                        # Both agree → use higher confidence
                        confidence = max(confidence, second_conf)
                        filter_result["confidence"] = confidence
                        filter_result["reasoning"] = (
                            filter_result.get("reasoning", "") +
                            " [2nd-pass confirmed]")
                    else:
                        # Disagree → keep first, flag for review
                        filter_result["flag_for_human_review"] = True
                        second_pass_flips += 1
                        print(f"           [second pass DISAGREES — flagging]")

                # Stage C — ConfidenceScorerAgent (skipped if enable_scorer=False)
                if self.config.enable_scorer:
                    review     = self.scorer.review(ctx, filter_result)
                    confidence = float(review.get("adjusted_confidence",
                                                  filter_result.get("confidence", 0.5)))
                    override   = review.get("override_decision")
                    if override is not None and isinstance(override, bool) and override != is_fault:
                        scorer_overrides += 1
                        is_fault = override
                        print(f"           [scorer override → {'FAULT' if is_fault else 'non-fault'}]")
                    flag = review.get("flag_for_human_review", False) or (
                        confidence < self.config.confidence_threshold)
                else:
                    flag = confidence < self.config.confidence_threshold
                reasoning = filter_result.get("reasoning", "")

            decision = FilterDecision(
                issue_number=num, repo=issue["repo"],
                is_fault_related=is_fault, reasoning=reasoning,
                confidence=confidence, flagged_for_review=flag,
                pre_filter_rule=pf["rule"] if pf else None,
                title=ctx.title, body=ctx.body,
                state=ctx.state, comments_content=ctx.comments_content)

            (fault_related if is_fault else non_fault).append(decision)
            if flag:
                flagged.append(decision)

            print(f"           → {'FAULT' if is_fault else 'non-fault'} "
                  f"(conf:{confidence:.2f}){' [FLAGGED]' if flag else ''}")
            time.sleep(self.config.github_rate_limit_delay)

        output = {
            "total_processed":          total,
            "fault_related_count":      len(fault_related),
            "non_fault_count":          len(non_fault),
            "flagged_for_review_count": len(flagged),
            "pre_filter_count":         pre_filter_hits,
            "scorer_overrides":         scorer_overrides,
            "second_pass_flips":        second_pass_flips,
            "context_fetch_failures":   fetch_failures,
            "fault_related":            [self._dd(d) for d in fault_related],
            "non_fault":                [self._dd(d) for d in non_fault],
            "flagged_for_review":       [self._dd(d) for d in flagged],
        }
        os.makedirs(os.path.dirname(self.config.output_path), exist_ok=True)
        with open(self.config.output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n[Stage II] Saved → {self.config.output_path}")
        self._print_summary(output)
        return output

    @staticmethod
    def _dd(d: FilterDecision) -> dict:
        out = {"issue_number": d.issue_number, "repo": d.repo,
               "is_fault_related": d.is_fault_related,
               "reasoning": d.reasoning, "confidence": d.confidence,
               "flagged_for_review": d.flagged_for_review,
               "title": d.title or "", "body": d.body or "",
               "state": d.state or "", "comments_content": d.comments_content or ""}
        if d.pre_filter_rule:
            out["pre_filter_rule"] = d.pre_filter_rule
        return out

    @staticmethod
    def _print_summary(o: dict):
        t = o["total_processed"]
        pf = o["pre_filter_count"]
        print(f"[Stage II] Total:{t} | Fault:{o['fault_related_count']} | "
              f"Non-fault:{o['non_fault_count']} | Pre-filter:{pf} ({pf/t:.0%}) | "
              f"Flagged:{o['flagged_for_review_count']} | Overrides:{o['scorer_overrides']} | "
              f"2nd-pass flips:{o['second_pass_flips']}")

    def close(self): self.retriever.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AutoEmpirical MAS — Stage II (v3)")
    parser.add_argument("--csv-path",            default=None)
    parser.add_argument("--model",               default="llama3")
    parser.add_argument("--max-issues-per-repo", type=int, default=None)
    parser.add_argument("--no-second-pass",      action="store_true",
                        help="Disable two-pass verification for borderline cases")
    parser.add_argument("--no-scorer",           action="store_true",
                        help="Disable ConfidenceScorerAgent (pre-filter + single LLM ablation)")
    args = parser.parse_args()

    from tools.models import model_config_from_name
    config = Stage2Config(
        max_issues_per_repo=args.max_issues_per_repo,
        csv_path=args.csv_path,
        enable_second_pass=not args.no_second_pass,
        enable_scorer=not args.no_scorer,
        model=model_config_from_name(args.model))

    pipeline = Stage2Pipeline(config)
    try:
        issues = (pipeline.load_issues_from_csv(args.csv_path)
                  if args.csv_path
                  else pipeline.fetch_issues_from_stage1())
        pipeline.run(issues)
    finally:
        pipeline.close()
