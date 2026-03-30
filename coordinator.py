import argparse
import json
import sys
from camel.agents import ChatAgent
from camel.messages import BaseMessage

from tools.models import ModelConfig, model_config_from_name, build_model, parse_json

from stage_1 import Stage1Pipeline, Stage1Config
from stage_2 import Stage2Pipeline, Stage2Config
from stage_3 import Stage3Pipeline, Stage3Config


# --------------------------------------------------
# Coordinator Agent
# --------------------------------------------------

class CoordinatorAgent:

    SYSTEM_PROMPT = (
        "You are a principal investigator coordinating a multi-stage "
        "empirical software engineering study.\n\n"
        "You oversee 3 stages:\n"
        "1. Repository selection + research questions\n"
        "2. Fault-related issue filtering\n"
        "3. Fault taxonomy classification\n\n"
        "Your responsibilities:\n"
        "- Decide whether outputs are sufficient to proceed\n"
        "- Detect weak or low-quality outputs\n"
        "- Suggest retries or adjustments\n"
        "- Ensure overall study quality\n\n"
        "Respond with valid JSON only.\n"
        'Schema: {"proceed": bool, "reasoning": str}'
    )

    def __init__(self, model_config: ModelConfig = None):
        if model_config is None:
            model_config = ModelConfig()
        model = build_model(model_config)
        self.agent = ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="Coordinator",
                content=self.SYSTEM_PROMPT
            ),
            model=model,
            token_limit=model_config.token_limit
        )

    def decide(self, context: str) -> dict:
        response = self.agent.step(
            BaseMessage.make_user_message(
                role_name="System",
                content=context
            )
        )
        self.agent.reset()
        return parse_json(response.msg.content,
                          default={"proceed": True,
                                   "reasoning": "parse failed — proceeding by default"})


# --------------------------------------------------
# MASTER PIPELINE
# --------------------------------------------------

class AutoEmpiricalMAS:

    def __init__(self, research_theme: str, model_config: ModelConfig,
                 max_issues_per_repo: int = None, csv_path: str = None):
        self.research_theme = research_theme
        self.model_config = model_config
        self.max_issues_per_repo = max_issues_per_repo
        self.csv_path = csv_path
        self.coordinator = CoordinatorAgent(model_config)

    # ------------------------------------------
    # Stage 1
    # ------------------------------------------

    def run_stage1(self):
        config = Stage1Config(
            research_theme=self.research_theme,
            model=self.model_config,
        )
        pipeline = Stage1Pipeline(config)
        output = pipeline.run()

        decision = self.coordinator.decide(
            f"""
            Stage 1 Output:
            Repositories: {len(output.selected_repos)}
            Research Questions: {len(output.research_questions)}

            Repos: {json.dumps(output.selected_repos, indent=2)}
            RQs: {output.research_questions}

            Is this sufficient to proceed to Stage 2?
            If not, explain what is wrong.
            """
        )

        print(f"\n[Coordinator - Stage 1] proceed={decision['proceed']} | {decision['reasoning']}")

        if not decision["proceed"]:
            print("[Coordinator] Halting pipeline after Stage 1.")
            sys.exit(1)

        return output

    # ------------------------------------------
    # Stage 2
    # ------------------------------------------

    def run_stage2(self):
        config = Stage2Config(
            max_issues_per_repo=self.max_issues_per_repo,
            csv_path=self.csv_path,
            model=self.model_config,
        )
        pipeline = Stage2Pipeline(config)

        if self.csv_path:
            issues = pipeline.load_issues_from_csv(self.csv_path)
        else:
            issues = pipeline.fetch_issues_from_stage1()
        output = pipeline.run(issues)

        decision = self.coordinator.decide(
            f"""
            Stage 2 Output Summary:
            Total: {output['total_processed']}
            Fault-related: {output['fault_related_count']}
            Non-fault: {output['non_fault_count']}
            Flagged: {output['flagged_for_review_count']}

            Is the filtering quality acceptable?
            Should we adjust confidence threshold or reprocess?
            """
        )

        print(f"\n[Coordinator - Stage 2] proceed={decision['proceed']} | {decision['reasoning']}")

        if not decision["proceed"]:
            print("[Coordinator] Halting pipeline after Stage 2.")
            sys.exit(1)

        return output

    # ------------------------------------------
    # Stage 3
    # ------------------------------------------

    def run_stage3(self):
        config = Stage3Config(
            model=self.model_config,
        )
        pipeline = Stage3Pipeline(config)

        with open(config.stage2_output_path) as f:
            stage2 = json.load(f)

        issues = stage2.get("fault_related", [])
        output = pipeline.run(issues)

        decision = self.coordinator.decide(
            f"""
            Stage 3 Output Summary:
            Total classified: {output['total_classified']}
            Average confidence: {output['avg_confidence']}
            Average debate rounds: {output['avg_debate_rounds']}

            Is the classification reliable enough for publication?
            """
        )

        print(f"\n[Coordinator - Stage 3] proceed={decision['proceed']} | {decision['reasoning']}")

        return output

    # ------------------------------------------
    # FULL PIPELINE
    # ------------------------------------------

    def run(self):
        print("\n===== AUTOEMPIRICAL MAS START =====\n", flush=True)

        if self.csv_path:
            print(f"[Coordinator] Skipping Stage 1 — loading issues from CSV: {self.csv_path}", flush=True)
        else:
            print("[Coordinator] Running Stage 1...", flush=True)
            self.run_stage1()

        print("[Coordinator] Running Stage 2...", flush=True)
        self.run_stage2()

        print("[Coordinator] Running Stage 3...", flush=True)
        self.run_stage3()

        print("\n===== PIPELINE COMPLETE =====\n", flush=True)


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AutoEmpirical MAS — automated empirical fault study pipeline"
    )
    parser.add_argument(
        "--research-theme", required=True,
        help='Research theme, e.g. "JavaScript-based deep learning system faults"'
    )
    parser.add_argument(
        "--model", default="llama3",
        help="Model name to use (default: llama3)"
    )
    parser.add_argument(
        "--max-issues-per-repo", type=int, default=None,
        help="Max issues to fetch per repo in Stage 2 (default: no cap)"
    )
    parser.add_argument(
        "--csv-path", default=None,
        help="Path to pre-collected issues CSV — skips Stage 1 entirely"
    )
    args = parser.parse_args()

    model_config = model_config_from_name(args.model)

    system = AutoEmpiricalMAS(
        research_theme=args.research_theme,
        model_config=model_config,
        max_issues_per_repo=args.max_issues_per_repo,
        csv_path=args.csv_path,
    )
    system.run()
