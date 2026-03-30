# AutoEmpirical MAS

A Multi-Agent System (MAS) built with the [CAMEL AI](https://github.com/camel-ai/camel) framework that automates empirical software fault studies on open-source repositories.

## Overview

The pipeline runs in three sequential stages:

| Stage | Description |
|-------|-------------|
| **Stage 1** | Selects representative GitHub repositories and formulates research questions for a given research theme |
| **Stage 2** | Fetches issues from the selected repos and classifies each as fault-related or not using an LLM filter + confidence scorer |
| **Stage 3** | Classifies fault-related issues into symptom and root cause taxonomy labels via a multi-agent debate loop |

A **Coordinator Agent** oversees each stage and decides whether the output is sufficient to proceed.

## Installation

```bash
pip install camel-ai
```

## Ollama Setup (for local models)

If you want to run locally without an API key, install [Ollama](https://ollama.com) and pull a model:

```bash
ollama pull llama3
ollama serve
```

Then run the pipeline with `--model llama3` (the default).

## Environment Setup

Copy `.env.example` to `.env` and fill in the required keys:

```bash
cp .env.example .env
```

You only need the API key for the model you intend to use. `GITHUB_TOKEN` is always required.

| Variable | Required for |
|----------|-------------|
| `GITHUB_TOKEN` | All runs (fetching GitHub issues) |
| `OPENAI_API_KEY` | `--model gpt-4o` or `--model o3` |
| `ANTHROPIC_API_KEY` | `--model claude-3-7-sonnet` |
| `GEMINI_API_KEY` | `--model gemini-2.5-flash` |
| `DEEPSEEK_API_KEY` | `--model deepseek-v3` |

## Usage

```bash
python coordinator.py --research-theme "<your research theme>" --model <model-name> [--max-issues-per-repo <n>]
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--research-theme` | Yes | — | The research theme to study (e.g. `"JavaScript-based deep learning system faults"`) |
| `--model` | No | `llama3` | Model to use (see supported models below) |
| `--max-issues-per-repo` | No | No cap | Max issues to fetch per repo in Stage 2 |

### Supported Models

| Name | Provider |
|------|----------|
| `llama3` | Ollama (local) |
| `gpt-4o` | OpenAI |
| `o3` | OpenAI |
| `claude-3-7-sonnet` | Anthropic |
| `gemini-2.5-flash` | Google |
| `deepseek-v3` | DeepSeek |

### Examples

```bash
# Run with GPT-4o, cap at 20 issues per repo
python coordinator.py \
  --research-theme "JavaScript-based deep learning system faults" \
  --model gpt-4o \
  --max-issues-per-repo 20

# Run with Claude, no issue cap
python coordinator.py \
  --research-theme "Python machine learning library faults" \
  --model claude-3-7-sonnet
```

## Outputs

Results are saved to the `outputs/` directory:

| File | Contents |
|------|----------|
| `outputs/stage1_output.json` | Selected repositories and research questions |
| `outputs/stage2_output.json` | Fault/non-fault classifications with confidence scores |
| `outputs/stage3_output.json` | Taxonomy labels (symptom + root cause) with debate transcripts |
