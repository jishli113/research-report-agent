# Research Report Agent

A CLI agent that turns a research topic into a cited markdown brief.

It plans 3–5 sub-questions, searches the web, fetches page text, writes per-topic notes, then synthesizes a report. Notes and reports land in `output/`. After each `save_notes` call, matching `search_web` dumps are compacted in the conversation so the context window does not keep full page text.

This is a standalone extraction of the research-report agent (not the full agents lab).

## Requirements

- Python 3.11+
- [Anthropic API key](https://console.anthropic.com/)
- Optional [Tavily API key](https://tavily.com/) for live search (mock results if omitted)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: ANTHROPIC_API_KEY, optional TAVILY_API_KEY
```

## Run

```bash
python report_agent.py "Impact of remote work on commercial real estate"
```

Or with no argument (interactive prompt):

```bash
python report_agent.py
```

Files are written under `output/` (basename only; paths are stripped).

## Pipeline

1. Plan 3–5 sub-questions
2. For each: `search_web` (Tavily + stripped page text, ~4k chars/page) → `save_notes`
3. `read_from_file` on those notes
4. `create_synthesis` → markdown report

`search_web` and `save_notes` share a `filename` so the agent can compact the right search dump after notes are saved. Max 40 model turns; tool calls retry up to 3 times on failure.

## Eval (manual)

Score 1–5 on accuracy, citations, and completeness. Suggested topics:

- Impact of remote work on commercial real estate
- State of open-source LLM fine-tuning in 2025
- How vector databases compare for RAG

Also try an obscure topic with poor search results and check that the report admits insufficient data.
