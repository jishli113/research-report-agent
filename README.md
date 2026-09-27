# Research Report Agent

A CLI agent that turns a research topic into a cited markdown brief.

It plans 3–5 sub-questions, searches the web, fetches page text, writes per-topic notes, then synthesizes a report. Two implementations live in this repo: a manual tool loop in `manual_agent/`, and a LangGraph loop in `langchain_agent/`. Each writes notes and reports under its own `output/` directory. Filenames are basenames only; directory paths are stripped.

## Requirements

- Python 3.11+
- [Anthropic API key](https://console.anthropic.com/)
- Optional [Tavily API key](https://tavily.com/) for live search (mock results if omitted)

The LangGraph agent also needs `langchain`, `langgraph`, and `langchain-anthropic`. Those are not listed in `requirements.txt`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install langchain langgraph langchain-anthropic
```

Create a `.env` in the repo root:

```bash
ANTHROPIC_API_KEY=...
TAVILY_API_KEY=...
```

`TAVILY_API_KEY` is optional. `ANTHROPIC_MODEL` is optional and defaults to `claude-3-5-haiku-20241022`.

## Manual agent

Anthropic Messages API loop. Tools: `search_web`, `save_notes`, `read_from_file`, `create_synthesis`.

```bash
python manual_agent/report_agent.py "Impact of remote work on commercial real estate"
```

With no argument, it prompts for a topic. Files land in `manual_agent/output/`.

`search_web` and `save_notes` share a `filename`. After each `save_notes`, the matching search dump in the conversation is replaced with a short stub. The same subtopic can be searched at most 3 times, each notes file can be saved once, and each file can be read once. Tool calls retry up to 3 times. The loop stops after 40 model turns, or when the model finishes without calling a tool.

## LangGraph agent

LangGraph loop. Tools: `search_web`, `write_to_file`, `read_from_file`, `create_synthesis`.

```bash
python langchain_agent/report_agent.py
```

It prompts for a topic. Files land in `langchain_agent/output/`.

Each model turn goes `llm` → `tools` → notes check → iteration check, then back to `llm`. `search_web` and `write_to_file` share a `filename`. After a successful `write_to_file`, the matching `search_web` page text in the message list is replaced with a short stub. The run stops when `create_synthesis` succeeds, when the model replies with no tool call, or after 40 model turns.

## Pipeline

1. Plan 3–5 sub-questions.
2. For each, search the web (Tavily, then plain text from each result, about 4,000 characters per page) and save notes.
3. Read each notes file once.
4. Call `create_synthesis` to write the markdown report, with sections, source URLs, and confidence notes.

## Eval (manual)

Score 1–5 on accuracy, citations, and completeness. Suggested topics:

- Impact of remote work on commercial real estate
- State of open-source LLM fine-tuning in 2025
- How vector databases compare for RAG

Also try an obscure topic with poor search results and check that the report admits insufficient data.

### Existing outputs

**Water bottles** (`langchain_agent/output/`). Four notes plus `water_bottles_comprehensive_report.md`. The report follows the notes (history, materials, health, market), and each section has a confidence line, source URLs, and a short gaps list. Completeness is about a 4. Citations are about a 2.

- The $9.70 billion to $15.24 billion market figure is repeated in the materials note, the market note, and the report. The URLs under it are a packaging blog, a paywalled `market.us` page, and a Yahoo article about kids’ bottles. That article does not support the global number. Confidence is still “medium-high.”
- The “240,000 nanoplastics per liter” line is a real 2024 result, and the notes never link the paper. The report’s high-confidence health URL, `PMC3210908`, is an older BPA article.
- History is marked high confidence off two brand blogs (`sjwave.com`, `justbottle.co`).
- The materials table puts PET and polycarbonate in one row and attaches BPA leaching to both. The 0.2–0.3 mg/L migration figure is about polycarbonate, not PET.

**Pokémon** (`manual_agent/output/pokemon-report.md`). Readable overview, and it flags thin spots such as unreleased titles and missing trading-card growth numbers. It has no source URLs, so it misses the citation bar.
