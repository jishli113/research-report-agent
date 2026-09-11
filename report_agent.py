import argparse
import os
from pathlib import Path

import anthropic
import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MAX_CYCLES = 40
MAX_TOOL_RETRIES = 3
MAX_PAGE_CHARS = 4_000
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

SYSTEM = (
    "Act as a researcher investigating a question and generate "
    "3-5 research sub-questions. For every sub-question search, then read, "
    "then save notes on the research (max 150 words). Create a filename for each "
    "subresearch question when calling search_web and use the same filename when "
    "calling save_notes. When all sub-questions have notes saved, read those files. "
    "Then synthesize a final markdown report. Name the report file after the topic. "
    "Include sections, source URLs, and confidence notes. If evidence is thin, say so."
)

TOOL_ERROR_MESSAGE = (
    "The tool failed. If it was for a sub-topic, skip that topic and continue. "
    "If it was the final report, stop after this or the next iteration."
)

client = anthropic.Anthropic()
tools = [
    {
        "name": "search_web",
        "description": (
            "Search the web for a sub-question, then fetch a short plain-text "
            "excerpt from each result page."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": (
                        "Notes filename for this sub-question. Must match save_notes later."
                    ),
                },
                "query": {
                    "type": "string",
                    "description": "The web search query.",
                },
            },
            "required": ["query", "filename"],
        },
    },
    {
        "name": "save_notes",
        "description": "Save notes for one sub-question to a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Must match the search_web filename for this sub-question.",
                },
                "content": {
                    "type": "string",
                    "description": "The notes to write (max ~150 words).",
                },
            },
            "required": ["filename", "content"],
        },
    },
    {
        "name": "create_synthesis",
        "description": "Write the final markdown report to a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Report filename, e.g. remote-work-report.md",
                },
                "content": {
                    "type": "string",
                    "description": "Full markdown report with sections, sources, confidence.",
                },
            },
            "required": ["filename", "content"],
        },
    },
    {
        "name": "read_from_file",
        "description": "Read notes or a report file and return its contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Filename to read from the output directory.",
                },
            },
            "required": ["filename"],
        },
    },
]


def output_path(filename: str) -> Path:
    name = Path(filename).name or "untitled.txt"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / name


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = "\n".join(
        line
        for line in soup.get_text(separator="\n", strip=True).splitlines()
        if line.strip()
    )
    if len(text) > MAX_PAGE_CHARS:
        return text[:MAX_PAGE_CHARS] + f"\n\n[Truncated at {MAX_PAGE_CHARS} characters]"
    return text


def fetch_page(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return "Error: url must start with http:// or https://"
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as http:
            response = http.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; research-report-agent/1.0)"},
            )
            response.raise_for_status()
        text = html_to_text(response.text)
        if not text:
            return "Error: page contained no readable text"
        return text
    except Exception as e:
        return f"Error: failed to fetch url ({e})"


def write_to_file(filename: str, content: str):
    try:
        path = output_path(filename)
        path.write_text(content, encoding="utf-8")
        return {"status": True, "content": f"Wrote notes to: {path.name}"}
    except Exception as e:
        return {"status": False, "content": f"Failed to write content to filename: {e}"}


def read_from_file(filename: str):
    try:
        path = output_path(filename)
        content = path.read_text(encoding="utf-8")
        return {"status": True, "content": content}
    except Exception as e:
        return {"status": False, "content": f"Failed to read filename: {e}"}


def search_web(query: str):
    if not query:
        return {"status": False, "content": "Error: missing required argument 'query'"}

    if TAVILY_API_KEY and TAVILY_API_KEY != "tvly-...":
        try:
            tavily = TavilyClient(api_key=TAVILY_API_KEY)
            response = tavily.search(query=query, max_results=3)
            results = response.get("results", [])
            if not results:
                return {"status": False, "content": "No results found."}

            lines = []
            for i, result in enumerate(results, 1):
                title = result.get("title", "Untitled")
                url = result.get("url", "")
                page = fetch_page(url) if url else "Error: missing url"
                lines.append(f"{i}. {title}\n   URL: {url}\n   {page}")
            return {"status": True, "content": "\n\n".join(lines)}
        except Exception as e:
            return {"status": False, "content": f"Error: search failed ({e})"}

    return {
        "status": True,
        "content": (
            f"Mock search results for: {query}\n\n"
            f"1. Example Article\n"
            f"   URL: https://example.com/article-1\n"
            f"   Summary: Placeholder search result about {query}.\n\n"
            f"2. Example Documentation\n"
            f"   URL: https://example.com/docs\n"
            f"   Summary: Another placeholder result for {query}."
        ),
    }


def run_tool(tool_name, tool_input):
    if tool_name == "search_web":
        return search_web(tool_input.get("query"))
    if tool_name == "save_notes" or tool_name == "create_synthesis":
        return write_to_file(tool_input.get("filename"), tool_input.get("content"))
    if tool_name == "read_from_file":
        return read_from_file(tool_input.get("filename"))
    return {"status": False, "content": "Invalid tool name input."}


def compact_research_notes(messages, filename):
    block_ids = set()
    for m in messages:
        if m["role"] == "user":
            continue
        for block in m["content"]:
            if (
                getattr(block, "type", None) == "tool_use"
                and getattr(block, "name", None) == "search_web"
                and (getattr(block, "input", None) or {}).get("filename") == filename
            ):
                block_ids.add(block.id)
    if not block_ids:
        return

    stub = f"Saved research note information to: {filename}"
    for m in messages:
        if m["role"] != "user" or not isinstance(m["content"], list):
            continue
        for item in m["content"]:
            if isinstance(item, dict) and item.get("tool_use_id") in block_ids:
                item["content"] = stub


def run_agent(topic: str) -> None:
    messages = [{"role": "user", "content": topic}]
    for i in range(MAX_CYCLES):
        response = client.messages.create(
            max_tokens=1024,
            model=ANTHROPIC_MODEL,
            tools=tools,
            messages=messages,
            system=SYSTEM,
        )
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                retries = 0
                output = None
                while retries < MAX_TOOL_RETRIES:
                    output = run_tool(block.name, block.input)
                    if output and output.get("status"):
                        break
                    retries += 1
                    print(f"retry {block.name} ({retries}/{MAX_TOOL_RETRIES})")
                if not output or not output.get("status"):
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": TOOL_ERROR_MESSAGE,
                        }
                    )
                else:
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(output.get("content", "")),
                        }
                    )
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            for block in response.content:
                if getattr(block, "type", None) == "tool_use" and block.name == "save_notes":
                    compact_research_notes(messages, block.input.get("filename"))
        if response.stop_reason == "end_turn" or i == MAX_CYCLES - 1:
            for block in response.content:
                if block.type == "text":
                    print(block.text)
            break


def main():
    parser = argparse.ArgumentParser(description="Research-report agent")
    parser.add_argument(
        "topic",
        nargs="?",
        help="Research topic. If omitted, you will be prompted.",
    )
    args = parser.parse_args()
    topic = args.topic or input("What topic would you like to research? ")
    if not topic.strip():
        raise SystemExit("A research topic is required.")
    print(f"Researching: {topic}\nOutput directory: {OUTPUT_DIR}\n")
    run_agent(topic.strip())


if __name__ == "__main__":
    main()
