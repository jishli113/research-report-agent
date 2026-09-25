from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

import os
from dotenv import load_dotenv

from pathlib import Path


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

SYSTEM = (
    "Act as a researcher investigating a question and generate "
    "3-5 research sub-questions by yourself given a topic. For every sub-question search, then read, "
    "then save notes on the research (max 300 words). Never call search_web for "
    "a subtopic after calling save_notes on it. Create a filename for each "
    "subresearch question when calling search_web and use the same filename when "
    "calling save_notes. When all sub-questions have notes saved, call read_from_file "
    "once per notes file, then immediately call create_synthesis. Do not call "
    "read_from_file again after you already received those file contents. Do not "
    "repeat the same tool calls. Name the report file after the topic. Include "
    "sections, source URLs, and confidence notes. If evidence is thin, say so. "
    "Do not write a full report from pretraining memory if tools fail."
    "or state that sources could not be retrieved."
)

client = ChatAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), model_name=ANTHROPIC_MODEL)
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

def output_path(filename: str) -> Path:
    name = Path(filename).name or "untitled.txt"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / name


def html_to_text(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")
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
    """Write the research summary contents to a file.

    Args:
        query: The filename of the file to write the research summary to.

    Returns:
        An object with status determining if it succeeded or not and content which is either the information to write or the error body.
    """
    try:
        path = output_path(filename)
        path.write_text(content, encoding="utf-8")
        return {"status": True, "content": f"Wrote notes to: {path.name}"}
    except Exception as e:
        return {"status": False, "content": f"Failed to write content to filename: {e}"}


def read_from_file(filename: str):
    """Read the research summary contents from a file.

    Args:
        query: The filename of the file containing the research summary.

    Returns:
        An object with status determining if it succeeded or not and content which is either the information or the error body.
    """
    try:
        path = output_path(filename)
        content = path.read_text(encoding="utf-8")
        return {"status": True, "content": content}
    except Exception as e:
        return {"status": False, "content": f"Failed to read filename: {e}"}


def search_web(query: str):
    """Search the web for a given query and return the top results.

    Args:
        query: The search term to look up.

    Returns:
        An object with status determining if it succeeded or not and content which is either the search return information or the error body.
    """
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

def run_agent(topic):
    tools = [search_web, read_from_file, write_to_file]
    prompt = {
        "messages":[{"role": "user", "content": topic}],
    }
    agent = create_agent(model=client, tools=tools)
    return agent.invoke(prompt, config={"recursion_limit":MAX_CYCLES})

def main():
    research_topic = input("What research topic would you like the agent to research about: ")
    if not research_topic.strip():
        raise SystemExit("A research topic is required.")
    print(run_agent(research_topic))
if __name__ == "__main__":
    main()