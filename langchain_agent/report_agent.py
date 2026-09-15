from langchain_anthropic import ChatAnthropic
import os
from dotenv import load_dotenv

import argparse
import os
from pathlib import Path
from collections import defaultdict

import anthropic
import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")

client = ChatAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


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

tools = [compact_research_notes, search_web, read_from_file, write_to_file]