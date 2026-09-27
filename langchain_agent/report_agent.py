from langchain.agents import AgentState
from langchain_anthropic import ChatAnthropic
from langgraph.graph import START, StateGraph, END
from langchain_core.tools import tool
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from typing_extensions import TypedDict
from typing import Annotated

import os
from dotenv import load_dotenv

from pathlib import Path
from operator import add, or_

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tavily import TavilyClient

from collections import defaultdict

load_dotenv()

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MAX_CYCLES = 40
MAX_TOOL_RETRIES = 3
MAX_PAGE_CHARS = 4_000
SEARCH_WEB_LIMITS = 3
SEARCH_WEB_ERROR_MESSAGE = "This subtopic has been researched enough. Please stop researching this, save the notes and continue on."

READ_FILE_ERROR_MESSAGE = "This file has already been read from. Please stop reading from this and instead reference the information" \
"read from this file from before."

SAVE_NOTES_ERROR_MESSAGE = "This file has already been saved to. Please stop saving to this and carry on."

SYSTEM = (
    "Act as a researcher investigating a question and generate "
    "3-5 research sub-questions by yourself given a topic. For every sub-question search, then read, "
    "then save notes on the research (max 300 words). Never call search_web for "
    "a subtopic after calling write_to_file on it. You can only call one write_to_file per cycle. Create a filename for each "
    "subresearch question when calling search_web and use the same filename when "
    "calling write_to_file. When all sub-questions have notes saved, call read_from_file "
    "once per notes file, then immediately call create_synthesis. Do not call "
    "read_from_file again after you already received those file contents. Do not "
    "repeat the same tool calls. Name the report file after the topic. Include "
    "sections, source URLs, and confidence notes. If evidence is thin, say so. "
    "Do not write a full report from pretraining memory if tools fail."
    "or state that sources could not be retrieved."
)
TOOL_ERROR_MESSAGE = (
    "The tool failed. If it was for a sub-topic, skip that topic and continue. "
    "If it was the final report, stop after this or the next iteration."
)
client = ChatAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), model_name=ANTHROPIC_MODEL)
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


# Custom Reducer Function for altering search_web_counts
def merge_counts(existing: dict[str, int], update: dict[str, int | None]) -> dict[str, int]:
    merged = dict(existing)
    for filename, count in update.items():
        if count is None:
            merged.pop(filename, None)
        else:
            merged[filename] = merged.get(filename, 0) + count
    return merged

class AgentState(TypedDict):
    messages:Annotated[list[str], add]
    iterations:int
    search_web_counts: Annotated[dict[str, int], merge_counts]
    files_read_from: Annotated[set[str], or_]
    files_saved_to: Annotated[set[str], or_]
    to_compress: list[str]
    respsonse:str


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

@tool
def write_to_file(filename: str, content: str):
    """Write the research summary contents to a file.

    Args:
        query: The filename of the file to write the research summary to.

    Returns:
        A confirmation that the file was written.

    Raises:
        Exception: If the file cannot be written. ToolNode records that as an error.
    """
    path = output_path(filename)
    path.write_text(content, encoding="utf-8")
    return f"Wrote notes to:{path.name}"

@tool
def read_from_file(filename: str):
    """Read the research summary contents from a file.

    Args:
        query: The filename of the file containing the research summary.

    Returns:
        The file contents.

    Raises:
        Exception: If the file cannot be read. ToolNode records that as an error.
    """
    path = output_path(filename)
    return path.read_text(encoding="utf-8")

@tool
def search_web(query: str, filename: str):
    """Search the web for a given query and return the top results.

    Args:
        query: The search term to look up.
        filename: Notes filename for this sub-question. Must match write_to_file later.

    Returns:
        Title, URL, and page text for each result.

    Raises:
        Exception: If the query is missing, the search fails, or nothing is found.
        ToolNode records that as an error.
    """
    if not query:
        raise ValueError("missing required argument 'query'")

    if TAVILY_API_KEY and TAVILY_API_KEY != "tvly-...":
        tavily = TavilyClient(api_key=TAVILY_API_KEY)
        response = tavily.search(query=query, max_results=3)
        results = response.get("results", [])
        if not results:
            raise RuntimeError("No results found.")

        lines = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            page = fetch_page(url) if url else "Error: missing url"
            lines.append(f"{i}. {title}\n   URL: {url}\n   {page}")
        return "\n\n".join(lines)

    return (
        f"Mock search results for: {query}\n\n"
        f"1. Example Article\n"
        f"   URL: https://example.com/article-1\n"
        f"   Summary: Placeholder search result about {query}.\n\n"
        f"2. Example Documentation\n"
        f"   URL: https://example.com/docs\n"
        f"   Summary: Another placeholder result for {query}."
    )

@tool
def create_synthesis(filename: str, content: str):
    """Write the final markdown report to a file.

    Args:
        filename: Report filename, named after the topic.
        content: Full markdown report with sections, sources, and confidence.

    Returns:
        A confirmation that the report was written.

    Raises:
        Exception: If the report cannot be written. ToolNode records that as an error.
    """
    path = output_path(filename)
    path.write_text(content, encoding="utf-8")
    return f"Wrote report to: {path.name}"


def compact_research_notes(messages, to_compact):
    filenames = set(to_compact.values())
    search_call_filenames = {}
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for tool_call in message.tool_calls:
            if tool_call.get("name") != "search_web":
                continue
            filename = (tool_call.get("args") or {}).get("filename")
            if filename in filenames:
                search_call_filenames[tool_call["id"]] = filename

    for message in messages:
        if not isinstance(message, ToolMessage) or message.status == "error":
            continue
        filename = search_call_filenames.get(message.tool_call_id)
        if filename is None:
            continue
        message.content = f"Saved research note information to: {filename}"
                
tools = [search_web, read_from_file, write_to_file, create_synthesis]

# Nodes
def llm_call(agent_state):
    model = client.bind_tools(tools)
    response = model.invoke([SystemMessage(content=SYSTEM), *agent_state["messages"]])
    return {"messages": [response], "iterations": agent_state.get("iterations", 0) + 1}

def check_if_notes_saved(agent_state):
    messages = agent_state["messages"]
    ai_message = None
    tool_messages = []
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            ai_message = message
            break
        if isinstance(message, ToolMessage):
            tool_messages.append(message)

    to_compact = {}
    if ai_message is not None:
        calls_by_id = {call["id"]: call for call in ai_message.tool_calls}
        for result in tool_messages:
            if result.name != "write_to_file" or result.status == "error":
                continue
            call = calls_by_id.get(result.tool_call_id)
            if call is None:
                continue
            to_compact[result.tool_call_id] = call["args"]["filename"]
    if to_compact:
        compact_research_notes(messages, to_compact)
    return "iteration_check"

def wrote_synthesis(messages):
    ai_message = None
    results = []
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            ai_message = message
            break
        if isinstance(message, ToolMessage):
            results.append(message)
    if ai_message is None:
        return False
    synthesis_ids = {
        call["id"]
        for call in ai_message.tool_calls
        if call["name"] == "create_synthesis"
    }
    return any(
        result.tool_call_id in synthesis_ids and result.status != "error"
        for result in results
    )

def iterations_check(agent_state):
    if wrote_synthesis(agent_state["messages"]):
        return END
    if agent_state.get("iterations", 0) >= MAX_CYCLES:
        return END
    return "llm"

def pass_through(agent_state):
    return {}

def retry_tool(request, execute):
    last_error = None
    for _ in range(MAX_TOOL_RETRIES):
        try:
            return execute(request)
        except Exception as exc:
            last_error = exc
    return ToolMessage(
        content=f"{last_error}\n\n{TOOL_ERROR_MESSAGE}",
        name=request.tool_call["name"],
        tool_call_id=request.tool_call["id"],
        status="error",
    )
                


def run_agent(topic):
    graph = StateGraph(AgentState)
    graph.add_node("llm", llm_call)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=False, wrap_tool_call=retry_tool))
    graph.add_node("iteration_check", pass_through)
    graph.add_edge(START, "llm")
    graph.add_conditional_edges(
        "llm",
        tools_condition,
        {"tools": "tools", "__end__": END},
    )
    graph.add_conditional_edges(
        "tools",
        check_if_notes_saved,
        {"iteration_check": "iteration_check"},
    )
    graph.add_conditional_edges(
        "iteration_check",
        iterations_check,
        {"llm": "llm", "__end__": END},
    )
    agent = graph.compile()
    return agent.invoke(
        {
            "messages": [HumanMessage(content=topic)],
            "iterations": 0,
        },
        config={"recursion_limit": MAX_CYCLES * 3},
    )

def main():
    research_topic = input("What research topic would you like the agent to research about: ")
    if not research_topic.strip():
        raise SystemExit("A research topic is required.")
    print(run_agent(research_topic))
if __name__ == "__main__":
    main()
