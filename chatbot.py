#!/usr/bin/env python3
"""
Claude Opus 4.6 Chatbot
Streaming CLI chatbot with web search, adaptive thinking, and conversation memory.

Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY='sk-ant-...'

Usage:
    python chatbot.py

Commands (during chat):
    clear   — erase conversation history and start fresh
    exit    — quit
"""

import json
import os
import sys
import anthropic

# ── Terminal colours (zero extra dependencies) ─────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
BLUE   = "\033[34m"
RED    = "\033[31m"

MODEL = "claude-opus-4-6"

SYSTEM_PROMPT = """\
You are Claude, an AI assistant made by Anthropic. You are helpful, harmless, and honest.

You have access to a web_search tool. Use it proactively when the user asks about:
- Current events, breaking news, or live data (prices, sports scores, weather)
- Specific facts you are uncertain about that may have changed since your training
- Any topic where up-to-date information would materially improve your answer

Always cite your sources when you rely on search results. Format your responses \
clearly, using markdown (headers, bullet points, code blocks) where it helps readability."""


# ── Client ─────────────────────────────────────────────────────────────────────

def make_client() -> anthropic.Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        sys.exit(
            f"{RED}Error:{RESET} ANTHROPIC_API_KEY is not set.\n"
            f"  Run: export ANTHROPIC_API_KEY='sk-ant-...'"
        )
    return anthropic.Anthropic(api_key=key)


# ── Streaming turn ─────────────────────────────────────────────────────────────

def stream_turn(client: anthropic.Anthropic, messages: list) -> list:
    """
    Stream one complete assistant turn.

    Handles:
      - Adaptive thinking   → shows a subtle indicator, suppresses raw content
      - Web search (server-side) → shows the live query before results arrive
      - pause_turn          → re-sends automatically so long searches finish
      - Error recovery      → pops the failed user message on exception

    Returns the updated messages list with the assistant reply appended.
    """
    while True:
        pending_tool_json = ""
        active_block_type: str | None = None
        last_was_thinking  = False

        with client.messages.stream(
            model=MODEL,
            max_tokens=64000,
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # cache across turns
            }],
            thinking={"type": "adaptive"},               # deep reasoning on demand
            tools=[{"type": "web_search_20260209", "name": "web_search"}],
            messages=messages,
        ) as stream:

            for event in stream:

                # ── Block opens ──────────────────────────────────────────────
                if event.type == "content_block_start":
                    active_block_type = event.content_block.type

                    if active_block_type == "thinking":
                        last_was_thinking = True
                        print(f"{DIM}[thinking…]{RESET}", end="", flush=True)

                    elif active_block_type == "text":
                        if last_was_thinking:
                            # Put response text on a new line after the indicator
                            print()
                            last_was_thinking = False

                    elif active_block_type == "server_tool_use":
                        pending_tool_json = ""

                # ── Streaming deltas ─────────────────────────────────────────
                elif event.type == "content_block_delta":
                    dt = event.delta.type

                    if dt == "text_delta":
                        print(event.delta.text, end="", flush=True)

                    elif dt == "input_json_delta" and active_block_type == "server_tool_use":
                        pending_tool_json += event.delta.partial_json

                    # thinking_delta: intentionally suppressed

                # ── Block closes ─────────────────────────────────────────────
                elif event.type == "content_block_stop":
                    if active_block_type == "server_tool_use" and pending_tool_json:
                        try:
                            query = json.loads(pending_tool_json).get("query", "…")
                        except json.JSONDecodeError:
                            query = "…"
                        print(f"\n{YELLOW}🔍  Searching: {query}{RESET}", flush=True)
                        pending_tool_json = ""

            final = stream.get_final_message()

        # Keep the FULL content list so thinking-block signatures are preserved
        # (the API requires them in subsequent turns).
        messages.append({"role": "assistant", "content": final.content})

        if final.stop_reason == "pause_turn":
            # Server-side tool loop reached its iteration limit; re-send to continue.
            print(f"\n{DIM}[continuing…]{RESET}", end="", flush=True)
            continue

        break  # end_turn or stop_sequence — we're done

    return messages


# ── Banner ─────────────────────────────────────────────────────────────────────

def print_banner():
    print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║   Claude Opus 4.6  ·  AI Chatbot         ║{RESET}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════╝{RESET}")
    print(f"{DIM}  'clear' to reset history  ·  'exit' to quit{RESET}\n")


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    client   = make_client()
    messages: list = []
    print_banner()

    while True:
        # ── Prompt ──────────────────────────────────────────────────────────
        try:
            user_input = input(f"{BOLD}{GREEN}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Goodbye!{RESET}")
            break

        if not user_input:
            continue

        # ── Built-in commands ────────────────────────────────────────────────
        if user_input.lower() in ("exit", "quit", "bye"):
            print(f"{DIM}Goodbye!{RESET}")
            break

        if user_input.lower() == "clear":
            messages = []
            print(f"  {DIM}✓ Conversation history cleared.{RESET}\n")
            continue

        # ── Send to Claude ───────────────────────────────────────────────────
        messages.append({"role": "user", "content": user_input})
        print(f"\n{BOLD}{BLUE}Claude:{RESET} ", end="", flush=True)

        try:
            messages = stream_turn(client, messages)

        except anthropic.BadRequestError as e:
            print(f"\n{RED}[Bad request: {e.message}]{RESET}")
            messages.pop()  # discard the failed user message

        except anthropic.RateLimitError:
            print(f"\n{RED}[Rate limited — please wait a moment and try again.]{RESET}")
            messages.pop()

        except anthropic.AuthenticationError:
            print(f"\n{RED}[Authentication failed — check your ANTHROPIC_API_KEY.]{RESET}")
            break

        except anthropic.APIError as e:
            print(f"\n{RED}[API error: {e}]{RESET}")
            messages.pop()

        print("\n")  # blank line between turns


if __name__ == "__main__":
    main()
