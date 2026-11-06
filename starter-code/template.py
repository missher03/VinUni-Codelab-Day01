"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import itertools
import os
import time
from typing import Any, Callable

# Tự động nạp biến môi trường từ file .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import openai
from google import genai
from google.genai import types
import anthropic

# ---------------------------------------------------------------------------
# Costs per 1M INPUT & OUTPUT tokens (USD) — cập nhật lại theo giá thật, kiểm tra 11/2026.
# (Template gốc ghi "as of March 2026" nhưng nhiều giá đã sai/lỗi thời.)
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    # OpenAI — gpt-4o: giá hiện tại là $2.50/$10.00
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    # Gemini 2.5 Flash 
    # ĐÃ BỊ GOOGLE TẮT ngày 16/10/2026 → không gọi được nữa.
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    # Gemini 3.5 Flash (Standard tier) — model thay thế 2.5 Flash, đang dùng làm GEMINI_MODEL.
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-3.5-flash"  # gemini-2.5-flash đã bị Google tắt ngày 16/10/2026
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The OpenAI model to use (default: gpt-4o).
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')
    """
    api_key = os.getenv("OPENAI_API_KEY") or "mock-key"
    client = openai.OpenAI(api_key=api_key)
    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start_time
    text = response.choices[0].message.content or ""
    usage = {
        "input_tokens": response.usage.prompt_tokens if response.usage else 0,
        "output_tokens": response.usage.completion_tokens if response.usage else 0,
    }
    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 3.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (using Gemini 2.5 Flash as standard) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Gemini model to use (default: gemini-2.5-flash).
        temperature: Sampling temperature.
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')
    """
    api_key = os.getenv("GEMINI_API_KEY") or "mock-key"
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
        thinking_config=types.ThinkingConfig(thinking_budget=0), # tắt thinking vì khi test model dùng gần hết max_token để suy luận nên trả câu trả lời không hoàn chỉnh 
    )
    start_time = time.time()
    models_to_try = list(dict.fromkeys([model, "gemini-3.5-flash"]))
    response = None
    last_err = None
    for m in models_to_try:
        try:
            response = client.models.generate_content(
                model=m,
                contents=prompt,
                config=config,
            )
            break
        except Exception as e:
            last_err = e
            if "404" in str(e) or "503" in str(e) or "no longer available" in str(e) or "high demand" in str(e):
                continue
            raise e

    if response is None:
        raise last_err

    latency = time.time() - start_time
    text = getattr(response, "text", "") or ""
    usage_metadata = getattr(response, "usage_metadata", None)
    usage = {
        "input_tokens": getattr(usage_metadata, "prompt_token_count", 0) if usage_metadata else 0,
        "output_tokens": getattr(usage_metadata, "candidates_token_count", 0) if usage_metadata else 0,
    }
    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API (using Claude 3.5 Haiku as default) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Claude model to use (default: claude-3-5-haiku).
        temperature: Sampling temperature (0.0 - 1.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum output tokens.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')
    """
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY") or "mock-key"
    client = anthropic.Anthropic(api_key=api_key)
    start_time = time.time()
    response = client.messages.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start_time
    text = ""
    if hasattr(response, "content") and response.content:
        text = getattr(response.content[0], "text", "") or ""
    usage_obj = getattr(response, "usage", None)
    usage = {
        "input_tokens": getattr(usage_obj, "input_tokens", 0) if usage_obj else 0,
        "output_tokens": getattr(usage_obj, "output_tokens", 0) if usage_obj else 0,
    }
    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 3.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str = "") -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 3.5 Flash (gemini-3.5-flash)
    with the same prompt and return a structured comparison dictionary.

    Calculate the exact USD token cost for input + output using the prices in PRICING_1M_TOKENS.

    Args:
        prompt: The user message to send to all models.

    Returns:
        A dictionary containing:
            - "gpt4o": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gpt4o_mini": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gemini_flash": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
    """
    # 1. Call gpt-4o
    gpt4o_text, gpt4o_lat, gpt4o_usage = call_openai(prompt, model="gpt-4o")
    gpt4o_cost = (
        gpt4o_usage["input_tokens"] * PRICING_1M_TOKENS["gpt-4o"]["input"]
        + gpt4o_usage["output_tokens"] * PRICING_1M_TOKENS["gpt-4o"]["output"]
    ) / 1_000_000

    # 2. Call gpt-4o-mini
    mini_text, mini_lat, mini_usage = call_openai(prompt, model="gpt-4o-mini")
    mini_cost = (
        mini_usage["input_tokens"] * PRICING_1M_TOKENS["gpt-4o-mini"]["input"]
        + mini_usage["output_tokens"] * PRICING_1M_TOKENS["gpt-4o-mini"]["output"]
    ) / 1_000_000

    # 3. Call gemini-3.5-flash
    gem_text, gem_lat, gem_usage = call_gemini(prompt, model=GEMINI_MODEL)
    gem_cost = (
        gem_usage["input_tokens"] * PRICING_1M_TOKENS[GEMINI_MODEL]["input"]
        + gem_usage["output_tokens"] * PRICING_1M_TOKENS[GEMINI_MODEL]["output"]
    ) / 1_000_000

    return {
        "gpt4o": {
            "response": gpt4o_text,
            "latency": gpt4o_lat,
            "cost": gpt4o_cost,
            "input_tokens": gpt4o_usage["input_tokens"],
            "output_tokens": gpt4o_usage["output_tokens"],
        },
        "gpt4o_mini": {
            "response": mini_text,
            "latency": mini_lat,
            "cost": mini_cost,
            "input_tokens": mini_usage["input_tokens"],
            "output_tokens": mini_usage["output_tokens"],
        },
        "gemini_flash": {
            "response": gem_text,
            "latency": gem_lat,
            "cost": gem_cost,
            "input_tokens": gem_usage["input_tokens"],
            "output_tokens": gem_usage["output_tokens"],
        },
    }


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 3.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 3.5.

    Behaviour:
        - Streams response tokens from Gemini 3.5 Flash as they arrive.
        - Maintains the last 3 turns of conversation history for context.
        - Typing 'quit' or 'exit' ends the session.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\033[93m[System Warning] GEMINI_API_KEY environment variable not set. Running in dummy mode.\033[0m")
        api_key = "mock-key"
    print("\n\033[94m==============================================")
    print("🤖 Vin Smart Future — Intelligent Chat Assistant")
    print("Powered by Google Gemini 3.5 Flash")
    print("Type 'quit' or 'exit' to end the session.")
    print("==============================================\033[0m\n")

    client = genai.Client(api_key=api_key)
    history = []

    print("Gemini 3.5 Chatbot initialized. Type 'quit' or 'exit' to stop.")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if not user_input:
            continue

        history.append({"role": "user", "parts": [{"text": user_input}]})
        # Keep last 3 full turns (3 user + 3 model = 6 messages) + the current question,
        # so the context always starts with a "user" message as Gemini expects
        active_history = history[-7:]

        print("Assistant: ", end="", flush=True)
        try:
            full_reply = ""
            models_to_try = list(dict.fromkeys([GEMINI_MODEL, "gemini-3.5-flash"]))
            response_stream = None
            first_chunk = None
            for m in models_to_try:
                try:
                    stream = iter(client.models.generate_content_stream(
                        model=m,
                        contents=active_history,
                    ))
                    # Stream là lazy: lỗi 404/503 chỉ xuất hiện khi lấy chunk đầu tiên
                    first_chunk = next(stream, None)
                    response_stream = stream
                    break
                except Exception as stream_err:
                    if "404" in str(stream_err) or "503" in str(stream_err) or "no longer available" in str(stream_err) or "high demand" in str(stream_err):
                        continue
                    raise stream_err

            if response_stream is not None:
                head = [first_chunk] if first_chunk is not None else []
                for chunk in itertools.chain(head, response_stream):  # vẫn in từng chunk ngay khi tới
                    chunk_text = getattr(chunk, "text", "") or ""
                    print(chunk_text, end="", flush=True)
                    full_reply += chunk_text
                print()
                history.append({"role": "model", "parts": [{"text": full_reply}]})
        except Exception as e:
            print(f"\n[Error]: {e}")
            if history and history[-1]["role"] == "user":
                history.pop()


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                time.sleep(base_delay * (2 ** attempt))
    if last_exception:
        raise last_exception


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    results = []
    for prompt in prompts:
        try:
            res = compare_models(prompt)
        except TypeError:
            res = compare_models()
        res = dict(res)
        res["prompt"] = prompt
        results.append(res)
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A beautiful Markdown table string with columns:
        | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    model_labels = [
        ("gpt4o", "GPT-4o"),
        ("gpt4o_mini", "GPT-4o-Mini"),
        ("gemini_flash", "Gemini-Flash"),
    ]
    for item in results:
        prompt_str = item.get("prompt", "")
        for key, label in model_labels:
            if key in item:
                stats = item[key]
                resp = str(stats.get("response", "")).replace("\n", " ")
                if len(resp) > 50:
                    resp = resp[:47] + "..."
                lat = f"{stats.get('latency', 0.0):.2f}s"
                tokens = f"{stats.get('input_tokens', 0)}/{stats.get('output_tokens', 0)}"
                cost = f"${stats.get('cost', 0.0):.6f}"
                lines.append(f"| {prompt_str} | {label} | {resp} | {lat} | {tokens} | {cost} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
