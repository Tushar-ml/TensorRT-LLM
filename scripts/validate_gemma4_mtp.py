#!/usr/bin/env python3
"""Greedy parity check: MTP vs non-MTP baseline on fixed prompts."""
import argparse
import json
import sys
import urllib.error
import urllib.request

PROMPTS = [
    "The capital of France is",
    "Explain gravity in one sentence:",
    "2 + 2 equals",
]


def chat(base_url: str, prompt: str, max_tokens: int = 32) -> str:
    payload = {
        "model": "dummy",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.load(resp)
    return body["choices"][0]["message"]["content"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mtp-url", default="http://127.0.0.1:8000")
    parser.add_argument("--baseline-url", default="http://127.0.0.1:8001")
    args = parser.parse_args()

    mismatches = 0
    for prompt in PROMPTS:
        try:
            mtp_out = chat(args.mtp_url, prompt)
            base_out = chat(args.baseline_url, prompt)
        except urllib.error.URLError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        match = mtp_out.strip() == base_out.strip()
        status = "MATCH" if match else "MISMATCH"
        print(f"[{status}] prompt={prompt!r}")
        print(f"  MTP:      {mtp_out!r}")
        print(f"  Baseline: {base_out!r}")
        if not match:
            mismatches += 1
    print(f"Summary: {len(PROMPTS) - mismatches}/{len(PROMPTS)} matched")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
