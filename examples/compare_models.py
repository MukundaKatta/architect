#!/usr/bin/env python3
"""Example: compare ontological fingerprints of Claude and GPT-4o.

Prerequisites:
    export ANTHROPIC_API_KEY="sk-..."
    export OPENAI_API_KEY="sk-..."

Usage:
    python examples/compare_models.py
"""

from __future__ import annotations

from pathlib import Path

from architect.models import AnthropicAdapter, OpenAIAdapter
from architect.fingerprint import generate_fingerprint
from architect.report import generate_full_report


def main() -> None:
    output_dir = Path("output/comparison")

    # Initialize model adapters
    claude = AnthropicAdapter(model="claude-sonnet-4-20250514")
    gpt4o = OpenAIAdapter(model="gpt-4o")

    # Generate fingerprints (skip cultural probes to save API calls)
    print("Generating fingerprint for Claude...")
    fp_claude = generate_fingerprint(claude, skip_cultural=True)

    print("Generating fingerprint for GPT-4o...")
    fp_gpt4o = generate_fingerprint(gpt4o, skip_cultural=True)

    # Generate full report with charts
    generate_full_report(
        [fp_claude, fp_gpt4o],
        output_dir=output_dir,
    )

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
