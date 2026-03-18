"""CLI for architect."""
import sys, json, argparse
from .core import Architect

def main():
    parser = argparse.ArgumentParser(description="The Architect's Blueprint — How training data constructs AI ontology, epistemology, and axiology. Mapping the world-building of LLMs.")
    parser.add_argument("command", nargs="?", default="status", choices=["status", "run", "info"])
    parser.add_argument("--input", "-i", default="")
    args = parser.parse_args()
    instance = Architect()
    if args.command == "status":
        print(json.dumps(instance.get_stats(), indent=2))
    elif args.command == "run":
        print(json.dumps(instance.generate(input=args.input or "test"), indent=2, default=str))
    elif args.command == "info":
        print(f"architect v0.1.0 — The Architect's Blueprint — How training data constructs AI ontology, epistemology, and axiology. Mapping the world-building of LLMs.")

if __name__ == "__main__":
    main()
