"""
Top-level CLI entry point.

Used as the ``llm-portfolio`` project script in ``pyproject.toml``.
"""

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="llm-portfolio",
        description="LLM Portfolio Platform CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # sop-builder
    sop = subparsers.add_parser("sop-build", help="Build SOP policies")
    sop.add_argument("--output", "-o", required=True)
    sop.add_argument("--validate", action="store_true")
    sop.add_argument("--registry", default=None)
    sop.add_argument("--source-id", default="owned_sop_v1")

    # canonical-cases
    cases = subparsers.add_parser("cases-build", help="Build canonical cases")
    cases.add_argument("--policies", required=True)
    cases.add_argument("--output", "-o", required=True)
    cases.add_argument("--validate", action="store_true")

    # conversations
    conv = subparsers.add_parser("conv-build", help="Generate conversations")
    conv.add_argument("--policies", required=True)
    conv.add_argument("--cases", required=True)
    conv.add_argument("--output", "-o", required=True)
    conv.add_argument("--seed", type=int, default=42)
    conv.add_argument("--source-id", default="owned_sop_v1")
    conv.add_argument("--mode", choices=["fixture", "training_release"], default="fixture")

    # pipeline
    pl = subparsers.add_parser("pipeline", help="Run the data pipeline")
    pl.add_argument("--input", required=True)
    pl.add_argument("--policies", required=True)
    pl.add_argument("--registry", required=True)
    pl.add_argument("--output", required=True)
    pl.add_argument("--mode", choices=["fixture", "training_release"], default="fixture")
    pl.add_argument("--seed", type=int, default=42)
    pl.add_argument("--near-dedup-threshold", type=float, default=0.7)
    pl.add_argument("--ratios", default="0.7,0.15,0.15")

    args = parser.parse_args()

    if args.command == "sop-build":
        from src.ecommerce.dataset.sop_builder import main as sop_main

        sys.argv = [
            "sop_builder.py",
            "--output",
            args.output,
            "--source-id",
            args.source_id,
        ]
        if args.validate:
            sys.argv.append("--validate")
        if args.registry:
            sys.argv += ["--registry", args.registry]
        return sop_main()
    if args.command == "cases-build":
        from src.ecommerce.dataset.canonical_cases import main as cases_main

        sys.argv = [
            "canonical_cases.py",
            "--policies",
            args.policies,
            "--output",
            args.output,
        ]
        if args.validate:
            sys.argv.append("--validate")
        return cases_main()
    if args.command == "conv-build":
        from src.ecommerce.dataset.conversation_generator import main as conv_main

        sys.argv = [
            "conversation_generator.py",
            "--policies",
            args.policies,
            "--cases",
            args.cases,
            "--output",
            args.output,
            "--seed",
            str(args.seed),
            "--source-id",
            args.source_id,
            "--mode",
            args.mode,
        ]
        return conv_main()
    if args.command == "pipeline":
        from src.ecommerce.dataset.pipeline import main as pl_main

        sys.argv = [
            "pipeline.py",
            "--input",
            args.input,
            "--policies",
            args.policies,
            "--registry",
            args.registry,
            "--output",
            args.output,
            "--mode",
            args.mode,
            "--seed",
            str(args.seed),
            "--near-dedup-threshold",
            str(args.near_dedup_threshold),
            "--ratios",
            args.ratios,
        ]
        return pl_main()

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    sys.exit(main())
