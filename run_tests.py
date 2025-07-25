#!/usr/bin/env python3
"""Comprehensive test runner that runs unit or integration tests."""

import os
import sys
import subprocess
import argparse
from dotenv import load_dotenv


def check_api_key():
    """Check if OpenAI API key is available."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        print("Please set your OpenAI API key:")
        print("  1. Add OPENAI_API_KEY=your_key to .env file, or")
        print("  2. Export it: export OPENAI_API_KEY=your_key")
        return False

    # Mask the key for security
    masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
    print(f"✅ OpenAI API key found: {masked_key}")
    return True


def run_unit_tests():
    """Run unit tests (non-integration)."""
    print("\n🧪 Running unit tests...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "-m", "not integration"],
        env=os.environ.copy(),
    )
    return result.returncode


def run_integration_tests():
    """Run integration tests that call real APIs."""
    print("\n🚀 Running integration tests with real OpenAI API...")
    print("⚠️  Warning: These tests will make actual API calls and incur costs")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-v",
            "-m",
            "integration",
            "-s",  # Don't capture output so we can see print statements
        ],
        env=os.environ.copy(),
    )
    return result.returncode


def run_all_tests():
    """Run both unit and integration tests."""
    print("\n🔍 Running all tests...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "-s"], env=os.environ.copy()
    )
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run tests for the agentic-metadata-converter app")
    parser.add_argument(
        "--type",
        choices=["unit", "integration", "all"],
        default="unit",
        help="Type of tests to run (default: unit)",
    )
    parser.add_argument(
        "--no-env-check", action="store_true", help="Skip environment variable checks"
    )

    args = parser.parse_args()

    # Load environment variables from .env file
    load_dotenv()

    print("Agentic Metadata Converter - Test Runner")
    print("=" * 50)

    # Check API key for integration tests
    if args.type in ["integration", "all"] and not args.no_env_check:
        if not check_api_key():
            sys.exit(1)

    # Run the appropriate tests
    if args.type == "unit":
        return_code = run_unit_tests()
    elif args.type == "integration":
        return_code = run_integration_tests()
    elif args.type == "all":
        unit_code = run_unit_tests()
        integration_code = run_integration_tests()
        return_code = unit_code or integration_code

    # Summary
    print("\n" + "=" * 50)
    if return_code == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")

    sys.exit(return_code)


if __name__ == "__main__":
    main()
