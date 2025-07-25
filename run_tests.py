#!/usr/bin/env python3
"""Test runner that loads environment variables from .env file."""

import os
import sys
import subprocess
from dotenv import load_dotenv


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Run pytest with the loaded environment
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v"], env=os.environ.copy()
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
