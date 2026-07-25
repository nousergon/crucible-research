"""
Local entry point for alpha-engine-research.
Delegates to local/run.py for full-featured CLI.

Usage: python main.py [--date YYYY-MM-DD] [--local] [--no-s3] [--offline] [--stub-llm]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from local.run import main

if __name__ == "__main__":
    main()
