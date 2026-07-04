"""
Entry point for running pine2pyne as a module.

Usage:
    python -m pine2pyne input.pine
    python -m pine2pyne input.pine -o output.py
"""
import sys
from .cli import main

if __name__ == '__main__':
    sys.exit(main())
