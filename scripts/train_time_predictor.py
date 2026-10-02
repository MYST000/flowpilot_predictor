#!/usr/bin/env python3
"""Compatibility entry point. Prefer python -m predictor.cli train."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predictor.cli import main
if __name__ == '__main__':
    sys.argv.insert(1, 'train')
    main()
