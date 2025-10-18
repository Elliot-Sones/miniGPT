"""
Thin wrapper to run machine_translation.setup_data with defaults.

Usage:
  python3 setup_data.py

This will download the default EN→FR corpus (WMT14), clean it, and write
machine_translation/archive/train.csv and machine_translation/archive/test.csv.
"""

import sys


def main():
    try:
        from machine_translation.setup_data import main as _mt_setup_main
    except Exception as e:
        print("Failed to import machine_translation.setup_data.\n"
              "Hint: run from the project root and ensure dependencies are installed (pip install -r requirements.txt).\n"
              f"Original error: {e}")
        sys.exit(1)
    _mt_setup_main()


if __name__ == "__main__":
    main()


