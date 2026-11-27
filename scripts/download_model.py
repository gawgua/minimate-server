import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.checkpoint import checkpoint


def main():
    parser = argparse.ArgumentParser(description="Download the original ViMD root checkpoint")
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    print(checkpoint(cache_dir=args.cache_dir))


if __name__ == "__main__":
    main()
