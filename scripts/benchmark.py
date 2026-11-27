import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

if __package__:
    from .test_client import DEFAULT_SERVER, request
else:
    from test_client import DEFAULT_SERVER, request


async def benchmark(args):
    source = args.input
    paths = sorted(source.rglob("*.wav")) if source.is_dir() else [source]
    if not paths:
        raise ValueError("No WAV files found")
    rows = []
    for path in paths:
        result = await request(path, args.server)
        name = path.relative_to(source).as_posix() if source.is_dir() else path.name
        row = {"file": name, **result, "rtf": result["inference_seconds"] / result["duration"]}
        print(json.dumps(row, ensure_ascii=False))
        rows.append(row)
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    if args.references:
        from jiwer import cer, wer

        refs = json.loads(args.references.read_text(encoding="utf-8"))
        truth = [refs[row["file"]] for row in rows]
        outputs = [row["text"] for row in rows]
        print(json.dumps({"wer": wer(truth, outputs), "cer": cer(truth, outputs)}))


def main():
    parser = argparse.ArgumentParser(description="Measure server inference and WebSocket latency")
    parser.add_argument("input", type=Path)
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--references", type=Path)
    try:
        asyncio.run(benchmark(parser.parse_args()))
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report any failure cleanly
        sys.exit(f"Benchmark failed: {exc}")


if __name__ == "__main__":
    main()
