"""Export original and preprocessed sentiment text using the training defaults."""

import argparse
import csv
from itertools import islice
from pathlib import Path

from sentiment_preprocessing import preprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", help="Custom text (defaults to dataset examples).")
    parser.add_argument("--limit", type=int, help="Maximum number of dataset rows (default: all).")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--url-token", action="store_true")
    parser.add_argument("--distinguish-mentions", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("preprocess_preview.csv"))
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be nonnegative")

    def texts():
        if args.text is not None:
            yield args.text
            return
        dataset = Path(__file__).parent / "dataset" / "dataset.csv"
        with dataset.open(encoding="utf-8-sig", newline="") as stream:
            for row in islice(csv.DictReader(stream), args.limit):
                yield row["text"]

    count = 0
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["number", "original", "preprocessed"])
        for count, original in enumerate(texts(), start=1):
            processed = preprocess(original, clean=args.clean, url_token=args.url_token,
                                   distinguish_mentions=args.distinguish_mentions)
            writer.writerow([count, original, processed])
    print(f"Wrote {count} rows to {args.output.resolve()}")


if __name__ == "__main__":
    main()
