"""Compare two HW3 runs with a paired bootstrap of macro-F1 differences."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


LABELS = ("negative", "neutral", "positive")
CLASS_IDS = tuple(range(len(LABELS)))


def paired_bootstrap(labels, pred_a, pred_b, n_boot):
    if n_boot < 1:
        raise ValueError("The number of bootstrap samples must be positive.")
    rng = np.random.default_rng(0)
    n = len(labels)
    differences = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.integers(0, n, n)
        truth = labels[sample]
        f1_a = f1_score(truth, pred_a[sample], labels=CLASS_IDS, average="macro", zero_division=0)
        f1_b = f1_score(truth, pred_b[sample], labels=CLASS_IDS, average="macro", zero_division=0)
        differences[i] = f1_b - f1_a
    return differences.mean(), np.percentile(differences, [2.5, 97.5]), (differences > 0).mean()


def compare_runs(run_a, run_b, n_boot=2000):
    with np.load(Path(run_a) / "probs.npz") as a, np.load(Path(run_b) / "probs.npz") as b:
        for split in ("val", "test"):
            np.testing.assert_array_equal(a[f"{split}_ids"], b[f"{split}_ids"],
                                          err_msg=f"{split} IDs must match in the same order.")
            np.testing.assert_array_equal(a[f"{split}_labels"], b[f"{split}_labels"],
                                          err_msg=f"{split} labels must match.")
        for title, splits in (("Test only (primary)", ("test",)),
                              ("Validation + test (secondary)", ("val", "test"))):
            labels = np.concatenate([a[f"{split}_labels"] for split in splits])
            pred_a, pred_b = (
                np.concatenate([run[f"{split}_member_probs"].mean(0).argmax(1) for split in splits])
                for run in (a, b)
            )
            scores_a = f1_score(labels, pred_a, labels=CLASS_IDS, average=None, zero_division=0)
            scores_b = f1_score(labels, pred_b, labels=CLASS_IDS, average=None, zero_division=0)
            mean_diff, (low, high), fraction_positive = paired_bootstrap(labels, pred_a, pred_b, n_boot)

            print(title)
            print(f"Macro-F1 | A {scores_a.mean():.4f} | B {scores_b.mean():.4f}")
            for name, score_a, score_b in zip(LABELS, scores_a, scores_b):
                print(f"{name} F1 | A {score_a:.4f} | B {score_b:.4f}")
            print(f"Mean B - A {mean_diff:+.4f} | 95% interval [{low:+.4f}, {high:+.4f}] | "
                  f"fraction above zero {fraction_positive:.3f}")
            if low <= 0 <= high:
                print("Verdict: no detectable difference")
            print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--n", type=int, default=2000, help="Number of paired bootstrap samples.")
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be positive")
    compare_runs(args.run_a, args.run_b, n_boot=args.n)


if __name__ == "__main__":
    main()
