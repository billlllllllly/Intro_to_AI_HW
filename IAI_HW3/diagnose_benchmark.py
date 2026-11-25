"""Summarize evaluation-pass predictions and benchmark distribution differences."""

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from compare_runs import LABELS


def em_prior(probs, train_prior, iters=100):
    prior = train_prior.copy()
    for _ in range(iters):
        adjusted = probs * (prior / train_prior)
        adjusted /= adjusted.sum(axis=1, keepdims=True)
        prior = adjusted.mean(axis=0)
    return prior


def class_values(values):
    return {label: round(float(value), 4) for label, value in zip(LABELS, values)}


def diagnose_split(title, member_probs, names):
    probs = member_probs.mean(axis=0)
    hard = probs.argmax(axis=1)
    shares = np.bincount(hard, minlength=len(LABELS)) / len(hard)
    log_probs = np.zeros_like(probs)
    np.log(probs, out=log_probs, where=probs > 0)
    entropy = -(probs * log_probs).sum(axis=1).mean()

    print(title)
    print("Predicted class shares", class_values(shares))
    print("Mean predicted probabilities", class_values(probs.mean(axis=0)))
    print(f"Mean max-probability {probs.max(axis=1).mean():.4f} | mean entropy {entropy:.4f} nats")
    individual = member_probs.argmax(axis=2)
    for i, j in combinations(range(len(names)), 2):
        disagreement = (individual[i] != individual[j]).mean()
        print(f"{names[i]} vs {names[j]} disagreement rate {disagreement:.2%}")
    print()


def diagnose_benchmark(run_dir, estimate_prior=False):
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    with np.load(run_dir / "probs.npz") as archive:
        benchmark = archive["benchmark_member_probs"]
        names = [member["name"] for member in config.get("members", [])]
        if not names:
            names = [f"Member {i + 1}" for i in range(len(benchmark))]
        diagnose_split("Validation", archive["val_member_probs"], names)
        diagnose_split("Test", archive["test_member_probs"], names)
        diagnose_split("Benchmark (evaluation pass)", benchmark, names)

    print("If benchmark and validation have similar mean probabilities but the benchmark has more "
          "negative predictions, check for a threshold issue. Lower mean benchmark probabilities "
          "for neutral or positive may indicate distribution shift.")
    if estimate_prior:
        priors = config.get("effective_class_priors")
        if priors is None:
            raise ValueError("The EM diagnostic requires effective class priors in config.json.")
        print("\nEM benchmark prior estimates (diagnostic only)")
        for name, probabilities in zip(names, benchmark):
            prior = np.asarray(priors[name], dtype=float)
            print(name, class_values(em_prior(probabilities, prior)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--em-prior", action="store_true",
                        help="Estimate benchmark priors from the saved effective training priors.")
    args = parser.parse_args()
    diagnose_benchmark(args.run_dir, estimate_prior=args.em_prior)


if __name__ == "__main__":
    main()
