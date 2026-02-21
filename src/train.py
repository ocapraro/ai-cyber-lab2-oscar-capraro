from __future__ import annotations

import argparse
from pathlib import Path

from .data import DEFAULT_DATA_PATH, FEATURE_NAMES, load_and_split
from .utils import save_json

DEFAULT_MODEL_PATH = Path("results/model.json")


def _fit_gaussian_nb(X_train: list[list[float]], y_train: list[int]) -> dict:
    """Fit a Gaussian Naive Bayes model with small-variance smoothing."""
    if not X_train:
        raise ValueError("No training samples were provided.")

    n_features = len(X_train[0])
    by_class = {0: [], 1: []}
    for features, label in zip(X_train, y_train):
        by_class[label].append(features)

    model = {"feature_names": FEATURE_NAMES, "classes": {}}
    total = len(X_train)

    for cls in (0, 1):
        rows = by_class[cls]
        if not rows:
            raise ValueError(f"Training split has no samples for class {cls}.")

        means = []
        variances = []
        for j in range(n_features):
            values = [row[j] for row in rows]
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            means.append(mean)
            variances.append(max(variance, 1e-9))

        model["classes"][str(cls)] = {
            "prior": len(rows) / total,
            "mean": means,
            "var": variances,
        }

    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline phishing detector (Gaussian NB).")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to phishing_site_urls.csv",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to save the trained model.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split fraction.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible splitting.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=300,
        help="Unused compatibility flag kept for lab command compatibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    X_train, X_test, y_train, y_test = load_and_split(
        csv_path=args.data_path,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # Keep computation deterministic and lightweight with a Gaussian NB baseline.
    _ = args.max_iter
    model = _fit_gaussian_nb(X_train, y_train)
    save_json(model, args.model_path)

    print(f"Train samples: {len(X_train):,}")
    print(f"Test samples:  {len(X_test):,}")
    print(f"Saved model to: {args.model_path}")
    print(f"Positive class ratio (train): {sum(y_train) / max(len(y_train), 1):.4f}")
    print(f"Positive class ratio (test):  {sum(y_test) / max(len(y_test), 1):.4f}")


if __name__ == "__main__":
    main()
