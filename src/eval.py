from __future__ import annotations

import argparse
import math
from pathlib import Path

from .data import DEFAULT_DATA_PATH, load_and_split
from .train import DEFAULT_MODEL_PATH
from .utils import load_json, save_json, write_png_rgb

DEFAULT_METRICS_PATH = Path("results/metrics.json")
DEFAULT_CM_PATH = Path("results/confusion_matrix.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained phishing model.")
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
        help="Path to trained model artifact.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Where to write JSON metrics.",
    )
    parser.add_argument(
        "--confusion-matrix-path",
        type=Path,
        default=DEFAULT_CM_PATH,
        help="Where to write confusion matrix PNG.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {args.model_path}. Run `python -m src.train` first."
        )

    _, X_test, _, y_test = load_and_split(
        csv_path=args.data_path,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    model = load_json(args.model_path)
    y_pred = [_predict_one(features, model) for features in X_test]

    tn, fp, fn, tp = _confusion_counts(y_test, y_pred)
    metrics = _classification_metrics(tn=tn, fp=fp, fn=fn, tp=tp)
    metrics["n_test_samples"] = len(y_test)
    metrics["confusion_matrix"] = [[tn, fp], [fn, tp]]

    save_json(metrics, args.metrics_path)
    _save_confusion_matrix_png([[tn, fp], [fn, tp]], args.confusion_matrix_path)

    print(f"Saved metrics: {args.metrics_path}")
    print(f"Saved confusion matrix: {args.confusion_matrix_path}")
    for name, value in metrics.items():
        if name.startswith("n_") or name == "confusion_matrix":
            print(f"{name}: {value}")
        else:
            print(f"{name}: {value:.4f}")


def _predict_one(features: list[float], model: dict) -> int:
    classes = model["classes"]
    scores = {}
    for cls in ("0", "1"):
        params = classes[cls]
        logp = math.log(max(params["prior"], 1e-12))
        for x, mean, var in zip(features, params["mean"], params["var"]):
            logp += -0.5 * math.log(2.0 * math.pi * var) - ((x - mean) ** 2) / (2.0 * var)
        scores[int(cls)] = logp
    return 1 if scores[1] > scores[0] else 0


def _confusion_counts(y_true: list[int], y_pred: list[int]) -> tuple[int, int, int, int]:
    tn = fp = fn = tp = 0
    for truth, pred in zip(y_true, y_pred):
        if truth == 0 and pred == 0:
            tn += 1
        elif truth == 0 and pred == 1:
            fp += 1
        elif truth == 1 and pred == 0:
            fn += 1
        else:
            tp += 1
    return tn, fp, fn, tp


def _classification_metrics(tn: int, fp: int, fn: int, tp: int) -> dict[str, float]:
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * precision * recall / (precision + recall)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def _save_confusion_matrix_png(cm: list[list[int]], output_path: Path) -> None:
    # Minimal heatmap-style confusion matrix image (no external plotting deps).
    width, height = 300, 300
    margin = 20
    cell_size = (width - 2 * margin) // 2
    image = [[(255, 255, 255) for _ in range(width)] for _ in range(height)]
    max_value = max(max(row) for row in cm) or 1

    for r in range(2):
        for c in range(2):
            value = cm[r][c]
            intensity = int(255 - 190 * (value / max_value))
            color = (intensity, intensity, 255)
            x0 = margin + c * cell_size
            y0 = margin + r * cell_size
            x1 = x0 + cell_size - 2
            y1 = y0 + cell_size - 2
            for y in range(y0, y1):
                for x in range(x0, x1):
                    image[y][x] = color

            border_color = (80, 80, 80)
            for x in range(x0, x1):
                image[y0][x] = border_color
                image[y1 - 1][x] = border_color
            for y in range(y0, y1):
                image[y][x0] = border_color
                image[y][x1 - 1] = border_color

    write_png_rgb(image, output_path)


if __name__ == "__main__":
    main()
