from __future__ import annotations

import csv
import random
import re
from pathlib import Path
from typing import List, Sequence, Tuple

DEFAULT_DATA_PATH = Path("data/raw/phishing_site_urls.csv")
FEATURE_NAMES = [
    "url_length",
    "digit_count",
    "letter_count",
    "symbol_count",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_question_marks",
    "num_equals",
    "num_at",
    "has_https",
    "has_ip_pattern",
    "has_suspicious_keyword",
    "digit_ratio",
    "symbol_ratio",
]

_IP_PATTERN = re.compile(r"(?:\d{1,3}\.){3}\d{1,3}")
_SUSPICIOUS_KEYWORDS = (
    "login",
    "verify",
    "secure",
    "update",
    "account",
    "password",
    "paypal",
    "bank",
    "signin",
)


def load_dataset(csv_path: Path = DEFAULT_DATA_PATH) -> List[Tuple[str, int]]:
    """Load URL samples and map labels to binary values."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Place phishing_site_urls.csv in data/raw/."
        )

    label_map = {
        "bad": 1,
        "phishing": 1,
        "malicious": 1,
        "good": 0,
        "benign": 0,
        "legitimate": 0,
    }

    samples: List[Tuple[str, int]] = []
    seen_urls = set()
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header in {csv_path}.")
        if "URL" not in reader.fieldnames or "Label" not in reader.fieldnames:
            raise ValueError(
                f"Expected URL and Label columns, found {reader.fieldnames} in {csv_path}."
            )

        for row in reader:
            raw_url = (row.get("URL") or "").strip().lower()
            raw_label = (row.get("Label") or "").strip().lower()
            if not raw_url:
                continue
            mapped = label_map.get(raw_label)
            if mapped is None:
                continue
            if raw_url in seen_urls:
                continue
            seen_urls.add(raw_url)
            samples.append((raw_url, mapped))

    if not samples:
        raise ValueError(f"No valid samples loaded from {csv_path}.")
    return samples


def build_url_features(url: str) -> List[float]:
    """Create lexical URL features for a single sample."""
    url = (url or "").strip().lower()
    url_len = max(len(url), 1)

    digit_count = sum(ch.isdigit() for ch in url)
    letter_count = sum(ch.isalpha() for ch in url)
    symbol_count = max(url_len - digit_count - letter_count, 0)

    has_ip_pattern = 1.0 if _IP_PATTERN.search(url) else 0.0
    has_suspicious_keyword = 1.0 if any(k in url for k in _SUSPICIOUS_KEYWORDS) else 0.0

    return [
        float(url_len),
        float(digit_count),
        float(letter_count),
        float(symbol_count),
        float(url.count(".")),
        float(url.count("-")),
        float(url.count("_")),
        float(url.count("/")),
        float(url.count("?")),
        float(url.count("=")),
        float(url.count("@")),
        float(url.startswith("https")),
        has_ip_pattern,
        has_suspicious_keyword,
        float(digit_count / url_len),
        float(symbol_count / url_len),
    ]


def _stratified_split(
    samples: Sequence[Tuple[str, int]], test_size: float, random_state: int
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    grouped = {0: [], 1: []}
    for sample in samples:
        grouped[sample[1]].append(sample)

    rng = random.Random(random_state)
    train_samples: List[Tuple[str, int]] = []
    test_samples: List[Tuple[str, int]] = []

    for label in (0, 1):
        group = grouped[label]
        rng.shuffle(group)
        if not group:
            continue
        raw_n_test = int(round(len(group) * test_size))
        if len(group) > 1:
            n_test = min(max(raw_n_test, 1), len(group) - 1)
        else:
            n_test = 1
        test_samples.extend(group[:n_test])
        train_samples.extend(group[n_test:])

    rng.shuffle(train_samples)
    rng.shuffle(test_samples)
    return train_samples, test_samples


def load_and_split(
    csv_path: Path = DEFAULT_DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[List[float]], List[List[float]], List[int], List[int]]:
    """Load data, extract features, and create a stratified train/test split."""
    samples = load_dataset(csv_path)
    train_samples, test_samples = _stratified_split(samples, test_size, random_state)

    X_train = [build_url_features(url) for url, _ in train_samples]
    y_train = [label for _, label in train_samples]
    X_test = [build_url_features(url) for url, _ in test_samples]
    y_test = [label for _, label in test_samples]
    return X_train, X_test, y_train, y_test
