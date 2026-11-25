from itertools import combinations
from pathlib import Path

import pandas as pd
import pytest

from splits import load_or_create_splits, split_key


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset/dataset.csv"
SPLITS = ROOT / "splits.json"


def test_frozen_split_has_disjoint_rows_and_groups():
    data = pd.read_csv(DATASET, keep_default_na=False)
    partitions = load_or_create_splits(data, DATASET, SPLITS)

    assert tuple(map(len, partitions)) == (7200, 900, 900)
    assert sorted(index for part in partitions for index in part) == list(range(len(data)))
    groups = data["text"].map(split_key)
    partition_groups = [set(groups.iloc[part]) for part in partitions]
    for left, right in combinations(partition_groups, 2):
        assert left.isdisjoint(right)


def test_frozen_split_rejects_a_changed_dataset(tmp_path):
    changed_dataset = tmp_path / "dataset.csv"
    changed_dataset.write_bytes(DATASET.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="dataset.csv changed"):
        load_or_create_splits(None, changed_dataset, SPLITS)
