"""Frozen train/validation/test split. Never change split_key or SPLIT_SEED.

The committed splits.json was generated with scikit-learn 1.9.0 and emoji 2.16.0.
Later runs load the saved indices without depending on the generator version.
"""

import hashlib
import json
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

import emoji
from sklearn.model_selection import StratifiedGroupKFold

SPLIT_SEED = 42


def split_key(text: str) -> str:
    """FROZEN copy of the preprocessing used when the 0.8946 run was made.
    Do NOT import preprocess() here and do NOT edit this function."""
    text = re.sub(r"(?:https?://|www\.)\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[\w.+%\-]+@gmail\.com(?![\w-]|\.[\w-])", " [GMAIL] ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\w)@\w+", " [CPY] ", text)
    text = emoji.replace_emoji(
        text, replace=lambda symbol, data: " " + data["en"].strip(":").replace("_", " ") + " ")
    return " ".join(text.split()).casefold()


def make_splits(data):
    groups = data["text"].map(split_key)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)
    train_pos, held_pos = next(outer.split(data, data["label"], groups))
    held = data.iloc[held_pos]
    inner = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=SPLIT_SEED)
    val_rel, test_rel = next(inner.split(held, held["label"], groups.iloc[held_pos]))
    return {"train": sorted(map(int, train_pos)),
            "validation": sorted(map(int, held_pos[val_rel])),
            "test": sorted(map(int, held_pos[test_rel]))}


def load_or_create_splits(data, dataset_path, split_file):
    split_file = Path(split_file)
    digest = hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()
    if split_file.is_file():
        saved = json.loads(split_file.read_text(encoding="utf-8"))
        if saved["dataset_sha256"] != digest:
            raise ValueError("dataset.csv changed since splits.json was created; refusing to continue.")
        return saved["train"], saved["validation"], saved["test"]
    splits = make_splits(data)
    split_file.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=split_file.parent, delete=False) as file:
        temporary = Path(file.name)
    try:
        temporary.write_text(json.dumps({"dataset_sha256": digest, **splits}), encoding="utf-8")
        temporary.replace(split_file)
    finally:
        temporary.unlink(missing_ok=True)
    return splits["train"], splits["validation"], splits["test"]
