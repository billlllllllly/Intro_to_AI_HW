"""Re-score the saved, unchanged fold-bagged baseline. Never trains a model."""
import hashlib
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoTokenizer, BertForMaskedLM

from fold_training import MeanClassifier, fit_offsets, holdout_split, log_probs
from main import BASE_MODEL, COMPANY_TOKEN, GMAIL_TOKEN, LABEL_TO_ID, ROOT
from main import has_content, make_loader, predict_probabilities, preprocess


def baseline_score(checkpoint_dir):
    directory = Path(checkpoint_dir)
    if not directory.is_dir() or not any(directory.glob("fold_*.pt")):
        print("Exact pooled baseline unavailable: previous fold checkpoints are missing.")
        return None
    data = pd.read_csv(ROOT / "dataset/dataset.csv", keep_default_na=False)
    groups = data.text.map(lambda text: preprocess(text).casefold())
    dev_ids, test_ids = holdout_split(data, groups)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    _, held = next(outer.split(data, data.label, groups))
    dev, test = data.iloc[dev_ids], data.iloc[test_ids]
    val_mask = np.isin(dev_ids, held)
    targets = dev.label.map(LABEL_TO_ID).to_numpy()
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
                 .split(dev, targets, groups.iloc[dev_ids]))
    signature = hashlib.sha256()
    # Original Colab checkpoint filenames include the exact three source blobs.
    for blob in ("a3c189563b5ed9bca70acfe9318503b3591cbb9a",
                 "495f6e99f357a387d09da1659e80526011c6c3a0",
                 "2ccffa163359438ccc1ec7a92cc445c4e6d889f8"):
        try:
            signature.update(subprocess.check_output(["git", "cat-file", "blob", blob], cwd=ROOT))
        except (OSError, subprocess.CalledProcessError):
            print("Exact baseline unavailable: original source blobs missing in this checkout.")
            return None
    signature.update((ROOT / "dataset/dataset.csv").read_bytes())
    signature.update(repr((3, 2, 16, 1e-5, 1.5e-4, 0.9, 0.1)).encode())
    paths = []
    for k, (training, held_fold) in enumerate(folds, 1):
        train = dev.iloc[training]
        digest = signature.copy()
        digest.update(train.loc[train.text.map(has_content)].index.to_numpy().tobytes())
        digest.update(dev.iloc[held_fold].index.to_numpy().tobytes())
        paths.append(directory / f"fold_{k}_{digest.hexdigest()[:24]}.pt")
    if not all(path.is_file() for path in paths):
        print("Exact baseline unavailable: missing matching fold checkpoints for this split.")
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.add_special_tokens({"additional_special_tokens": [COMPANY_TOKEN, GMAIL_TOKEN]})
    options = dict(tokenizer=tokenizer, batch_size=16, max_length=128, clean=False)
    oof, test_members = torch.empty(len(dev), 3), []
    for path, (_, held_fold) in zip(paths, folds):
        encoder = BertForMaskedLM.from_pretrained(BASE_MODEL).bert
        encoder.resize_token_embeddings(len(tokenizer))
        model = MeanClassifier(encoder)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
        model.to(device)
        oof[held_fold] = predict_probabilities(model, make_loader(dev.iloc[held_fold], **options), device)
        test_members.append(predict_probabilities(model, make_loader(test, **options), device))
        del model, encoder
    logs = log_probs(oof)
    crossfit = np.empty(len(dev), dtype=int)
    for training, held_fold in folds:
        offsets = fit_offsets(logs[training], targets[training])
        crossfit[held_fold] = (logs[held_fold] + offsets).argmax(1)
    test_pred = (log_probs(torch.stack(test_members).mean(0)) + fit_offsets(logs, targets)).argmax(1)
    val_test = np.r_[targets[val_mask], test.label.map(LABEL_TO_ID).to_numpy()]
    score = f1_score(val_test, np.r_[crossfit[val_mask], test_pred],
                     labels=[0, 1, 2], average="macro", zero_division=0)
    print(f"Unchanged fold-bagged pooled validation + test macro-F1 {score:.6f}")
    return score
