"""Three fold-bagged BERTs with fold-local MLM adaptation and OOF calibration."""
import hashlib
from itertools import product
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling
from transformers import get_linear_schedule_with_warmup

from main import (BASE_MODEL, LABELS, LABEL_TO_ID, ROOT, has_content, make_loader,
                  predict_probabilities, run_epoch)
from sentiment_preprocessing import COMPANY_TOKEN, GMAIL_TOKEN, preprocess


class MeanClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(encoder.config.hidden_size, len(LABELS))

    def forward(self, inputs):
        hidden = self.encoder(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.head(self.dropout(pooled))


def adapt(mlm, tokenizer, texts, epochs, batch_size, device):
    if not epochs:
        return
    encoded = tokenizer([preprocess(t) for t in texts], max_length=128, truncation=True,
                        return_special_tokens_mask=True)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15)
    loader = DataLoader([encoded["input_ids"][i] for i in range(len(texts))],
                        batch_size=batch_size, shuffle=True, collate_fn=collator)
    optimizer = torch.optim.AdamW(mlm.parameters(), lr=2e-5, weight_decay=0.01)
    mlm.train()
    for epoch in range(epochs):
        total, tokens = 0.0, 0
        for batch in loader:
            count = int(batch["labels"].ne(-100).sum())
            if not count:
                continue
            optimizer.zero_grad()
            loss = mlm(**{k: v.to(device) for k, v in batch.items()}).loss
            loss.backward()
            nn.utils.clip_grad_norm_(mlm.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * count
            tokens += count
        print(f"MLM epoch {epoch+1}: masked-token loss {total/max(tokens, 1):.4f}", flush=True)


def optimizer_for(model, encoder_lr, head_lr, layer_decay):
    layers = [model.encoder.embeddings, *model.encoder.encoder.layer]
    groups = [{"params": layer.parameters(), "lr": encoder_lr * layer_decay ** depth}
              for depth, layer in enumerate(reversed(layers))]
    groups.append({"params": model.head.parameters(), "lr": head_lr})
    return torch.optim.AdamW(groups, weight_decay=0.01)


def holdout_split(data, groups):
    # Reproduce the previous test split for comparable reporting.
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train, held = next(outer.split(data, data.label, groups))
    inner = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
    val, test = next(inner.split(data.iloc[held], data.label.iloc[held], groups.iloc[held]))
    return np.sort(np.concatenate([train, held[val]])), held[test]


def fit_offsets(log_probs, targets):
    # Negative is the reference. Prefer smaller offsets on equal macro-F1.
    grid = sorted(product((0., -.1, .1, -.2, .2), repeat=2),
                  key=lambda pair: sum(abs(value) for value in pair))
    best, best_score = np.zeros(3), -1.0
    for neutral, positive in grid:
        offsets = np.array([0., neutral, positive])
        pred = (log_probs + offsets).argmax(1)
        score = f1_score(targets, pred, labels=[0, 1, 2], average="macro", zero_division=0)
        if score > best_score:
            best, best_score = offsets, score
    return best


def log_probs(probabilities):
    return probabilities.clamp_min(1e-12).log().numpy()


def run_bagged(*, checkpoint_dir, review_path, epochs=3, mlm_epochs=2, batch_size=16,
               encoder_lr=1e-5, head_lr=1.5e-4, layer_decay=0.9, smoothing=0.1):
    if epochs < 1 or mlm_epochs < 0 or batch_size < 1 or not 0 < layer_decay <= 1:
        raise ValueError("Invalid epoch count, batch size, or layer decay")
    if min(encoder_lr, head_lr) <= 0 or not 0 <= smoothing < 1:
        raise ValueError("Invalid learning rate or label smoothing")
    data = pd.read_csv(ROOT / "dataset/dataset.csv", keep_default_na=False)
    benchmark = pd.read_csv(ROOT / "dataset/benchmark.csv", keep_default_na=False)
    groups = data.text.map(lambda t: preprocess(t).casefold())
    dev_ids, test_ids = holdout_split(data, groups)
    dev, test = data.iloc[dev_ids], data.iloc[test_ids]
    targets = dev.label.map(LABEL_TO_ID).to_numpy()
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
                 .split(dev, targets, groups.iloc[dev_ids]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.add_special_tokens({"additional_special_tokens": [COMPANY_TOKEN, GMAIL_TOKEN]})
    options = dict(tokenizer=tokenizer, batch_size=batch_size, max_length=128)
    output_dir = Path(checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256()
    for path in (Path(__file__), ROOT / "main.py", ROOT / "sentiment_preprocessing.py",
                 ROOT / "dataset/dataset.csv"):
        signature.update(path.read_bytes())
    signature.update(repr((epochs, mlm_epochs, batch_size, encoder_lr, head_lr,
                           layer_decay, smoothing)).encode())
    oof = torch.empty(len(dev), len(LABELS))
    test_probs, benchmark_probs, histories = [], [], {}
    print(f"Development: {len(dev)}, held-out test: {len(test)}", flush=True)
    for fold, (train_ids, valid_ids) in enumerate(folds):
        train = dev.iloc[train_ids]
        train = train.loc[train.text.map(has_content)]
        valid = dev.iloc[valid_ids]
        name = f"fold_{fold+1}"
        digest = signature.copy()
        digest.update(train.index.to_numpy().tobytes())
        digest.update(valid.index.to_numpy().tobytes())
        path = output_dir / f"{name}_{digest.hexdigest()[:24]}.pt"
        torch.manual_seed(42 + fold)
        print(f"{name}: {len(train)} training, {len(valid)} out-of-fold", flush=True)
        mlm = BertForMaskedLM.from_pretrained(BASE_MODEL)
        mlm.resize_token_embeddings(len(tokenizer))
        if not path.exists():
            adapt(mlm.to(device), tokenizer, train.text, mlm_epochs, batch_size, device)
        model = MeanClassifier(mlm.bert).to(device)
        del mlm
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(saved["model"])
            history = saved["history"]
            del saved
        else:
            loader = make_loader(train, **options, shuffle=True)
            validation_loader = make_loader(valid, **options)
            counts = train.label.value_counts().reindex(LABELS)
            weights = torch.tensor(np.sqrt(len(train) / (3 * counts.to_numpy())),
                                   dtype=torch.float32, device=device)
            objective = nn.CrossEntropyLoss(weight=weights, label_smoothing=smoothing)
            optimizer = optimizer_for(model, encoder_lr, head_lr, layer_decay)
            steps = epochs * len(loader)
            scheduler = get_linear_schedule_with_warmup(optimizer, round(0.1 * steps), steps)
            history = []
            for epoch in range(epochs):
                train_metrics = run_epoch(model, loader, objective, device, optimizer, scheduler)
                val_metrics = run_epoch(model, validation_loader, objective, device)
                history.append((train_metrics, val_metrics))
                print(f"{name} epoch {epoch+1}: train F1 {train_metrics[1]:.4f}, "
                      f"fold F1 {val_metrics[1]:.4f}", flush=True)
            del optimizer, scheduler
            with NamedTemporaryFile(dir=output_dir, delete=False) as file:
                tmp = Path(file.name)
            try:
                torch.save({"model": model.state_dict(), "history": history}, tmp)
                tmp.replace(path)
            finally:
                tmp.unlink(missing_ok=True)
        histories[name] = history
        oof[valid_ids] = predict_probabilities(model, make_loader(valid, **options), device)
        test_probs.append(predict_probabilities(model, make_loader(test, **options), device))
        benchmark_probs.append(predict_probabilities(model, make_loader(benchmark, **options), device))
        del model

    logs = log_probs(oof)
    crossfit = np.empty(len(dev), dtype=int)
    for train_ids, valid_ids in folds:
        offsets = fit_offsets(logs[train_ids], targets[train_ids])
        crossfit[valid_ids] = (logs[valid_ids] + offsets).argmax(1)
    offsets = fit_offsets(logs, targets)
    test_mean = torch.stack(test_probs).mean(0)
    test_targets = test.label.map(LABEL_TO_ID).to_numpy()
    for name, truth, pred in (
        ("Raw OOF", targets, logs.argmax(1)),
        ("Cross-calibrated OOF", targets, crossfit),
        ("Raw held-out ensemble", test_targets, test_mean.argmax(1).numpy()),
        ("Calibrated held-out ensemble", test_targets, (log_probs(test_mean) + offsets).argmax(1)),
    ):
        print(f"{name} macro-F1 {f1_score(truth, pred, average='macro', labels=[0,1,2], zero_division=0):.4f}")
    print(f"OOF logit offsets {dict(zip(LABELS, offsets))}")
    print(classification_report(targets, crossfit, labels=[0,1,2], target_names=LABELS, digits=4))
    print(confusion_matrix(targets, crossfit, labels=[0,1,2]))
    errors = dev[["id", "text", "label"]].copy()
    errors["prediction"] = np.asarray(LABELS)[crossfit]
    errors["p_neutral"] = oof[:, LABEL_TO_ID["neutral"]].numpy()
    errors = errors.loc[(targets != crossfit) & (targets < 2) & (crossfit < 2)]
    if review_path:
        path = Path(review_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        errors.to_csv(path, index=False)
    pred = (log_probs(torch.stack(benchmark_probs).mean(0)) + offsets).argmax(1)
    return benchmark[["id"]].assign(label=np.asarray(LABELS)[pred]), histories
