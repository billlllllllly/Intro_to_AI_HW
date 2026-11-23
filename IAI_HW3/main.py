"""Train a three-class sentiment classifier on the HW3 labeled dataset."""

import argparse
import html
from pathlib import Path
import re

import emoji
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, BertModel, get_linear_schedule_with_warmup


ROOT = Path(__file__).resolve().parent
BASE_MODEL = "google-bert/bert-base-uncased"
COMPANY_TOKEN = "[CPY]"
GMAIL_TOKEN = "[GMAIL]"
LABELS = ("negative", "neutral", "positive")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def preprocess(text, clean=False, url_token=False):
    if clean:
        text = html.unescape(text)
        text = re.sub(r"Cancelled Flight(?:led|ed)", "cancelled", text, flags=re.I)
        text = re.sub(r"Late Flightr\b", "later", text, flags=re.I)
    text = re.sub(r"(?:https?://|www\.)\S+", " [URL] " if url_token else " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[\w.+%\-]+@gmail\.com(?![\w-]|\.[\w-])",
                  f" {GMAIL_TOKEN} ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\w)@\w+", f" {COMPANY_TOKEN} ", text)
    text = emoji.replace_emoji(
        text, replace=lambda symbol, data: " " + data["en"].strip(":").replace("_", " ") + " ",
    )
    return " ".join(text.split())


def has_content(text):
    cleaned = preprocess(text).replace(COMPANY_TOKEN, "").replace(GMAIL_TOKEN, "")
    return any(character.isalnum() for character in cleaned)


class SentimentModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = BertModel.from_pretrained(BASE_MODEL, add_pooling_layer=False)
        self.encoder.resize_token_embeddings(vocab_size)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, len(LABELS)),
        )

    def forward(self, inputs, embeddings=None):
        if embeddings is not None:
            inputs = {name: tensor for name, tensor in inputs.items() if name != "input_ids"}
            inputs["inputs_embeds"] = embeddings
        hidden = self.encoder(**inputs).last_hidden_state
        return self.head(hidden[:, 0])


def make_loader(frame, tokenizer, batch_size, max_length, shuffle=False, sampler=None, clean=False, url_token=False):
    labels = frame["label"].map(LABEL_TO_ID) if "label" in frame else [-1] * len(frame)
    examples = list(zip(frame["text"], labels))

    def collate(batch):
        texts, labels = zip(*batch)
        inputs = tokenizer(
            [preprocess(text, clean, url_token) for text in texts],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return inputs, torch.tensor(labels, dtype=torch.long)

    return DataLoader(examples, batch_size=batch_size, shuffle=shuffle and sampler is None,
                      sampler=sampler, collate_fn=collate)


def run_epoch(model, loader, loss_fn, device, optimizer=None, scheduler=None,
              class_weights=None, fgsm_epsilon=0.0):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    predictions = []
    targets = []
    with torch.set_grad_enabled(training):
        for inputs, labels in loader:
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            labels = labels.to(device)
            if training:
                optimizer.zero_grad()
            embeddings = None
            if training and fgsm_epsilon > 0:
                embeddings = model.encoder.get_input_embeddings()(inputs["input_ids"])
            logits = model(inputs) if embeddings is None else model(inputs, embeddings)
            loss = loss_fn(logits, labels)
            if training:
                objective = loss
                if class_weights is not None:
                    objective = nn.functional.cross_entropy(logits, labels, weight=class_weights)
                if embeddings is not None:
                    gradient, = torch.autograd.grad(objective, embeddings, retain_graph=True)
                    perturbation = fgsm_epsilon * gradient.sign()
                    perturbation = perturbation * inputs["attention_mask"].unsqueeze(-1)
                    adversarial_logits = model(inputs, embeddings + perturbation)
                    adversarial_loss = nn.functional.cross_entropy(
                        adversarial_logits, labels, weight=class_weights,
                    )
                    objective = (objective + adversarial_loss) / 2
                objective.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            loss_sum += loss.item() * labels.size(0)
            predictions.extend(logits.detach().argmax(dim=1).tolist())
            targets.extend(labels.tolist())

    macro_f1 = f1_score(
        targets, predictions, labels=list(range(len(LABELS))),
        average="macro", zero_division=0,
    )
    return loss_sum / len(targets), macro_f1


@torch.no_grad()
def predict_probabilities(model, loader, device):
    model.eval()
    batches = []
    for inputs, _ in loader:
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        batches.append(model(inputs).softmax(dim=1).cpu())
    return torch.cat(batches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", action="store_true", help="Select preprocessing, weighting, FGSM and ensemble on validation data.")
    parser.add_argument("--epochs", type=int, default=5, help="Maximum number of epochs.")
    parser.add_argument("--patience", type=int, default=2, help="Stop after this many epochs without a new best validation macro-F1.")
    parser.add_argument("--class-weighting", choices=("sqrt", "inverse", "none"), default="sqrt",
                        help="Training loss weighting based on training split class frequencies.")
    parser.add_argument("--fgsm-epsilon", type=float, default=0.0,
                        help="Embedding perturbation strength for optional FGSM training.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--balanced-sampling", action="store_true",
                        help="Sample training examples inversely to class frequency.")
    parser.add_argument("--ensemble", action="store_true", help="Train three class specialists and combine their predictions.")
    parser.add_argument("--max-length", type=int, default=128)
    return run(**vars(parser.parse_args()))


def run(*, epochs=5, patience=2, class_weighting="sqrt", fgsm_epsilon=0.0,
        batch_size=16, balanced_sampling=False, ensemble=False, max_length=128, experiment=False):
    """Train BERT and return benchmark predictions and training histories in memory."""
    if not 0 <= fgsm_epsilon < float("inf"):
        raise ValueError("FGSM epsilon must be finite and nonnegative.")
    if patience < 1:
        raise ValueError("Patience must be positive.")
    if epochs < 1 or batch_size < 1 or not 3 <= max_length <= 512:
        raise ValueError("Use positive epochs and batch size, and a max length from 3 to 512.")
    if class_weighting not in ("sqrt", "inverse", "none"):
        raise ValueError("Class weighting must be sqrt, inverse, or none.")

    seed = 42
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pd.read_csv(ROOT / "dataset" / "dataset.csv", keep_default_na=False)
    if set(data["label"]) != set(LABELS):
        raise ValueError(f"Expected exactly these sentiment labels: {LABELS}")

    groups = data["text"].map(lambda text: preprocess(text, clean=experiment).casefold())
    outer_split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_ids, held_out_ids = next(outer_split.split(data, data["label"], groups))
    train, held_out = data.iloc[train_ids], data.iloc[held_out_ids]
    inner_split = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    val_ids, test_ids = next(
        inner_split.split(held_out, held_out["label"], groups.iloc[held_out_ids])
    )
    validation, test = held_out.iloc[val_ids], held_out.iloc[test_ids]
    keep = train["text"].map(has_content)
    removed = train.loc[~keep, "label"].value_counts().reindex(LABELS, fill_value=0)
    train = train.loc[keep]
    if set(train["label"]) != set(LABELS):
        raise ValueError("Filtering nearly empty posts left a training class without examples.")
    print(f"Removed nearly empty training posts by class: {removed.to_dict()}")
    print(f"Device: {device} | train: {len(train)} | validation: {len(validation)} | test: {len(test)}", flush=True)

    benchmark = pd.read_csv(ROOT / "dataset" / "benchmark.csv", keep_default_na=False)
    targets = validation["label"].map(LABEL_TO_ID).tolist()
    counts = train["label"].value_counts().reindex(LABELS)
    cache = {}

    def fit(config, model_seed, specialist=None, final=False):
        weighting, balanced, clean, url_token, epsilon = config
        key = (config, model_seed, specialist)
        if not final and key in cache:
            return cache[key]
        torch.manual_seed(model_seed)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        tokens = [COMPANY_TOKEN, GMAIL_TOKEN] + (["[URL]"] if url_token else [])
        tokenizer.add_special_tokens({"additional_special_tokens": tokens})
        options = dict(tokenizer=tokenizer, batch_size=batch_size, max_length=max_length,
                       clean=clean, url_token=url_token)
        sampler = None
        if balanced:
            sampler = WeightedRandomSampler(train["label"].map(1 / counts).to_numpy(),
                                            len(train), replacement=True)
        train_loader = make_loader(train, **options, shuffle=True, sampler=sampler)
        val_loader = make_loader(validation, **options)
        weights = torch.ones(len(LABELS), device=device)
        if weighting != "none":
            weights = torch.tensor(len(train) / (len(LABELS) * counts.to_numpy()),
                                   dtype=torch.float32, device=device)
            if weighting == "sqrt":
                weights = weights.sqrt()
        if specialist is not None:
            weights[LABEL_TO_ID[specialist]] *= 2
        print(f"Training {config}, seed {model_seed}, specialist {specialist}", flush=True)
        model, history = train_model(tokenizer, train_loader, val_loader, device,
                                     nn.CrossEntropyLoss(), weights, epochs=epochs,
                                     patience=patience, fgsm_epsilon=epsilon)
        val = predict_probabilities(model, val_loader, device)
        if final:
            return (val, predict_probabilities(model, make_loader(test, **options), device),
                    predict_probabilities(model, make_loader(benchmark, **options), device), history)
        cache[key] = val
        return val

    def validation_score(probs):
        predicted = probs.argmax(dim=1).tolist()
        score = f1_score(targets, predicted, average="macro", labels=[0, 1, 2], zero_division=0)
        print(classification_report(targets, predicted, labels=[0, 1, 2], target_names=LABELS,
                                    digits=4, zero_division=0))
        print(confusion_matrix(targets, predicted, labels=[0, 1, 2]))
        return score

    config = (class_weighting, balanced_sampling, False, False, fgsm_epsilon)
    members = [(seed + i, label) for i, label in enumerate(LABELS)] if ensemble else [(seed, None)]
    if experiment:
        def select(candidates):
            return max(candidates, key=lambda c: validation_score(fit(c, seed)))

        config = select([(w, b, False, False, 0.0) for w, b in
                         [("sqrt", False), ("inverse", False), ("none", True)]])
        config = select([config, (*config[:2], True, False, 0.0)])
        config = select([config, (*config[:3], True, 0.0)])
        config = select([config, (*config[:4], 0.001), (*config[:4], 0.003)])
        candidates = [[(seed, None)]]
        if ensemble:
            candidates.extend([[(seed + i, None) for i in range(3)], members])

        def ensemble_score(group):
            return validation_score(torch.stack([fit(config, n, label) for n, label in group]).mean(0))

        members = max(candidates, key=ensemble_score)
        print(f"Selected configuration {config} and members {members}", flush=True)

    results = [fit(config, n, label, final=True) for n, label in members]
    histories = {(label or ("bert" if len(members) == 1 else f"bert_seed_{n}")): result[3]
                 for (n, label), result in zip(members, results)}
    validation_probs, test_probs, benchmark_probs = (
        torch.stack([result[i] for result in results]).mean(0) for i in range(3)
    )
    for name, frame, scores in (
        ("Validation", validation, validation_probs), ("Held-out test", test, test_probs)
    ):
        score = f1_score(
            frame["label"].map(LABEL_TO_ID), scores.argmax(dim=1).tolist(),
            labels=list(range(len(LABELS))), average="macro", zero_division=0,
        )
        print(f"{name} macro-F1 {score:.4f}")

    predictions = benchmark[["id"]].assign(
        label=[LABELS[i] for i in benchmark_probs.argmax(dim=1).tolist()]
    )
    return predictions, histories


def train_model(tokenizer, train_loader, val_loader, device, loss_fn, class_weights,
                *, epochs, patience, fgsm_epsilon):
    model = SentimentModel(len(tokenizer)).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": 1e-5},
            {"params": model.head.parameters(), "lr": 5e-4},
        ],
        weight_decay=0.01,
    )
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=round(0.1 * total_steps), num_training_steps=total_steps,
    )

    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, loss_fn, device, optimizer, scheduler,
            class_weights, fgsm_epsilon,
        )
        val_metrics = run_epoch(model, val_loader, loss_fn, device)
        history.append((train_metrics, val_metrics))
        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss {train_metrics[0]:.4f}, macro-F1 {train_metrics[1]:.4f} | "
            f"val loss {val_metrics[0]:.4f}, macro-F1 {val_metrics[1]:.4f}",
            flush=True,
        )
        if val_metrics[1] > best_f1:
            best_f1 = val_metrics[1]
            best_epoch = epoch
            best_state = {
                name: tensor.detach().to(device="cpu", copy=True)
                for name, tensor in model.state_dict().items()
            }
        if epoch - best_epoch >= patience:
            print(f"Early stopping after epoch {epoch}: validation macro-F1 stalled.")
            break

    print(f"Best validation macro-F1 {best_f1:.4f} at epoch {best_epoch}.")
    model.load_state_dict(best_state)
    return model, history


if __name__ == "__main__":
    main()
