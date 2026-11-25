"""Train a three-class sentiment classifier on the HW3 labeled dataset."""

import argparse
from dataclasses import dataclass
import hashlib
from itertools import combinations
from pathlib import Path
from tempfile import NamedTemporaryFile

from sentiment_preprocessing import COMPANY_TOKEN, GMAIL_TOKEN, preprocess
from splits import load_or_create_splits
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    BertModel,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


ROOT = Path(__file__).resolve().parent
BASE_MODEL = "google-bert/bert-base-uncased"
LABELS = ("negative", "neutral", "positive")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ENCODER_LR = 2e-5
HEAD_LR = 2e-4


def has_content(text):
    cleaned = preprocess(text).replace(COMPANY_TOKEN, "").replace(GMAIL_TOKEN, "")
    return any(character.isalnum() for character in cleaned)


class CNNHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.widths = (2, 3, 4)
        channels = 64
        self.convolutions = nn.ModuleList(
            nn.Conv1d(hidden_size, channels, width) for width in self.widths
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(channels * len(self.widths), len(LABELS))

    def forward(self, hidden, attention_mask):
        padding = max(0, max(self.widths) - hidden.size(1))
        tokens = nn.functional.pad(hidden.transpose(1, 2), (0, padding))
        mask = nn.functional.pad(attention_mask.bool(), (0, padding), value=False)
        features = []
        for width, convolution in zip(self.widths, self.convolutions):
            activations = convolution(tokens).relu()
            valid = mask.unfold(1, width, 1).all(dim=-1).unsqueeze(1)
            pooled = activations.masked_fill(~valid, 0).amax(dim=-1)
            features.append(pooled)
        return self.classifier(self.dropout(torch.cat(features, dim=1)))


class SentimentModel(nn.Module):
    def __init__(self, vocab_size, head_type="mlp"):
        super().__init__()
        self.encoder = BertModel.from_pretrained(BASE_MODEL, add_pooling_layer=False)
        self.encoder.resize_token_embeddings(vocab_size)
        if head_type == "cnn":
            self.head = CNNHead(self.encoder.config.hidden_size)
            return
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
        if isinstance(self.head, CNNHead):
            return self.head(hidden, inputs["attention_mask"])
        return self.head(hidden[:, 0])


def make_loader(frame, tokenizer, batch_size, max_length, shuffle=False, sampler=None, clean=False, url_token=False, distinguish_mentions=False):
    labels = frame["label"].map(LABEL_TO_ID) if "label" in frame else [-1] * len(frame)
    examples = list(zip(frame["text"], labels))

    def collate(batch):
        texts, labels = zip(*batch)
        inputs = tokenizer(
            [preprocess(text, clean, url_token, distinguish_mentions) for text in texts],
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
                    objective = nn.functional.cross_entropy(logits, labels, weight=class_weights, label_smoothing=0.05)
                if embeddings is not None:
                    gradient, = torch.autograd.grad(objective, embeddings, retain_graph=True)
                    perturbation = fgsm_epsilon * gradient.sign()
                    perturbation = perturbation * inputs["attention_mask"].unsqueeze(-1)
                    adversarial_logits = model(inputs, embeddings + perturbation)
                    adversarial_loss = nn.functional.cross_entropy(
                        adversarial_logits, labels, weight=class_weights, label_smoothing=0.05,
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
    parser.add_argument("--checkpoint-dir", type=Path, help="Reuse completed models from this directory.")
    parser.add_argument("--distinguish-mentions", action="store_true")
    parser.add_argument("--review-path", type=Path, help="Save validation neutral/negative disagreements here.")
    parser.add_argument("--epochs", type=int, default=3, help="Maximum number of epochs.")
    parser.add_argument("--patience", type=int, default=2, help="Stop after this many epochs without a new best validation macro-F1.")
    parser.add_argument("--class-weighting", choices=("sqrt", "inverse", "none"), default="sqrt",
                        help="Training loss weighting based on training split class frequencies.")
    parser.add_argument("--fgsm-epsilon", type=float, default=0.0,
                        help="Embedding perturbation strength for optional FGSM training.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--balanced-sampling", action="store_true",
                        help="Sample training examples inversely to class frequency.")
    parser.add_argument("--ensemble", action="store_true",
                        help="Train one CNN specialist and two MLP specialists.")
    parser.add_argument("--schedule", choices=("linear", "cosine"), default="linear")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--split-file", type=Path, default="splits.json",
                        help="Load or create frozen split indices at this path.")
    return run(**vars(parser.parse_args()))


def run(*, epochs=3, patience=2, class_weighting="sqrt", fgsm_epsilon=0.0,
        batch_size=16, balanced_sampling=False, ensemble=False, max_length=128,
        checkpoint_dir=None, schedule="linear", distinguish_mentions=False, review_path=None,
        split_file="splits.json"):
    """Train BERT and return benchmark predictions and training histories."""
    if schedule not in ("linear", "cosine"):
        raise ValueError("Schedule must be linear or cosine.")
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
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

    train_pos, val_pos, test_pos = load_or_create_splits(
        data, ROOT / "dataset/dataset.csv", ROOT / split_file,
    )
    train, validation, test = data.iloc[train_pos], data.iloc[val_pos], data.iloc[test_pos]
    keep = train["text"].map(has_content)
    removed = train.loc[~keep, "label"].value_counts().reindex(LABELS, fill_value=0)
    train = train.loc[keep]
    if set(train["label"]) != set(LABELS):
        raise ValueError("Filtering nearly empty posts left a training class without examples.")
    print(f"Removed nearly empty training posts by class: {removed.to_dict()}")
    print(f"Device: {device} | train: {len(train_pos)} | validation: {len(validation)} | test: {len(test)}", flush=True)

    benchmark = pd.read_csv(ROOT / "dataset" / "benchmark.csv", keep_default_na=False)
    targets = validation["label"].map(LABEL_TO_ID).tolist()
    counts = train["label"].value_counts().reindex(LABELS)
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(Path(__file__).read_bytes())
    fingerprint.update((ROOT / "sentiment_preprocessing.py").read_bytes())
    fingerprint.update((ROOT / "dataset/dataset.csv").read_bytes())
    fingerprint.update(repr((epochs, patience, batch_size, max_length, str(device),
                             train.index.tolist(), validation.index.tolist())).encode())

    def fit(model_seed, specialist, head_type):
        key = (class_weighting, balanced_sampling, fgsm_epsilon, model_seed, specialist, head_type, schedule)
        if distinguish_mentions:
            key += ("distinct_mentions",)
        checkpoint = None
        if checkpoint_dir is not None:
            digest = fingerprint.copy()
            digest.update(repr(key).encode())
            checkpoint = checkpoint_dir / (digest.hexdigest() + ".pt")
        torch.manual_seed(model_seed)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        tokens = [COMPANY_TOKEN, GMAIL_TOKEN] + (["[OTHER_USER]"] if distinguish_mentions else [])
        tokenizer.add_special_tokens({"additional_special_tokens": tokens})
        options = dict(tokenizer=tokenizer, batch_size=batch_size, max_length=max_length,
                       distinguish_mentions=distinguish_mentions, clean=True)
        val_loader = make_loader(validation, **options)
        if checkpoint is not None and checkpoint.is_file():
            print(f"Reusing {head_type.upper()} checkpoint | seed {model_seed} | specialist {specialist}", flush=True)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model = SentimentModel(len(tokenizer), head_type=head_type)
            model.load_state_dict(saved["model"])
            model.to(device)
            return (saved["validation"],
                    predict_probabilities(model, make_loader(test, **options), device),
                    predict_probabilities(model, make_loader(benchmark, **options), device),
                    saved["history"])

        sampler = None
        if balanced_sampling:
            sampler = WeightedRandomSampler(train["label"].map(1 / counts).to_numpy(),
                                            len(train), replacement=True)
        train_loader = make_loader(train, **options, shuffle=True, sampler=sampler)
        weights = torch.ones(len(LABELS), device=device)
        if class_weighting != "none":
            weights = torch.tensor(len(train) / (len(LABELS) * counts.to_numpy()),
                                   dtype=torch.float32, device=device)
            if class_weighting == "sqrt":
                weights = weights.sqrt()
        if specialist is not None:
            weights[LABEL_TO_ID[specialist]] *= 2
        print(f"Training {head_type.upper()} head | encoder LR {ENCODER_LR:g} | head LR {HEAD_LR:g} | "
              f"schedule {schedule} | seed {model_seed} | specialist {specialist}", flush=True)
        model, history = train_model(tokenizer, train_loader, val_loader, device,
                                     nn.CrossEntropyLoss(), weights, epochs=epochs,
                                     patience=patience, fgsm_epsilon=fgsm_epsilon, head_type=head_type,
                                     encoder_lr=ENCODER_LR, head_lr=HEAD_LR, schedule=schedule)
        val = predict_probabilities(model, val_loader, device)
        if checkpoint is not None:
            saved = {"model": {name: tensor.cpu() for name, tensor in model.state_dict().items()},
                     "validation": val, "history": history}
            with NamedTemporaryFile(dir=checkpoint_dir, suffix=".tmp", delete=False) as file:
                temporary = Path(file.name)
            try:
                torch.save(saved, temporary)
                temporary.replace(checkpoint)
            finally:
                temporary.unlink(missing_ok=True)
        return (val, predict_probabilities(model, make_loader(test, **options), device),
                predict_probabilities(model, make_loader(benchmark, **options), device), history)

    if ensemble:
        members = [(seed, "negative", "cnn"), (seed + 1, "neutral", "mlp"),
                   (seed + 2, "positive", "mlp")]
    else:
        members = [(seed, None, "mlp")]
    results = [fit(*member) for member in members]
    histories = {label or "bert": result[3]
                 for (_, label, _), result in zip(members, results)}

    voting = VotingRule("equal soft")
    validation_probs, test_probs, benchmark_probs = (
        vote([result[i] for result in results], voting) for i in range(3)
    )
    if ensemble:
        names = [f"{label}_{head}" for _, label, head in members]
        report_members([result[0] for result in results], names, validation, targets,
                       validation_probs, review_path)
    predicted = validation_probs.argmax(dim=1).tolist()
    print(classification_report(targets, predicted, labels=list(range(len(LABELS))),
                                target_names=LABELS, digits=4, zero_division=0))
    print(confusion_matrix(targets, predicted, labels=list(range(len(LABELS)))))
    for name, frame, scores in (
        ("Validation", validation, validation_probs), ("Held-out test", test, test_probs),
        ("Pooled validation + test", pd.concat([validation, test]),
         torch.cat([validation_probs, test_probs]))
    ):
        score = f1_score(
            frame["label"].map(LABEL_TO_ID), scores.argmax(dim=1).tolist(),
            labels=list(range(len(LABELS))), average="macro", zero_division=0,
        )
        print(f"{name} macro-F1 {score:.4f}")

    predictions = benchmark[["id"]].assign(
        label=[LABELS[i] for i in benchmark_probs.argmax(dim=1).tolist()]
    )
    predictions.attrs["pooled_macro_f1"] = score
    return predictions, histories


@dataclass(frozen=True)
class VotingRule:
    name: str
    weights: tuple = ()
    neutral_multiplier: float = 1.0


def vote(probabilities, rule):
    members = torch.stack(probabilities)
    if rule.weights:
        weights = members.new_tensor(rule.weights).view(-1, 1, 1)
        scores = (members * weights).sum(dim=0)
    else:
        scores = members.mean(dim=0)
    if rule.name == "hard":
        counts = nn.functional.one_hot(members.argmax(dim=2), num_classes=len(LABELS)).sum(dim=0)
        scores = scores.masked_fill(counts != counts.amax(dim=1, keepdim=True), -torch.inf)
    scores[:, LABEL_TO_ID["neutral"]] *= rule.neutral_multiplier
    return scores


def report_members(probabilities, names, validation, targets, ensemble_probs, review_path):
    targets = torch.tensor(targets)
    individual = [probs.argmax(dim=1) for probs in probabilities]
    for name, predictions in zip(names, individual):
        print(name, classification_report(targets, predictions, labels=list(range(len(LABELS))),
                                          target_names=LABELS, digits=4, zero_division=0))
    for i, j in combinations(range(len(names)), 2):
        print(f"{names[i]} vs {names[j]}: {int((individual[i] != individual[j]).sum())}"
              f"/{len(targets)} validation predictions differ")

    correct = torch.stack(individual) == targets
    correct_count = correct.sum(dim=0)
    for name, member_correct in zip(names, correct):
        alone = (member_correct & (correct_count == 1)).sum().item()
        others = (~member_correct & (correct_count == len(names) - 1)).sum().item()
        print(f"{name} alone correct: {alone} | both others correct and {name} wrong: {others}")

    selected = ensemble_probs.argmax(dim=1)
    negative, neutral = LABEL_TO_ID["negative"], LABEL_TO_ID["neutral"]
    negative_to_neutral = (targets == negative) & (selected == neutral)
    neutral_to_negative = (targets == neutral) & (selected == negative)
    print(f"Negative -> neutral: {negative_to_neutral.sum().item()} | "
          f"neutral -> negative: {neutral_to_negative.sum().item()}")
    if review_path is not None:
        errors = negative_to_neutral | neutral_to_negative
        review = validation.loc[errors.numpy(), ["id", "text", "label"]].copy()
        for name, predictions in zip(names, individual):
            review[name] = [LABELS[i] for i in predictions[errors].tolist()]
        review["ensemble"] = [LABELS[i] for i in selected[errors].tolist()]
        path = Path(review_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        review.to_csv(path, index=False)
        print(f"Saved {len(review)} validation disagreements to {path}")


def train_model(tokenizer, train_loader, val_loader, device, loss_fn, class_weights,
                *, epochs, patience, fgsm_epsilon, head_type, encoder_lr, head_lr, schedule):
    model = SentimentModel(len(tokenizer), head_type=head_type).to(device)
    from fold_training import optimizer_for
    optimizer = optimizer_for(model, encoder_lr, head_lr, layer_decay=0.9)
    total_steps = epochs * len(train_loader)
    make_scheduler = (get_cosine_schedule_with_warmup if schedule == "cosine"
                      else get_linear_schedule_with_warmup)
    scheduler = make_scheduler(
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
