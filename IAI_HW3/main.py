"""Train a three-class sentiment classifier on the HW3 labeled dataset."""

import argparse
from dataclasses import dataclass
import hashlib
import html
from pathlib import Path
import re
import subprocess
from tempfile import NamedTemporaryFile

import emoji
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
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
COMPANY_TOKEN = "[CPY]"
GMAIL_TOKEN = "[GMAIL]"
LABELS = ("negative", "neutral", "positive")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ENCODER_LR = 1e-5
HEAD_LR = 5e-4


def preprocess(text, clean=False, url_token=False, distinguish_mentions=False):
    if clean:
        text = html.unescape(text)
        text = re.sub(r"Cancelled Flight(?:led|ed)", "cancelled", text, flags=re.I)
        text = re.sub(r"Late Flightr\b", "later", text, flags=re.I)
    text = re.sub(r"(?:https?://|www\.)\S+", " [URL] " if url_token else " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[\w.+%\-]+@gmail\.com(?![\w-]|\.[\w-])",
                  f" {GMAIL_TOKEN} ", text, flags=re.IGNORECASE)
    mentions = r"(?<!\w)@\w+"
    first = re.search(mentions, text) if distinguish_mentions else None
    if first:
        addressed = first.group().casefold()
        text = re.sub(mentions, lambda m: " [CPY] " if m.group().casefold() == addressed
                      else " [OTHER_USER] ", text)
    else:
        text = re.sub(mentions, f" {COMPANY_TOKEN} ", text)
    text = emoji.replace_emoji(
        text, replace=lambda symbol, data: " " + data["en"].strip(":").replace("_", " ") + " ",
    )
    return " ".join(text.split())


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
    parser.add_argument("--checkpoint-dir", type=Path, help="Reuse completed models from this directory.")
    parser.add_argument("--distinguish-mentions", action="store_true")
    parser.add_argument("--review-path", type=Path, help="Save validation neutral/negative disagreements here.")
    parser.add_argument("--epochs", type=int, default=5, help="Maximum number of epochs.")
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
    return run(**vars(parser.parse_args()))


def run(*, epochs=5, patience=2, class_weighting="sqrt", fgsm_epsilon=0.0,
        batch_size=16, balanced_sampling=False, ensemble=False, max_length=128,
        checkpoint_dir=None, schedule="linear", distinguish_mentions=False, review_path=None,
        sweep_path=None, tune_neutral=False):
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

    groups = data["text"].map(lambda text: preprocess(text).casefold())
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
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # The previous Colab version hashed the entire script into its filenames.
    # Its training recipe is compatible only with the original mention handling.
    sources = [Path(__file__).read_bytes()]
    if checkpoint_dir and any(checkpoint_dir.glob("*.pt")):
        compatible = ["57eaa643d4bc977cf8ffee1496c1acb8e0fbe183"]
        if not distinguish_mentions:
            compatible.append("236a5a13c653087e41a9e8a79f8fad60aa780546")
        for blob in compatible:
            try:
                old = subprocess.run(["git", "cat-file", "blob", blob], cwd=ROOT,
                                     capture_output=True, check=True).stdout
                sources.append(old)
            except (OSError, subprocess.CalledProcessError):
                continue
    data_bytes = (ROOT / "dataset" / "dataset.csv").read_bytes()
    run_key = repr((epochs, patience, batch_size, max_length, str(device),
                    train.index.tolist(), validation.index.tolist())).encode()
    fingerprints = []
    for source in sources:
        digest = hashlib.sha256(source)
        digest.update(data_bytes)
        digest.update(run_key)
        fingerprints.append(digest)

    def fit(model_seed, specialist, head_type):
        key = (class_weighting, balanced_sampling, fgsm_epsilon, model_seed, specialist, head_type, schedule)
        if distinguish_mentions:
            key += ("distinct_mentions",)
        checkpoint = None
        if checkpoint_dir is not None:
            paths = []
            for fingerprint in fingerprints:
                digest = fingerprint.copy()
                digest.update(repr(key).encode())
                paths.append(checkpoint_dir / (digest.hexdigest() + ".pt"))
            checkpoint = next((path for path in paths if path.is_file()), paths[0])
        torch.manual_seed(model_seed)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        tokens = [COMPANY_TOKEN, GMAIL_TOKEN] + (["[OTHER_USER]"] if distinguish_mentions else [])
        tokenizer.add_special_tokens({"additional_special_tokens": tokens})
        options = dict(tokenizer=tokenizer, batch_size=batch_size, max_length=max_length,
                       distinguish_mentions=distinguish_mentions)
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

    validation_members = [result[0] for result in results]
    voting = VotingRule("equal soft")
    if ensemble:
        voting = select_vote(validation_members, validation, targets, groups.loc[validation.index],
                             review_path, sweep_path, tune_neutral)
    validation_probs, test_probs, benchmark_probs = (
        vote([result[i] for result in results], voting) for i in range(3)
    )
    predicted = validation_probs.argmax(dim=1).tolist()
    print(classification_report(targets, predicted, labels=list(range(len(LABELS))),
                                target_names=LABELS, digits=4, zero_division=0))
    print(confusion_matrix(targets, predicted, labels=list(range(len(LABELS)))))
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


def select_vote(probabilities, validation, targets, groups, review_path, sweep_path, tune_neutral):
    targets = torch.tensor(targets)
    names = ("negative_cnn", "neutral_mlp", "positive_mlp")
    individual = [probs.argmax(dim=1) for probs in probabilities]
    for name, predictions in zip(names, individual):
        print(name, classification_report(targets, predictions, labels=[0, 1, 2],
                                          target_names=LABELS, digits=4, zero_division=0))
    for i, j in ((0, 1), (0, 2), (1, 2)):
        print(f"{names[i]} vs {names[j]}: {int((individual[i] != individual[j]).sum())}"
              f"/{len(targets)} validation predictions differ")
    cnn, neutral, positive = [pred == targets for pred in individual]
    print(f"CNN alone correct: {int((cnn & ~neutral & ~positive).sum())}; "
          f"both MLPs correct and CNN wrong: {int((~cnn & neutral & positive).sum())}")

    grid = [VotingRule("grid", (i / 10, j / 10, (10 - i - j) / 10))
            for i in range(11) for j in range(11 - i)]
    # Prefer the equal soft control on ties; among grid ties prefer balanced weights.
    grid.sort(key=lambda rule: sum((w - 1 / 3) ** 2 for w in rule.weights))
    rules = [VotingRule("equal soft"), VotingRule("hard"), *grid]
    predictions = {}

    def labels(rule):
        if rule not in predictions:
            predictions[rule] = vote(probabilities, rule).argmax(dim=1)
        return predictions[rule]

    def score(rule, rows):
        return f1_score(targets[rows], labels(rule)[rows], labels=[0, 1, 2],
                        average="macro", zero_division=0)

    def select(rows):
        best = max(rules, key=lambda rule: score(rule, rows))
        if not tune_neutral:
            return best
        # Keep 1.0 first so a tie preserves the unadjusted decision.
        candidates = [VotingRule(best.name, best.weights, m) for m in (1., .95, 1.05, .90, 1.10)]
        return max(candidates, key=lambda rule: score(rule, rows))

    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
                 .split(validation, targets.numpy(), groups))
    crossfit = torch.empty_like(targets)
    for k, (train, held) in enumerate(folds, 1):
        rule = select(train)
        crossfit[held] = labels(rule)[held]
        print(f"Fold {k} selected {rule} | held macro-F1 {score(rule, held):.4f}")
    all_rows = slice(None)
    crossfit_f1 = f1_score(targets, crossfit, labels=[0, 1, 2], average="macro", zero_division=0)
    print(f"Grouped out-of-fold voting selection {crossfit_f1:.4f} | "
          f"equal soft {score(rules[0], all_rows):.4f} | hard {score(rules[1], all_rows):.4f}")
    chosen = select(all_rows)
    selected = labels(chosen)
    if sweep_path is not None:
        records = []
        report_rules = list(rules)
        if tune_neutral:
            report_rules.extend(VotingRule(chosen.name, chosen.weights, m) for m in (.90, .95, 1.05, 1.10))
        for rule in report_rules:
            predicted = labels(rule)
            positive = predicted == LABEL_TO_ID["neutral"]
            actual = targets == LABEL_TO_ID["neutral"]
            correct = int((positive & actual).sum())
            weights = rule.weights or ((1/3,) * 3 if rule.name == "equal soft" else (None,) * 3)
            records.append(dict(rule=rule.name, cnn=weights[0], neutral_mlp=weights[1],
                                positive_mlp=weights[2], multiplier=rule.neutral_multiplier,
                                macro_f1=score(rule, all_rows), selected=rule == chosen,
                                neutral_precision=correct / max(1, int(positive.sum())),
                                neutral_recall=correct / max(1, int(actual.sum())),
                                **{f"fold_{k}": score(rule, held)
                                   for k, (_, held) in enumerate(folds, 1)}))
        path = Path(sweep_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(path, index=False)
    review = validation[["id", "text", "label"]].copy()
    for name, pred in zip(names, individual):
        review[name] = [LABELS[i] for i in pred.tolist()]
    review["ensemble"] = [LABELS[i] for i in selected.tolist()]
    errors = review[((targets != selected) & (targets < 2) & (selected < 2)).numpy()]
    print(f"Negative -> neutral: {int(((targets == 0) & (selected == 1)).sum())}; "
          f"neutral -> negative: {int(((targets == 1) & (selected == 0)).sum())}")
    if review_path is not None:
        path = Path(review_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        errors.to_csv(path, index=False)
        print(f"Saved {len(errors)} validation disagreements to {path}")
    return chosen


def train_model(tokenizer, train_loader, val_loader, device, loss_fn, class_weights,
                *, epochs, patience, fgsm_epsilon, head_type, encoder_lr, head_lr, schedule):
    model = SentimentModel(len(tokenizer), head_type=head_type).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": encoder_lr},
            {"params": model.head.parameters(), "lr": head_lr},
        ],
        weight_decay=0.01,
    )
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
