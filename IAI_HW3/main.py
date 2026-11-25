"""Train a three-class sentiment classifier on the HW3 labeled dataset."""

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
from itertools import combinations
import json
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile
from typing import Optional

import numpy as np
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


@dataclass(frozen=True)
class MemberSpec:
    name: str
    seed: int
    head: str = "mlp"
    base_model: str = BASE_MODEL
    boost_class: Optional[str] = None
    subsample: float = 1.0


PRESETS = {
    "legacy": [
        MemberSpec("negative_cnn", 42, "cnn", boost_class="negative"),
        MemberSpec("neutral_mlp", 43, "mlp", boost_class="neutral"),
        MemberSpec("positive_mlp", 44, "mlp", boost_class="positive"),
    ],
    "diverse": [
        MemberSpec("cls_mlp", 42, "mlp"),
        MemberSpec("mean_pool", 43, "mean"),
        MemberSpec("cls_mean", 44, "clsmean"),
    ],
    # Opt in only if the assignment allows the cased BERT-base checkpoint.
    "diverse-cased": [
        MemberSpec("cls_mlp", 42, "mlp"),
        MemberSpec("mean_pool", 43, "mean"),
        MemberSpec("cased_cls_mean", 44, "clsmean", base_model="google-bert/bert-base-cased"),
    ],
}


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


def masked_mean(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


class MeanPoolHead(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, len(LABELS))

    def forward(self, hidden, attention_mask):
        return self.classifier(self.dropout(masked_mean(hidden, attention_mask)))


class ClsMeanHead(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden_size, len(LABELS))

    def forward(self, hidden, attention_mask):
        pooled = torch.cat([hidden[:, 0], masked_mean(hidden, attention_mask)], dim=-1)
        return self.classifier(self.dropout(pooled))


class SentimentModel(nn.Module):
    def __init__(self, vocab_size, head_type="mlp", base_model=BASE_MODEL):
        super().__init__()
        self.encoder = BertModel.from_pretrained(base_model, add_pooling_layer=False)
        self.encoder.resize_token_embeddings(vocab_size)
        if head_type == "cnn":
            self.head = CNNHead(self.encoder.config.hidden_size)
            return
        if head_type == "mean":
            self.head = MeanPoolHead(self.encoder.config.hidden_size)
            return
        if head_type == "clsmean":
            self.head = ClsMeanHead(self.encoder.config.hidden_size)
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
        if isinstance(self.head, (CNNHead, MeanPoolHead, ClsMeanHead)):
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
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--select", choices=("last", "best-f1"), default="last",
                        help="Keep the last epoch or select by validation macro-F1.")
    parser.add_argument("--patience", type=int, default=2, help="Early-stopping patience for --select best-f1.")
    parser.add_argument("--class-weighting", choices=("sqrt", "inverse", "none"), default="sqrt",
                        help="Training loss weighting based on training split class frequencies.")
    parser.add_argument("--fgsm-epsilon", type=float, default=0.0,
                        help="Embedding perturbation strength for optional FGSM training.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--balanced-sampling", action="store_true",
                        help="Sample training examples inversely to class frequency.")
    parser.add_argument("--ensemble", action="store_true",
                        help="Train the three members of the selected preset.")
    parser.add_argument("--preset", choices=PRESETS, default="legacy",
                        help="Ensemble member preset. Use diverse-cased only if permitted.")
    parser.add_argument("--subsample", type=float,
                        help="Override the stratified training fraction for every member.")
    parser.add_argument("--schedule", choices=("linear", "cosine"), default="linear")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--split-file", type=Path, default="splits.json",
                        help="Load or create frozen split indices at this path.")
    parser.add_argument("--run-dir", type=Path, help="Save run configuration, probabilities, and predictions.")
    parser.add_argument("--final-fit", action="store_true",
                        help="Refit on the labeled train, validation, and test rows for submission.")
    return run(**vars(parser.parse_args()))


def run(*, epochs=3, patience=2, class_weighting="sqrt", fgsm_epsilon=0.0,
        batch_size=16, balanced_sampling=False, ensemble=False, max_length=128,
        checkpoint_dir=None, schedule="linear", distinguish_mentions=False, review_path=None,
        split_file="splits.json", select="last", preset="legacy", subsample=None, run_dir=None,
        final_fit=False):
    """Train BERT and return benchmark predictions and training histories."""
    run_args = locals().copy()
    if schedule not in ("linear", "cosine"):
        raise ValueError("Schedule must be linear or cosine.")
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
    if not 0 <= fgsm_epsilon < float("inf"):
        raise ValueError("FGSM epsilon must be finite and nonnegative.")
    if select not in ("last", "best-f1"):
        raise ValueError("Epoch selection must be last or best-f1.")
    if preset not in PRESETS:
        raise ValueError(f"Unknown ensemble preset: {preset}")
    if subsample is not None and not 0 < subsample <= 1:
        raise ValueError("Subsample must be greater than 0 and at most 1.")
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

    split_path = ROOT / split_file
    train_pos, val_pos, test_pos = load_or_create_splits(data, ROOT / "dataset/dataset.csv", split_path)
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
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(Path(__file__).read_bytes())
    fingerprint.update((ROOT / "sentiment_preprocessing.py").read_bytes())
    fingerprint.update((ROOT / "dataset/dataset.csv").read_bytes())
    fingerprint.update(repr((epochs, patience, batch_size, max_length, str(device),
                             distinguish_mentions)).encode())

    effective_class_priors = {}
    final_effective_class_priors = {}

    def fit(spec, frame, final=False):
        key = repr((class_weighting, balanced_sampling, fgsm_epsilon, schedule, select, final, spec))
        checkpoint = None
        if checkpoint_dir is not None:
            digest = fingerprint.copy()
            digest.update(repr((frame.index.tolist(), None if final else validation.index.tolist())).encode())
            digest.update(key.encode())
            checkpoint = checkpoint_dir / (digest.hexdigest() + ".pt")

        member_train = frame
        if spec.subsample < 1:
            member_train = frame.groupby("label", group_keys=False).sample(
                frac=spec.subsample, random_state=spec.seed,
            )
        counts = member_train["label"].value_counts().reindex(LABELS, fill_value=0)
        if counts.eq(0).any():
            raise ValueError(f"Training data for {spec.name} must include every sentiment class.")
        weights = torch.ones(len(LABELS), device=device)
        if class_weighting != "none":
            weights = torch.tensor(len(member_train) / (len(LABELS) * counts.to_numpy()),
                                   dtype=torch.float32, device=device)
            if class_weighting == "sqrt":
                weights = weights.sqrt()
        if spec.boost_class is not None:
            weights[LABEL_TO_ID[spec.boost_class]] *= 2
        if run_dir is not None:
            prior = counts.to_numpy() * weights.cpu().numpy()
            priors = final_effective_class_priors if final else effective_class_priors
            priors[spec.name] = (prior / prior.sum()).tolist()

        torch.manual_seed(spec.seed)
        tokenizer = AutoTokenizer.from_pretrained(spec.base_model)
        tokens = [COMPANY_TOKEN, GMAIL_TOKEN] + (["[OTHER_USER]"] if distinguish_mentions else [])
        tokenizer.add_special_tokens({"additional_special_tokens": tokens})
        options = dict(tokenizer=tokenizer, batch_size=batch_size, max_length=max_length,
                       distinguish_mentions=distinguish_mentions, clean=True)

        if checkpoint is not None and checkpoint.is_file():
            print(f"Reusing checkpoint | {spec} | final fit {final}", flush=True)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model = SentimentModel(len(tokenizer), head_type=spec.head, base_model=spec.base_model)
            model.load_state_dict(saved["model"])
            model.to(device)
            val_probs, history = saved["validation"], saved["history"]
        else:
            sampler = None
            if balanced_sampling:
                sampler = WeightedRandomSampler(member_train["label"].map(1 / counts).to_numpy(),
                                                len(member_train), replacement=True)
            train_loader = make_loader(member_train, **options, shuffle=True, sampler=sampler)
            val_loader = None if final else make_loader(validation, **options)
            print(f"Training {spec} | final fit {final} | encoder LR {ENCODER_LR:g} | "
                  f"head LR {HEAD_LR:g} | schedule {schedule}", flush=True)
            model, history = train_model(
                tokenizer, train_loader, val_loader, device, nn.CrossEntropyLoss(), weights,
                epochs=epochs, patience=patience, fgsm_epsilon=fgsm_epsilon, head_type=spec.head,
                encoder_lr=ENCODER_LR, head_lr=HEAD_LR, schedule=schedule, select=select,
                base_model=spec.base_model,
            )
            val_probs = None if final else predict_probabilities(model, val_loader, device)
            if checkpoint is not None:
                saved = {"model": {name: tensor.cpu() for name, tensor in model.state_dict().items()},
                         "validation": val_probs, "history": history}
                with NamedTemporaryFile(dir=checkpoint_dir, suffix=".tmp", delete=False) as file:
                    temporary = Path(file.name)
                try:
                    torch.save(saved, temporary)
                    temporary.replace(checkpoint)
                finally:
                    temporary.unlink(missing_ok=True)

        test_probs = None if final else predict_probabilities(model, make_loader(test, **options), device)
        benchmark_probs = predict_probabilities(model, make_loader(benchmark, **options), device)
        return val_probs, test_probs, benchmark_probs, history

    members = PRESETS[preset] if ensemble else [MemberSpec("bert", seed)]
    if ensemble:
        assert len(members) == 3, "assignment allows at most three models in the ensemble"
    if subsample is not None:
        members = [replace(spec, subsample=subsample) for spec in members]

    results = [fit(spec, train) for spec in members]
    histories = {spec.name: result[3] for spec, result in zip(members, results)}

    voting = VotingRule("equal soft")
    validation_probs, test_probs, benchmark_probs = (
        vote([result[i] for result in results], voting) for i in range(3)
    )
    if ensemble:
        names = [spec.name for spec in members]
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
    if final_fit:
        full = pd.concat([train, validation, test])
        final_results = [fit(spec, full, final=True) for spec in members]
        final_benchmark = vote([result[2] for result in final_results], voting)
        final_labels = final_benchmark.argmax(dim=1)
        final_predictions = benchmark[["id"]].assign(
            label=[LABELS[i] for i in final_labels.tolist()],
        )
        final_predictions.attrs["pooled_macro_f1"] = score
        final_predictions.attrs["evaluation_predictions"] = predictions

        agreement = (final_labels == benchmark_probs.argmax(dim=1)).float().mean().item()
        print(f"Final-fit vs evaluation-pass benchmark agreement: {agreement:.3f}")
        print("Final-fit predicted class shares:", {
            name: round((final_labels == i).float().mean().item(), 3)
            for i, name in enumerate(LABELS)
        })
        if agreement < 0.93:
            print("Warning: benchmark agreement below 0.93 suggests an unstable recipe.")

    if run_dir is not None:
        run_path = Path(run_dir)
        run_path.mkdir(parents=True, exist_ok=True)
        try:
            git_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            git_commit = ""
        config = {
            "args": run_args,
            "members": [asdict(spec) for spec in members],
            "git_commit": git_commit,
            "splits_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            "effective_class_priors": effective_class_priors,
        }
        arrays = {}
        for index, (name, frame) in enumerate((("val", validation), ("test", test), ("benchmark", benchmark))):
            ids = frame["id"].to_numpy()
            arrays[f"{name}_ids"] = ids.astype(str) if ids.dtype.hasobject else ids
            arrays[f"{name}_member_probs"] = torch.stack([result[index] for result in results]).numpy()
            if "label" in frame:
                arrays[f"{name}_labels"] = frame["label"].map(LABEL_TO_ID).to_numpy(dtype=np.int64)
        if final_fit:
            config["final_effective_class_priors"] = final_effective_class_priors
            arrays["final_benchmark_member_probs"] = torch.stack([result[2] for result in final_results]).numpy()
            final_predictions.to_csv(run_path / "predictions_final.csv", index=False)
        np.savez(run_path / "probs.npz", **arrays)
        predictions.to_csv(run_path / "predictions.csv", index=False)
        (run_path / "config.json").write_text(
            json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8",
        )
    return (final_predictions if final_fit else predictions), histories


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
                *, epochs, patience, fgsm_epsilon, head_type, encoder_lr, head_lr,
                schedule, select="last", base_model=BASE_MODEL):
    """Train the model, accepting None for val_loader when validation is unavailable."""
    if select not in ("last", "best-f1"):
        raise ValueError("Epoch selection must be last or best-f1.")
    keep_best = select == "best-f1" and val_loader is not None
    model = SentimentModel(len(tokenizer), head_type=head_type, base_model=base_model).to(device)
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
        val_metrics = (float("nan"), float("nan"))
        val_log = ""
        if val_loader is not None:
            val_metrics = run_epoch(model, val_loader, loss_fn, device)
            val_log = f" | val loss {val_metrics[0]:.4f}, macro-F1 {val_metrics[1]:.4f}"
        history.append((train_metrics, val_metrics))
        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss {train_metrics[0]:.4f}, macro-F1 {train_metrics[1]:.4f}{val_log}",
            flush=True,
        )
        if not keep_best:
            continue
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

    if keep_best:
        print(f"Best validation macro-F1 {best_f1:.4f} at epoch {best_epoch}.")
        model.load_state_dict(best_state)
    return model, history


if __name__ == "__main__":
    main()
