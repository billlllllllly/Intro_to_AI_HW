"""Train, evaluate, and package the HW3 airline-tweet sentiment system."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import platform
import re
import sys
import tempfile
import unicodedata
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import emoji
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader


AIRLINE_NAMES = {
    "americanair": "american airlines",
    "jetblue": "jetblue",
    "southwestair": "southwest airlines",
    "united": "united airlines",
    "usairways": "us airways",
    "virginamerica": "virgin america",
}
FORMAT_CUE_NAMES = (
    "airline", "length", "reply", "punctuation",
    "airline_length", "reply_punctuation", "length_punctuation",
)
FORMAT_LENGTH_CUTOFFS = (40, 70, 98, 119, 133, 139)
FORMAT_INTERACTIONS = ((0, 1), (2, 3), (1, 3))
QUOTE_TRANSLATION = str.maketrans(
    dict.fromkeys("‘’‚‛", "'") | dict.fromkeys("“”„‟", '"')
)
URL_PATTERN = r"""(?:https?://|www\.)[^\s<>"“”]+"""
HANDLE_PATTERN = r"(?<!\w)@[A-Za-z0-9_]+"
HASHTAG_PATTERN = r"(?<!\w)#(\w+)"
URL_RE = re.compile(URL_PATTERN, re.IGNORECASE)
HANDLE_RE = re.compile(HANDLE_PATTERN)
URL_OR_HANDLE_RE = re.compile(f"{URL_PATTERN}|{HANDLE_PATTERN}", re.IGNORECASE)
WORD_COUNT_EXCLUSIONS_RE = re.compile(
    f"{URL_PATTERN}|{HANDLE_PATTERN}|{HASHTAG_PATTERN}", re.IGNORECASE
)
HASHTAG_RE = re.compile(HASHTAG_PATTERN)
CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")
JOINED_TAG_RE = re.compile(r"[A-Za-z]{6,32}\Z")
QUOTE_RE = re.compile(r'\b(?:RT|MT)\s+@[A-Za-z0-9_]+\b|"\s*@[A-Za-z0-9_]+\b', re.IGNORECASE)
LEADING_RECIPIENTS_RE = re.compile(
    r"(?P<public>\.\s*)?(?P<recipients>@[A-Za-z0-9_]+(?:[ ,]+@[A-Za-z0-9_]+)*)"
)
ASCII_FACE_RE = re.compile(
    r"(?:"
    r"(?P<heart>(?:<|&lt;|&#0*60;|&#[xX]0*3[cC];)3)|"
    r"(?P<wink>;[-^]?[)\]dDpP])|"
    r"(?P<sad>[:=]'?[-^]?[(\[]|(?<!\w)[tT]_[tT]|;_;)|"
    r"(?P<playful>[:=][-^]?[pP])|"
    r"(?P<unsure>:[-^]?/)|"
    r"(?P<smile>[:=][-^]?[)\]dD]|\^_?\^)"
    r")(?![A-Za-z0-9_])"
)
ASCII_FACE_WORDS = {
    "heart": "heart", "wink": "winking face", "sad": "sad face",
    "playful": "playful face", "unsure": "unsure face", "smile": "smiling face",
}


def _canonical_tweet(text):
    return unicodedata.normalize("NFKC", html.unescape(text)).translate(QUOTE_TRANSLATION)


def _url_suffix(url):
    end = len(url)
    closing_pairs = {")": "(", "]": "[", "}": "{"}
    while end:
        last = url[end - 1]
        prefix = url[:end]
        if last in ".,!?;:'…":
            end -= 1
        elif last in closing_pairs and prefix.count(last) > prefix.count(closing_pairs[last]):
            end -= 1
        else:
            break
    return url[end:]


def _replace_urls(text, replacement):
    return URL_RE.sub(lambda match: replacement + _url_suffix(match.group()), text)


def _unicode_words(text):
    fragments = []
    for char in text:
        if not char.isascii() and unicodedata.category(char).startswith("S"):
            name = unicodedata.name(char, "")
            fragments.append(f" {name} " if name else char)
        elif char not in "\u200d\ufe0e\ufe0f":
            fragments.append(char)
    return "".join(fragments)


def normalize_text(text):
    """Normalize one airline tweet before BERT WordPiece tokenization."""
    text = _canonical_tweet(text)
    text = _replace_urls(text, " link")
    text = HANDLE_RE.sub(lambda match: " " + AIRLINE_NAMES.get(match.group()[1:].lower(), "user") + " ", text)
    text = HASHTAG_RE.sub(lambda match: " " + CAMEL_RE.sub(" ", match[1]).replace("_", " ") + " ", text)
    text = _unicode_words(text)
    return " ".join(text.lower().split())


def _tweet_format(text):
    original = _canonical_tweet(text)
    without_urls = _replace_urls(original, "")
    handles = [match.group()[1:].lower() for match in HANDLE_RE.finditer(without_urls)]
    airline = next((AIRLINE_NAMES[handle] for handle in handles if handle in AIRLINE_NAMES), "none")
    length = len(_replace_urls(original, "x" * 23))
    length_band = str(sum(length > cutoff for cutoff in FORMAT_LENGTH_CUTOFFS))

    stripped = without_urls.strip()
    quote = QUOTE_RE.search(stripped)
    recipients = LEADING_RECIPIENTS_RE.match(stripped)
    if quote:
        quoted_text = quote.group().startswith('"')
        after_quote = stripped.find('"', quote.end()) if quoted_text else -1
        has_comment = quote.start() > 0 or (after_quote >= 0 and bool(stripped[after_quote + 1:].strip()))
        reply = "quote_comment" if has_comment else "quote"
    elif recipients and recipients["recipients"].count("@") > 1:
        reply = "multiple_recipients"
    elif recipients:
        reply = "public_reply" if recipients["public"] else "reply"
    else:
        reply = "broadcast" if handles else "unaddressed"

    punctuation = "question" if "?" in without_urls else "exclamation" if "!" in without_urls else "plain"
    has_symbol = any(unicodedata.category(char).startswith("S") for char in without_urls)
    content = "link" if URL_RE.search(original) else "symbol" if has_symbol else "ordinary"
    return airline, length_band, reply, f"{punctuation}_{content}"


def _interaction_key(left, right):
    return json.dumps([left, right], separators=(",", ":"))


def _format_keys(text):
    main = _tweet_format(text)
    return main + tuple(_interaction_key(main[left], main[right]) for left, right in FORMAT_INTERACTIONS)


def _smoothed_distribution(counts, target, pseudocounts):
    return (counts + pseudocounts * target) / (counts.sum() + pseudocounts)


def _centered_format_ratio(distribution, target):
    log_ratio = np.log(distribution / target)
    return np.clip(log_ratio - log_ratio.mean(), -2.5, 2.5).tolist()


def fit_format_tables(texts, labels):
    """Fit per-member format-cue tables on its own fitting rows."""
    labels = np.asarray(labels, dtype=np.int64)
    global_counts = np.bincount(labels, minlength=3)
    if global_counts.shape != (3,) or np.any(global_counts == 0):
        raise ValueError("Format calibration requires fitting examples for all three sentiment classes.")
    prior = global_counts / global_counts.sum()
    group_counts = [{} for _ in FORMAT_CUE_NAMES]
    for text, label in zip(texts, labels, strict=True):
        for counts, key in zip(group_counts, _format_keys(text), strict=True):
            counts.setdefault(key, np.zeros(3, dtype=np.int64))[label] += 1

    main_distributions = [
        {key: _smoothed_distribution(counts, prior, 40) for key, counts in groups.items()}
        for groups in group_counts[:4]
    ]
    tables = {
        name: {key: _centered_format_ratio(distribution, prior) for key, distribution in groups.items()}
        for name, groups in zip(FORMAT_CUE_NAMES[:4], main_distributions, strict=True)
    }
    for name, counts, (left, right) in zip(
        FORMAT_CUE_NAMES[4:], group_counts[4:], FORMAT_INTERACTIONS, strict=True,
    ):
        ratios = {}
        for key, values in counts.items():
            left_key, right_key = json.loads(key)
            target = main_distributions[left][left_key] * main_distributions[right][right_key] / prior
            target /= target.sum()
            distribution = _smoothed_distribution(values, target, 60)
            ratios[key] = _centered_format_ratio(distribution, target)
        tables[name] = ratios
    return tables


def format_features(texts, tables):
    """Return per-row, per-cue, per-class log-ratio features."""
    zero = (0.0, 0.0, 0.0)
    rows = [
        [tables[name].get(key, zero) for name, key in zip(FORMAT_CUE_NAMES, _format_keys(text), strict=True)]
        for text in texts
    ]
    return np.asarray(rows, dtype=np.float64).reshape(-1, len(FORMAT_CUE_NAMES), 3)


def format_adjustments(cues, coefficients):
    return np.einsum("nfc,f->nc", cues, coefficients)


def fit_format_coefficients(logits, cues, labels):
    """Fit shared bounded weights using pooled original-head stopping logits."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    label_counts = np.bincount(labels, minlength=3)
    class_weights = label_counts[labels].astype(np.float64) ** -0.20
    class_weights /= class_weights.mean()
    indices = np.arange(len(labels))
    start = np.full(len(FORMAT_CUE_NAMES), 0.15)

    def objective(coefficients):
        adjusted = logits + format_adjustments(cues, coefficients)
        normalizer = logsumexp(adjusted, axis=1, keepdims=True)
        offset = coefficients - start
        loss = class_weights @ (normalizer[:, 0] - adjusted[indices, labels]) + 5 * (offset @ offset)
        residual = np.exp(adjusted - normalizer)
        residual[indices, labels] -= 1
        residual *= class_weights[:, None]
        gradient = np.einsum("nfc,nc->f", cues, residual) + 10 * offset
        return loss, gradient

    result = minimize(
        objective, start, method="L-BFGS-B", jac=True,
        bounds=[(0.0, 1.25)] * len(FORMAT_CUE_NAMES),
        options={"ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"Tweet-format coefficient optimization failed: {result.message}")
    return result.x


def fit_word_counts(shared_texts):
    """Count plain words and adjacent word pairs in the shared fitting rows."""
    words = Counter()
    pairs = Counter()
    for text in shared_texts:
        text = WORD_COUNT_EXCLUSIONS_RE.sub(" ", _canonical_tweet(text))
        tokens = re.findall(r"[a-z]+", text.lower())
        words.update(tokens)
        pairs.update(f"{left} {right}" for left, right in zip(tokens, tokens[1:]))
    return {"words": dict(words), "pairs": dict(pairs), "total": sum(words.values())}


def _split_hashtag(tag, word_counts, bert_vocab):
    if not JOINED_TAG_RE.fullmatch(tag) or CAMEL_RE.search(tag):
        return None
    tag = tag.lower()
    words = word_counts["words"]
    if tag in bert_vocab or words.get(tag, 0) >= 3:
        return None
    total = word_counts["total"]
    pairs = word_counts["pairs"]
    candidates = []

    def search(start, split, score):
        if start == len(tag):
            if len(split) >= 2:
                candidates.append((score, split))
            return
        if len(split) == 4:
            return
        for end in range(start + 2, len(tag) + 1):
            word = tag[start:end]
            count = words.get(word, 0)
            if count < 3:
                continue
            p = count / total
            term = math.log(p)
            if split:
                previous = split[-1]
                q = (pairs.get(f"{previous} {word}", 0) + 24 * p) / (words[previous] + 24)
                term += 0.5 * max(-2.0, min(2.0, math.log(q / p)))
            search(end, (*split, word), score + term)

    search(0, (), 0.0)
    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    if candidates and (len(candidates) == 1 or candidates[0][0] - candidates[1][0] >= math.log(3)):
        return " ".join(candidates[0][1])
    return None


def _shorten_emoji_runs(text):
    found = emoji.emoji_list(text)
    parts = []
    previous = None
    end = 0
    run_length = 0
    for item in found:
        start = item["match_start"]
        symbol = item["emoji"]
        run_length = run_length + 1 if start == end and symbol == previous else 1
        parts.append(text[end:start])
        if run_length <= 2:
            parts.append(symbol)
        previous = symbol
        end = item["match_end"]
    parts.append(text[end:])
    return "".join(parts)


def alternate_text(text, word_counts, bert_vocab):
    """Return the raw alternate reading, preserving URLs and handles."""
    def update(part):
        part = ASCII_FACE_RE.sub(lambda match: f" {ASCII_FACE_WORDS[match.lastgroup]} ", part)
        part = _shorten_emoji_runs(part)

        def update_hashtag(match):
            split = _split_hashtag(match[1], word_counts, bert_vocab)
            return f" {split} " if split is not None else match.group()

        return HASHTAG_RE.sub(update_hashtag, part)

    parts = []
    end = 0
    for match in URL_OR_HANDLE_RE.finditer(text):
        protected = match.group()
        suffix = "" if protected.startswith("@") else _url_suffix(protected)
        prefix = protected[:-len(suffix)] if suffix else protected
        parts.extend((update(text[end:match.start()]), prefix))
        end = match.end() - len(suffix)
    parts.append(update(text[end:]))
    return "".join(parts)


BASE_MODEL = "google-bert/bert-base-uncased"
BASE_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
MAX_LENGTH = 80
BATCH_SIZE = 24
NUM_LABELS = 3


@dataclass(frozen=True)
class MemberConfig:
    seed: int
    learning_rate: float
    exponent: float
    mlm_seed: int


MEMBER_CONFIGS = (
    MemberConfig(113, 2.0e-5, 0.18, 2140),
    MemberConfig(197, 1.6e-5, 0.35, 2224),
    MemberConfig(511, 2.5e-5, 0.00, 2538),
)


def load_tokenizer(path=None):
    from transformers import AutoTokenizer

    if path is not None:
        return AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    return AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION, use_fast=True)


def _text_loader(texts, tokenizer, labels=None, shuffle_seed=None):
    data = list(texts) if labels is None else list(zip(texts, labels, strict=True))

    def collate(batch):
        if labels is None:
            batch_texts = batch
            targets = None
        else:
            batch_texts, targets = zip(*batch)
            targets = torch.tensor(targets, dtype=torch.long)
        inputs = tokenizer(
            list(batch_texts),
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        return inputs, targets

    generator = None if shuffle_seed is None else torch.Generator().manual_seed(shuffle_seed)
    return DataLoader(
        data, batch_size=BATCH_SIZE, shuffle=generator is not None,
        generator=generator, collate_fn=collate,
    )


def _warmup_and_decay(optimizer, steps):
    from transformers import get_linear_schedule_with_warmup

    return get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=round(0.10 * steps), num_training_steps=steps,
    )


def _mask_for_mlm(input_ids, special_ids, ordinary_ids, mask_id):
    eligible = ~torch.isin(input_ids, special_ids)
    selected = eligible & (torch.rand(input_ids.shape, device=input_ids.device) < 0.15)
    labels = input_ids.masked_fill(~selected, -100)
    draws = torch.rand(input_ids.shape, device=input_ids.device)
    masked = input_ids.masked_fill(selected & (draws < 0.80), mask_id)
    replace_with_random = selected & (draws >= 0.80) & (draws < 0.90)
    random_indices = torch.randint(
        ordinary_ids.numel(), (int(replace_with_random.sum()),), device=input_ids.device,
    )
    masked[replace_with_random] = ordinary_ids[random_indices]
    return masked, labels


def _adapt_encoder(config, tokenizer, fit_texts, device):
    from transformers import AutoModelForMaskedLM

    torch.manual_seed(config.mlm_seed)
    model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL, revision=BASE_REVISION).to(device)
    loader = _text_loader(fit_texts, tokenizer, shuffle_seed=config.mlm_seed)
    special_id_set = set(tokenizer.all_special_ids)
    special_id_set.update(
        token_id for token, token_id in tokenizer.get_vocab().items()
        if token.startswith("[unused") and token.endswith("]")
    )
    special_ids = torch.tensor(sorted(special_id_set), device=device)
    ordinary_ids = torch.tensor(
        [i for i in range(model.config.vocab_size) if i not in special_id_set],
        device=device,
    )
    epochs = 4
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = _warmup_and_decay(optimizer, epochs * len(loader))
    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        selected_total = 0
        loss_total = 0.0
        for inputs, _ in loader:
            inputs = inputs.to(device)
            inputs["input_ids"], targets = _mask_for_mlm(
                inputs["input_ids"], special_ids, ordinary_ids, tokenizer.mask_token_id,
            )
            selected_count = int((targets != -100).sum())
            if selected_count == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = model(**inputs, labels=targets).loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            selected_total += selected_count
            loss_total += loss.item() * selected_count
        if selected_total == 0:
            raise ValueError("No ordinary tokens were selected during the MLM epoch.")
        mean_loss = loss_total / selected_total
        history.append({"epoch": epoch, "loss": mean_loss, "target_tokens": selected_total})
        print(f"Seed {config.seed} | MLM {epoch}/{epochs} | loss {mean_loss:.4f}", flush=True)
    model.zero_grad(set_to_none=True)
    return model.base_model, history


class SentimentModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.15)
        self.head = nn.Linear(2 * encoder.config.hidden_size, NUM_LABELS)
        self.corrected_head = None

    def features(self, inputs):
        hidden = self.encoder(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return torch.cat((hidden[:, 0], mean), dim=1)

    def forward(self, inputs):
        return self.head(self.dropout(self.features(inputs)))


def _class_weights(labels, exponent, device):
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=NUM_LABELS)
    if len(counts) != NUM_LABELS or np.any(counts == 0):
        raise ValueError("A sentiment fitting split must contain all three classes.")
    weights = counts.astype(np.float64) ** (-exponent)
    weights /= np.average(weights, weights=counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _macro_f1(labels, logits):
    return float(f1_score(
        labels, logits.argmax(axis=1), labels=np.arange(NUM_LABELS),
        average="macro", zero_division=0,
    ))


def _mean_class_loss(logits, labels):
    labels = np.asarray(labels, dtype=np.int64)
    labels_by_row = logits[np.arange(len(labels)), labels]
    losses = np.logaddexp.reduce(logits, axis=1) - labels_by_row
    counts = np.bincount(labels, minlength=NUM_LABELS)
    if np.any(counts == 0):
        raise ValueError("A validation split must contain all three sentiment classes.")
    return float((np.bincount(labels, weights=losses, minlength=NUM_LABELS) / counts).mean())


@torch.inference_mode()
def _collect_member_outputs(model, loader, device, include_features=False):
    model.eval()
    raw_batches = []
    corrected_batches = []
    feature_batches = []
    for inputs, _ in loader:
        features = model.features(inputs.to(device))
        raw_logits = model.head(features).cpu().numpy()
        raw_batches.append(raw_logits)
        if model.corrected_head is not None:
            corrected_batches.append(model.corrected_head(features).cpu().numpy())
        if include_features:
            feature_batches.append(features.cpu().numpy())

    plain = np.concatenate(raw_batches) if raw_batches else np.empty((0, NUM_LABELS), np.float32)
    corrected = np.concatenate(corrected_batches) if corrected_batches else plain
    features = None
    if include_features:
        features = (
            np.concatenate(feature_batches) if feature_batches
            else np.empty((0, model.head.in_features), np.float32)
        )
    return plain, corrected, features


def member_outputs(model, tokenizer, texts, device, include_features=False):
    return _collect_member_outputs(
        model, _text_loader(texts, tokenizer), device, include_features=include_features,
    )


def _sentiment_optimizer(model, learning_rate):
    decay = []
    no_decay = []
    for parameter in model.parameters():
        (decay if parameter.ndim > 1 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}],
        lr=learning_rate,
    )


@torch.no_grad()
def _update_ema(ema_model, model):
    for averaged, current in zip(ema_model.parameters(), model.parameters(), strict=True):
        averaged.lerp_(current, 0.10)


def _write_training_plot(history, path):
    from matplotlib.figure import Figure

    validation = history["validation"]
    train = history["sentiment"]
    mlm = history["mlm"]
    figure = Figure(figsize=(12.5, 3.5), layout="constrained")
    loss_axes, f1_axes, mlm_axes = figure.subplots(1, 3)
    train_x = [row["epoch_position"] for row in train]
    validation_x = [row["epoch_position"] for row in validation]
    loss_axes.plot(train_x, [row["weighted_loss"] for row in train], "o-", label="Train")
    loss_axes.plot(
        validation_x, [row["mean_class_loss"] for row in validation],
        ".-", label="Stopping set (EMA)",
    )
    loss_axes.set(title="Sentiment cross-entropy", xlabel="Epoch")
    loss_axes.legend()
    f1_axes.plot(train_x, [row["macro_f1"] for row in train], "o-", label="Train")
    f1_axes.plot(
        validation_x, [row["macro_f1"] for row in validation], ".-", label="Stopping set (EMA)",
    )
    f1_axes.set(title="Sentiment macro-F1", xlabel="Epoch")
    f1_axes.legend()
    mlm_axes.plot([row["epoch"] for row in mlm], [row["loss"] for row in mlm], "o-")
    mlm_axes.set(title="Masked-language-model loss", xlabel="Epoch", xticks=[1, 2, 3, 4])
    figure.savefig(path, dpi=150)


def train_member(config, tokenizer, fit_texts, fit_labels, stop_texts, stop_labels, output_dir, device):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder, mlm_history = _adapt_encoder(config, tokenizer, fit_texts, device)
    torch.manual_seed(config.seed)
    model = SentimentModel(encoder).to(device)
    train_loader = _text_loader(fit_texts, tokenizer, fit_labels, shuffle_seed=config.seed)
    stop_loader = _text_loader(stop_texts, tokenizer)
    stop_labels = np.asarray(stop_labels, dtype=np.int64)
    class_weights = _class_weights(fit_labels, config.exponent, device)
    optimizer = _sentiment_optimizer(model, config.learning_rate)
    epochs = 6
    steps_per_epoch = len(train_loader)
    scheduler = _warmup_and_decay(optimizer, epochs * steps_per_epoch)

    best_path = output_dir / "best.pt"
    best_macro_f1 = -1.0
    best_stopping_loss = float("inf")
    stale_checks = 0
    steps = 0
    ema_model = None
    history = {"mlm": mlm_history, "sentiment": [], "validation": []}
    check_batches = {(steps_per_epoch + 1) // 2, steps_per_epoch}
    for epoch in range(1, epochs + 1):
        model.train()
        train_logits = []
        train_labels = []
        weighted_loss = 0.0
        total_examples = 0
        for batch, (inputs, labels) in enumerate(train_loader, 1):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            losses = nn.functional.cross_entropy(logits, labels, reduction="none")
            loss = (losses * class_weights[labels]).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            steps += 1
            if ema_model is None:
                ema_model = deepcopy(model).requires_grad_(False)
            elif (steps - 1) % 8 == 0:
                _update_ema(ema_model, model)

            total_examples += labels.numel()
            weighted_loss += loss.item() * labels.numel()
            train_logits.append(logits.detach().cpu().numpy())
            train_labels.append(labels.cpu().numpy())
            if batch not in check_batches:
                continue

            stop_logits, _, _ = _collect_member_outputs(ema_model, stop_loader, device)
            stopping_loss = _mean_class_loss(stop_logits, stop_labels)
            macro_f1 = _macro_f1(stop_labels, stop_logits)
            epoch_position = epoch - 1 + batch / steps_per_epoch
            history["validation"].append({
                "epoch_position": epoch_position,
                "step": steps,
                "mean_class_loss": stopping_loss,
                "macro_f1": macro_f1,
            })
            print(
                f"Seed {config.seed} | sentiment epoch {epoch_position:.2f} | "
                f"mean per-class stopping loss {stopping_loss:.4f} | macro-F1 {macro_f1:.5f}",
                flush=True,
            )
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                torch.save({
                    "encoder_config": model.encoder.config.to_dict(),
                    "member": asdict(config),
                    "base_model": BASE_MODEL,
                    "base_revision": BASE_REVISION,
                    "max_length": MAX_LENGTH,
                    "model_state": ema_model.state_dict(),
                    "macro_f1": macro_f1,
                    "step": steps,
                }, best_path)
            if best_stopping_loss - stopping_loss > 0.002:
                best_stopping_loss = stopping_loss
                stale_checks = 0
            else:
                stale_checks += 1
            if stale_checks >= 2:
                break

        history["sentiment"].append({
            "epoch_position": epoch - 1 + batch / steps_per_epoch,
            "weighted_loss": weighted_loss / total_examples,
            "macro_f1": _macro_f1(np.concatenate(train_labels), np.concatenate(train_logits)),
        })
        if stale_checks >= 2:
            break

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    ema_model.load_state_dict(checkpoint["model_state"])
    stop_logits, _, stop_features = _collect_member_outputs(
        ema_model, stop_loader, device, include_features=True,
    )
    np.savez_compressed(
        output_dir / "stop_outputs.npz", logits=stop_logits,
        features=stop_features, labels=stop_labels,
    )
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    _write_training_plot(history, output_dir / "training.png")
    return {"stop_logits": stop_logits, "stop_features": stop_features, "best_macro_f1": best_macro_f1}


def fit_head_update(features, logits, labels, fit_indices, select_indices):
    features = np.asarray(features, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    fit_vectors = features[fit_indices]
    select_vectors = features[select_indices]
    fit_targets = labels[fit_indices]
    select_targets = labels[select_indices]
    original_fit = logits[fit_indices]
    original_select = logits[select_indices]

    rms = float(np.sqrt(np.square(fit_vectors).sum(axis=1).mean()))
    if not np.isfinite(rms) or rms <= 0:
        raise ValueError("Invalid RMS norm for head calibration.")
    fit_design = np.column_stack((fit_vectors / rms, np.ones(len(fit_vectors))))
    select_design = np.column_stack((select_vectors / rms, np.ones(len(select_vectors))))
    example_weights = np.where(original_fit.argmax(axis=1) == fit_targets, 1.0, 2.0)
    example_weights /= example_weights.sum()
    update = np.zeros((fit_design.shape[1], NUM_LABELS))
    selected_update = update.copy()
    selected_loss = _mean_class_loss(original_select, select_targets)
    selected_step = 0
    stale_checks = 0
    checks = [{"step": 0, "mean_class_loss": selected_loss}]

    for step in range(1, 101):
        scores = original_fit + fit_design @ update
        probabilities = np.exp(scores - np.logaddexp.reduce(scores, axis=1, keepdims=True))
        probabilities[np.arange(len(fit_targets)), fit_targets] -= 1
        gradient = fit_design.T @ (probabilities * example_weights[:, None]) + 0.01 * update
        update -= gradient
        if step % 5 != 0:
            continue
        loss = _mean_class_loss(original_select + select_design @ update, select_targets)
        checks.append({"step": step, "mean_class_loss": loss})
        if selected_loss - loss > 0.001:
            selected_loss = loss
            selected_step = step
            selected_update = update.copy()
            stale_checks = 0
        else:
            stale_checks += 1
        if stale_checks >= 4:
            break

    weight_delta = (selected_update[:-1] / rms).T.astype(np.float32)
    bias_delta = selected_update[-1].astype(np.float32)
    metadata = {
        "fit_rms_norm": rms,
        "selected_step": selected_step,
        "selected_mean_class_loss": selected_loss,
        "checks": checks,
    }
    return weight_delta, bias_delta, metadata


def write_head_update(output_dir, weight_delta, bias_delta):
    np.savez_compressed(Path(output_dir) / "head_update.npz", weight=weight_delta, bias=bias_delta)


def load_member(output_dir, device, require_correction=True):
    from transformers import BertConfig, BertModel

    output_dir = Path(output_dir)
    checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True)
    encoder = BertModel(BertConfig.from_dict(checkpoint["encoder_config"]), add_pooling_layer=False)
    model = SentimentModel(encoder)
    model.load_state_dict(checkpoint["model_state"])
    correction_path = output_dir / "head_update.npz"
    if require_correction or correction_path.is_file():
        with np.load(correction_path) as correction, torch.no_grad():
            model.corrected_head = deepcopy(model.head)
            model.corrected_head.weight.add_(torch.from_numpy(correction["weight"]))
            model.corrected_head.bias.add_(torch.from_numpy(correction["bias"]))
    return model.requires_grad_(False).to(device).eval()


LABELS = ("negative", "neutral", "positive")
LABEL_IDS = {label: index for index, label in enumerate(LABELS)}
STAGES = ("plain", "corrected_heads", "format_adjustment", "two_readings")
ROOT = Path(__file__).resolve().parent
RECIPE_VERSION = 1


@dataclass(frozen=True)
class Tweets:
    ids: np.ndarray
    raw: np.ndarray
    normalized: np.ndarray
    labels: np.ndarray | None
    digest: str


@dataclass(frozen=True)
class MemberSplit:
    fitting: np.ndarray
    stopping: np.ndarray
    calibration_fit: np.ndarray
    calibration_select: np.ndarray


@dataclass(frozen=True)
class ExperimentSplit:
    fold: int
    test: np.ndarray
    members: tuple[MemberSplit, ...]
    shared_fit: np.ndarray


def read_tweets(path, labeled):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"id", "text", "label"} if labeled else {"id", "text"}
        if len(reader.fieldnames or ()) != len(expected) or set(reader.fieldnames or ()) != expected:
            raise ValueError(f"{path} must contain columns {', '.join(sorted(expected))}.")
        rows = list(reader)
    if not rows or any(
        set(row) != expected or any(value is None for value in row.values()) for row in rows
    ):
        raise ValueError(f"{path} has no rows or contains a malformed CSV row.")
    ids = np.array([row["id"] for row in rows])
    if "" in ids or len(set(ids)) != len(ids):
        raise ValueError(f"{path} contains missing or duplicate IDs.")
    texts = np.array([row["text"] for row in rows], dtype=object)
    labels = None
    if labeled:
        seen = {row["label"] for row in rows}
        if seen != set(LABELS):
            raise ValueError(f"{path} must contain exactly these labels: {LABELS}.")
        labels = np.array([LABEL_IDS[row["label"]] for row in rows], dtype=np.int64)
    return Tweets(
        ids, texts, np.array([normalize_text(value) for value in texts], dtype=object),
        labels, hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def grouped_folds(data, indices, count, seed):
    splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=seed)
    labels = data.labels[indices]
    groups = data.normalized[indices]
    for fit, test in splitter.split(indices, labels, groups):
        if set(labels[fit]) != set(LABEL_IDS.values()) or set(labels[test]) != set(LABEL_IDS.values()):
            raise ValueError(f"A {count}-fold partition is missing a sentiment class.")
        if set(groups[fit]) & set(groups[test]):
            raise RuntimeError("Identical normalized tweets crossed a split.")
        yield np.sort(indices[fit]), np.sort(indices[test])


def experiment_splits(data, outer_folds):
    requested = set(outer_folds)
    outer = grouped_folds(data, np.arange(len(data.ids)), count=10, seed=113)
    for number, (outer_train, test) in enumerate(outer, start=1):
        if number not in requested:
            continue
        members = []
        inner = grouped_folds(data, outer_train, count=18, seed=113)
        for config, (fit, stop) in zip(MEMBER_CONFIGS, inner):
            cal_fit, cal_select = next(grouped_folds(data, stop, count=3, seed=config.seed + 614))
            members.append(MemberSplit(fit, stop, cal_fit, cal_select))
        shared = members[0].fitting
        for member in members[1:]:
            shared = np.intersect1d(shared, member.fitting, assume_unique=True)
        yield ExperimentSplit(number, test, tuple(members), shared)


def split_manifest(data, split):
    return {
        "recipe_version": RECIPE_VERSION,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "max_length": MAX_LENGTH,
        "members_config": [asdict(config) for config in MEMBER_CONFIGS],
        "dataset_sha256": data.digest,
        "labels": list(LABELS),
        "outer_fold": split.fold,
        "test_ids": data.ids[split.test].tolist(),
        "shared_fitting_ids": data.ids[split.shared_fit].tolist(),
        "members": [
            {
                "fitting_ids": data.ids[member.fitting].tolist(),
                "stopping_ids": data.ids[member.stopping].tolist(),
                "head_update_fit_ids": data.ids[member.calibration_fit].tolist(),
                "head_update_select_ids": data.ids[member.calibration_select].tolist(),
            }
            for member in split.members
        ],
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_splits(data, split, run_dir):
    manifest = split_manifest(data, split)
    destination = run_dir / "splits.json"
    if destination.exists() and read_json(destination) != manifest:
        raise ValueError(f"{destination} belongs to different data or an outer fold. Use a fresh run directory.")
    if not destination.exists():
        write_json(destination, manifest)
    print(f"Outer fold {split.fold}: {len(split.test)} test / {len(split.shared_fit)} shared fitting")
    for number, member in enumerate(split.members, start=1):
        print(
            f"Member {number}: {len(member.fitting)} fitting / {len(member.stopping)} stopping "
            f"({len(member.calibration_fit)} head fit / {len(member.calibration_select)} head selection)"
        )


def report_environment(device):
    packages = ("torch", "transformers", "scikit-learn", "numpy", "scipy", "matplotlib", "emoji")
    return {
        "python": platform.python_version(),
        **{package: version(package) for package in packages},
        "device": str(device),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this environment. Use a CUDA-enabled Colab runtime or --device cpu.")
    return torch.device(name)


def member_dir(run_dir, number):
    return run_dir / "saved_models" / f"member_{number}"


def check_run(run_dir):
    if not (run_dir / "system.json").is_file():
        raise FileNotFoundError(f"{run_dir} has no completed system. Run the train command first.")
    if not (run_dir / "tokenizer" / "tokenizer_config.json").is_file():
        raise FileNotFoundError(f"{run_dir} does not have its saved tokenizer.")
    for number in range(1, len(MEMBER_CONFIGS) + 1):
        for name in ("best.pt", "head_update.npz"):
            if not (member_dir(run_dir, number) / name).is_file():
                raise FileNotFoundError(f"{run_dir} is missing member {number}'s {name}.")


def train_system(data, split, run_dir, device):
    save_splits(data, split, run_dir)
    if (run_dir / "system.json").is_file():
        check_run(run_dir)
        print(f"Using completed training run at {run_dir}.")
        return
    environment = report_environment(device)
    print("Environment:", environment, flush=True)
    write_json(run_dir / "environment.json", environment)
    tokenizer_path = run_dir / "tokenizer"
    if (tokenizer_path / "tokenizer_config.json").is_file():
        tokenizer = load_tokenizer(tokenizer_path)
    else:
        tokenizer = load_tokenizer()
        tokenizer.save_pretrained(tokenizer_path)

    format_tables = []
    stop_logits, stop_cues, stop_labels = [], [], []
    for number, (config, partition) in enumerate(zip(MEMBER_CONFIGS, split.members), start=1):
        fitting = partition.fitting
        stopping = partition.stopping
        output = member_dir(run_dir, number)
        output.mkdir(parents=True, exist_ok=True)
        saved_stop = output / "stop_outputs.npz"
        # A complete base checkpoint and its paired outputs can be used after an interrupted calibration.
        if (output / "best.pt").is_file() and (output / "training.png").is_file() and saved_stop.is_file():
            with np.load(saved_stop, allow_pickle=False) as arrays:
                result = {"stop_logits": arrays["logits"], "stop_features": arrays["features"]}
                if not np.array_equal(arrays["labels"], data.labels[stopping]):
                    raise ValueError(f"Saved stopping labels disagree for member {number}.")
        else:
            print(f"Training member {number} in outer fold {split.fold}", flush=True)
            result = train_member(
                config, tokenizer, data.normalized[fitting].tolist(), data.labels[fitting],
                data.normalized[stopping].tolist(), data.labels[stopping], output, device,
            )
        logits = result["stop_logits"]
        weight, bias, calibration = fit_head_update(
            result["stop_features"], logits, data.labels[stopping],
            np.searchsorted(stopping, partition.calibration_fit),
            np.searchsorted(stopping, partition.calibration_select),
        )
        write_head_update(output, weight, bias)
        write_json(output / "calibration.json", calibration)
        tables = fit_format_tables(data.raw[fitting].tolist(), data.labels[fitting])
        format_tables.append(tables)
        stop_logits.append(logits)
        stop_cues.append(format_features(data.raw[stopping].tolist(), tables))
        stop_labels.append(data.labels[stopping])

    coefficients = fit_format_coefficients(
        np.concatenate(stop_logits), np.concatenate(stop_cues), np.concatenate(stop_labels)
    )
    write_json(run_dir / "system.json", {
        "format_tables": format_tables,
        "format_coefficients": coefficients.tolist(),
        "word_counts": fit_word_counts(data.raw[split.shared_fit].tolist()),
    })


def score_system(raw, normalized, run_dir, device):
    check_run(run_dir)
    system = read_json(run_dir / "system.json")
    tokenizer = load_tokenizer(run_dir / "tokenizer")
    vocab = tokenizer.get_vocab()
    alternates = [
        normalize_text(alternate_text(text, system["word_counts"], vocab)) for text in raw
    ]
    eligible = np.flatnonzero(np.asarray(alternates, dtype=object) != normalized)
    alternate_inputs = [alternates[index] for index in eligible]
    probabilities = {stage: [] for stage in STAGES}
    member_adjustments = []
    for number, tables in enumerate(system["format_tables"], start=1):
        model = load_member(member_dir(run_dir, number), device)
        original, corrected, _ = member_outputs(model, tokenizer, normalized.tolist(), device)
        adjustments = format_adjustments(format_features(raw.tolist(), tables), system["format_coefficients"])
        if not np.isfinite(adjustments).all():
            raise ValueError("Format adjustments contain non-finite values.")
        member_adjustments.append(adjustments)
        formatted = softmax(corrected + adjustments, axis=1)
        combined = formatted.copy()
        if eligible.size:
            _, alternate, _ = member_outputs(model, tokenizer, alternate_inputs, device)
            combined[eligible] = (formatted[eligible] + softmax(alternate + adjustments[eligible], axis=1)) / 2
        probabilities["plain"].append(softmax(original, axis=1))
        probabilities["corrected_heads"].append(softmax(corrected, axis=1))
        probabilities["format_adjustment"].append(formatted)
        probabilities["two_readings"].append(combined)
        del model
    probabilities = {stage: np.mean(values, axis=0) for stage, values in probabilities.items()}
    if any(not np.isfinite(values).all() for values in probabilities.values()):
        raise ValueError("The model returned non-finite prediction probabilities.")
    return probabilities, eligible, alternates, np.stack(member_adjustments, axis=1)


def evaluate_predictions(data, indices, probabilities, eligible, alternates, adjustments):
    expected = data.labels[indices]
    predicted = {stage: probabilities[stage].argmax(axis=1) for stage in STAGES}
    report = {
        "row_count": len(expected),
        "second_reading_eligible": len(eligible),
        "comparison_references": {
            "single_900_row_holdout_macro_f1": 0.91026,
            "oof_macro_f1_format_to_two_readings": [0.89931, 0.90045],
            "oof_neutral_f1_format_to_two_readings": [0.82380, 0.82491],
        },
        "labels": list(LABELS),
        "stages": {},
    }
    previous = None
    for stage in STAGES:
        ids = predicted[stage]
        per_class_f1 = f1_score(expected, ids, labels=[0, 1, 2], average=None, zero_division=0)
        metrics = {
            "macro_f1": float(per_class_f1.mean()),
            "neutral_f1": float(per_class_f1[LABEL_IDS["neutral"]]),
            "confusion_matrix": confusion_matrix(expected, ids, labels=[0, 1, 2]).tolist(),
        }
        if previous is not None:
            before = predicted[previous]
            metrics.update({
                "compared_with": previous,
                "mistakes_fixed": int(np.sum((before != expected) & (ids == expected))),
                "new_mistakes": int(np.sum((before == expected) & (ids != expected))),
            })
        report["stages"][stage] = metrics
        previous = stage

    example = None
    explanations = {
        "corrected_heads": "The head update selected on stopping examples changed the averaged probabilities.",
        "format_adjustment": "The tweet-format logit adjustments changed the averaged probabilities.",
        "two_readings": "Averaging the alternate and original readings changed the winning label.",
    }
    for previous, current in reversed(list(zip(STAGES, STAGES[1:]))):
        before, after = predicted[previous], predicted[current]
        changed = np.flatnonzero(before != after)
        if changed.size:
            fixed = changed[(before[changed] != expected[changed]) & (after[changed] == expected[changed])]
            row = int(fixed[0] if fixed.size else changed[0])
            original_index = indices[row]
            example = {
                "id": str(data.ids[original_index]),
                "text": data.raw[original_index],
                "gold": LABELS[expected[row]],
                "before_stage": previous,
                "after_stage": current,
                "before_label": LABELS[before[row]],
                "after_label": LABELS[after[row]],
                "before_probabilities": dict(zip(LABELS, probabilities[previous][row].tolist())),
                "after_probabilities": dict(zip(LABELS, probabilities[current][row].tolist())),
                "explanation": explanations[current],
            }
            if current == "two_readings":
                example["normalized_original"] = data.normalized[original_index]
                example["normalized_alternate"] = alternates[row]
            if current == "format_adjustment":
                example["format_logit_delta_by_member"] = adjustments[row].tolist()
            break
    report["changed_prediction_example"] = example
    return report, predicted


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    temporary.replace(path)


def save_evaluation(data, indices, scored, run_dir, name):
    report, predictions = evaluate_predictions(data, indices, *scored)
    write_json(run_dir / "report.json", report)
    write_csv(
        run_dir / name, ("id", "gold", *STAGES),
        (
            (data.ids[original_index], LABELS[data.labels[original_index]],
             *(LABELS[predictions[stage][row]] for stage in STAGES))
            for row, original_index in enumerate(indices)
        ),
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))


def infer_benchmark(data_dir, run_dir, output, device):
    benchmark = read_tweets(data_dir / "benchmark.csv", labeled=False)
    probabilities, _, _, _ = score_system(benchmark.raw, benchmark.normalized, run_dir, device)
    labels = [LABELS[index] for index in probabilities["two_readings"].argmax(axis=1)]
    write_csv(output, ("id", "label"), zip(benchmark.ids, labels))
    with output.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header, rows = next(reader), list(reader)
    if (header != ["id", "label"] or any(len(row) != 2 or row[1] not in LABEL_IDS for row in rows)
            or [row[0] for row in rows] != benchmark.ids.tolist()):
        raise ValueError("Prediction CSV failed its schema, label, or benchmark-order check.")
    print(f"Verified {len(rows)} benchmark predictions at {output}.")
    return output


def run_oof(data, run_dir, device, reuse_holdout):
    count = len(data.ids)
    probabilities = {stage: np.empty((count, len(LABELS))) for stage in STAGES}
    adjustments = np.empty((count, len(MEMBER_CONFIGS), len(LABELS)))
    alternates = np.empty(count, dtype=object)
    eligible = np.zeros(count, dtype=bool)
    evaluated = np.zeros(count, dtype=bool)
    for split in experiment_splits(data, range(1, 11)):
        fold_dir = (reuse_holdout if split.fold == 2 and reuse_holdout is not None
                    else run_dir / f"outer_{split.fold:02}")
        train_system(data, split, fold_dir, device)
        rows = split.test
        if evaluated[rows].any():
            raise RuntimeError("An evaluation row was already scored by another outer fold.")
        scores, active, text, delta = score_system(data.raw[rows], data.normalized[rows], fold_dir, device)
        for stage in STAGES:
            probabilities[stage][rows] = scores[stage]
        eligible[rows[active]] = True
        evaluated[rows] = True
        alternates[rows] = text
        adjustments[rows] = delta
    if not evaluated.all():
        raise RuntimeError("Some labeled rows have no out-of-fold prediction.")
    save_evaluation(data, np.arange(count), (probabilities, np.flatnonzero(eligible), alternates, adjustments),
                    run_dir, "oof_predictions.csv")


def homework_package(data_dir, run_dir, output, student_id, device):
    if not student_id.isascii() or not student_id.isdecimal():
        raise ValueError("Provide a numeric student ID from the course roster.")
    check_run(run_dir)
    training_manifest = read_json(run_dir / "splits.json")
    if hashlib.sha256((data_dir / "dataset.csv").read_bytes()).hexdigest() != training_manifest["dataset_sha256"]:
        raise ValueError("The dataset to package differs from the one used for training.")
    output.mkdir(parents=True, exist_ok=True)
    original_csv = infer_benchmark(data_dir, run_dir, output / "predictions.csv", device)
    package_name = f"HW3_{student_id}"
    archive_path = output / f"{package_name}.zip"
    with tempfile.TemporaryDirectory(prefix=".hw3-package-", dir=output) as temporary:
        temporary = Path(temporary)
        archive = temporary / archive_path.name
        with ZipFile(archive, "w", compression=ZIP_STORED) as bundle:
            files = {
                Path("main.py"): ROOT / "main.py",
                Path("IAI_HW3.ipynb"): ROOT / "IAI_HW3.ipynb",
                Path("dataset/dataset.csv"): data_dir / "dataset.csv",
                Path("dataset/benchmark.csv"): data_dir / "benchmark.csv",
            }
            for name in ("system.json", "environment.json", "splits.json"):
                files[Path(name)] = run_dir / name
            for file in (run_dir / "tokenizer").iterdir():
                if file.is_file():
                    files[Path("tokenizer") / file.name] = file
            for number in range(1, len(MEMBER_CONFIGS) + 1):
                folder = member_dir(run_dir, number)
                for name in ("best.pt", "head_update.npz", "training.png", "history.json", "calibration.json"):
                    if (folder / name).is_file():
                        files[folder.relative_to(run_dir) / name] = folder / name
            for name in ("report.json", "heldout_predictions.csv"):
                if (run_dir / name).is_file():
                    files[Path(name)] = run_dir / name
            for relative, source in files.items():
                bundle.write(source, (Path(package_name) / relative).as_posix())
            bundle.writestr(f"{package_name}/requirements.txt", PACKAGE_REQUIREMENTS)
            bundle.writestr(f"{package_name}/README.md", f"""# HW3 airline-tweet sentiment

Install with `python -m pip install -r requirements.txt`. From this folder:

```text
python main.py predict --run-dir . --output predictions.csv
python main.py evaluate --run-dir .
```

The packaged weights use the course dataset and outer fold {training_manifest["outer_fold"]}.
The training and validation plots are inside `saved_models`.
""")
        extracted = temporary / "reload"
        with ZipFile(archive) as bundle:
            bundle.extractall(extracted)
        project = extracted / package_name
        regenerated = infer_benchmark(project / "dataset", project, temporary / "regenerated.csv", device)
        if original_csv.read_bytes() != regenerated.read_bytes():
            raise RuntimeError("Reloading the packaged system produced a different CSV.")
        archive.replace(archive_path)
    print(f"Verified reloaded submission: {archive_path}")


PACKAGE_REQUIREMENTS = """torch>=2.6,<3
transformers==4.57.6
scikit-learn>=1.5,<2
numpy>=1.26
scipy>=1.11
matplotlib>=3.8,<4
emoji>=2.12,<3
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    descriptions = {
        "prepare": "Save the grouped holdout, stopping, and calibration IDs.",
        "train": "Train and calibrate three independent BERTs for the chosen outer fold.",
        "evaluate": "Compare all stages on the run's labeled held-out rows.",
        "oof": "Repeat the full recipe for all ten outer folds.",
        "predict": "Generate and validate the benchmark submission CSV.",
        "package": "Create the coursework ZIP and verify its reloaded predictions.",
    }
    for name, description in descriptions.items():
        command = commands.add_parser(name, help=description)
        command.add_argument("--run-dir", type=Path, required=True)
        command.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
        if name in {"prepare", "train"}:
            command.add_argument("--outer-fold", type=int, choices=range(1, 11), default=2)
        if name != "prepare":
            command.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
        if name == "oof":
            command.add_argument("--reuse-holdout", type=Path, help="Reuse a completed, matching outer fold 2 run.")
        if name in {"predict", "package"}:
            command.add_argument("--output", type=Path, required=True,
                                 help="Destination CSV file." if name == "predict" else "Destination directory.")
        if name == "package":
            command.add_argument("--student-id", required=True)
    args = parser.parse_args()

    if args.command == "predict":
        infer_benchmark(args.data_dir, args.run_dir, args.output, choose_device(args.device))
        return
    if args.command == "package":
        homework_package(args.data_dir, args.run_dir, args.output, args.student_id, choose_device(args.device))
        return

    data = read_tweets(args.data_dir / "dataset.csv", labeled=True)
    if args.command == "oof":
        run_oof(data, args.run_dir, choose_device(args.device), args.reuse_holdout)
        return

    fold = read_json(args.run_dir / "splits.json")["outer_fold"] if args.command == "evaluate" else args.outer_fold
    split = next(experiment_splits(data, [fold]))
    if args.command == "prepare":
        save_splits(data, split, args.run_dir)
    elif args.command == "train":
        train_system(data, split, args.run_dir, choose_device(args.device))
    else:
        save_splits(data, split, args.run_dir)
        rows = split.test
        scores = score_system(data.raw[rows], data.normalized[rows], args.run_dir, choose_device(args.device))
        save_evaluation(data, rows, scores, args.run_dir, "heldout_predictions.csv")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"HW3: {error}", file=sys.stderr)
        raise SystemExit(1) from error
