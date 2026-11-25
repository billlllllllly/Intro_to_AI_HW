# HW3 sentiment project: change specification (v2)

Audience: Codex (or any coding agent) working inside the repo `Intro_to_AI_HW/IAI_HW3`.
Written against `main_c8d939.py` (the run that scored 0.8922 on Kaggle). Line numbers are approximate; **function names are authoritative**.

## 0. Ground rules (do not violate)

1. **Model restriction:** only BERT-base-size models. Default checkpoint stays `google-bert/bert-base-uncased`. Any other checkpoint (e.g. `google-bert/bert-base-cased`) must be behind an opt-in preset and OFF by default (the user must confirm the assignment allows it).
2. **Ensemble cap: exactly 3 members.** Add `assert len(members) == 3` (message: "assignment allows at most three models in the ensemble"). Models used for the submission = the 3 final models only.
3. **Do not delete or rename** these public names, because `baseline_evaluation.py`, `fold_training.py` and the notebook import them: `run`, `BASE_MODEL`, `COMPANY_TOKEN`, `GMAIL_TOKEN`, `LABEL_TO_ID`, `LABELS`, `ROOT`, `has_content`, `make_loader`, `predict_probabilities`, `preprocess`, `SentimentModel`, `train_model`, `vote`, `VotingRule`.
4. `run()` must keep returning `(predictions_dataframe, histories)` and keep `predictions.attrs["pooled_macro_f1"]`.
5. Do not touch `fold_training.py` logic or `baseline_evaluation.py` except to fix imports if something moves.
6. Commit after each numbered step in section 3 so each change can be reverted on its own.

## 1. Why these changes (evidence from the 0.8922 run)

| Observation | Number | Meaning |
|---|---|---|
| Split sizes changed between runs | 7202/901/897 (0.8946 run) vs 7200/905/895 (0.8922 run) | The split key is `preprocess(text).casefold()` (line ~212), so editing preprocessing silently reshuffles the split. Runs are not comparable. |
| Baseline re-score impossible | "previous fold checkpoints are missing" | No controlled before/after comparison exists. |
| Kaggle noise | 1000 rows, SE of macro-F1 about 0.01 | 0.8918 / 0.8922 / 0.8946 are statistically indistinguishable. |
| Members almost identical | 28-29 of 905 validation predictions differ (about 3%) | Best single model 0.8948, ensemble 0.8981 (+0.003). Ensemble adds almost nothing. |
| CNN member hurts | "CNN alone correct: 3; both MLPs correct and CNN wrong: 11" | Weakest member. |
| "Specialists" do not specialize | positive-specialist positive-F1 0.8856 < neutral-MLP's 0.8913 | The x2 class weight has no measurable effect. |
| Vote selection overfits | per-fold picks: hard / grid(0.8,0.2,0.0) x1.1 / hard; cross-fit 0.8935 < equal-soft 0.8981 | `tune_neutral` + 66-weight grid on 905 samples fits noise. |
| Best-epoch selection is noisy | CNN val F1 0.8604 -> 0.8828 -> 0.8716 | Choosing the "best" epoch on validation is noise-chasing. |
| Validation vs test gap | val 0.8981, test 0.8773, pooled 0.8885 | Validation is used for early stopping and voting, so it is optimistic. |
| Benchmark leans negative | predicted 72.4% neg / 14.0% neu / 13.6% pos on benchmark vs 68.7 / 15.6 / 15.7 on validation | Either real shift or a threshold issue; needs a diagnostic, not a guess. |
| Label ambiguity | 13 of 20 shown error rows: all three members agree against the label | Likely a ceiling near 0.90 for this data; do not expect 0.92 from tuning. |

Goal of v2: (a) make experiments **comparable and statistically honest**, (b) stop fitting noise, (c) create **real ensemble diversity** inside the 3-model cap, (d) use **more training data** for the final submission. Realistic target: 0.895-0.905 on Kaggle. Do not promise more.

## 2. Read these files first (they were NOT available when this spec was written)

- `sentiment_preprocessing.py`: check exactly what `preprocess(text, clean, url_token, distinguish_mentions)` already does for emoji, ASCII emoticons, hashtags, elongated words, HTML entities.
- `fold_training.py`: contains `MeanClassifier`, `optimizer_for`, `holdout_split`, `fit_offsets`, `log_probs`. Reuse `optimizer_for` as-is. Reuse the pooling logic of `MeanClassifier` if it is a masked mean pool.
- `requirements.txt`, `dataset/dataset.csv` (confirm columns `id,text,label`), `dataset/benchmark.csv` (`id,text`), any existing `tests/` folder.

## 3. Changes

### 3.1 Frozen evaluation split (new file `splits.py`)

**Why:** make train/validation/test identical across all future runs, independent of preprocessing edits.

Create `IAI_HW3/splits.py`:

```python
"""Frozen train/validation/test split. Never change split_key or SPLIT_SEED."""
import hashlib, json, re
from pathlib import Path
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
        saved = json.loads(split_file.read_text())
        if saved["dataset_sha256"] != digest:
            raise ValueError("dataset.csv changed since splits.json was created; refusing to continue.")
        return saved["train"], saved["validation"], saved["test"]
    splits = make_splits(data)
    split_file.write_text(json.dumps({"dataset_sha256": digest, **splits}))
    return splits["train"], splits["validation"], splits["test"]
```

Edit `run()` in `main.py`:
- Replace lines ~212-218 (`groups = ...` through the `validation, test = ...` assignment) with:
  ```python
  train_pos, val_pos, test_pos = load_or_create_splits(data, ROOT / "dataset/dataset.csv", ROOT / split_file)
  train, validation, test = data.iloc[train_pos], data.iloc[val_pos], data.iloc[test_pos]
  ```
- `groups` is still used later (`groups.loc[validation.index]` in `select_vote`). Keep a `groups = data["text"].map(split_key)` line for that, importing `split_key` from `splits`.
- Add `split_file="splits.json"` to the `run()` signature and a `--split-file` CLI arg.
- Commit `splits.json` to git.

**Acceptance:** with a fresh `splits.json`, the printed sizes are `train: 7202 | validation: 901 | test: 897` (the sizes of the 0.8946 run). If they differ, STOP and report; do not "fix" by editing `split_key`. Also assert in a test that no `split_key` value appears in more than one partition.

### 3.2 Remove vote-rule tuning

**Why:** it overfits (section 1).

- In `run()`: always use `voting = VotingRule("equal soft")`. Delete the `if ensemble: voting = select_vote(...)` selection.
- Remove the `tune_neutral` and `sweep_path` parameters from `run()` and `select_vote()`; delete the 66-weight grid, the `hard` rule search, `select()`, the `crossfit` loop and the sweep CSV writing.
- **Keep** the reporting part of `select_vote` and rename it `report_members(...)`: the pairwise "x vs y: n/N validation predictions differ" lines (generalize member names from the spec list, no hard-coded "negative_cnn"), the "alone correct / both others correct" line (generalize to "member i alone correct: k"), and the `review_path` CSV with columns `id, text, label, <one column per member>, ensemble`.
- Keep `VotingRule` and `vote()` (other code imports them); `weights=()` and `neutral_multiplier=1.0` must still mean plain mean of probabilities.
- Update the notebook call (section 3.10).

**Acceptance:** `vote(probs, VotingRule("equal soft"))` equals `torch.stack(probs).mean(0)` (unit test).

### 3.3 Epoch selection policy: `--select last|best-f1`

**Why:** best-epoch picking on 905 samples is noise.

In `train_model(...)`:
- Add keyword `select="last"`. Signature allows `val_loader=None`.
- If `select == "last"`: train exactly `epochs`, no early stopping, keep final weights (do not load `best_state`).
- If `select == "best-f1"`: current behaviour (best validation macro-F1, patience).
- If `val_loader is None`: skip validation; append `(train_metrics, (float("nan"), float("nan")))` to `history` so the notebook plotting cell (which indexes `val[index]`) still works.
- Add `select` to `run()` and CLI (`--select`, default `last`), and add it to the checkpoint `key` in `fit` so cached checkpoints do not collide.
- Keep `epochs=3`, label smoothing 0.05, `ENCODER_LR=2e-5`, `HEAD_LR=2e-4`, `layer_decay=0.9` unchanged (no evidence they hurt; do not change recipe and protocol at once).

### 3.4 Member specs, presets and heads (core diversity change)

**Why:** members currently share data, encoder and recipe, so they agree 97% of the time.

Add to `main.py`:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class MemberSpec:
    name: str
    seed: int
    head: str = "mlp"                      # "mlp" | "cnn" | "mean" | "clsmean"
    base_model: str = BASE_MODEL
    boost_class: Optional[str] = None      # legacy x2 class-weight "specialist"
    subsample: float = 1.0                 # fraction of training rows (stratified)

PRESETS = {
    # exactly the current ensemble, for a controlled baseline
    "legacy": [MemberSpec("negative_cnn", 42, "cnn", boost_class="negative"),
               MemberSpec("neutral_mlp", 43, "mlp", boost_class="neutral"),
               MemberSpec("positive_mlp", 44, "mlp", boost_class="positive")],
    # same encoder, different heads/seeds, no x2 boost
    "diverse": [MemberSpec("cls_mlp", 42, "mlp"),
                MemberSpec("mean_pool", 43, "mean"),
                MemberSpec("cls_mean", 44, "clsmean")],
    # OPT-IN ONLY (user must confirm the assignment allows another BERT-base checkpoint)
    "diverse-cased": [MemberSpec("cls_mlp", 42, "mlp"),
                      MemberSpec("mean_pool", 43, "mean"),
                      MemberSpec("cased_cls_mean", 44, "clsmean",
                                 base_model="google-bert/bert-base-cased")],
}
```

New heads (next to `CNNHead`):

```python
class MeanPoolHead(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, len(LABELS))
    def forward(self, hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.classifier(self.dropout(pooled))

class ClsMeanHead(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden_size, len(LABELS))
    def forward(self, hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        mean = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.classifier(self.dropout(torch.cat([hidden[:, 0], mean], dim=-1)))
```

Edit `SentimentModel`:
- `__init__(self, vocab_size, head_type="mlp", base_model=BASE_MODEL)`; use `base_model` in `BertModel.from_pretrained(base_model, add_pooling_layer=False)`.
- Build `MeanPoolHead(hidden)` for `"mean"` and `ClsMeanHead(hidden)` for `"clsmean"`.
- In `forward`, change `if isinstance(self.head, CNNHead)` to `if isinstance(self.head, (CNNHead, MeanPoolHead, ClsMeanHead))` (they all take `(hidden, attention_mask)`).
- Keep the attribute names `encoder` and `head` (`optimizer_for` and FGSM depend on them).
- Keep the CNN head and MLP head byte-for-byte compatible so old checkpoints still load.

Edit `run()` / `fit()`:
- Add `preset="legacy"` and `subsample=None` (override for all members) to `run()` and CLI (`--preset`, `--subsample`). The old `ensemble` flag: keep; `ensemble=False` means a single member `MemberSpec("bert", 42)`; `ensemble=True` uses the preset. Replace lines ~301-304 (`members = [...]`) accordingly, then `assert len(members) == 3` when `ensemble`.
- Change `fit(model_seed, specialist, head_type)` to `fit(spec, frame, final=False)`.
  - Tokenizer: `AutoTokenizer.from_pretrained(spec.base_model)` (per member, because a cased member needs its own tokenizer; add the special tokens exactly as now).
  - Model: `SentimentModel(len(tokenizer), head_type=spec.head, base_model=spec.base_model)`, also in the checkpoint-reload branch.
  - Subsample: `member_train = frame.groupby("label", group_keys=False).sample(frac=spec.subsample, random_state=spec.seed)` when `spec.subsample < 1`; compute `counts`, class weights and the sampler from `member_train`, not from `train`.
  - Class weights: `weights[LABEL_TO_ID[spec.boost_class]] *= 2` only if `spec.boost_class` is not None.
  - Checkpoint `key`: use `repr((class_weighting, balanced_sampling, fgsm_epsilon, schedule, select, final, spec))` (the dataclass repr includes every spec field); fingerprint must use `frame.index.tolist()` instead of `train.index.tolist()`.
  - Return `(val_probs, test_probs, benchmark_probs, history)`; in `final=True` mode return `None` for `val_probs` and `test_probs` and skip val loader / val predictions.
- Update the training log line to print `spec`.

**Acceptance:** `preset="legacy", select="best-f1"` reproduces the previous ensemble behaviour (same member list, same weights). Run `python -c "from main import PRESETS; print({k: len(v) for k, v in PRESETS.items()})"` and confirm 3 for all.

### 3.5 Persist artifacts per run (`run_dir`)

**Why:** later comparison and diagnostics need saved probabilities.

Add `run_dir=None` to `run()` and CLI (`--run-dir`). If set, create it and write:
- `config.json`: all `run()` arguments, member specs (`dataclasses.asdict`), git commit (`subprocess.run(["git","rev-parse","HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()`, empty string on failure), `splits.json` sha256, and the per-member **effective class prior** (`counts * weights / (counts * weights).sum()`).
- `probs.npz` via `np.savez`, arrays: `val_ids`, `val_labels` (int), `val_member_probs` (shape `[3, N_val, 3]`), `test_ids`, `test_labels`, `test_member_probs`, `benchmark_ids`, `benchmark_member_probs`, and if `final_fit`: `final_benchmark_member_probs`.
- `predictions.csv` (id,label) from the evaluation-pass ensemble, and `predictions_final.csv` if `final_fit`.
- Use the dataset `id` column for `*_ids`.

### 3.6 Final fit on more data (`final_fit=True`, CLI `--final-fit`)

**Why:** +25% training rows for the submission (7202 -> 9000 minus `has_content` filtering) is the most reliable gain available, and `select="last"` makes a no-validation refit well defined.

In `run()`, after the evaluation pass and its reports:

```python
if final_fit:
    full = pd.concat([train, validation, test])        # train already has has_content applied
    final_results = [fit(spec, full, final=True) for spec in members]
    final_benchmark = vote([r[2] for r in final_results], VotingRule("equal soft"))
    final_predictions = benchmark[["id"]].assign(
        label=[LABELS[i] for i in final_benchmark.argmax(dim=1).tolist()])
    agreement = (final_benchmark.argmax(1) == benchmark_probs.argmax(1)).float().mean().item()
    print(f"Final-fit vs evaluation-pass benchmark agreement: {agreement:.3f}")
    print("Final-fit predicted class shares:", {LABELS[i]: round((final_benchmark.argmax(1) == i).float().mean().item(), 3) for i in range(3)})
```

- Never report test/validation scores for the final-fit models (they trained on them).
- Return `final_predictions` instead of `predictions` when `final_fit=True`, but still attach `pooled_macro_f1` from the evaluation pass in `.attrs`; also expose the evaluation-pass predictions as `final_predictions.attrs["evaluation_predictions"]`.
- Print a warning if agreement < 0.93 (indicates an unstable recipe).

### 3.7 Preprocessing gaps (only after reading `sentiment_preprocessing.py`)

Dataset statistics from the notebook: emoticons 1.98%, hashtags 17.38%, elongated words 0.62%, HTML entities 5.28% of 9000 tweets. Benchmark rates are similar (2.0%, 16.2%, 0.7%, 5.7%), so there is no preprocessing shift.

For each item below, implement it ONLY if `preprocess(..., clean=True)` does not already do it. Order inside `preprocess`: URL removal -> gmail token -> mention token -> `emoji.replace_emoji` -> (new) emoticons -> hashtags -> elongation -> whitespace normalization.

```python
EMOTICONS = [   # ">:(" must be matched before ":("
    (r"(?<!\w)(?:>:-?\(|-_-|:-?/)(?!\w)", " annoyed face "),
    (r"(?<!\w)(?::'?-?[(\[]|=\(|T_T|;_;)(?!\w)", " sad face "),
    (r"(?<!\w)<3(?!\w)", " heart "),
    (r"(?<!\w)(?:[:=;]-?[)\]D]|\^_?\^)(?!\w)", " smiling face "),
]
for pattern, words in EMOTICONS:
    text = re.sub(pattern, words, text)
text = re.sub(r"#(\w+)", lambda m: " " + re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", m.group(1)) + " ", text)
text = re.sub(r"([A-Za-z])\1{2,}", r"\1\1", text)   # sooooo -> soo; digits untouched
```

- `html.unescape` already runs first when `clean=True` (so `&lt;3` becomes `<3`); keep that order.
- These edits must NOT affect the split (that is why `split_key` is frozen in 3.1).
- Add `tests/test_preprocessing.py` with at least: `"@united #AspenBaggageFail pls fix"` contains "Aspen Baggage Fail"; `"flight cancelled :("` contains "sad face"; `"I love it <3"` contains "heart"; `"soooo late"` becomes `"soo late"`; `"1000 miles"` unchanged; `"http://x.co/a:/"` produces no emoticon token.
- Expect this to be a small effect (affects about 2% of tweets for emoticons); do not claim a gain without section 5 evidence.

### 3.8 Paired comparison tool (new file `compare_runs.py`)

**Why:** decide whether a change helped, with uncertainty.

CLI: `python compare_runs.py RUN_A RUN_B [--n 2000]`. Load both `probs.npz`; assert identical `test_ids` and `val_ids`; ensemble prediction = `argmax(member_probs.mean(0))`. Compute for TWO subsets: **test only** (primary) and **val+test pooled** (secondary). Paired bootstrap:

```python
rng = np.random.default_rng(0)
n = len(labels)
diffs = []
for _ in range(n_boot):
    idx = rng.integers(0, n, n)
    fa = f1_score(labels[idx], pred_a[idx], labels=[0, 1, 2], average="macro", zero_division=0)
    fb = f1_score(labels[idx], pred_b[idx], labels=[0, 1, 2], average="macro", zero_division=0)
    diffs.append(fb - fa)
print(mean, np.percentile(diffs, [2.5, 97.5]), (np.array(diffs) > 0).mean())
```

Also print per-class F1 for both runs. Print the verdict "no detectable difference" when the 95% interval contains 0.

### 3.9 Benchmark diagnostics (new file `diagnose_benchmark.py`)

CLI: `python diagnose_benchmark.py RUN_DIR`. Using `probs.npz` print, for validation, test and benchmark: predicted class shares of the ensemble, mean predicted probability per class, mean max-probability, mean entropy; and the members' pairwise disagreement rate on each split. Interpretation line to print: if benchmark mean probabilities match validation but hard predictions lean negative -> threshold issue; if benchmark mean neutral/positive probability is lower than validation -> distribution shift.

Optional (default OFF, diagnostic only, never applied to submissions): Saerens EM prior estimate

```python
def em_prior(probs, train_prior, iters=100):
    prior = train_prior.copy()
    for _ in range(iters):
        adjusted = probs * (prior / train_prior)
        adjusted /= adjusted.sum(1, keepdims=True)
        prior = adjusted.mean(0)
    return prior
```

`train_prior` must be the **effective prior** saved in `config.json` (class counts times class weights, normalized), not the raw class frequencies, because training used sqrt-inverse class weights.

### 3.10 Notebook changes (`IAI_HW3.ipynb`)

- Cell with the `run(...)` call becomes:
  ```python
  predictions, histories = run(
      ensemble=True, preset="diverse", select="last", epochs=3, final_fit=True,
      checkpoint_dir="/content/hw3_ckpt", run_dir="/content/runs/diverse_v1",
      review_path="/content/neutral_errors.csv",
  )
  ```
  Remove `tune_neutral=...` and `sweep_path=...`.
- Keep or delete the `baseline_score` cell (it only prints "unavailable"); if kept, leave the guarded `if baseline is not None` line.
- Download cell: save `predictions` (already the final-fit predictions when `final_fit=True`) to `/content/predictions.csv`; add a cell that zips `/content/runs/<name>` for download.
- Add cells to run `compare_runs.py` and `diagnose_benchmark.py` with `!python`.
- Update the markdown header cell to say members are chosen by preset and the vote is plain equal-weight soft voting.

## 4. Tests (add `tests/`, run with `pytest`; CPU only, no model downloads)

1. `test_splits.py`: partitions are disjoint; no `split_key` appears in two partitions; sizes equal 7202/901/897 for the shipped dataset; `load_or_create_splits` raises if `dataset.csv` hash changes.
2. `test_voting.py`: equal soft equals the mean of member probabilities.
3. `test_heads.py`: `MeanPoolHead`/`ClsMeanHead` output shape `[B, 3]`; padding tokens do not change the output (compare a batch with and without extra padded positions).
4. `test_presets.py`: every preset has exactly 3 members, unique names and unique seeds.
5. `test_preprocessing.py`: cases listed in 3.7.
6. `test_compare.py`: comparing a run with itself gives difference 0 and an interval containing 0.

## 5. Experiment protocol (after the code is merged)

Run each config once on the frozen split (each run trains 3 models; with `final_fit` it is 6). Compare every run to **E0** with `compare_runs.py`, using the **test-only** interval as the primary evidence.

| Run | Command arguments | Question it answers |
|---|---|---|
| E0 | `preset=legacy select=best-f1` | Baseline on the frozen split (old behaviour, equal soft vote) |
| E1 | `preset=legacy select=last` | Does dropping best-epoch selection hurt/help? |
| E2 | `preset=diverse select=last subsample=1.0` | Does head diversity help? |
| E3 | `preset=diverse select=last subsample=0.9` | Does data bagging help on top of E2? |
| E4 | `preset=diverse-cased select=last` | Only if the user confirms cased checkpoints are allowed |

Decision rules:
- A change counts only if the 95% paired-bootstrap interval on the **test set** excludes 0 or the pooled interval excludes 0 **and** the test-only mean difference is positive. Differences under about 0.01 are noise at these sample sizes.
- Pick the best run, re-run it with `final_fit=True`, run `diagnose_benchmark.py`, and only then submit to Kaggle. Spend Kaggle submissions only on that final candidate (and optionally E0 final-fit as a reference).
- If no run beats E0 with confidence, submit the E0 `final_fit` predictions anyway: more training data is the one change expected to help without needing proof, but say so explicitly.

## 6. Things NOT to do

- Do not re-introduce vote-weight grids, neutral multipliers or logit offsets tuned on validation.
- Do not change the split key or seeds after `splits.json` exists.
- Do not use `benchmark.csv` text for any training signal (no pseudo-labels, no MLM on it) unless the user explicitly approves it.
- Do not add a fourth model, checkpoint-snapshot ensembling, or test-time augmentation that creates extra models.
- Do not change the training recipe (LR, epochs, smoothing, layer decay) in the same commit as protocol changes.
- Do not enable `diverse-cased` by default.

## 7. Open questions the user must answer before enabling optional parts

1. Does the assignment allow `bert-base-cased` (or other BERT-base-size checkpoints), or only `bert-base-uncased`? (Controls preset `diverse-cased`.)
2. Is retraining the final three models on train+validation+test (our own labeled split, not the benchmark) acceptable? (Controls `--final-fit`; the evaluation-pass models are used only for measurement and are not part of the submission.)
3. Which GPU does Colab give? (Runtime per model is roughly 5-10 minutes on a T4 at batch 16, max length 128; this is an estimate, not measured.)
