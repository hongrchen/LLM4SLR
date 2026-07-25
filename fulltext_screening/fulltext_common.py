"""Shared helpers for full-text screening scripts.

Mirrors the dataset loading / evaluation conventions used in tiab_analysis,
adapted for full-text markdown inputs and a 2-category
("Relevant" / "Not Relevant") classification scheme.
"""

import os
import sys
from datetime import datetime

import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

# Where the per-run evaluation summary is appended (relative to the project root)
EVALUATION_SUMMARY_FILE = "results/full_text_results/evaluation_summary.csv"


def load_paper_data(input_filename, dataset_type="test"):
    """Load the 300-paper dataset and select a subset.

    dataset_type:
        "train"  -> first 200 rows
        "test"   -> last 100 rows
        "trial"  -> 20 random rows (seed=42)
        "all"    -> all rows
    """
    # Pick the reader by file extension so an .xlsx path isn't fed to read_csv
    # (which raises a ParserError on the binary content, not FileNotFoundError).
    ext = os.path.splitext(input_filename)[1].lower()
    if ext in (".xlsx", ".xls"):
        candidates = [(input_filename, "excel"),
                      (input_filename.rsplit(".", 1)[0] + ".csv", "csv")]
    else:
        candidates = [(input_filename, "csv"),
                      (input_filename.rsplit(".", 1)[0] + ".xlsx", "excel")]

    df = None
    for path, kind in candidates:
        if not os.path.exists(path):
            continue
        df = pd.read_excel(path) if kind == "excel" else pd.read_csv(path, encoding="latin-1")
        break

    if df is None:
        tried = ", ".join(path for path, _ in candidates)
        print(f"Error: File not found. Tried {tried}")
        sys.exit(1)

    dataset_type = dataset_type.lower()
    if dataset_type == "train":
        df = df.iloc[:200]
        print(f"Using training data: {len(df)} rows")
    elif dataset_type == "test":
        df = df.iloc[-100:]
        print(f"Using test data: {len(df)} rows")
    elif dataset_type == "trial":
        df = df.sample(n=20, random_state=42)
        print(f"Using trial data: {len(df)} rows")
    elif dataset_type == "all":
        print(f"Using all data: {len(df)} rows")
    else:
        print(f"Invalid dataset_type: {dataset_type}. Using 'train' by default.")
        df = df.iloc[:200]
        print(f"Using training data: {len(df)} rows")

    return df.reset_index(drop=True)


def read_full_text(paper_id, md_dir, max_chars=None):
    """Read the markdown full text for a single paper. Returns None if the file doesn't exist."""
    md_path = os.path.join(md_dir, f"{paper_id}.md")
    if not os.path.exists(md_path):
        return None
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars]
    return text


def filter_papers_with_fulltext(df, md_dir, id_column="No."):
    """Drop rows whose markdown file doesn't exist, warning about missing ones."""
    mask = df[id_column].astype(str).apply(
        lambda pid: os.path.exists(os.path.join(md_dir, f"{pid}.md"))
    )
    missing = df.loc[~mask, id_column].tolist()
    if missing:
        print(f"Warning: {len(missing)} full-text file(s) not found and will be skipped: {missing}")
    return df[mask].reset_index(drop=True)


# Full-text markdown files are large (tens to hundreds of thousands of
# characters), far beyond what fits in a single prompt, so we truncate before
# building it. Token-dense papers (tables/math) can still OOM the GPU at a given
# char budget, so we shrink *only that paper* and retry until it fits.
# These are defaults only; each screening script sets its own max_chars/min_chars
# so the budget can be tuned per model (context window, free VRAM).
FULLTEXT_MAX_CHARS = 30000
FULLTEXT_MIN_CHARS = 4000


def generate_with_oom_retry(pipe, build_messages, paper_id, md_dir,
                            max_chars=FULLTEXT_MAX_CHARS,
                            min_chars=FULLTEXT_MIN_CHARS, **gen_kwargs):
    """Run `pipe` on one paper's (truncated) full text, shrinking the text and
    retrying on CUDA OOM.

    `build_messages(full_text)` returns the messages list, letting each script
    keep its own chat format / system prompt while sharing the truncation and
    retry logic. Most papers run at the full `max_chars` budget; only the few
    un-fittable ones get trimmed (logged via tqdm.write). Fragmented CUDA memory
    is released after each paper so varying prompt lengths don't accumulate
    reserved-but-unusable fragments across iterations until they OOM.
    """
    char_limit = max_chars
    while True:
        full_text = read_full_text(paper_id, md_dir, max_chars=char_limit) or ""
        messages = build_messages(full_text)
        try:
            outputs = pipe(messages, **gen_kwargs)
            break
        except torch.OutOfMemoryError:
            # Release the failed allocation's fragments, then retry this paper
            # with ~2/3 the text. Bail only if even a short prompt won't fit.
            torch.cuda.empty_cache()
            if char_limit <= min_chars:
                raise
            char_limit = char_limit * 2 // 3
            tqdm.write(f"OOM on paper {paper_id}; retrying with max_chars={char_limit}")

    # Release fragmented CUDA memory before the next (possibly longer) prompt.
    torch.cuda.empty_cache()
    return outputs


def evaluate(df, true_col="Human_Fulltext_Classification", pred_col="LLM_Classification",
              id_column="No."):
    """Compare the LLM predictions against the human full-text ground truth
    (already present in df), print raw and relaxed accuracy, and attach
    correctness columns to df."""

    print("\nStarting evaluation...")

    df = df.copy()
    df[true_col] = df[true_col].astype(str).str.strip()

    # Drop rows without a usable ground-truth label (e.g. "x" / missing)
    valid = df[true_col].isin(["Relevant", "Not Relevant"])
    if not valid.all():
        dropped = df.loc[~valid, id_column].tolist()
        print(f"Warning: {len(dropped)} row(s) without a valid ground-truth label will be excluded from evaluation: {dropped}")
        df = df[valid].reset_index(drop=True)

    y_true = df[true_col]
    y_pred = df[pred_col]

    # ----------------------------------------------------
    # 1. Raw Accuracy
    # ----------------------------------------------------
    raw_accuracy = accuracy_score(y_true, y_pred)

    df["correct_raw"] = (y_true == y_pred)
    raw_correct = df["correct_raw"].sum()
    raw_total = len(df)
    raw_errors = raw_total - raw_correct

    print("\n=== Raw Accuracy ===")
    print(f"Correct: {raw_correct} / {raw_total}")
    print(f"Errors:  {raw_errors}")
    print(f"Raw Accuracy: {raw_accuracy:.4f}")

    # ----------------------------------------------------
    # 2. Relaxed Accuracy
    # ----------------------------------------------------
    def relaxed_is_correct(row):
        true = row[true_col]
        pred = row[pred_col]

        if true != "Not Relevant" and pred == "Not Relevant":
            return False

        return True

    df["correct_relaxed"] = df.apply(relaxed_is_correct, axis=1)

    relaxed_correct = df["correct_relaxed"].sum()
    relaxed_total = len(df)
    relaxed_errors = relaxed_total - relaxed_correct
    relaxed_accuracy = relaxed_correct / relaxed_total

    print("\n=== Relaxed Accuracy ===")
    print(f"Correct: {relaxed_correct} / {relaxed_total}")
    print(f"Errors:  {relaxed_errors}")
    print(f"Relaxed Accuracy: {relaxed_accuracy:.4f}")

    return df


def log_run(df, project_dir, model_id, output_filename, dataset_type, elapsed_seconds,
            true_col="Human_Fulltext_Classification", pred_col="LLM_Classification",
            summary_file=EVALUATION_SUMMARY_FILE, md_source=None,
            max_chars=FULLTEXT_MAX_CHARS):
    """Append a row summarizing this run's evaluation results and running
    time to a shared CSV, so results don't need to be copied by hand.

    `max_chars` records the full-text truncation budget used for the run
    (defaults to the shared FULLTEXT_MAX_CHARS the screening helper applies).
    """

    raw_accuracy = accuracy_score(df[true_col], df[pred_col])
    relaxed_accuracy = df["correct_relaxed"].mean()

    row = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_id": model_id,
        "dataset_type": dataset_type,
        "md_source": md_source,
        "max_chars": max_chars,
        "n_papers": len(df),
        "raw_accuracy": round(raw_accuracy, 4),
        "relaxed_accuracy": round(relaxed_accuracy, 4),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "elapsed_minutes": round(elapsed_seconds / 60, 2),
        "output_file": os.path.relpath(output_filename, project_dir),
    }])

    summary_path = os.path.join(project_dir, summary_file)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    # Read-modify-write rather than a plain append: adding a new column (e.g.
    # max_chars) to a summary that predates it would otherwise misalign against
    # the old header. Concatenating realigns by column name, so older rows just
    # get a blank value for the new column.
    if os.path.exists(summary_path):
        existing = pd.read_csv(summary_path)
        combined = pd.concat([existing, row], ignore_index=True)
    else:
        combined = row
    combined.to_csv(summary_path, index=False)

    print(f"\nRun summary written to '{summary_path}'")
    return row


# ---------------------------------------------------------------------------
# Prompt templates (full-text, 2-category versions of the tiab prompts)
# ---------------------------------------------------------------------------

# Used by the "criteria-based" tiab prompts (gpt-oss, gemma3, gemma-4-*-it,
# llama, mistral, qwen scripts).
FULLTEXT_PROMPT_CRITERIA = """You are reviewing the full text of a research article to determine its relevance based on predefined criteria. Carefully read the provided full text and evaluate whether it meets each of the criteria below. Base your judgment only on the information provided in the text.

We define the following three criteria:
1. Use of AI Methods
The study must use an algorithmic learning approach rather than a fixed statistical formula. A method counts as AI if it meets at least one of the following: it automatically models non-linear relationships and interactions (e.g., Trees, Neural Nets, Bayesian models); or it uses algorithmic optimization like Boosting, Bagging, or Cross-Validated tuning; or it uses regularization (like Lasso) to handle high-dimensional variable sets. Standard "Fixed-Effect" models (OLS, Logistic Regression, 2-Stage Least Squares) and standard Hypothesis Tests (T-tests, ANOVA) are not AI.
2. Estimation of Treatment Effects
The study must attempt to answer a "What if?" question (Counterfactual effect such as ATE, CATE, HTE, ITE). It must measure the impact of an intervention on an outcome using AI methods. Exclude purely prognostic/predictive models. For example, a model that predicts "Risk of Heart Attack" is not a treatment effect study unless it specifically measures how a specific medication changes that risk.
3. Use of Real-World Medical Data
The study must use real-world medical data (e.g., observational data or clinical trial data related to medical interventions or health outcomes), and the AI methods used for estimating treatment effects must be applied to this data.

Classification Rules:
Classify as "Relevant" if all three criteria are clearly satisfied.
Classify as "Not Relevant" if any one of the three criteria is not satisfied or cannot be clearly confirmed from the full text.

Output Format:
Your response must be exactly one of the following: "Relevant", "Not Relevant".

Now evaluate the following article:
Title: {title}

Full Text:
{full_text}
"""

