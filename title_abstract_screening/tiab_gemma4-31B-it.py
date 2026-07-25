import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm
import numpy as np
import os
from sklearn.metrics import accuracy_score
import time
import sys

# --- 1. Setup and Configuration ---

start_time = time.time()

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

input_filename = os.path.join(project_dir, "data/300p_human_tiab_final.xlsx")
output_filename = os.path.join(project_dir, "results/tiab_results/tiab_gemma4-31B-it.csv")

# Choose dataset: "train" for first 200 rows, "test" for last 100 rows,
# "trial" for 20 random papers, "all" for all 300 papers
dataset_type = "all"

# Embed the dataset type in the output filename (e.g. "..._test.csv")
_base, _ext = os.path.splitext(output_filename)
output_filename = f"{_base}_{dataset_type}{_ext}"

# Generation settings: the model only needs to emit a short label.
max_new_tokens_length = 10


# Specify the model ID for the Gemma-4-31B-it model
model_id = "google/gemma-4-31B-it"


# Load the dataset (Excel, falling back to CSV)
try:
    df = pd.read_excel(input_filename)
except FileNotFoundError:
    try:
        input_filename_csv = input_filename.replace(".xlsx", ".csv")
        df = pd.read_csv(input_filename_csv, encoding="latin-1")
    except FileNotFoundError:
        print(f"Error: File not found. Tried {input_filename} and {input_filename_csv}")
        sys.exit(1)

# Select dataset based on configuration
if dataset_type.lower() == "train":
    df = df.iloc[:200]  # First 200 rows for training
    print(f"Using training data: {len(df)} rows")
elif dataset_type.lower() == "test":
    df = df.iloc[-100:]  # Last 100 rows for testing
    print(f"Using test data: {len(df)} rows")
elif dataset_type.lower() == "trial":
    df = df.sample(n=20, random_state=42)  # 20 random papers for testing
    print(f"Using trial data: {len(df)} rows")
elif dataset_type.lower() == "all":
    print(f"Using all data: {len(df)} rows")
else:
    print(f"Invalid dataset_type: {dataset_type}. Using 'all' by default.")
    print(f"Using all data: {len(df)} rows")

# Choose a Gemma 4 instruction-tuned model
# Good smaller option: google/gemma-4-E2B-it
# Better quality / still manageable: google/gemma-4-E4B-it
# Larger options: google/gemma-4-26B-A4B-it, google/gemma-4-31B-it

print("Initializing the text-generation pipeline...")
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)
print("Pipeline initialized successfully.")

# --- 2. Prompt Definition ---

prompt_template = """You are reviewing texts to determine its relevance based on predefined criteria. Carefully read the provided texts and evaluate whether it meets each of the criteria below. Base your judgment only on the information provided in the texts. If information is missing or unclear, do not assume—use the "Uncertain" category when appropriate.

We define the following three criteria:
1.	Use of AI Methods
The study must use an algorithmic learning approach rather than a fixed statistical formula. A method counts as AI if it meets at least one of the following: it automatically models non-linear relationships and interactions (e.g., Trees, Neural Nets, Bayesian models); or it uses algorithmic optimization like Boosting, Bagging, or Cross-Validated tuning; or it uses regularization (like Lasso) to handle high-dimensional variable sets. Standard "Fixed-Effect" models (OLS, Logistic Regression, 2-Stage Least Squares) and standard Hypothesis Tests (T-tests, ANOVA) are not AI.
2.	Estimation of Treatment Effects
The study must attempt to answer a "What if?" question (Counterfactual effect such as ATE, CATE, HTE, ITE). It must measure the impact of an intervention on an outcome using AI methods. Exclude purely prognostic/predictive models. For example, a model that predicts "Risk of Heart Attack" is not a treatment effect study unless it specifically measures how a specific medication changes that risk.
3.	Use of Real-World Medical Data
The study must use real-world medical data (e.g., observational data or clinical trial data related to medical interventions or health outcomes), and the AI methods used for estimating treatment effects must be applied to this data.

Classification Rules
•	Classify as "Relevant" if all three criteria are clearly satisfied.
•	Classify as "Not Relevant" if any one of the three criteria is clearly not satisfied.
•	Classify as "Uncertain" if at least one criterion is unclear, and the remaining criteria are satisfied. For example, classify as Uncertain if the methodology or data source is not mentioned, or if the medical nature of the data is ambiguous. However, if the context clearly implies a non-medical domain or a purely statistical (non-AI) approach, classify as Not Relevant.

Output Format
Your response must be exactly one of the following: "Relevant", "Not Relevant", "Uncertain". 

Now evaluate the following article:
Title: {title}
Abstract: {abstract}
"""

# --- 3. Processing and Classification ---

results = []
LLMoutputs = []
print("Starting classification of articles...")

for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Classifying Articles"):
    title = str(row.get("Title", ""))
    abstract = str(row.get("Abstract", ""))

    prompt = prompt_template.format(title=title, abstract=abstract)

    messages = [
        {
            "role": "system",
            "content": "You are a strict research screening assistant. Output exactly one label: Relevant, Not Relevant, or Uncertain."
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    outputs = pipe(
        messages,
        max_new_tokens=max_new_tokens_length,
        do_sample=False,
    )

    LLMoutputs.append(outputs[0])
    generated_text = outputs[0]["generated_text"][-1]["content"].strip()

    # Normalize and extract the label robustly
    text = generated_text.lower()

    if "not relevant" in text:
        classification = "Not Relevant"
    elif "uncertain" in text:
        classification = "Uncertain"
    elif "relevant" in text:
        classification = "Relevant"
    else:
        classification = generated_text  # fallback for unexpected output

    results.append(classification)

print("Classification complete.")

# --- 4. Save Results ---

df["LLM_Classification"] = results
df["LLM_Output"] = LLMoutputs

os.makedirs(os.path.dirname(output_filename), exist_ok=True)
df.to_csv(output_filename, index=False)

end_time = time.time()
elapsed = end_time - start_time
print(f"\nTotal time: {elapsed:.2f} seconds")
print(f"Total time: {elapsed/60:.2f} minutes")
print(f"Total time: {elapsed/3600:.2f} hours")



# --- 5. Evaluation ---

print("\nStarting evaluation...")

df = pd.read_csv(output_filename)  # Load the results for evaluation

# Ground truth vs prediction
y_true = df["HUMAN_Classification"]
y_pred = df["LLM_Classification"]

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
    true = row["HUMAN_Classification"]
    pred = row["LLM_Classification"]

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


