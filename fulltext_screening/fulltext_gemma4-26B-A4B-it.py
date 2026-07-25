import os
import time

# Reduce CUDA memory fragmentation across iterations (varying prompt lengths
# otherwise leave reserved-but-unusable fragments that accumulate until OOM).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd
from transformers import pipeline
from tqdm import tqdm

from fulltext_common import (
    FULLTEXT_PROMPT_CRITERIA,
    evaluate,
    filter_papers_with_fulltext,
    generate_with_oom_retry,
    load_paper_data,
    log_run,
)

start_time = time.time()

# --- 1. Setup and Configuration ---

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

input_filename = os.path.join(project_dir, "data/300p_human_fulltext_final.xlsx")
output_filename = os.path.join(project_dir, "results/full_text_results/fulltext_gemma4-26B-A4B-it.csv")

# Which OCR/markdown extraction to use for the full text: "docling" or "pymupdf4llm"
md_source = "docling"  # Change to "pymupdf4llm" to use the other extraction
md_dir = os.path.join(project_dir, "data/300p_full_text_md", md_source)

# Choose dataset: "train" for first 200 rows, "test" for last 100 rows, "trial" for 20 random papers
dataset_type = "all"  # Change to "train", "test", or "trial" to use different data

# Truncate each paper's full text to at most max_chars before prompting; on a
# CUDA OOM, retry this paper with progressively less text down to min_chars.
# Tune per model (smaller context window / less free VRAM -> lower max_chars).
max_chars = 40000
min_chars = 4000

# Embed the md source and dataset type in the output filename (e.g. "..._docling_test.csv")
_base, _ext = os.path.splitext(output_filename)
output_filename = f"{_base}_{md_source}_{dataset_type}{_ext}"

# Generation settings tuned to allow a brief reasoning step.
max_new_tokens_length = 10

# Specify the model ID for the Gemma-4-26B-A4B model
model_id = "google/gemma-4-26B-A4B-it"

df = load_paper_data(input_filename, dataset_type)
df = filter_papers_with_fulltext(df, md_dir)

print("Initializing the text-generation pipeline...")
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)
print("Pipeline initialized successfully.")

# --- 2. Prompt Definition ---
prompt_template = FULLTEXT_PROMPT_CRITERIA

# --- 3. Processing and Classification ---

results = []
LLMoutputs = []
print("Starting classification of articles...")

for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Classifying Articles"):
    title = str(row.get("Title", ""))

    def build_messages(full_text):
        prompt = prompt_template.format(title=title, full_text=full_text)
        return [
            {
                "role": "system",
                "content": "You are a strict research screening assistant. Output exactly one label: Relevant or Not Relevant."
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

    # Generate the classification using the pipeline, truncating the (large)
    # full text and shrinking/retrying this paper if it OOMs the GPU.
    outputs = generate_with_oom_retry(
        pipe,
        build_messages,
        str(row["No."]),
        md_dir,
        max_chars=max_chars,
        min_chars=min_chars,
        max_new_tokens=max_new_tokens_length,
        do_sample=False,
    )

    # Store only the assistant's generated text, not the full conversation
    # (the input prompt embeds the full text, which we don't want to re-save).
    generated_text = outputs[0]["generated_text"][-1]["content"].strip()
    LLMoutputs.append(generated_text)

    # Normalize and extract the label robustly
    text = generated_text.lower()

    if "not relevant" in text:
        classification = "Not Relevant"
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
df = pd.read_csv(output_filename)  # Load the results for evaluation
df = evaluate(df)

# --- 6. Log run summary ---
log_run(df, project_dir, model_id, output_filename, dataset_type, elapsed, md_source=md_source, max_chars=max_chars)
