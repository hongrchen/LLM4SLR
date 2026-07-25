import os
import re
import sys
import time
import warnings

# Reduce CUDA memory fragmentation across iterations (varying prompt lengths
# otherwise leave reserved-but-unusable fragments that accumulate until OOM).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from transformers import pipeline
from tqdm.auto import tqdm
from transformers.utils import logging

from fulltext_common import (
    FULLTEXT_PROMPT_CRITERIA,
    evaluate,
    filter_papers_with_fulltext,
    generate_with_oom_retry,
    load_paper_data,
    log_run,
)

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

start_time = time.time()

# --- 1. Setup and Configuration ---

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

input_filename = os.path.join(project_dir, "data/300p_human_fulltext_final.xlsx")
output_filename = os.path.join(project_dir, "results/full_text_results/fulltext_mistral3-3b-it.csv")

# Which OCR/markdown extraction to use for the full text: "docling" or "pymupdf4llm"
md_source = "docling"  # Change to "pymupdf4llm" to use the other extraction
md_dir = os.path.join(project_dir, "data/300p_full_text_md", md_source)

# Choose dataset: "train" for first 200 rows, "test" for last 100 rows, "trial" for 20 random papers
dataset_type = "all"  # Change to "train", "test", or "trial" to use different data

# Truncate each paper's full text to at most max_chars before prompting; on a
# CUDA OOM, retry this paper with progressively less text down to min_chars.
# Tune per model (smaller context window / less free VRAM -> lower max_chars).
max_chars = 30000
min_chars = 4000

# Embed the md source and dataset type in the output filename (e.g. "..._docling_test.csv")
_base, _ext = os.path.splitext(output_filename)
output_filename = f"{_base}_{md_source}_{dataset_type}{_ext}"

df = load_paper_data(input_filename, dataset_type)
df = filter_papers_with_fulltext(df, md_dir)
paper_ids = df["No."].astype(str).tolist()

# Specify the model ID for the text generation pipeline
model_id = "mistralai/Ministral-3-3B-Instruct-2512"

# Initialize the text-generation pipeline
# This will automatically use a GPU if available
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
reason = []
modeloutput = []
# Use tqdm for a progress bar to monitor the classification process
print("Starting classification of articles...")
for i, row in enumerate(tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc="Classifying Articles",
        dynamic_ncols=True,
        file=sys.stdout
)):
    def build_messages(full_text):
        prompt = prompt_template.format(
            title=row.Title,
            full_text=full_text
        )
        return [
            {"role": "user", "content": prompt},
        ]

    # Generate the classification using the pipeline, truncating the (large)
    # full text and shrinking/retrying this paper if it OOMs the GPU.
    outputs = generate_with_oom_retry(
        pipe,
        build_messages,
        paper_ids[i],
        md_dir,
        max_chars=max_chars,
        min_chars=min_chars,
        max_new_tokens=2000,
        return_full_text=False,
        do_sample=False,
    )

    # Extract the generated text, which is the model's classification
    generated_text = outputs[0]["generated_text"].strip()

    reason.append(generated_text)
    modeloutput.append(outputs)

    # Find the most likely classification in the response
    matches = re.findall(r"\b(Not Relevant|Relevant)\b", generated_text)

    if matches:
        classification = matches[0]  # first match
    else:
        classification = "ERROR"

    results.append(classification)

print("Classification complete.")

# --- 4. Save Results ---

# Add the list of results as a new column to the DataFrame
df["LLM_reasoning"] = reason
df['LLM_output'] = modeloutput
df['LLM_Classification'] = results

# Save the updated DataFrame to a new CSV file
os.makedirs(os.path.dirname(output_filename), exist_ok=True)
df.to_csv(output_filename, index=False)

print(f"\nClassification results have been saved to '{output_filename}'")
print("\nFirst 5 rows of the new file:")
print(df.head())

end_time = time.time()
elapsed = end_time - start_time
print(f"\nTotal time: {elapsed:.2f} seconds")
print(f"Total time: {elapsed/60:.2f} minutes")
print(f"Total time: {elapsed/3600:.2f} hours")

# --- 5. Evaluation ---
df = evaluate(df)

# --- 6. Log run summary ---
log_run(df, project_dir, model_id, output_filename, dataset_type, elapsed, md_source=md_source, max_chars=max_chars)
