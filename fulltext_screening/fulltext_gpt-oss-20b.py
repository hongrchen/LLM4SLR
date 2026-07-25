import os
import sys
import time
import warnings

# Reduce CUDA memory fragmentation across iterations (varying prompt lengths
# otherwise leave reserved-but-unusable fragments that accumulate until OOM).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import pipeline
from tqdm.auto import tqdm
from transformers.utils import logging

from fulltext_common import (
    FULLTEXT_PROMPT_CRITERIA,
    evaluate,
    filter_papers_with_fulltext,
    load_paper_data,
    log_run,
    read_full_text,
)

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

start_time = time.time()

# --- 1. Setup and Configuration ---

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

input_filename = os.path.join(project_dir, "data/300p_human_fulltext_final.xlsx")
output_filename = os.path.join(project_dir, "results/full_text_results/fulltext_gpt-oss-20b.csv")

# Which OCR/markdown extraction to use for the full text: "docling" or "pymupdf4llm"
md_source = "docling"  # Change to "pymupdf4llm" to use the other extraction
md_dir = os.path.join(project_dir, "data/300p_full_text_md", md_source)

# Choose dataset: "train" for first 200 rows, "test" for last 100 rows, "trial" for 20 random papers, "all" for all 300 papers
dataset_type = "all"  # Change to "train", "test", or "trial" to use different data

# Truncate each paper's full text to at most max_chars before prompting; on a
# CUDA OOM, retry this paper with progressively less text down to min_chars.
# Tune per model (smaller context window / less free VRAM -> lower max_chars).
max_chars = 20000
min_chars = 4000

# Embed the md source and dataset type in the output filename (e.g. "..._docling_test.csv")
_base, _ext = os.path.splitext(output_filename)
output_filename = f"{_base}_{md_source}_{dataset_type}{_ext}"

df = load_paper_data(input_filename, dataset_type)
df = filter_papers_with_fulltext(df, md_dir)
paper_ids = df["No."].astype(str).tolist()

# Specify the model ID
# Note: the original MXFP4 checkpoint's triton kernels assume a Hopper+ GPU
# (sm90) and crash with "illegal memory access" on older GPUs (e.g. L40S,
# sm89). On such GPUs, convert the checkpoint to plain bf16 locally first
# and point model_id at that folder instead (e.g. "models/gpt-oss-20b-bf16").
model_id = "openai/gpt-oss-20b"

# Initialize the text-generation pipeline
# This will automatically use a GPU if available
print("Initializing the text-generation pipeline...")
# Cap per-GPU memory so the model spreads across all 4 GPUs instead of being
# crammed onto GPU 0 with no room left for activations (gpt_oss only
# supports "eager" attention, which materializes an O(seq_len^2) attention
# matrix per head, so the full text is also truncated above).
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
    model_kwargs={
        "max_memory": {i: "12GiB" for i in range(torch.cuda.device_count())},
    },
)
print("Pipeline initialized successfully.")

# --- 2. Prompt Definition ---
prompt_template = FULLTEXT_PROMPT_CRITERIA

# --- 3. Processing and Classification ---
results = []
reason = []

# Use tqdm for a progress bar to monitor the classification process
print("Starting classification of articles...")
for i, row in enumerate(tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc="Classifying Articles",
        dynamic_ncols=True,
        file=sys.stdout
)):
    # Start at the full truncation limit. Token-dense papers (tables/math) pack
    # far more tokens into the same char count, so their eager O(seq_len^2)
    # attention matrix can OOM even a whole GPU. If that happens, shorten *only
    # this paper* and retry until it fits; most papers run at the full max_chars budget and
    # only the few un-fittable ones get trimmed (logged below).
   
    while True:
        full_text = read_full_text(paper_ids[i], md_dir, max_chars=max_chars)

        prompt = prompt_template.format(
            title=row.Title,
            full_text=full_text
        )

        messages = [
            {"role": "user", "content": prompt},
        ]

        # Generate the classification using the pipeline
        try:
            outputs = pipe(
                messages,
                max_new_tokens=2000,
                return_full_text=False,
                do_sample=False,
            )
            break
        except torch.OutOfMemoryError:
            # Release the failed allocation's fragments, then retry this paper
            # with ~2/3 the text. Bail only if even a short prompt won't fit.
            torch.cuda.empty_cache()
            if max_chars <= min_chars:
                raise
            max_chars = max_chars * 2 // 3
            tqdm.write(
                f"OOM on paper {paper_ids[i]}; retrying with max_chars={max_chars}"
            )

    # Extract the generated text, which is the model's classification
    # We clean up the output to get only the classification word
    generated_text = outputs[0]["generated_text"].strip()

    reason.append(generated_text)


    # Find the most likely classification in the response
    if "Relevant" in generated_text[-20:] and "Not Relevant" not in generated_text[-20:]:
        classification = "Relevant"
    elif "Not Relevant" in generated_text[-20:]:
        classification = "Not Relevant"
    else:
        classification = generated_text[-20:]  # Fallback for unexpected output

    results.append(classification)

    # Release fragmented CUDA memory before the next (possibly longer) prompt.
    torch.cuda.empty_cache()

print("Classification complete.")

# --- 4. Save Results ---
# Add the list of results as a new column to the DataFrame
df["LLM_reasoning"] = reason
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
