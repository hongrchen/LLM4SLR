# Deploying Local Large Language Models for Automated Article Screening in Scientific Literature Reviews

This repository contains the code for evaluating local large language models (LLMs) as automated screeners in a scientific literature review workflow. Screening is evaluated at the two standard stages of study selection:

1. **Title/abstract (TIAB) screening** — the model reads a paper's title and abstract and classifies it as `Relevant`, `Not Relevant`, or `Uncertain`.
2. **Full-text screening** — the model reads the paper's full text (converted from PDF to markdown) and classifies it as `Relevant` or `Not Relevant`.

Each script runs one model over a dataset labeled by human reviewers, saves the per-paper predictions, and assess agreement with the human labels.

## Repository structure

```
.
├── README.md
├── LICENSE                       # MIT license
├── pyproject.toml                # project metadata + dependencies (for uv)
├── requirements.txt              # same dependencies, for pip users
├── title_abstract_screening/     # one script per model (self-contained)
│   ├── tiab_gemma3-12b-it.py
│   ├── ...
│   └── tiab_qwen3-4b-it.py
├── fulltext_screening/
│   ├── fulltext_common.py        # shared helpers (data loading, OOM retry, evaluation, run logging)
│   ├── fulltext_gemma3-12b-it.py # one script per model
│   ├── ...
│   └── fulltext_qwen3-4b-it.py
├── data/
│   ├── 300p_human_tiab_final.xlsx      # human title/abstract labels (included)
│   ├── 300p_human_fulltext_final.xlsx  # human full-text labels (included)
│   └── 300p_full_text_md/              # full texts as markdown (not included, see below)
└── results/                      # created automatically by the scripts
```

## Setup

Requirements: Python ≥ 3.14 (developed on 3.14) and a CUDA GPU (models are loaded with `device_map="auto"`; the larger models need multiple GPUs or substantial VRAM).

The recommended way to set up the environment is with [uv](https://docs.astral.sh/uv/), which creates the virtual environment and installs all dependencies from `pyproject.toml` in one step:

```bash
uv sync
```

Alternatively, install into your own environment with pip:

```bash
pip install -r requirements.txt
```

Some models (Llama, Gemma) are gated on Hugging Face — request access on their model pages and log in before running:

```bash
uv run hf auth login    # or `hf auth login` with pip
```

## Data

The dataset consists of 300 papers screened for a systematic review on AI methods for estimating treatment effects from real-world medical data (the inclusion criteria are spelled out in each script's prompt). Each paper is identified by the `No.` column, which holds the string IDs `a1`–`a100`, `b1`–`b100` and `c1`–`c100`; the `train` / `test` splits described below follow this order.

### Included: human screening labels

The two human-labeled spreadsheets are part of this repository, so both screening stages can be evaluated against the same reference labels used in the study.

| Path                                    | Description                                                                                                                                                                                                                |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data/300p_human_tiab_final.xlsx`     | The 300 papers with columns `No.`, `Title`, `Abstract`, `Authors` and `HUMAN_Classification` — the human title/abstract screening label, available for all 300 papers.                                          |
| `data/300p_human_fulltext_final.xlsx` | The same 300 papers, in the same order, with the additional column `Human_Fulltext_Classification` — the human full-text screening label (`Relevant` / `Not Relevant`). Full texts from 7 papers are not available. |

For either file, a `.csv` with the same name is accepted as a fallback to `.xlsx`. 

### Not included: full texts

The full-text markdown files contain licensed article content and are therefore not part of this repository. To run full-text screening, obtain the PDFs, convert them to markdown and place them at the following paths relative to the repository root:

| Path                                            | Description                                                                                                                                                              |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `data/300p_full_text_md/docling/<No.>.md`     | Full text of each paper, converted from PDF with [Docling](https://github.com/docling-project/docling), one markdown file per paper named by its `No.` (e.g. `a1.md`) |
| `data/300p_full_text_md/pymupdf4llm/<No.>.md` | The same full texts converted with [PyMuPDF4LLM](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/)                                                                   |

Papers whose markdown file is missing are skipped with a warning, so a partial set of full texts is enough for a test run.

## Running title/abstract screening

```bash
uv run title_abstract_screening/tiab_gemma3-12b-it.py    # or `python ...` with pip
```

All scripts share the same configuration block at the top:

- `dataset_type` — which slice of the 300-paper dataset to screen: `"train"` (first 200), `"test"` (last 100), `"trial"` (20 random papers, useful as a smoke test), or `"all"` (default).

Results are written to `results/tiab_results/tiab_<model>_<dataset_type>.csv`, containing the input columns plus the model's raw output and the parsed `LLM_Classification` label. After the run, the script prints raw and relaxed accuracy against `HUMAN_Classification` (see [Evaluation](#evaluation)).

## Running full-text screening

```bash
uv run fulltext_screening/fulltext_gemma3-12b-it.py    # or `python ...` with pip
```

Shared configuration block at the top of each script:

- `md_source` — which PDF-to-markdown extraction to screen: `"docling"` (default) or `"pymupdf4llm"`.
- `dataset_type` — same options as above (default `"all"`).
- `max_chars` / `min_chars` — each paper's full text is truncated to `max_chars` characters before prompting. On a CUDA out-of-memory error, only the offending paper is retried with progressively less text (down to `min_chars`). `max_chars` is tuned per model; lower it if you have less VRAM.

Results are written to `results/full_text_results/fulltext_<model>_<md_source>_<dataset_type>.csv`. The accuracies are computed over the 293 papers that have a human full-text label (see [Data](#data)). Each run also appends a summary row (model, settings, accuracies, runtime) to `results/full_text_results/evaluation_summary.csv`, so runs can be compared without collecting numbers by hand.

## Models

Each screening stage has one script per model; the script name encodes the model.

| Script suffix              | Hugging Face model                           |
| -------------------------- | -------------------------------------------- |
| `gemma3-12b-it`          | `google/gemma-3-12b-it`                    |
| `gemma3-27b-it`          | `google/gemma-3-27b-it`                    |
| `gemma4-26B-A4B-it`      | `google/gemma-4-26B-A4B-it`                |
| `gemma4-31B-it`          | `google/gemma-4-31B-it`                    |
| `gpt-oss-20b`            | `openai/gpt-oss-20b`                       |
| `llama3.1-8b-it`         | `meta-llama/Meta-Llama-3.1-8B-Instruct`    |
| `mistral3-14b-it`        | `mistralai/Ministral-3-14B-Instruct-2512`  |
| `mistral3-14b-reasoning` | `mistralai/Ministral-3-14B-Reasoning-2512` |
| `mistral3-3b-it`         | `mistralai/Ministral-3-3B-Instruct-2512`   |
| `qwen3-14b`              | `Qwen/Qwen3-14B`                           |
| `qwen3-4b-it`            | `Qwen/Qwen3-4B-Instruct-2507`              |

The scripts intentionally differ slightly per model (chat/message format, output parsing, generation settings) to match each model family's conventions; the data handling and evaluation are identical.

## Evaluation

Two agreement metrics are printed after every run:

- **Raw accuracy** — exact match between the model label and the human label.
- **Relaxed accuracy** — a prediction only counts as an error when the model would *wrongly exclude* a paper, i.e. it predicts `Not Relevant` while the human label is not `Not Relevant`. This reflects the screening use case, where we would like to not miss a relevant study.

## Tips

- Full-text runs over all 300 papers can take hours. Run them inside `tmux` (or with `nohup`) so they survive an SSH disconnect.
- `gpt-oss-20b` note: the original MXFP4 checkpoint's triton kernels require a Hopper-or-newer GPU (sm90) and may crash on older GPUs (e.g. L40S, sm89). On such GPUs, convert the checkpoint to plain bf16 locally and point `model_id` in the script at the converted folder.

## Authors

- Ling-Hsuan Lao — Epidemiology, Biostatistics and Prevention Institute, University of Zurich, Zurich, Switzerland
- Hongruyu Chen — Epidemiology, Biostatistics and Prevention Institute, University of Zurich, Zurich, Switzerland

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
