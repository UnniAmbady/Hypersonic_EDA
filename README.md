# Hypersonic — EDA Notebook Generator

**Build a complete Exploratory Data Analysis notebook in minutes.**  
Data scientists and ML folks are often evaluated on their EDA chops—but wiring up the same scaffolding each time is repetitive. **Hypersonic** lets you quick-build a clean, structured Jupyter notebook so you can spend your time on **insights**, not boilerplate.

- ⚡ **Fast**: point it at CSV/Parquet/SQLite and get an EDA notebook.
- 🧭 **Structured**: title/overview, reproducible loading, helper plots, and per-feature sections.
- 🧹 **Practical**: includes basic string normalization and “check/correct category typos” prompts.
- 🧰 **Flexible**: use it as a CLI or import as a Python library.

> Name inspiration: hypersonic jets—because your EDA should take off quickly.

---

## Installation

Python **3.9+** recommended.

### From source (this repo)
```bash
# from the repo root
pip install -e .
```

This installs a console command `hypersonic` and the `hypersonic` Python package.

---

## Quick Start (CLI)

### Basic (CSV)
```bash
hypersonic --input data/my_data.csv --output eda.ipynb
```

### With a target column
```bash
hypersonic --input data/my_data.csv --target "label" --output eda_label.ipynb
```

### Parquet
```bash
hypersonic --input data/my_data.parquet --output eda_parquet.ipynb
```

### SQLite (.db), first table auto-detected
```bash
hypersonic --input data/my_database.db --output eda_db.ipynb
```

### SQLite with a specific table
```bash
hypersonic --input data/my_database.db --table events --output eda_events.ipynb
```

### Remote files (HTTP/HTTPS)
```bash
hypersonic --input "https://example.com/data.csv" --output eda_remote.ipynb
```

---

## Command-line Options

```bash
hypersonic --help
```

- `--input` (required): CSV/Parquet/SQLite .db (path or URL)
- `--table` (optional): Table name if input is SQLite
- `--target` (optional): Target column name (excluded from feature loops)
- `--max-cat` (default: 30): Max number of categorical features to include
- `--max-num` (default: 30): Max number of numeric features to include
- `--output` (default: hypersonic_eda.ipynb): Output notebook path

---

## What the Generated Notebook Contains

1. **Title & Overview**  
   Source info (path/URL, table, target) and timestamp.

2. **Data Loading (reproducible)**  
   Handles CSV/Parquet/SQLite; downloads remote files to a temp path for SQLite.

3. **Dataset Snapshot**  
   A compact `describe(include="all")` view (transposed) of the dataframe.

4. **Category Text Hygiene**  
   A cell that normalizes string columns (lowercase, trims whitespace, removes stray spaces/underscores/periods).  
   A Markdown reminder to “carefully check and correct category typos.”  
   Per-categorical-feature cells that print unique values and counts to help you spot issues.

5. **Helper Functions**  
   Simple plotting utilities (`plot_categorical`, `plot_numeric`) and tabulation helpers.

6. **Per-Feature Sections**  
   For each **categorical** feature: a plot cell and a value-counts table.  
   For each **numeric** feature: a histogram and a summary-stats table (with IQR fences).

7. **Notes**  
   Pointers and next-steps you can extend.

> Tip: For very long-tail categoricals, consider grouping infrequent classes into “Other” right in the notebook.

---

## Python API

Use Hypersonic programmatically if you prefer.

```python
import pandas as pd
from hypersonic import notebook_builder as hnb

# Infer feature types from a small preview (mimics CLI behavior)
df = pd.read_csv("data/my_data.csv")
cats, nums = hnb.infer_feature_types(df, max_unique_for_categorical=40)

# Build the notebook
hnb.build_notebook(
    input_source="data/my_data.csv",  # path or URL
    output_path="eda.ipynb",
    target=None,                      # or "label"
    table=None,                       # set for SQLite
    features_categorical=cats,
    features_numeric=nums,
)
```

Or just call the CLI entrypoint from Python:
```python
from hypersonic.notebook_builder import main
# Emulates: hypersonic --input data.csv --output eda.ipynb
main()
```

---

## Examples

Create a notebook focused on a target, with more features:
```bash
hypersonic --input data/train.csv --target Outcome --max-cat 50 --max-num 50 --output eda_outcome.ipynb
```

Generate EDA for the first table in a remote SQLite DB:
```bash
hypersonic --input "https://host/path/data.db" --output eda_db.ipynb
```

---

## Why this design?

- **One file in, one notebook out.** Clear contract, low ceremony.
- **Reproducibility first.** The “Data Loading” cell records exactly how the data was pulled.
- **No heavy dependencies.** Pure `pandas`/`matplotlib`/`nbformat` (+ `requests` for remote files).

---

## Limitations & Notes

- Extremely wide datasets (thousands of columns) will generate large notebooks—use `--max-cat/--max-num` to cap feature counts.
- The default string cleaning is intentionally simple; adapt it in the notebook if your domain needs different rules.
- Plots are basic on purpose—tune them to your style post-generation.

---

## Roadmap

- Subcommands for different flows (e.g., `hypersonic eda target ...`)
- Optional profiling (timings, memory)
- “Other” binning for long-tail categories
- Skew detection and auto-transform hints

---

## Contributing

Issues and PRs welcome!  
Please open an issue with a small sample dataset and the expected behavior.

---

## License

MIT © Krish Ambady
