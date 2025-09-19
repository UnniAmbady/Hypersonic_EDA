# Hypersonic — EDA Notebook Generator

![Quick Start](img/QuickStart.png)

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

This installs the console command `hy` and the `hypersonic` Python package (distribution name: `hypersonic-eda`).

---

## Quick Start (CLI)

### Basic (CSV)
```bash
hy --input data/my_data.csv --output eda.ipynb
```

### With a target column
```bash
hy --input data/my_data.csv --target "label" --output eda_label.ipynb
```

### Parquet
```bash
hy --input data/my_data.parquet --output eda_parquet.ipynb
```

### SQLite (.db), first table auto-detected
```bash
hy --input data/my_database.db --output eda_db.ipynb
```

### SQLite with a specific table
```bash
hy --input data/my_database.db --table events --output eda_events.ipynb
```

### Remote files (HTTP/HTTPS)
```bash
hy --input "https://example.com/data.csv" --output eda_remote.ipynb
```

---

## Command-line Options

```bash
hy --help
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
## Call from Python (optional)

If you want to trigger the CLI from Python:

**A) Use `subprocess` (recommended)**
```python
import subprocess
subprocess.run(["hy", "--input", "data/my_data.csv", "--output", "eda.ipynb"], check=True)
```

B) Or set `sys.argv` then call `main()`
```python
import sys
from hypersonic.hy import main
sys.argv = ["hy", "--input", "data/my_data.csv", "--output", "eda.ipynb"]
main()
```
---

## Examples

Create a notebook focused on a target, with more features:
```bash
hy --input data/train.csv --target Outcome --max-cat 50 --max-num 50 --output eda_outcome.ipynb
```

Generate EDA for the first table in a remote SQLite DB:
```bash
hy --input "https://host/path/data.db" --output eda_db.ipynb
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
- Do not use quotes in Target names
- It does primitive typographic corrctions only in catagorical Text.

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

---
## 📌 Why Hypersonic-EDA?</b></span>

Most “auto-EDA” tools today (like YData-Profiling, Sweetviz, AutoViz, DataPrep.EDA) focus on generating static HTML reports or inside-notebook widgets. Those are useful for quick looks—but they don’t give you an editable, reproducible notebook you can extend with your own code.  

**Hypersonic-EDA is different:** it’s a CLI tool that generates a structured Jupyter Notebook (`.ipynb`) with clear sections for loading, statistics, helper functions, and per-feature plots/tables. This means you don’t just *view* a report—you start with a clean, living notebook that you can immediately modify, annotate, and share.

### Key Differentiators
- **Notebook output, not HTML:** Produces an editable `.ipynb` scaffold with code + markdown.  
- **Source flexibility:** Works with CSV, Parquet, and SQLite (local or HTTP URL, with auto-table detection).  
- **Lightweight & CLI-friendly:** Just run `hy --input data.csv --output eda.ipynb`.  
- **Built-in data hygiene:** Normalizes text columns and prompts you to check/correct categorical typos.  
- **Balanced defaults:** Per-feature plots (categorical + numeric), summary stats, outlier counts, and a notes section ready for domain context.  

### Comparison
## 🔍 Comparison with Other Auto-EDA Tools

| Feature / Tool        | Hypersonic-EDA | YData Profiling | Sweetviz | AutoViz | DataPrep.EDA |
|------------------------|----------------|-----------------|----------|---------|--------------|
| **Output format**      | **Jupyter Notebook** (`.ipynb`) scaffold | Static HTML/JSON report | Static HTML dashboard | Inline plots / notebook widget | Inline interactive plots |
| **CLI usage**          | ✅ (`hy --input …`) | ❌ | ⚠️ (only via wrapper, HTML output) | ❌ | ❌ |
| **Editable code**      | ✅ (full notebook you can extend) | ❌ | ❌ | Partial | Partial |
| **Input sources**      | CSV, Parquet, SQLite (local/URL) | CSV, Parquet, DF | CSV, DF | CSV, DF | CSV, Parquet, DF |
| **Typo/cleaning guidance** | ✅ text normalization + typo-check prompts | ❌ | ❌ | ❌ | ❌ |
| **Target column aware** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Dependencies**       | Lightweight (`pandas`, `matplotlib`, `nbformat`, `requests`) | Heavier (`pandas`, `matplotlib`, `visions`, `phik`, …) | Moderate | Moderate | Moderate |

