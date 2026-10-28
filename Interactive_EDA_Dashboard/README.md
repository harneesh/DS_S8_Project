# Interactive EDA Dashboard

A web dashboard for **Exploratory Data Analysis** of any tabular dataset. Upload a CSV, Excel or JSON file and explore it without writing any code.

Built with **Python, Streamlit, pandas, Plotly and SciPy**.

## Features

| Page | What it does |
|---|---|
| 🏠 **Home** | Quick summary and preview of the loaded dataset |
| 📋 **Data Overview** | Row and column counts, column types, memory use, summary statistics, and **automatic insights** (missing data, duplicates, ID columns, skewness, strong correlations, imbalance) |
| 🧹 **Data Quality & Cleaning** | Missing-value table and heatmap, duplicate detection, plus cleaning tools: drop or fill missing values (mean, median, mode, constant, forward fill, backward fill), change column types, drop or rename columns, tidy text. Includes a cleaning log and **CSV download** |
| 📈 **Univariate Analysis** | Numeric columns: histogram, box and violin plots, Q-Q plot with Shapiro-Wilk normality test, ECDF. Categorical columns: bar and pie charts of value counts. Date columns: counts over time |
| 🔗 **Bivariate & Multivariate** | Scatter plot with trend line and r, correlation heatmap (Pearson, Spearman or Kendall), numeric vs category (box plot, violin plot, ANOVA test), category vs category (crosstab, chi-square test, Cramér's V), pair plot, trends over time |
| 🎯 **Outlier Detection** | IQR and Z-score methods with adjustable thresholds, per-column summary, highlighted outlier points, **remove or cap** outliers |

The sidebar also has:
- **Filters**: filter rows by category, numeric range or date range. Every page updates.
- **Undo / Reset**: revert any cleaning step.
- Every chart is interactive (zoom, hover, pan) and can be saved as PNG with the 📷 icon.

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## Run

```bash
python -m streamlit run app.py   # use "python -m": Windows may block streamlit.exe
```

The dashboard opens at <http://localhost:8501>.

## Sample datasets

You can try the app without uploading anything (sidebar → *Sample dataset*):
- **Employees (messy demo data)**: `data/sample_employees.csv`, a synthetic HR dataset with missing values, duplicates and outliers added on purpose to demonstrate the cleaning tools
- **Iris**, **Restaurant tips**, **Gapminder**: classic datasets bundled with Plotly

## Project structure

```
app.py                  # entry point: sidebar (data source, filters, undo) and page navigation
pages/
  home.py
  overview.py           # dataset summary and automatic insights
  data_quality.py       # missing values, duplicates, cleaning, export
  univariate.py         # one-column analysis
  bivariate.py          # relationships between columns, statistical tests
  outliers.py           # IQR and Z-score outlier detection and treatment
utils/
  data_loader.py        # file reading (encoding/delimiter detection, date parsing), column type helpers
  analysis.py           # statistics, outlier logic, insight rules
  plots.py              # reusable Plotly charts
  state.py              # session state: working dataset, undo history, cleaning log
data/
  sample_employees.csv
```

## How it works

- The loaded dataset is kept in Streamlit's `session_state`. Cleaning actions replace the working copy and push the previous version onto an undo stack (last 15 steps).
- Sidebar filters create a filtered **view** that the analysis pages use. Cleaning always applies to the full dataset.
- CSV files are read with automatic encoding fallback (UTF-8 → Latin-1) and delimiter sniffing, so semicolon- and tab-separated files work. Columns whose names suggest dates are parsed automatically.
