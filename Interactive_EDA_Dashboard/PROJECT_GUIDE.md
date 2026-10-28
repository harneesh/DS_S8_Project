# Interactive EDA Dashboard: Project Guide

## What is this project?

**EDA (Exploratory Data Analysis)** is the first step of any data science work. Before building a model, you have to understand your data: what it contains, whether it has errors, how values are spread out, and how columns relate to each other.

Doing EDA usually means writing a lot of repetitive pandas and matplotlib code for every new dataset. This project replaces that with an **interactive web dashboard**: upload any dataset and explore it with clicks instead of code.

**Why it is useful:**
- Saves time, because the same analysis works on any CSV, Excel or JSON file.
- Finds data problems (missing values, duplicates, outliers) before they cause wrong results.
- Charts are interactive (zoom, hover, filter), so patterns are easier to spot.
- People who don't code can still analyse data.

**Tools used:** Python, Streamlit (web app), pandas (data handling), Plotly (interactive charts), SciPy (statistical tests).

---

## How to run it

Open a terminal in the project folder and run:

```powershell
.venv\Scripts\activate
python -m streamlit run app.py
```

The dashboard opens in your browser at **http://localhost:8501**. Press **Ctrl + C** in the terminal to stop it.

> Use `python -m streamlit`, not plain `streamlit`. Windows may block `streamlit.exe`.

---

## How to use it

### 1. Load data (sidebar)
- **Upload file**: CSV, Excel or JSON.
- **Sample dataset**: try it without your own file. *Employees (messy demo data)* is best for a demo because it contains missing values, duplicates and outliers on purpose.

### 2. Pages

| Page | What you can do | Why it is used |
|---|---|---|
| 🏠 **Home** | See dataset size and the first rows | A quick check that the data loaded correctly |
| 📋 **Data Overview** | Column types, summary statistics (mean, std, min, max...), and **automatic insights** | Understand the structure of the data and get a list of issues found automatically |
| 🧹 **Data Quality & Cleaning** | Find missing values and duplicates; fill or drop missing values, change column types, drop or rename columns; download the cleaned CSV | Real data is messy, and dirty data gives wrong analysis results |
| 📈 **Univariate Analysis** | Choose one column and see its histogram, box plot, Q-Q plot and normality test (or bar and pie charts for categories) | Understand how a single variable is distributed: its centre, spread, skewness and normality |
| 🔗 **Bivariate & Multivariate** | Scatter plots, correlation heatmap, compare groups (ANOVA), compare categories (chi-square), pair plot, time trends | Find relationships between variables, for example "does salary increase with experience?" |
| 🎯 **Outlier Detection** | Detect unusual values with the IQR or Z-score method; remove or cap them | Outliers can distort averages and models, so they should be found and handled |

### 3. Helpful extras
- **Filter rows** (sidebar): for example, show only one department. Every page updates.
- **Undo / Reset** (sidebar): reverse any cleaning step.
- **Save a chart**: click the 📷 icon on any chart to download it as PNG.
- **Download data**: *Data Quality & Cleaning → History & export*.

---

## Typical workflow

1. **Load** the dataset.
2. **Data Overview**: read the automatic insights.
3. **Data Quality**: fix missing values and remove duplicates.
4. **Univariate**: study each important column.
5. **Bivariate**: look for relationships and correlations.
6. **Outliers**: detect and treat extreme values.
7. **Download** the cleaned dataset for modelling.
