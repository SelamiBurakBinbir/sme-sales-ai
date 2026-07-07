<img width="1920" height="2028" alt="screencapture-selamiburakbinbir-sme-sales-ai-streamlit-app-2026-07-07-22_38_23" src="https://github.com/user-attachments/assets/98d26e1b-c411-464a-b7da-78d9f7d8e658" />
# SME Sales AI

SME Sales AI is a Streamlit-based sales analytics application designed for small and medium-sized businesses. It helps users upload product sales data, clean messy datasets, analyze sales performance, detect unusual revenue days, generate weekly revenue forecasts, and receive short AI-powered business insights.

Live app:
https://selamiburakbinbir-sme-sales-ai.streamlit.app/

---

## What This Project Does

Many small businesses keep their sales data in Excel or CSV files, but these files are often inconsistent, messy, or difficult to analyze directly. SME Sales AI turns raw sales data into a cleaner and more understandable business dashboard.

The application helps with:

* Uploading CSV or Excel sales files
* Matching messy source columns to a standard sales format
* Cleaning dates, product names, quantities, prices, and revenue values
* Detecting possible revenue inconsistencies
* Analyzing total revenue, quantity, product variety, and daily trends
* Finding top-selling and low-selling products
* Running Pareto / ABC product analysis
* Detecting unusual revenue days statistically
* Simulating simple price and quantity changes
* Forecasting weekly revenue with machine learning models
* Generating short AI business insights with Gemini

---

## Supported File Types

The app supports:

```text
.csv
.xlsx
.xls
```

CSV files are read with multiple encoding attempts, including:

```text
utf-8
utf-8-sig
cp1254
latin1
```

This makes the app more tolerant of Turkish and English datasets exported from different systems.

---

## Expected Sales Data

The app works by converting uploaded data into a simple standard sales schema:

```text
date
product_name
quantity
unit_price
revenue
```

The source file does not need to use these exact column names. During column mapping, the app suggests likely matches and lets the user manually select the correct columns.

Revenue can either be selected from an existing column or derived from:

```text
quantity × unit_price
```

An optional discount column can also be selected for revenue consistency checks, but discount is not required and is not added to the final cleaned dataset.

---

## Main Features

### 1. Upload & Preview

The user uploads a raw sales dataset and previews its structure.

This section shows:

* Row count
* Column count
* Missing cell count
* Data preview
* Column summary
* Upload information

The app can also restore the latest uploaded dataset after refresh if the stored upload still exists.

![Uploading screencapture-selamiburakbinbir-sme-sales-ai-streamlit-app-2026-07-07-22_38_23.png…]()
![Uploading screencapture-selamiburakbinbir-sme-sales-ai-streamlit-app-2026-07-07-22_40_51.png…]()

---

### 2. Column Mapping

The app analyzes source columns and suggests likely matches for the standard fields.

The user maps the uploaded dataset to:

```text
date
product_name
quantity
unit_price
revenue
```

For revenue, the user can either select a revenue column or let the app calculate revenue from quantity and unit price.

The app also checks whether the selected revenue column is consistent with expected formulas such as:

```text
quantity × unit_price
quantity × unit_price × (1 - discount)
```

---

### 3. Cleaning & Standardization

After mapping, the app cleans the dataset in memory.

Cleaning includes:

* Date parsing
* Date order detection
* Product name normalization
* Quantity cleaning
* Unit price cleaning
* Revenue cleaning
* Missing revenue fill from quantity × unit price
* Revenue consistency checks
* Schema validation

The app does not automatically create output folders or save generated files. Instead, cleaned results can be downloaded manually from the interface.

Downloadable outputs include:

```text
Standardized CSV
Cleaned CSV
Cleaning report JSON
```

---

### 4. Analysis Dashboard

The dashboard analyzes the cleaned sales data for a selected date range.

It includes:

* Total revenue
* Total quantity
* Product variety
* Average unit price
* Average daily revenue
* Daily revenue chart
* Daily quantity chart
* Top products by revenue
* Top selling products
* Least selling products
* Downloadable text report

The dashboard is divided into subsections so that only the selected part is rendered. This improves performance on larger datasets.

---

### 5. Statistical Anomaly Detection

The app can detect unusual revenue days using the IQR method.

This is not a machine learning feature. It is a statistical method that compares daily revenue values against a normal range.

The user can choose the IQR multiplier and optionally include unusually low revenue days.

The anomaly section shows:

* Number of anomaly days
* Anomaly ratio
* Highest anomaly date
* Highest anomaly revenue
* Normal upper bound
* Chart with anomaly days marked
* Top unusual revenue days table
* Downloadable anomaly CSV

---

### 6. Pareto / ABC Product Analysis

The app groups products by revenue and classifies them into A, B, and C groups.

The goal is to show which products generate most of the revenue.

ABC classification is based on cumulative revenue share:

```text
A = products that make up roughly the first 80% of revenue
B = products that make up the next 15%
C = remaining products
```

This helps identify the most important products in the dataset.

---

### 7. What-if Simulator

The What-if Simulator is a simple decision-support tool.

The user can change:

```text
Price change (%)
Quantity change (%)
```

The app then estimates how total revenue would change under that scenario.

This is not a machine learning model. It is a simple business simulation based on the selected date range.

---

### 8. ML Forecast

The ML Forecast tab generates weekly revenue forecasts.

The app compares multiple models:

```text
Moving Average Baseline
Ridge Regression
Random Forest Regressor
```

The best model is selected mainly by the lowest sMAPE value, with RMSE and MAE used as tie-breakers.

The forecast section includes:

* Selected model
* MAE
* RMSE
* sMAPE
* Relative MAE
* Relative RMSE
* Forecasted total revenue
* Previous period comparison
* Conservative / expected / optimistic scenarios
* Forecast trend summary
* Actual vs predicted chart
* Future forecast table
* Model comparison table
* Used and excluded features

Forecast horizons available in the UI:

```text
1 week
2 weeks
4 weeks
8 weeks
12 weeks
```

The model generates a 12-week background forecast, and the selected horizon only changes the displayed slice. Changing the horizon does not rerun the model.

---

### 9. AI Insights

The AI Insights tab generates a short business explanation from the analysis and forecast results.

It uses Gemini through the Google GenAI client.

The app does not send the full dataset to the AI model. It only sends compact summary data such as:

* KPI summary
* Top products
* Pareto / ABC summary
* Anomaly summary
* Forecast metrics
* Multi-horizon forecast summary

The AI output is designed to be short, clear, and business-oriented.

AI Insights sections:

```text
General Summary
Revenue Drivers
Risks and Anomalies
Forecast Interpretation
Recommended Actions
```

---

## Environment Variables

AI Insights requires a Gemini API key.

Create a `.env` file or export the variable manually:

```bash
GEMINI_API_KEY=your_api_key_here
```

Optional model override:

```bash
GEMINI_MODEL=gemini-3.1-flash-lite
```

If no model is provided, the app uses:

```text
gemini-3.1-flash-lite
```

---

## Running Locally

Clone the repository:

```bash
git clone https://github.com/SelamiBurakBinbir/sme-sales-ai.git
cd sme-sales-ai
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

Or run it with a specific port:

```bash
streamlit run streamlit_app.py --server.headless true --server.port 8502
```

If the port is busy, choose another port:

```bash
streamlit run streamlit_app.py --server.headless true --server.port 8503
```

---

## Project Structure

```text
streamlit_app.py      Main Streamlit user interface
file_reader.py        File reading helpers
column_profiler.py    Column profiling and type detection
field_scorer.py       Column scoring and revenue validation
output_writer.py      Standardized dataframe creation
data_cleaner.py       Data cleaning and validation
sales_analysis.py     Sales analysis and report generation
ml_forecast.py        Weekly revenue forecasting logic
ai_insights.py        Gemini-based business insight generation
main.py               Streamlit entrypoint note
```

---

## Notes

SME Sales AI runs as a Streamlit web application. Generated outputs can be downloaded manually from the interface.
