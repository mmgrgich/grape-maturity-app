# 🍇 Grape Maturity Tracker

A Streamlit app for visualizing and predicting grape maturity data during harvest season. Upload Excel files with lab results, adjust harvest readiness rules, and compare vintages for optimal decision-making.

---

## 🚀 Features

- Upload multiple Excel lab files (same format each year)
- Customize harvest readiness thresholds (Brix, pH, TA)
- Compare trends across blocks, varieties, and vintages
- Forecast future ripeness using linear regression
- Automatically detects and graphs results by vintage
- Works with single or multiple years of data

---

## 📂 File Format

Each Excel file should:

- Be in `.xlsx` format
- Have a single collection date per file
- Include the following columns (after cleaning):


- Start reading data from **row 8** (header row on line 8, data starts on line 9)

---

## 🔧 Installation

1. Clone the repo:

```bash
git clone https://github.com/YOUR_USERNAME/grape-maturity-app.git
cd grape-maturity-app
pip install -r requirements.txt
streamlit run app.py
USER = "myuser"
PASS = "mypassword"

---

Would you like this saved as a file (`README.md`) in your project folder or zipped for upload to GitHub?

  
