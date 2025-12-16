# 🚗 Spinny 2.0 – Car Price Predictor

Spinny 2.0 is a machine learning project that predicts the selling price of used cars listed on Quikr using features such as car name, company, fuel type, year of manufacture, and kilometers driven.
The project contains a complete workflow from data cleaning and model training in a Jupyter notebook to an interactive Streamlit web app for real‑time price estimation.

---

## 🔍 Features

- Cleaned Quikr used‑car dataset with consistent numeric fields and standardized text labels. 
- scikit‑learn pipeline using OneHotEncoder for categorical features and Linear Regression for price prediction.  
- Streamlit UI with dropdowns populated directly from `Cleaned_Car_data.csv` to avoid category mismatches.  
- Sensible defaults for year and kilometers based on dataset statistics, plus instant estimated price in INR.

---

## ⚙️ Installation

1. **Clone the repository**:
 Clone the repository
 git clone https://github.com/<your-username>/car-price-prediction.git
 cd car-price-prediction

2. **(Optional) Create and activate a virtual environment**:
 python -m venv venv
 venv\Scripts\activate # Windows
 source venv/bin/activate # macOS / Linux

3. **Install dependencies**:
 pip install -r requirements.txt

---

## 🖥️ Running the App

 streamlit run app.py

