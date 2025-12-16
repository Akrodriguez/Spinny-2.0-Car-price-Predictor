import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st


# ---------- Load model, metadata, and CSV ----------
@st.cache_resource
def load_artifacts():
    with open("quikr_price_model.pkl", "rb") as f:
        model = pickle.load(f)

    try:
        with open("quikr_columns.json", "r") as f:
            cols = json.load(f)
    except FileNotFoundError:
        cols = ["name", "company", "fuel_type", "year", "kms_driven"]

    df = pd.read_csv("Cleaned_Car_data.csv")
    return model, cols, df


model, feature_cols, car_df = load_artifacts()


# ---------- Basic page config ----------
st.set_page_config(
    page_title="Car price Predictor",
    page_icon="🚗",
    layout="centered",
)
st.title("🚗 Spinny 2.0-Car Price Predictor")
st.write("Enter car details to get an estimated selling price (in INR).")


# ---------- Dropdown options from CSV ----------
company_options = sorted(car_df["company"].dropna().unique().tolist())[0:]
fuel_options = sorted(car_df["fuel_type"].dropna().unique().tolist())
name_options = sorted(car_df["name"].dropna().unique().tolist())  # all car names


# ---------- Input form ----------
with st.form("car_input_form"):
    st.subheader("Car details")

    # Car name dropdown from CSV
    selected_name = st.selectbox("Car name", name_options)

    # Filter row(s) matching this name to prefill company/fuel (optional)
    matched_rows = car_df[car_df["name"] == selected_name]
    default_company = matched_rows["company"].iloc[0] if not matched_rows.empty else company_options[0]
    default_fuel = matched_rows["fuel_type"].iloc[0] if not matched_rows.empty else fuel_options[0]

    col1, col2 = st.columns(2)

    with col1:
        company = st.selectbox("Company", company_options, index=company_options.index(default_company))
        year = st.number_input(
            "Year of manufacture",
            min_value=int(car_df["year"].min()),
            max_value=int(car_df["year"].max()),
            value=int(car_df["year"].median()),
            step=1,
        )
        kms_driven = st.number_input(
            "Kilometers driven",
            min_value=0,
            max_value=int(car_df["kms_driven"].max()),
            value=int(car_df["kms_driven"].median()),
            step=1000,
        )

    with col2:
        fuel = st.selectbox("Fuel type", fuel_options, index=fuel_options.index(default_fuel))

    submitted = st.form_submit_button("Predict Price")


# ---------- Prediction ----------
if submitted:
    input_dict = {
        "name": selected_name,
        "company": company,
        "fuel_type": fuel,
        "year": int(year),
        "kms_driven": int(kms_driven),
    }

    input_df = pd.DataFrame([input_dict])

    try:
        input_df = input_df[feature_cols]
    except KeyError:
        input_df = input_df[["name", "company", "fuel_type", "year", "kms_driven"]]

    try:
        pred_price = model.predict(input_df)[0]
        st.success(f"Estimated price: ₹ {pred_price:,.0f}")
    except Exception as e:
        st.error(
            "Prediction failed. The model expects columns: "
            "name, company, fuel_type, year, kms_driven."
        )
        st.exception(e)


st.caption(
    "Car name, company, and fuel type dropdowns are populated from Cleaned_Car_data.csv "
    "to exactly match the training data categories."
)
