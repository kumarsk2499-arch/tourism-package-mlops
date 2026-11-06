"""
Streamlit app for the Wellness Tourism Package purchase model.

Loads the model that the GitHub Actions pipeline trained and committed next to
this file, collects one customer's details, arranges them into a dataframe with
the exact columns the model was trained on, and reports the purchase
probability and the resulting decision at the model's tuned threshold.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "best_tourism_model.joblib"
META_PATH = APP_DIR / "model_meta.json"

st.set_page_config(page_title="Wellness Package Purchase Predictor", layout="centered")


@st.cache_resource
def load_artifacts():
    """Load the model and its metadata once per app session."""
    return joblib.load(MODEL_PATH), json.loads(META_PATH.read_text())


model, meta = load_artifacts()
categories = meta["categories"]
defaults = meta["category_defaults"]
ranges = meta["numeric_ranges"]
threshold = meta["decision_threshold"]


def choice(label, col):
    """Dropdown limited to the levels seen in training, opened on the most common one."""
    options = categories[col]
    return st.selectbox(label, options, index=options.index(defaults[col]))


def int_range(col):
    """Integer slider bounds and default taken from the training data."""
    r = ranges[col]
    return int(r["min"]), int(r["max"]), int(r["median"])


st.title("Wellness Tourism Package: Purchase Predictor")
st.write(
    "Enter a customer's details to estimate how likely they are to buy the "
    "Wellness Tourism Package. Input options and ranges match the data the "
    "model was trained on."
)

with st.form("customer_form"):
    st.subheader("Customer profile")
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", *int_range("Age"))
        gender = choice("Gender", "Gender")
        marital_status = choice("Marital status", "MaritalStatus")
        occupation = choice("Occupation", "Occupation")
        designation = choice("Designation", "Designation")
        monthly_income = st.number_input(
            "Monthly income", min_value=0.0, value=ranges["MonthlyIncome"]["median"], step=500.0
        )
    with c2:
        city_tier = st.selectbox("City tier", [1, 2, 3])
        passport = st.radio("Holds a valid passport", ["No", "Yes"], horizontal=True)
        own_car = st.radio("Owns a car", ["No", "Yes"], horizontal=True)
        trips = st.slider("Trips per year", *int_range("NumberOfTrips"))
        persons = st.slider("People on the trip", *int_range("NumberOfPersonVisiting"))
        children = st.slider("Children under 5 on the trip", *int_range("NumberOfChildrenVisiting"))
        property_star = st.selectbox("Preferred hotel rating", [3, 4, 5])

    st.subheader("Sales interaction")
    c3, c4 = st.columns(2)
    with c3:
        contact_type = choice("Type of contact", "TypeofContact")
        pitch_duration = st.slider("Pitch duration (minutes)", *int_range("DurationOfPitch"))
    with c4:
        followups = st.slider("Follow-ups after the pitch", *int_range("NumberOfFollowups"))
        satisfaction = st.slider("Pitch satisfaction score", 1, 5, 3)

    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    # One row with the same column names and order used during training
    input_data = pd.DataFrame([{
        "Age": age,
        "TypeofContact": contact_type,
        "CityTier": city_tier,
        "DurationOfPitch": pitch_duration,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": persons,
        "NumberOfFollowups": followups,
        "PreferredPropertyStar": property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": trips,
        "Passport": 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore": satisfaction,
        "OwnCar": 1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": children,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
    }])[meta["feature_order"]]

    probability = float(model.predict_proba(input_data)[0, 1])
    st.metric("Purchase probability", f"{probability:.1%}")
    if probability >= threshold:
        st.success("Likely buyer: prioritize this customer for the Wellness package pitch.")
    else:
        st.info("Unlikely buyer at this time: lower priority for outreach.")
    st.caption(f"Decision threshold: {threshold:.2f}, tuned on out-of-fold training predictions.")

    with st.expander("Model input"):
        st.dataframe(input_data, hide_index=True)
