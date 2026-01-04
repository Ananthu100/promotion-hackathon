import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open("model/model.pkl", "rb"))

st.title("Employee Promotion Predictor")

employee_id = st.number_input("Employee ID", value=1)
department = st.selectbox("Department", ["Sales & Marketing","Operations","Technology","Analytics","HR","R&D","Procurement","Finance","Legal"])
region = st.selectbox("Region", [f"region_{i}" for i in range(1,35)])
education = st.selectbox("Education", ["Below Secondary","Bachelor's","Master's & above"])
gender = st.selectbox("Gender", ["m","f"])
recruitment_channel = st.selectbox("Recruitment Channel", ["sourcing","referred","other"])
no_of_trainings = st.number_input("No of Trainings", min_value=1, max_value=10, value=1)
age = st.number_input("Age", 20, 60, value=30)
previous_year_rating = st.number_input("Previous Year Rating", 1, 5, value=3)
length_of_service = st.number_input("Length of Service", 1, 40, value=5)
kpi = st.selectbox("KPIs Met > 80%", [0,1])
awards = st.selectbox("Awards Won", [0,1])
avg_training_score = st.number_input("Avg Training Score", 0, 100, value=70)

if st.button("Predict"):
    df = pd.DataFrame([{
        "employee_id": employee_id,
        "department": department,
        "region": region,
        "education": education,
        "gender": gender,
        "recruitment_channel": recruitment_channel,
        "no_of_trainings": no_of_trainings,
        "age": age,
        "previous_year_rating": previous_year_rating,
        "length_of_service": length_of_service,
        "KPIs_met >80%": kpi,
        "awards_won?": awards,
        "avg_training_score": avg_training_score
    }])

    prob = model.predict_proba(df)[0][1]
    st.success(f"Promotion Probability: {prob:.2f}")
