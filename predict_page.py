import streamlit as st
import numpy as np
import joblib

def load_model():
    data = joblib.load('saved_steps.joblib')
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

def show_predict_page():
    st.title("Software Developer Salary Prediction.")

    st.write("""Select some information to predict the salary""")

    countries = (
        "United States",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    education_levels = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Associate degree",
        "Professional degree",
        "Other doctoral degree",
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education_levels)

    experience = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")
    if ok:
        # Convert input to array for prediction
        X = np.array([[country, education, experience]])
        X[:, 0] = le_country.transform(X[:, 0])
        X[:, 1] = le_education.transform(X[:, 1])
        X = X.astype(float)

        # Predict salary
        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")

# Run the app
if __name__ == "__main__":
    show_predict_page()
