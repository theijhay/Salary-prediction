import streamlit as st
import numpy as np
import joblib

@st.cache_data
def load_model():
    with open('saved_steps.joblib', 'rb') as file:
        loaded_data = joblib.load(file)
    return loaded_data

data = load_model()

regressor_loaded = data["model"]
le_country_loaded = data["le_country"]
le_education_loaded = data["le_education"]

def show_predict_page():
    st.title("Software Developer Salary Prediction.")

    st.write("""Input the necessary information for salary prediction""")

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

    ok = st.button("Calculate the Salary")
    if ok:
        # Convert input to array for prediction
        X = np.array([[country, education, experience]])
        X[:, 0] = le_country_loaded.transform(X[:, 0])
        X[:, 1] = le_education_loaded.transform(X[:, 1])
        X = X.astype(float)

        # Predict salary
        salary = regressor_loaded.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")

# Run the app
if __name__ == "__main__":
    show_predict_page()
