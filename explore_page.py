import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

# Load the model and encoders
@st.cache_data
def load_model():
    with open('saved_steps.joblib', 'rb') as file:
        loaded_data = joblib.load(file)
    return loaded_data

data = load_model()

regressor_loaded = data["model"]
le_country_loaded = data["le_country"]
le_education_loaded = data["le_education"]

# Function to preprocess data
def preprocess_data(df):
    def shorten_categories(categories, cutoff):
        categorical_map = {k: k if v >= cutoff else 'Other' for k, v in categories.items()}
        return categorical_map

    def clean_experience(x):
        if x == 'More than 50 years':
            return 50
        if x == 'Less than 1 year':
            return 0.5
        return float(x)

    def clean_education(x):
        if 'Other doctoral degree' in x:
            return 'Other doctoral degree'
        if 'Professional degree' in x:
            return 'Professional degree'
        if 'Associate degree' in x:
            return 'Associate degree'
        if 'Bachelor’s degree' in x:
            return 'Bachelor’s degree'
        if 'Master’s degree' in x:
            return 'Master’s degree'
        if 'Professional degree' in x or 'Other doctoral' in x:
            return 'Post grad'
        return 'Less than a Bachelors'

    df = df[["Country", "EdLevel", "YearsCodePro", "Employment", "ConvertedComp"]]
    df = df[df["ConvertedComp"].notnull()]
    df = df.dropna()
    df = df[df["Employment"] == "Employed full-time"]
    df = df.drop("Employment", axis=1)

    country_map = shorten_categories(df['Country'].value_counts(), 400)
    df['Country'] = df['Country'].map(country_map)

    df = df[(df["ConvertedComp"] <= 250000) & (df["ConvertedComp"] >= 10000) & (df['Country'] != 'Other')]

    df['YearsCodePro'] = df['YearsCodePro'].apply(clean_experience)
    df['EdLevel'] = df['EdLevel'].apply(clean_education)
    df = df.rename({"ConvertedComp": "Salary"}, axis=1)
    return df

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("survey_results_public.csv")
    df = preprocess_data(df)
    return df

df = load_data()

def show_explore_page():
    st.title("Number of Data from different countries")

    data_countries = df["Country"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(data_countries, labels=data_countries.index, autopct="%1.1f%%", shadow=True, startangle=90)
    ax1.axis("equal")
    st.pyplot(fig1)

    st.write("""#### Salary Based On Country""")
    data_country_mean_salary = df.groupby(["Country"])["Salary"].mean().sort_values(ascending=True)
    st.bar_chart(data_country_mean_salary)

    st.write("""#### Salary Based On Experience""")
    data_experience_mean_salary = df.groupby(["YearsCodePro"])["Salary"].mean().sort_values(ascending=True)
    st.line_chart(data_experience_mean_salary)

# Run the app
if __name__ == "__main__":
    show_explore_page()