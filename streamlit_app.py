import streamlit as st
from predict_page import show_predict_page
from explore_page import show_explore_page

# Display developer name at the top
selected_page = st.sidebar.title("Dev Isaac")

# Use a different variable name for the page selection
selected_page = st.sidebar.selectbox("Explore Or Predict", ("Predict", "Explore"))

if selected_page == "Predict":
    show_predict_page()
else:
    show_explore_page()