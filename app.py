import streamlit as st
from predict_page import show_predict_page
from explore_page import show_explore_page

# Display developer name at the top
st.sidebar.title("Dev Isaac")

# GitHub logo URL
github_logo_url = "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"

# GitHub account URL
github_account_url = "https://github.com/theijhay"

# Display GitHub logo as a clickable link with reduced size
st.sidebar.markdown(f'<a href="{github_account_url}"><img src="{github_logo_url}" width="30"></a>', unsafe_allow_html=True)
st.sidebar.markdown(f'[View the source code](https://github.com/theijhay/Salary-prediction)')

# Use a different variable name for the page selection
selected_page = st.sidebar.selectbox("Explore Or Predict", ("Predict", "Explore"))

if selected_page == "Predict":
    show_predict_page()
else:
    show_explore_page()
