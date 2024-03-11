# Machine Learning Web Application with Streamlit

- This project demonstrates how to build a Machine Learning web application from scratch in Python using the Streamlit framework. The application utilizes real-world data to train a machine learning model. The README provides a step-by-step guide covering data analysis, model development, and the creation of a Streamlit web app.

## Requirements

- pandas==2.2.1
- numpy==1.26.4
- scikit-learn==1.2.2
- streamlit==1.31.1
- matplotlib==3.8.0

$ pip install -r requirements.txt

# Setup
Create the project directory

$ mkdir Salary-prediction 

# Navigate to the project directory
$ cd Salary-prediction

# Create a virtue evironment for the project
$ python3 -m venv venv

# Active the virtue environment
$ source venv/bin/activate 

But in this case I used conda environment.
- Click on this link [conda](https://docs.anaconda.com/free/miniconda/#quick-command-line-install) to install it.

## Create a new Conda environment
$ conda create -n ml python=3.11

# Activate the environment
$ conda activate ml

4. Data Analysis and Model Development
Data Collection:
Download the real-world dataset(s) and place it in the data directory.

[Stack Overflow Survey Data](https://insights.stackoverflow.com/survey)