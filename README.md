# Machine Learning Web Application

# Introduction
Welcome to the Salary Prediction Web App! This project utilizes machine learning to predict salaries based on user input such as years of experience, education level, and job title. The web app provides an intuitive interface for users to obtain estimated salary ranges for various job positions.

# Project Overview:

The Machine Learning Web Application provides a user-friendly interface for interacting with a machine learning model trained on real-world data. Built using Python and the Streamlit framework, this application allows users to select input features, input data, and obtain predictions from the model in real-time. It also includes features for data visualization and model evaluation.

# Project Goals:

- Provide an intuitive web-based interface for users to interact with a machine learning model.
- Enable users to select input features, input data, and obtain predictions from the model.
- Enhance user understanding of the underlying data and model performance through data visualization and model evaluation features.

Deployed Site: [Link to Deployed Web App](https://sdprediction.streamlit.app/)
Final Project Blog Article: Link to Blog Article
Author(s) LinkedIn: [Author's LinkedIn Profile](https://www.linkedin.com/in/olawaleisaac/)


## Installation:

# Requirements
- pandas==2.2.1
- numpy==1.26.4
- scikit-learn==1.4.1.post1
- streamlit==1.32.2
- matplotlib==3.8.0
- joblib==1.3.2

$ pip install -r requirements.txt

# Setup
- Create the project directory

$ mkdir Salary-prediction 

- Navigate to the project directory

$ cd Salary-prediction

# Data Collection

Download the real-world dataset(s) and place it into the data directory.

[Stack Overflow Survey Data](https://insights.stackoverflow.com/survey)


- Create a virtue evironment for the project

$ python3 -m venv venv

- Active the virtue environment

$ source venv/bin/activate 

But in this case I used conda environment.
- Click on this link [conda](https://docs.anaconda.com/free/miniconda/#quick-command-line-install) to install it.

- Create a new environment name "ml" with Python3.11 version

$ conda create -n ml python=3.11

- Activate the environment

$ conda activate ml

# Install the packages

$ conda install streamlit

$ conda install numpy pandas

So we need some more so we say

$ conda install matplotlib scikit-learn

So we want to create a jupyter nootebook and 
there we play around with the data and train our model.

So we also want to install a kernel for this virtue environment.
but maybe in your own case you may need to install ipython before running this command.

$ ipyhon kernel install --user --name=ml

Now we can start our jupyter notebook server

$ jupyter notebook

Note: This command will automatically take you to where you will train your model
and then you click on new notebook and select ml. You can find the file in the repository.

# Data Analysis

- Navigate to the project directory
$ cd Salary-prediction

- Run Jupyter notebooks for data analysis and model development.
$ jupyter notebook


# Data cleaning
Firstly, we only want to keep a few columns so we want to keep the 
column 'Country', 'EdLevel' which is the education level, 'YearsCodePro' which is the number 
of the years of the professional experience, 'Employment' for example if the developer is working full time,'ConvertedComp' so this is the salary converted to US dollars, and the 'salary' so this is what we want to keep.

![alt text](<DataCleaning.png>)


