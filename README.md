## Salary Prediction Web Application

![alt text](https://i.imgur.com/eLke1Xd.png)

### Introduction
Welcome to the Salary Prediction Web App! This project utilizes machine learning to predict salaries based on user input such as years of experience, education level, and job title. The web app provides an intuitive interface for users to obtain estimated salary ranges for various job positions.

### Project Overview:

The Machine Learning Web Application provides a user-friendly interface for interacting with a machine learning model trained on real-world data. Built using Python and the Streamlit framework, this application allows users to select input features, input data, and obtain predictions from the model in real-time. It also includes features for data visualization and model evaluation.

### Project Goals:

- Provide an intuitive web-based interface for users to interact with a machine learning model.
- Enable users to select input features, input data, and obtain predictions from the model.
- Enhance user understanding of the underlying data and model performance through data visualization and model evaluation features.

Deployed Site: [Link to Deployed Web App](https://sdprediction.streamlit.app/)

Final Project Blog Article: [Link to Blog Article](https://www.linkedin.com/pulse/empowering-careers-building-data-driven-salary-prediction-isaac-8bijf/)

Author(s) LinkedIn: [Author's LinkedIn Profile](https://www.linkedin.com/in/olawaleisaac/)


# Installation:

#### Requirements
- pandas==2.2.1
- numpy==1.26.4
- scikit-learn==1.4.1.post1
- streamlit==1.32.2
- matplotlib==3.8.0
- joblib==1.3.2

```
$ pip install -r requirements.txt 
```

#### Setup
- Create the project directory
```
$ mkdir Salary-prediction 
``` 

- Navigate to the project directory
```
$ cd Salary-prediction
```

#### Data Collection

Download the real-world dataset(s) and place it into the project directory.

[Stack Overflow Survey Data](https://insights.stackoverflow.com/survey)


- Create a virtue evironment for the project
```
$ python3 -m venv venv
```

- Active the virtue environment
```
$ source venv/bin/activate 
```

But in this case I used conda environment.
- Click on this link [conda](https://docs.anaconda.com/free/miniconda/#quick-command-line-install) to install it. 
Anaconda provides a simple double-click installer for your
convenience.

- Create a new environment name "ml" with Python3.11 version
```
$ conda create -n ml python=3.11
```

- Activate the environment
```
$ conda activate ml
```

#### Install the packages
```
$ conda install streamlit
$ conda install numpy pandas
$ conda install matplotlib scikit-learn
```

So we want to create a jupyter nootebook and 
there we play around with the data and train our model.
So we also want to install a kernel for this virtue environment.
but maybe in your own case you may need to install ipython before running this command.

```
$ ipyhon kernel install --user --name=ml
```

Now we can start our jupyter notebook server

```
$ jupyter notebook
```

This notebook uses several Python packages that come standard with the Anaconda
Python distribution. The primary libraries that we'll be using are:

**NumPy**: Provides a fast numerical array structure and helper functions.

**pandas**: Provides a DataFrame structure to store data in memory and work with
it easily and efficiently.

**scikit-learn**: The essential Machine Learning package for a variaty of
supervised learning models, in Python.

**matplotlib**: Basic plotting library in Python; most other Python plotting
libraries are built on top of it.


#### Data cleaning
Firstly, we only want to keep a few columns so we want to keep the 
column 'Country', 'EdLevel' which is the education level, 'YearsCodePro' which is the number 
of the years of the professional experience, 'Employment' for example if the developer is working full time,'ConvertedComp' so this is the salary converted to US dollars, and the 'salary' so this is what we want to keep.

```
# Select relevant columns
df = df[["Country", "EdLevel", "YearsCodePro", "Employment", "ConvertedComp"]]
df = df.rename({"ConvertedComp": "Salary"}, axis=1)
```

#### Data Analysis
- Navigate to the project directory

```
$ cd Salary-prediction
```
- Run Jupyter notebooks for data analysis and training of the model.

```
$ jupyter notebook
```
This command will automatically take you to where you will train your model
and then you click on new notebook and select ml, and start importing the librabries needed for project. You can find the file in the repository ends with ipynb.

```
# models

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np
```


#### Training & Testing

For the Salary Prediction, I simply trained three models to predict the salary.

```
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Random Forest Regression:** This ensemble learning algorithm can be used for regression tasks, such as predicting salaries based on input features like years of experience, education level, and job title. Random forests are known for their robustness and ability to handle complex relationships in the data.
```
# Train a Random Forest model

random_forest_reg = RandomForestRegressor(random_state=0)
random_forest_reg.fit(X_train, y_train)
y_pred_rf = random_forest_reg.predict(X_test)
error_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
```

**Linear Regression:** Linear regression is another suitable algorithm for predicting salaries, especially when there's a linear relationship between the input features and the target variable (salary). It's a simple yet effective algorithm commonly used for regression tasks.
```
# Train a Linear Regression model

linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_pred_linear = linear_reg.predict(X_test)
error_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
```

**Decision Trees:** Decision trees can also be employed for predicting salaries. They're useful for capturing nonlinear relationships between features and the target variable. Decision trees are easy to interpret and can handle both numerical and categorical data.
```
# Train a Decision Tree model

dec_tree_reg = DecisionTreeRegressor(random_state=0)
dec_tree_reg.fit(X_train, y_train)
y_pred_tree = dec_tree_reg.predict(X_test)
error_tree = np.sqrt(mean_squared_error(y_test, y_pred_tree))
```
#### Hyperparameter tuning for Decision Tree using GridSearchCV
```
# Hyperparameter tuning for Decision Tree using GridSearchCV

max_depth = [None, 2, 4, 6, 8, 10, 12]
parameters = {"max_depth": max_depth}
regressor = DecisionTreeRegressor(random_state=0)
gs = GridSearchCV(regressor, parameters, scoring='neg_mean_squared_error')
gs.fit(X_train, y_train)
regressor_tuned = gs.best_estimator_

y_pred_tuned = regressor_tuned.predict(X_test)
error_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
```
#### Save the trained model

You can save the trained models using either the pickle module or the joblib library in Python. But here I saved the trained model using joblib because I'm dealing with a large dataset, so joblib is my champion. Its speed and memory efficiency are invaluable assets for serious machine learning endeavors
```
# Save the trained model and encoders

data = {"model": regressor_tuned, "le_country": le_country, "le_education": le_education}
with open('saved_steps.joblib', 'wb') as file:
    joblib.dump(data, file)
```
#### Load the model and encoders

Let's load the model and encoders
```
# Load the model and encoders

with open('saved_steps.joblib', 'rb') as file:
    loaded_data = joblib.load(file)

regressor_loaded = loaded_data["model"]
le_country_loaded = loaded_data["le_country"]
le_education_loaded = loaded_data["le_education"]
```
#### Testing

Let's test the loaded model
```
# Test the loaded model

X_example = np.array([[le_country_loaded.transform(['United States'])[0], le_education_loaded.transform(['Master’s degree'])[0], 15]])
X_example = X_example.astype(float)
y_pred_example = regressor_loaded.predict(X_example)
print(f"Predicted Salary: ${y_pred_example[0]:,.2f}")
```
#### The result

Now, you can see the output of our trained models.
```
Predicted Salary: $143,285.07

/home/jhay-tech/miniconda3/envs/ml/lib/python3.11/site-packages/sklearn/base.py:493: UserWarning: X does not have valid feature names, but DecisionTreeRegressor was fitted with feature names
  warnings.warn(
``` 
### To run the web app locally, follow these steps:

#### Clone the repository
```
$ git clone https://github.com/theijhay/Salary-prediction.git
```
#### Navigate to the project directory
```
$ cd salary-prediction
```
##### Install the required dependencies
```
$ pip install -r requirements.txt
```
#### Usage
- Start the Streamlit app
```
$ streamlit run app.py
```

- Open a web browser and navigate to the provided local URL to access the web app.
- Enter the required information such as years of experience, education level, and job title.
- Click on the "Predict Salary" button to view the estimated salary range for the specified job parameters.

# Contributing

- Contributions to the project are welcome! To contribute, follow these steps:

1. Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3. Make your changes and commit them (git commit -am 'Add new feature').
4. Push to the branch (git push origin feature-branch).
5. Create a new Pull Request and describe your changes.

#### Bug Fixes
None yet

### Licensing
This project is licensed under the MIT License - see the LICENSE file for details.