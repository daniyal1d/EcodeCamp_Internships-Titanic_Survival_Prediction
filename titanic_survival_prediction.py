# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load the Titanic dataset
tdata = pd.read_csv('train.csv')

# Preprocess the data
# Drop the 'Cabin' column
tdata = tdata.drop(columns='Cabin', axis=1)

# Fill missing values in 'Age' with the mean
tdata['Age'].fillna(tdata['Age'].mean(), inplace=True)

# Replace missing values in 'Embarked' with the mode
tdata['Embarked'].fillna(tdata['Embarked'].mode()[0], inplace=True)

# Encode categorical variables
tdata.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)

# Separate features and target variable
X = tdata.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
Y = tdata['Survived']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Initialize and train the logistic regression model
model = LogisticRegression(max_iter=1000)  # Increased max_iter to avoid convergence issues
model.fit(X_train, Y_train)

# Evaluate the model on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data:', training_data_accuracy)

# Evaluate the model on testing data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of testing data:', test_data_accuracy)

# Save the trained model using pickle
with open('titanic_survival_prediction.pkl', 'wb') as file:
    pickle.dump(model, file)
