# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 16:08:11 2024

@author: Daniyal
"""

from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('titanic_survival_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return "Welcome to the Titanic Survival Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Ensure the input features are in the same order as your training data
    features = np.array([data['Pclass'], data['Sex'], data['Age'], data['SibSp'], data['Parch'], data['Fare'], data['Embarked']]).reshape(1, -1)
    
    prediction = model.predict(features)

    return jsonify({'prediction': int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)