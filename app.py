from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

app = Flask(__name__)

# Load the dataset
insurance_dataset = pd.read_csv('insurance.csv')

# Data preprocessing
insurance_dataset.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
insurance_dataset.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
insurance_dataset.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)

X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Model training
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


# Predict function
def predict_charges(age, sex, bmi, children, smoker, region):
    input_data = np.array([[age, sex, bmi, children, smoker, region]])
    prediction = regressor.predict(input_data)
    return prediction[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    age = int(data['age'])
    sex = 1 if data['sex'] == 'female' else 0
    bmi = float(data['bmi'])
    children = int(data['children'])
    smoker = 1 if data['smoker'] == 'yes' else 0
    region = int(data['region'])

    predicted_charge = predict_charges(age, sex, bmi, children, smoker, region)

    return render_template('result.html', prediction=predicted_charge)

if __name__ == '__main__':
    app.run(debug=True)
