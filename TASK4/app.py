from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    age = int(request.form['age'])
    gender = 1 if request.form['gender'] == 'Male' else 0
    chest_pain = int(request.form['chest_pain'])
    bp = int(request.form['bp'])
    cholesterol = int(request.form['cholesterol'])
    sugar = 1 if request.form['sugar'] == 'Yes' else 0
    ecg = int(request.form['ecg'])
    max_heart_rate = int(request.form['max_heart_rate'])
    angina = int(request.form['angina'])
    oldpeak = float(request.form['oldpeak'])
    st_slope = int(request.form['st_slope'])
    vessels = int(request.form['vessels'])
    thalassemia = int(request.form['thalassemia'])

    # Create an input array
    input_data = np.array([[age, gender, chest_pain, bp, cholesterol, sugar, ecg, max_heart_rate, angina, oldpeak, st_slope, vessels, thalassemia]])

    # Predict using the loaded model
    prediction = model.predict(input_data)[0]
    result = "Positive" if prediction == 1 else "Negative"

    # Redirect to the result page with the prediction result
    return redirect(url_for('result', result=result))

@app.route('/result')
def result():
    prediction = request.args.get('result')
    return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)