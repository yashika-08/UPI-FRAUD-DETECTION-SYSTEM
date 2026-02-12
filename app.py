from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load model and feature names
model = joblib.load('fraud_detection_model.pkl')
feature_names = joblib.load('feature_names.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form.to_dict()

        # Convert Amount to float
        amount = float(data.get("amount"))

        # Create DataFrame with correct feature name
        input_data = pd.DataFrame([[amount]], columns=feature_names)

        # Predict
        prediction = model.predict(input_data)

        result = "Fraudulent" if prediction[0] == 1 else "Not Fraudulent"

        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

