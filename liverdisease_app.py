from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
rfm = joblib.load('best_rf_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return "LiverDisease Classifier API"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    print(f"Received features: {features}")

    # Standardize the input features
    features = scaler.transform(features)
    print(f"Transformed features: {features}")

    # Make a prediction
    prediction = rfm.predict(features)
    print(f"Prediction: {prediction}")
    predicted_class = int(prediction[0])

    # Map the prediction to the Iris target names
    target_names = ['No', 'Yes']
    response = {
        'prediction': target_names[predicted_class]
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)