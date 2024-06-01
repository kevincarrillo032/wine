from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model (assuming your pickle file is named 'winequalitymodel.pkl')
try:
    model = pickle.load(open('winequalitymodel.pkl', 'rb'))
except FileNotFoundError:
    print("Error: Model file 'winequalitymodel.pkl' not found.")
    exit(1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_quality', methods=['POST'])
def predict_quality():
    try:
        # Retrieve data from the incoming request in JSON format
        data = request.json
        
        # Ensure all required features are present
        required_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                             'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'type']
        for feature in required_features:
            if feature not in data:
                # Provide default values if a feature is missing
                if feature == 'type':
                    data[feature] = 0  # Assuming default type value is 0
                else:
                    return jsonify({'error': f'Missing feature: {feature}'}), 400

        # Create DataFrame for prediction
        X_new = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(X_new)[0]

        # Convert prediction to native Python int for JSON serialization
        prediction = int(prediction)

        # Return the prediction as JSON
        return jsonify({'predicted_quality': prediction})
    except Exception as e:
        print(f"Error during prediction: {e}")  # Log specific error messages
        return jsonify({'error': 'An error occurred during prediction.'}), 400

if __name__ == '__main__':
    port = int(os.environ.get('port', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

