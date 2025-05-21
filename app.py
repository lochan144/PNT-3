"""
Flask web application for GNSS LOS/NLOS classification.
"""
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import json
from pathlib import Path
import sys
import os

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.data_loader import GNSSDataLoader
from feature_engineering.feature_extractor import GNSSFeatureExtractor
from models.classifier import GNSSClassifier

app = Flask(__name__)
CORS(app)

# Initialize components
data_loader = None
feature_extractor = None
classifier = None

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/api/initialize', methods=['POST'])
def initialize():
    """Initialize the classification system with specified parameters."""
    try:
        data = request.get_json()
        data_dir = data.get('data_dir', 'data')
        model_type = data.get('model_type', 'rf')
        model_params = data.get('model_params', {})

        global data_loader, feature_extractor, classifier
        data_loader = GNSSDataLoader(data_dir)
        feature_extractor = GNSSFeatureExtractor()
        classifier = GNSSClassifier(model_type=model_type, model_params=model_params)

        return jsonify({'status': 'success', 'message': 'System initialized successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train():
    """Train the classifier on provided data."""
    try:
        data = request.get_json()
        train_file = data.get('train_file')
        val_file = data.get('val_file')

        # Load and preprocess training data
        train_data = data_loader.load_data(train_file)
        train_data = data_loader.preprocess_data(train_data)
        X_train = feature_extractor.extract_features(train_data)
        y_train = train_data['LOS/NLOS']

        # Load and preprocess validation data if provided
        X_val, y_val = None, None
        if val_file:
            val_data = data_loader.load_data(val_file)
            val_data = data_loader.preprocess_data(val_data)
            X_val = feature_extractor.extract_features(val_data)
            y_val = val_data['LOS/NLOS']

        # Train the model
        metrics = classifier.train(X_train, y_train, X_val, y_val)

        return jsonify({
            'status': 'success',
            'message': 'Model trained successfully',
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make predictions on new data."""
    try:
        data = request.get_json()
        test_file = data.get('test_file')

        # Load and preprocess test data
        test_data = data_loader.load_data(test_file)
        test_data = data_loader.preprocess_data(test_data)
        X_test = feature_extractor.extract_features(test_data)

        # Make predictions
        predictions = classifier.predict(X_test)
        probabilities = classifier.predict_proba(X_test)

        # If ground truth is available, calculate metrics
        metrics = None
        if 'LOS/NLOS' in test_data.columns:
            metrics = classifier.evaluate(X_test, test_data['LOS/NLOS'])

        return jsonify({
            'status': 'success',
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/feature_importance', methods=['GET'])
def feature_importance():
    """Get feature importance scores."""
    try:
        if classifier.model_type != 'rf':
            return jsonify({
                'status': 'error',
                'message': 'Feature importance is only available for Random Forest models'
            }), 400

        importance_scores = classifier.get_feature_importance()
        feature_names = feature_extractor.get_feature_names()
        
        importance_dict = dict(zip(feature_names, importance_scores.values()))
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )

        return jsonify({
            'status': 'success',
            'feature_importance': sorted_importance
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 