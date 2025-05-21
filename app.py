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

from data_loader import GNSSDataLoader
from feature_extractor import GNSSFeatureExtractor
from classifier import GNSSClassifier

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
        data_dir = data.get('data_dir', '.')  # Use current directory as default
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
        # Check if files were uploaded
        if 'train_file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No training file provided'}), 400
        
        train_file = request.files['train_file']
        test_file = request.files.get('test_file')

        # Save uploaded files temporarily
        train_path = Path('temp_train.xlsx')
        train_file.save(train_path)

        # Extract features from training data
        train_features = feature_extractor.extract_features_from_excel(train_path)
        if 'LOS/NLOS' not in train_features.columns:
            return jsonify({'status': 'error', 'message': 'Training data must contain LOS/NLOS labels'}), 400

        X_train = train_features.drop('LOS/NLOS', axis=1)
        y_train = train_features['LOS/NLOS']

        # Handle test file if provided
        X_test, y_test = None, None
        if test_file:
            test_path = Path('temp_test.xlsx')
            test_file.save(test_path)
            test_features = feature_extractor.extract_features_from_excel(test_path)
            
            if 'LOS/NLOS' in test_features.columns:
                X_test = test_features.drop('LOS/NLOS', axis=1)
                y_test = test_features['LOS/NLOS']
            
            # Clean up test file
            test_path.unlink(missing_ok=True)

        # Train the model
        metrics = classifier.train(X_train, y_train, X_test, y_test)

        # Clean up training file
        train_path.unlink(missing_ok=True)

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
        if 'test_file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No test file provided'}), 400

        test_file = request.files['test_file']
        test_path = Path('temp_test.xlsx')
        test_file.save(test_path)

        # Extract features from test data
        test_features = feature_extractor.extract_features_from_excel(test_path)
        X_test = test_features.drop('LOS/NLOS', axis=1) if 'LOS/NLOS' in test_features.columns else test_features

        # Make predictions
        predictions = classifier.predict(X_test)
        probabilities = classifier.predict_proba(X_test)

        # Calculate metrics if ground truth is available
        metrics = None
        if 'LOS/NLOS' in test_features.columns:
            metrics = classifier.evaluate(X_test, test_features['LOS/NLOS'])

        # Clean up
        test_path.unlink(missing_ok=True)

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
        if not classifier or not classifier.is_trained:
            return jsonify({
                'status': 'error',
                'message': 'Model must be trained before getting feature importance'
            }), 400

        importance_scores = classifier.get_feature_importance()
        feature_names = feature_extractor.get_feature_names()
        
        importance_dict = dict(zip(feature_names, importance_scores['importance']))
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