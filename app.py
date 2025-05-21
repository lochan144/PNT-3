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
import time
import shutil

from data_loader import GNSSDataLoader
from feature_extractor import GNSSFeatureExtractor
from classifier import GNSSClassifier

app = Flask(__name__)
CORS(app)

# Initialize components
data_loader = None
feature_extractor = None
classifier = None

def safe_remove_file(file_path: Path, max_retries: int = 3, delay: float = 1.0):
    """
    Safely remove a file with retries.

    Args:
        file_path (Path): Path to the file to remove
        max_retries (int): Maximum number of removal attempts
        delay (float): Delay between retries in seconds
    """
    for attempt in range(max_retries):
        try:
            if file_path.exists():
                file_path.unlink()
            break
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Warning: Could not remove temporary file {file_path}: {str(e)}")
            time.sleep(delay)

def save_uploaded_file(file, prefix: str = "temp") -> Path:
    """
    Save an uploaded file with a unique name.

    Args:
        file: Uploaded file object
        prefix (str): Prefix for the temporary file name

    Returns:
        Path: Path to the saved file
    """
    timestamp = int(time.time() * 1000)
    temp_path = Path(f"{prefix}_{timestamp}.xlsx")
    file.save(temp_path)
    return temp_path

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
    train_los_path = None
    train_nlos_path = None
    test_los_path = None
    test_nlos_path = None
    
    try:
        # Check if files were uploaded
        if 'train_los_file' not in request.files or 'train_nlos_file' not in request.files:
            return jsonify({'status': 'error', 'message': 'Both LOS and NLOS training files are required'}), 400
        
        train_los_file = request.files['train_los_file']
        train_nlos_file = request.files['train_nlos_file']
        test_los_file = request.files.get('test_los_file')
        test_nlos_file = request.files.get('test_nlos_file')

        # Save uploaded files with unique names
        train_los_path = save_uploaded_file(train_los_file, "train_los")
        train_nlos_path = save_uploaded_file(train_nlos_file, "train_nlos")

        # Combine training data
        train_features = feature_extractor.combine_los_nlos_data(train_los_path, train_nlos_path)
        
        X_train = train_features.drop('LOS/NLOS', axis=1)
        y_train = train_features['LOS/NLOS']

        # Handle test files if provided
        X_test, y_test = None, None
        if test_los_file and test_nlos_file:
            test_los_path = save_uploaded_file(test_los_file, "test_los")
            test_nlos_path = save_uploaded_file(test_nlos_file, "test_nlos")
            
            test_features = feature_extractor.combine_los_nlos_data(test_los_path, test_nlos_path)
            X_test = test_features.drop('LOS/NLOS', axis=1)
            y_test = test_features['LOS/NLOS']

        # Train the model
        metrics = classifier.train(X_train, y_train, X_test, y_test)

        return jsonify({
            'status': 'success',
            'message': 'Model trained successfully',
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        # Clean up temporary files
        for path in [train_los_path, train_nlos_path, test_los_path, test_nlos_path]:
            if path:
                safe_remove_file(path)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make predictions on new data."""
    test_path = None
    
    try:
        if 'test_file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No test file provided'}), 400

        test_file = request.files['test_file']
        test_path = save_uploaded_file(test_file, "predict")

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

        return jsonify({
            'status': 'success',
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        # Clean up temporary file
        if test_path:
            safe_remove_file(test_path)

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