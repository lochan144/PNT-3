"""
Demo script for GNSS LOS/NLOS Classification System.
"""
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_processing.data_loader import GNSSDataLoader
from feature_engineering.feature_extractor import GNSSFeatureExtractor
from models.classifier import GNSSClassifier

def main():
    """Run a demonstration of the GNSS LOS/NLOS Classification System."""
    print("GNSS LOS/NLOS Classification System Demo")
    print("=======================================")

    # Initialize components
    data_dir = Path("data")
    data_loader = GNSSDataLoader(str(data_dir))
    feature_extractor = GNSSFeatureExtractor()
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    data = data_loader.load_data("sample_data.csv")
    processed_data = data_loader.preprocess_data(data)
    
    # Split data
    print("\nSplitting data into train/validation/test sets...")
    train_data, val_data, test_data = data_loader.split_data(
        processed_data,
        test_size=0.2,
        validation_size=0.2,
        random_state=42
    )
    
    # Extract features
    print("\nExtracting features...")
    X_train = feature_extractor.extract_features(train_data)
    y_train = train_data['LOS/NLOS']
    
    X_val = feature_extractor.extract_features(val_data)
    y_val = val_data['LOS/NLOS']
    
    X_test = feature_extractor.extract_features(test_data)
    y_test = test_data['LOS/NLOS']
    
    # Train and evaluate Random Forest model
    print("\nTraining Random Forest model...")
    rf_classifier = GNSSClassifier(model_type='rf', model_params={
        'n_estimators': 100,
        'max_depth': None,
        'random_state': 42
    })
    
    rf_metrics = rf_classifier.train(X_train, y_train, X_val, y_val)
    print("\nRandom Forest Training Metrics:")
    for metric, value in rf_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Train and evaluate Neural Network model
    print("\nTraining Neural Network model...")
    nn_classifier = GNSSClassifier(model_type='nn', model_params={
        'input_dim': len(X_train.columns),
        'epochs': 50,
        'batch_size': 32
    })
    
    nn_metrics = nn_classifier.train(X_train, y_train, X_val, y_val)
    print("\nNeural Network Training Metrics:")
    print(f"Final epoch accuracy: {nn_metrics['accuracy'][-1]:.4f}")
    print(f"Final epoch loss: {nn_metrics['loss'][-1]:.4f}")
    
    # Make predictions using both models
    print("\nMaking predictions on test set...")
    rf_predictions = rf_classifier.predict(X_test)
    nn_predictions = nn_classifier.predict(X_test)
    
    # Evaluate models on test set
    print("\nTest Set Performance:")
    print("\nRandom Forest:")
    rf_test_metrics = rf_classifier.evaluate(X_test, y_test)
    for metric, value in rf_test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nNeural Network:")
    nn_test_metrics = nn_classifier.evaluate(X_test, y_test)
    for metric, value in nn_test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot feature importance for Random Forest
    if rf_classifier.model_type == 'rf':
        print("\nPlotting feature importance...")
        importance_scores = rf_classifier.get_feature_importance()
        feature_names = feature_extractor.get_feature_names()
        
        plt.figure(figsize=(12, 6))
        sns.barplot(
            x=list(importance_scores.values()),
            y=feature_names,
            palette='viridis'
        )
        plt.title('Feature Importance (Random Forest)')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        
        # Save plot
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        plt.savefig(plots_dir / "feature_importance.png")
        print(f"\nFeature importance plot saved to {plots_dir / 'feature_importance.png'}")

if __name__ == "__main__":
    main() 