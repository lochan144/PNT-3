# GNSS LOS/NLOS Classification System

This system implements a machine learning-based approach for classifying GNSS (Global Navigation Satellite Systems) signals as either Line-of-Sight (LOS) or Non-Line-of-Sight (NLOS) based on signal characteristics.

## Features

- Data loading and preprocessing for GNSS signal data
- Feature extraction from raw GNSS measurements
- Machine learning model training and evaluation
- Web interface for interactive classification
- Performance metrics visualization
- Support for multiple machine learning algorithms
- Real-time and batch processing capabilities

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

## Installation

1. Clone this repository or download the source code:
```bash
git clone <repository-url>
cd gnss-los-nlos-classifier
```

2. Create and activate a virtual environment (recommended):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/MacOS
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Data Format

The system expects GNSS data in Excel format with the following columns:
- Year, Month, Date: Date components
- Hour, Min, Sec: Time components
- PRN: Satellite identifier
- Elevation: Satellite elevation angle (0-90 degrees)
- Azimuth: Satellite azimuth angle (0-360 degrees)
- SNR: Signal-to-Noise Ratio (0-60 dB-Hz)
- LOS/NLOS: Ground truth labels (for training data)

## Usage

1. Start the web application:
```bash
python app.py
```

2. Open a web browser and navigate to:
```
http://localhost:5000
```

3. Use the web interface to:
   - Upload GNSS data files
   - Train classification models
   - Make predictions on new data
   - View performance metrics and visualizations

## API Endpoints

The system provides the following REST API endpoints:

- `POST /api/initialize`: Initialize the classification system
- `POST /api/train`: Train the classifier on provided data
- `POST /api/predict`: Make predictions on new data
- `GET /api/feature_importance`: Get feature importance scores

## Model Types

The system supports multiple classification algorithms:
- Random Forest (default)
- Support Vector Machine
- Neural Network
- Gradient Boosting

## Performance Metrics

The system calculates and reports:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

## Directory Structure

```
gnss-los-nlos-classifier/
├── app.py                 # Main Flask application
├── data_loader.py         # Data loading and preprocessing
├── feature_extractor.py   # Feature engineering
├── classifier.py          # Machine learning models
├── requirements.txt       # Package dependencies
├── README.md             # This file
└── static/               # Static web assets
    └── templates/        # HTML templates
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- IIT Tirupati Navavishkar I-Hub Foundation for project sponsorship
- Contributors and maintainers

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Format Errors**
   - Ensure input files follow the required format
   - Check for missing or invalid values

3. **Model Performance Issues**
   - Try different model types
   - Adjust hyperparameters
   - Ensure sufficient training data

### Support

For issues and support:
1. Check the documentation
2. Search existing issues
3. Create a new issue with:
   - Description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - System information 