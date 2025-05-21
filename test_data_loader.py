"""
Tests for the GNSS data loader module.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.data_loader import GNSSDataLoader

@pytest.fixture
def sample_data():
    """Create sample GNSS data for testing."""
    return pd.DataFrame({
        'Year': [2023] * 5,
        'Month': [1] * 5,
        'Date': [1] * 5,
        'Hour': [12] * 5,
        'Min': [30] * 5,
        'Sec': [0] * 5,
        'PRN': ['GPS/1', 'GPS/2', 'GPS/3', 'GPS/4', 'GPS/5'],
        'Elevation': [45.0, 60.0, 30.0, 75.0, 15.0],
        'Azimuth': [120.0, 240.0, 60.0, 300.0, 180.0],
        'SNR': [35.0, 40.0, 25.0, 45.0, 20.0],
        'LOS/NLOS': [1, 1, 0, 1, 0]
    })

@pytest.fixture
def data_loader(tmp_path):
    """Create a data loader instance with temporary directory."""
    return GNSSDataLoader(str(tmp_path))

def test_data_validation(data_loader, sample_data, tmp_path):
    """Test data validation functionality."""
    # Save sample data to temporary file
    file_path = tmp_path / 'test_data.csv'
    sample_data.to_csv(file_path, index=False)
    
    # Test loading valid data
    loaded_data = data_loader.load_data('test_data.csv')
    assert isinstance(loaded_data, pd.DataFrame)
    assert len(loaded_data) == len(sample_data)

def test_invalid_ranges(data_loader, sample_data, tmp_path):
    """Test validation of invalid data ranges."""
    # Create invalid data
    invalid_data = sample_data.copy()
    invalid_data.loc[0, 'Month'] = 13  # Invalid month

    # Save invalid data
    file_path = tmp_path / 'invalid_data.csv'
    invalid_data.to_csv(file_path, index=False)

    # Test loading invalid data
    with pytest.raises(ValueError):
        data_loader.load_data('invalid_data.csv')

def test_missing_columns(data_loader, sample_data, tmp_path):
    """Test handling of missing columns."""
    # Create data with missing column
    invalid_data = sample_data.drop('SNR', axis=1)
    
    # Save invalid data
    file_path = tmp_path / 'missing_column.csv'
    invalid_data.to_csv(file_path, index=False)

    # Test loading data with missing column
    with pytest.raises(ValueError):
        data_loader.load_data('missing_column.csv')

def test_data_preprocessing(data_loader, sample_data):
    """Test data preprocessing functionality."""
    # Add some missing values
    sample_data.loc[0, 'SNR'] = np.nan
    
    # Preprocess data
    processed_data = data_loader.preprocess_data(sample_data)
    
    # Check that missing values are handled
    assert not processed_data['SNR'].isna().any()
    
    # Check that numerical features are normalized
    assert processed_data['SNR'].between(0, 1).all()
    assert processed_data['Elevation'].between(0, 1).all()
    assert processed_data['Azimuth'].between(0, 1).all()

def test_data_splitting(data_loader, sample_data):
    """Test data splitting functionality."""
    # Split data
    train, val, test = data_loader.split_data(
        sample_data,
        test_size=0.2,
        validation_size=0.2,
        random_state=42
    )
    
    # Check split sizes
    total_size = len(sample_data)
    assert len(train) + len(val) + len(test) == total_size
    assert abs(len(test) - total_size * 0.2) <= 1  # Allow for rounding
    assert abs(len(val) - total_size * 0.2) <= 1  # Allow for rounding

if __name__ == '__main__':
    pytest.main([__file__]) 