"""
Module for loading and preprocessing GNSS signal data.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from pathlib import Path


class GNSSDataLoader:
    """Class for loading and preprocessing GNSS signal data."""

    def __init__(self, data_dir: str):
        """
        Initialize the data loader.

        Args:
            data_dir (str): Directory containing GNSS data files
        """
        self.data_dir = Path(data_dir)
        self.required_columns = [
            'Year', 'Month', 'Date', 'Hour', 'Min', 'Sec',
            'PRN', 'Elevation', 'Azimuth', 'SNR'
        ]

    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Load GNSS data from a CSV file.

        Args:
            filename (str): Name of the file to load

        Returns:
            pd.DataFrame: Loaded and validated data
        """
        file_path = self.data_dir / filename
        try:
            data = pd.read_csv(file_path)
            self._validate_data(data)
            return data
        except Exception as e:
            raise ValueError(f"Error loading data from {filename}: {str(e)}")

    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate that the data contains required columns and values are in valid ranges.

        Args:
            data (pd.DataFrame): Data to validate

        Raises:
            ValueError: If data validation fails
        """
        # Check required columns
        missing_cols = set(self.required_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Validate ranges
        self._validate_ranges(data)

    def _validate_ranges(self, data: pd.DataFrame) -> None:
        """
        Validate that numeric values are within expected ranges.

        Args:
            data (pd.DataFrame): Data to validate

        Raises:
            ValueError: If values are outside valid ranges
        """
        # Time validations
        if not all(data['Year'].between(1980, 2100)):
            raise ValueError("Year values outside valid range (1980-2100)")
        if not all(data['Month'].between(1, 12)):
            raise ValueError("Month values outside valid range (1-12)")
        if not all(data['Date'].between(1, 31)):
            raise ValueError("Date values outside valid range (1-31)")
        if not all(data['Hour'].between(0, 23)):
            raise ValueError("Hour values outside valid range (0-23)")
        if not all(data['Min'].between(0, 59)):
            raise ValueError("Minute values outside valid range (0-59)")
        if not all(data['Sec'].between(0, 59)):
            raise ValueError("Second values outside valid range (0-59)")

        # Signal validations
        if not all(data['Elevation'].between(0, 90)):
            raise ValueError("Elevation values outside valid range (0-90)")
        if not all(data['Azimuth'].between(0, 360)):
            raise ValueError("Azimuth values outside valid range (0-360)")
        if not all(data['SNR'].between(0, 60)):
            raise ValueError("SNR values outside valid range (0-60)")

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the GNSS data by handling missing values and normalizing features.

        Args:
            data (pd.DataFrame): Raw GNSS data

        Returns:
            pd.DataFrame: Preprocessed data
        """
        # Create a copy to avoid modifying the original data
        processed_data = data.copy()

        # Handle missing values
        processed_data = self._handle_missing_values(processed_data)

        # Normalize numerical features
        processed_data = self._normalize_features(processed_data)

        return processed_data

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            data (pd.DataFrame): Data containing missing values

        Returns:
            pd.DataFrame: Data with missing values handled
        """
        # For numeric columns, fill missing values with median
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            data[col] = data[col].fillna(data[col].median())

        # For categorical columns (like PRN), fill with mode
        categorical_columns = data.select_dtypes(exclude=[np.number]).columns
        for col in categorical_columns:
            data[col] = data[col].fillna(data[col].mode()[0])

        return data

    def _normalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize numerical features to [0, 1] range.

        Args:
            data (pd.DataFrame): Data to normalize

        Returns:
            pd.DataFrame: Normalized data
        """
        # Select numerical columns for normalization
        numeric_columns = ['Elevation', 'Azimuth', 'SNR']
        
        # Create a copy of the data
        normalized_data = data.copy()
        
        # Normalize each numeric column
        for col in numeric_columns:
            min_val = data[col].min()
            max_val = data[col].max()
            normalized_data[col] = (data[col] - min_val) / (max_val - min_val)
        
        return normalized_data

    def split_data(self, data: pd.DataFrame, test_size: float = 0.2, 
                  validation_size: float = 0.1, random_state: Optional[int] = None
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into training, validation, and test sets.

        Args:
            data (pd.DataFrame): Data to split
            test_size (float): Proportion of data for testing
            validation_size (float): Proportion of data for validation
            random_state (Optional[int]): Random seed for reproducibility

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Training, validation, and test sets
        """
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)

        # Calculate split indices
        total_size = len(data)
        test_idx = int(total_size * (1 - test_size))
        val_idx = int(test_idx * (1 - validation_size))

        # Shuffle data
        shuffled_data = data.sample(frac=1, random_state=random_state)

        # Split data
        train_data = shuffled_data[:val_idx]
        val_data = shuffled_data[val_idx:test_idx]
        test_data = shuffled_data[test_idx:]

        return train_data, val_data, test_data 