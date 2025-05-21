"""
Module for extracting features from GNSS signal data.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Union
from pathlib import Path


class GNSSFeatureExtractor:
    """Class for extracting features from GNSS signal data."""

    def __init__(self):
        """Initialize the feature extractor."""
        self.feature_names = [
            'elevation',
            'azimuth',
            'snr',
            'elevation_sin',
            'elevation_cos',
            'azimuth_sin',
            'azimuth_cos',
            'time_of_day',
            'snr_gradient',
            'elevation_rate'
        ]

    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from GNSS signal data.

        Args:
            data (pd.DataFrame): Raw GNSS data

        Returns:
            pd.DataFrame: Extracted features
        """
        features = pd.DataFrame()

        # Basic features
        features['elevation'] = data['Elevation']
        features['azimuth'] = data['Azimuth']
        features['snr'] = data['SNR']

        # Trigonometric features
        features['elevation_sin'] = np.sin(np.radians(data['Elevation']))
        features['elevation_cos'] = np.cos(np.radians(data['Elevation']))
        features['azimuth_sin'] = np.sin(np.radians(data['Azimuth']))
        features['azimuth_cos'] = np.cos(np.radians(data['Azimuth']))

        # Time features
        features['time_of_day'] = (data['Hour'] * 3600 + data['Min'] * 60 + data['Sec']) / 86400.0

        # Signal dynamics
        features['snr_gradient'] = self._calculate_gradient(data['SNR'])
        features['elevation_rate'] = self._calculate_gradient(data['Elevation'])

        return features

    def _calculate_gradient(self, series: pd.Series) -> pd.Series:
        """
        Calculate the gradient of a time series.

        Args:
            series (pd.Series): Input time series

        Returns:
            pd.Series: Gradient of the time series
        """
        gradient = np.gradient(series)
        return pd.Series(gradient, index=series.index)

    def get_feature_names(self) -> List[str]:
        """
        Get the list of feature names.

        Returns:
            List[str]: List of feature names
        """
        return self.feature_names

    def load_excel_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from an Excel file.

        Args:
            file_path (Union[str, Path]): Path to the Excel file

        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            # Read all sheets from the Excel file
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names

            # Initialize an empty list to store DataFrames from each sheet
            dfs = []

            # Process each sheet
            for sheet_name in sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                # Validate required columns
                required_columns = [
                    'Year', 'Month', 'Date', 'Hour', 'Min', 'Sec',
                    'PRN', 'Elevation', 'Azimuth', 'SNR'
                ]
                missing_cols = set(required_columns) - set(df.columns)
                if missing_cols:
                    print(f"Warning: Sheet '{sheet_name}' missing columns: {missing_cols}")
                    continue

                dfs.append(df)

            # Concatenate all valid DataFrames
            if not dfs:
                raise ValueError("No valid sheets found in the Excel file")
            
            combined_data = pd.concat(dfs, ignore_index=True)
            return combined_data

        except Exception as e:
            raise ValueError(f"Error loading Excel file: {str(e)}")

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data before feature extraction.

        Args:
            data (pd.DataFrame): Raw data

        Returns:
            pd.DataFrame: Preprocessed data
        """
        processed = data.copy()

        # Handle missing values
        numeric_columns = ['Elevation', 'Azimuth', 'SNR']
        for col in numeric_columns:
            processed[col] = processed[col].fillna(processed[col].median())

        # Remove outliers
        for col in numeric_columns:
            q1 = processed[col].quantile(0.25)
            q3 = processed[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            processed[col] = processed[col].clip(lower_bound, upper_bound)

        return processed

    def extract_features_from_excel(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from Excel and extract features.

        Args:
            file_path (Union[str, Path]): Path to the Excel file

        Returns:
            pd.DataFrame: Extracted features
        """
        # Load data from Excel
        raw_data = self.load_excel_data(file_path)

        # Preprocess data
        processed_data = self.preprocess_data(raw_data)

        # Extract features
        features = self.extract_features(processed_data)

        # Add labels if available
        if 'LOS/NLOS' in raw_data.columns:
            features['LOS/NLOS'] = raw_data['LOS/NLOS']

        return features 