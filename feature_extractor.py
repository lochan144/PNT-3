"""
Module for extracting features from GNSS signal data.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple, Optional
from pathlib import Path
import os
import warnings
import csv
import tempfile
import shutil
import logging
from datetime import datetime
import subprocess
import sys
import importlib
from io import StringIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gnss_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def ensure_dependencies():
    """Ensure all required dependencies are installed."""
    dependencies = {
        'openpyxl': 'openpyxl>=3.0.7',
        'xlrd': 'xlrd>=2.0.1',
        'odfpy': 'odfpy>=1.4.1',
        'pyxlsb': 'pyxlsb>=1.0.9'
    }
    
    missing = []
    for package, version in dependencies.items():
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(version)
    
    if missing:
        logger.info(f"Installing missing dependencies: {missing}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            logger.info("Successfully installed missing dependencies")
        except Exception as e:
            logger.error(f"Failed to install dependencies: {e}")
            raise RuntimeError(f"Failed to install required dependencies: {e}")

class FileValidationError(Exception):
    """Custom exception for file validation errors."""
    pass

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
        self.temp_dir = Path(tempfile.mkdtemp(prefix='gnss_'))
        ensure_dependencies()
        logger.info(f"Initialized GNSSFeatureExtractor with temp directory: {self.temp_dir}")

    def __del__(self):
        """Cleanup temporary files on object destruction."""
        try:
            if hasattr(self, 'temp_dir') and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary directory: {e}")

    def _validate_file(self, file_path: Union[str, Path]) -> Path:
        """
        Validate the input file and return a Path object.

        Args:
            file_path: Input file path

        Returns:
            Path: Validated file path

        Raises:
            FileValidationError: If file validation fails
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileValidationError(f"File does not exist: {file_path}")
            if not file_path.is_file():
                raise FileValidationError(f"Path is not a file: {file_path}")
            if file_path.stat().st_size == 0:
                raise FileValidationError(f"File is empty: {file_path}")
            return file_path
        except Exception as e:
            raise FileValidationError(f"File validation failed: {str(e)}")

    def _create_backup(self, file_path: Path) -> Path:
        """
        Create a backup of the input file.

        Args:
            file_path: Original file path

        Returns:
            Path: Backup file path
        """
        backup_path = self.temp_dir / f"{file_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_path.suffix}"
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup file: {backup_path}")
        return backup_path

    def _convert_to_csv(self, file_path: Path) -> Path:
        """
        Convert Excel file to CSV as a fallback option.

        Args:
            file_path: Excel file path

        Returns:
            Path: CSV file path
        """
        csv_path = self.temp_dir / f"{file_path.stem}.csv"
        try:
            # Try different Excel engines
            engines = ['openpyxl', 'xlrd', 'odf']
            for engine in engines:
                try:
                    df = pd.read_excel(file_path, engine=engine)
                    df.to_csv(csv_path, index=False)
                    logger.info(f"Successfully converted Excel to CSV using {engine} engine")
                    return csv_path
                except Exception as e:
                    logger.warning(f"Failed to convert using {engine} engine: {e}")
                    continue
            
            raise ValueError("All Excel engines failed")
        except Exception as e:
            logger.error(f"Failed to convert Excel to CSV: {e}")
            raise

    def _detect_file_format(self, file_path: Path) -> str:
        """
        Detect the format of the input file.

        Args:
            file_path: Path to the file

        Returns:
            str: Detected format ('xlsx', 'xls', 'ods', 'csv', or 'unknown')
        """
        try:
            # Check file signature (magic numbers)
            with open(file_path, 'rb') as f:
                header = f.read(4)
                
            # Excel 2007+ (.xlsx)
            if header.startswith(b'PK\x03\x04'):
                return 'xlsx'
            
            # Excel 97-2003 (.xls)
            if header.startswith(b'\xD0\xCF\x11\xE0'):
                return 'xls'
            
            # Try to detect CSV
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    sample = f.read(1024)
                    dialect = csv.Sniffer().sniff(sample)
                    return 'csv'
            except:
                pass

            # Check file extension as fallback
            suffix = file_path.suffix.lower()
            if suffix in ['.xlsx', '.xls', '.ods', '.csv']:
                return suffix[1:]  # Remove the dot

            return 'unknown'
        except Exception as e:
            logger.warning(f"Error detecting file format: {e}")
            return 'unknown'

    def _read_with_engine(self, file_path: Path, engine: str) -> Optional[pd.DataFrame]:
        """
        Try to read Excel file with specific engine.

        Args:
            file_path: Path to the file
            engine: Excel engine to use

        Returns:
            Optional[pd.DataFrame]: DataFrame if successful, None otherwise
        """
        try:
            if engine == 'csv':
                # Try different encodings for CSV
                encodings = ['utf-8', 'latin1', 'cp1252']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        if not df.empty:
                            return df
                    except:
                        continue
                return None

            df = pd.read_excel(file_path, engine=engine)
            return df if not df.empty else None
        except Exception as e:
            logger.warning(f"Failed to read with {engine} engine: {e}")
            return None

    def load_excel_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from an Excel file with comprehensive error handling.

        Args:
            file_path: Path to the Excel file

        Returns:
            pd.DataFrame: Loaded data

        Raises:
            ValueError: If file cannot be loaded
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise ValueError(f"File does not exist: {file_path}")

            # Create backup
            backup_path = self.temp_dir / f"{file_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_path.suffix}"
            shutil.copy2(file_path, backup_path)
            logger.info(f"Created backup at {backup_path}")

            # Detect file format
            file_format = self._detect_file_format(file_path)
            logger.info(f"Detected file format: {file_format}")

            # Try reading based on detected format first
            df = None
            errors = []

            if file_format == 'xlsx':
                df = self._read_with_engine(file_path, 'openpyxl')
            elif file_format == 'xls':
                df = self._read_with_engine(file_path, 'xlrd')
            elif file_format == 'ods':
                df = self._read_with_engine(file_path, 'odf')
            elif file_format == 'csv':
                df = self._read_with_engine(file_path, 'csv')

            # If initial attempt fails, try all available engines
            if df is None:
                engines = ['openpyxl', 'xlrd', 'odf']
                for engine in engines:
                    df = self._read_with_engine(file_path, engine)
                    if df is not None:
                        break

            # If still no success, try CSV conversion
            if df is None:
                try:
                    # Try to convert to CSV using Excel if available
                    csv_path = self.temp_dir / f"{file_path.stem}.csv"
                    df = self._convert_to_csv(file_path)
                except Exception as e:
                    errors.append(f"CSV conversion failed: {str(e)}")

            if df is None:
                error_msg = "Failed to load file with all available methods:\n" + "\n".join(errors)
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Process the DataFrame
            df.columns = df.columns.str.strip()
            return self._process_single_dataframe(df)

        except Exception as e:
            logger.error(f"Error loading file: {str(e)}")
            raise ValueError(f"Error loading Excel file: {str(e)}")

    def _process_single_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a single DataFrame with enhanced error handling.

        Args:
            df: Input DataFrame

        Returns:
            pd.DataFrame: Processed DataFrame
        """
        try:
            # Make a copy to avoid modifying the original
            df = df.copy()
            logger.info(f"Processing DataFrame with initial shape: {df.shape}")

            # Handle missing columns
            required_columns = {
                'Year': lambda: datetime.now().year,
                'Month': lambda: datetime.now().month,
                'Date': lambda: datetime.now().day,
                'Hour': lambda: datetime.now().hour,
                'Min': lambda: datetime.now().minute,
                'Sec': lambda: datetime.now().second,
                'PRN': lambda: 0,
                'Elevation': lambda: df['Elevation'].median() if 'Elevation' in df else 0,
                'SNR': lambda: df['SNR'].median() if 'SNR' in df else 0
            }

            # Add missing columns with default values
            for col, default_func in required_columns.items():
                if col not in df.columns:
                    logger.warning(f"Adding missing column {col} with default values")
                    df[col] = default_func()

            # Process Label column if it exists
            if 'Label' in df.columns:
                logger.info("Processing Label column")
                snr_values, los_nlos_labels = self._process_label_column(df)
                logger.info(f"Label distribution from Label column:\n{los_nlos_labels.value_counts()}")
                
                if 'Azimuth SNR' not in df.columns:
                    df['SNR'] = snr_values
                else:
                    try:
                        azimuth_snr = df['Azimuth SNR'].astype(str).str.split(expand=True)
                        if len(azimuth_snr.columns) >= 2:
                            df['Azimuth'] = pd.to_numeric(azimuth_snr[0], errors='coerce')
                            df['SNR'] = pd.to_numeric(azimuth_snr[1], errors='coerce')
                    except Exception as e:
                        logger.warning(f"Failed to process Azimuth SNR column: {e}")
                
                df['LOS/NLOS'] = los_nlos_labels

            # Clean up LOS/NLOS labels
            if 'LOS/NLOS' in df.columns:
                logger.info("Cleaning LOS/NLOS labels")
                original_dist = df['LOS/NLOS'].value_counts()
                logger.info(f"Original label distribution:\n{original_dist}")

                # Convert to uppercase and strip whitespace
                df['LOS/NLOS'] = df['LOS/NLOS'].str.strip().str.upper()
                
                # Map variations to standard values
                label_map = {
                    'L': 'LOS',
                    'N': 'NLOS',
                    'LOS': 'LOS',
                    'NLOS': 'NLOS',
                    'LINE-OF-SIGHT': 'LOS',
                    'NON-LINE-OF-SIGHT': 'NLOS',
                    '0': 'LOS',
                    '1': 'NLOS'
                }
                df['LOS/NLOS'] = df['LOS/NLOS'].map(label_map)
                
                # Log any unmapped values
                unmapped = df[~df['LOS/NLOS'].isin(['LOS', 'NLOS'])]
                if not unmapped.empty:
                    logger.warning(f"Found {len(unmapped)} rows with unmapped labels: {unmapped['LOS/NLOS'].unique()}")
                
                # Remove rows with invalid labels
                valid_mask = df['LOS/NLOS'].isin(['LOS', 'NLOS'])
                if not valid_mask.all():
                    invalid_count = (~valid_mask).sum()
                    logger.warning(f"Removing {invalid_count} rows with invalid labels")
                    df = df[valid_mask]

                final_dist = df['LOS/NLOS'].value_counts()
                logger.info(f"Final label distribution:\n{final_dist}")

            # Ensure all numeric columns are properly converted
            numeric_columns = ['Year', 'Month', 'Date', 'Hour', 'Min', 'Sec', 'PRN', 'Elevation', 'SNR']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Fill missing values
            df = df.fillna(method='ffill').fillna(method='bfill')

            # Validate the final DataFrame
            if df.empty:
                raise ValueError("All rows were filtered out during processing")

            if 'LOS/NLOS' not in df.columns:
                raise ValueError("Missing LOS/NLOS labels after processing")

            if df['LOS/NLOS'].isnull().any():
                null_count = df['LOS/NLOS'].isnull().sum()
                raise ValueError(f"Found {null_count} null values in LOS/NLOS labels after processing")

            logger.info(f"Final DataFrame shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error processing DataFrame: {str(e)}")
            raise ValueError(f"Error processing DataFrame: {str(e)}")

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

    def _process_label_column(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Process and validate the label column.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            Tuple[pd.Series, pd.Series]: Processed labels and mask of valid labels
        """
        # Find label column
        label_columns = [col for col in df.columns if col.lower() in ['label', 'los/nlos', 'los_nlos', 'class']]
        if not label_columns:
            raise ValueError("No label column found. Expected columns: 'label', 'los/nlos', 'los_nlos', or 'class'")
        
        label_col = label_columns[0]
        labels = df[label_col].copy()
        
        # Convert labels to string and uppercase for standardization
        labels = labels.astype(str).str.upper()
        
        # Create mapping for various label formats
        label_mapping = {
            'LOS': 'LOS',
            'NLOS': 'NLOS',
            'L': 'LOS',
            'N': 'NLOS',
            '0': 'LOS',
            '1': 'NLOS',
            'LINE-OF-SIGHT': 'LOS',
            'NON-LINE-OF-SIGHT': 'NLOS'
        }
        
        # Apply mapping and create mask for valid labels
        valid_mask = labels.isin(label_mapping.keys())
        labels = labels.map(label_mapping)
        
        # Log label distribution
        label_dist = labels[valid_mask].value_counts()
        logger.info(f"Label distribution: {label_dist.to_dict()}")
        
        # Validate that we have both classes
        unique_labels = labels[valid_mask].unique()
        if len(unique_labels) < 2:
            available_classes = ', '.join(unique_labels)
            logger.error(f"Insufficient classes in data. Found only: {available_classes}")
            raise ValueError(f"Data must contain both LOS and NLOS classes. Found only: {available_classes}")
        
        # Log number of invalid labels
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} rows with invalid labels. These will be excluded from processing.")
        
        return labels, valid_mask

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

    def combine_los_nlos_data(self, los_file: Union[str, Path], nlos_file: Union[str, Path]) -> pd.DataFrame:
        """
        Combine LOS and NLOS data files into a single dataset.

        Args:
            los_file: Path to the LOS data file
            nlos_file: Path to the NLOS data file

        Returns:
            pd.DataFrame: Combined dataset with proper labels
        """
        # Validate input files
        los_path = self._validate_file(los_file)
        nlos_path = self._validate_file(nlos_file)

        # Create backups
        los_backup = self._create_backup(los_path)
        nlos_backup = self._create_backup(nlos_path)

        logger.info("Loading LOS and NLOS data files...")

        try:
            # Load LOS data
            los_df = self.load_excel_data(los_backup)
            if los_df.empty:
                raise ValueError("LOS data file is empty after loading")
            
            # Add LOS label if not present
            if not any(col.lower() in ['label', 'los/nlos', 'los_nlos', 'class'] for col in los_df.columns):
                los_df['Label'] = 'LOS'
            
            # Load NLOS data
            nlos_df = self.load_excel_data(nlos_backup)
            if nlos_df.empty:
                raise ValueError("NLOS data file is empty after loading")
            
            # Add NLOS label if not present
            if not any(col.lower() in ['label', 'los/nlos', 'los_nlos', 'class'] for col in nlos_df.columns):
                nlos_df['Label'] = 'NLOS'

            # Ensure both dataframes have the same columns
            common_cols = set(los_df.columns) & set(nlos_df.columns)
            if not common_cols:
                raise ValueError("No common columns found between LOS and NLOS data files")
            
            los_df = los_df[list(common_cols)]
            nlos_df = nlos_df[list(common_cols)]

            # Combine the datasets
            combined_df = pd.concat([los_df, nlos_df], ignore_index=True)
            
            # Validate the combined dataset
            if combined_df.empty:
                raise ValueError("Combined dataset is empty")
            
            # Process and validate labels
            labels, valid_mask = self._process_label_column(combined_df)
            if valid_mask.sum() == 0:
                raise ValueError("No valid labels found in the combined dataset")
            
            # Apply the valid mask to the combined dataset
            combined_df = combined_df[valid_mask].copy()
            
            # Log dataset statistics
            logger.info(f"Combined dataset statistics:")
            logger.info(f"Total samples: {len(combined_df)}")
            logger.info(f"Label distribution: {labels[valid_mask].value_counts().to_dict()}")
            logger.info(f"Features: {list(combined_df.columns)}")

            return combined_df

        except Exception as e:
            logger.error(f"Error combining LOS/NLOS data: {str(e)}")
            raise RuntimeError(f"Failed to combine LOS/NLOS data: {str(e)}")

    def extract_features_from_excel(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from Excel and extract features.

        Args:
            file_path (Union[str, Path]): Path to the Excel file

        Returns:
            pd.DataFrame: Extracted features
        """
        try:
            # Load data from Excel
            raw_data = self.load_excel_data(file_path)
            logger.info(f"Loaded data from {file_path} with shape {raw_data.shape}")

            # Check if we have LOS/NLOS labels before processing
            if 'LOS/NLOS' not in raw_data.columns:
                # Try to infer from filename
                file_path = str(file_path).lower()
                if 'los' in file_path and 'nlos' not in file_path:
                    logger.info("Inferring LOS labels from filename")
                    raw_data['LOS/NLOS'] = 'LOS'
                elif 'nlos' in file_path:
                    logger.info("Inferring NLOS labels from filename")
                    raw_data['LOS/NLOS'] = 'NLOS'
                else:
                    raise ValueError("Could not determine LOS/NLOS labels from data or filename")

            # Log label distribution
            label_dist = raw_data['LOS/NLOS'].value_counts()
            logger.info(f"Label distribution before processing:\n{label_dist}")

            # Preprocess data
            processed_data = self.preprocess_data(raw_data)
            logger.info(f"Preprocessed data shape: {processed_data.shape}")

            # Extract features
            features = self.extract_features(processed_data)
            
            # Add labels
            features['LOS/NLOS'] = processed_data['LOS/NLOS']
            
            # Validate final label distribution
            final_dist = features['LOS/NLOS'].value_counts()
            logger.info(f"Final label distribution:\n{final_dist}")
            
            if len(final_dist) < 2:
                logger.error(f"Only found one class in the data: {final_dist.index[0]}")
                raise ValueError(f"Data contains only one class: {final_dist.index[0]}. Both LOS and NLOS samples are required.")

            return features

        except Exception as e:
            logger.error(f"Error extracting features from Excel: {str(e)}")
            raise ValueError(f"Error extracting features from Excel: {str(e)}") 