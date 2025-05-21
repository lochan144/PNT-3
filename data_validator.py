"""
Module for validating GNSS data quality and handling edge cases.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataValidator:
    """Class for validating GNSS data quality."""

    def __init__(self):
        """Initialize the data validator."""
        self.validation_rules = {
            'Elevation': {
                'range': (-90, 90),
                'type': 'float',
                'required': True
            },
            'Azimuth': {
                'range': (0, 360),
                'type': 'float',
                'required': True
            },
            'SNR': {
                'range': (-50, 100),  # Typical SNR range in dB
                'type': 'float',
                'required': True
            },
            'LOS/NLOS': {
                'values': ['LOS', 'NLOS'],
                'type': 'category',
                'required': True
            }
        }

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Validate the input DataFrame against defined rules.

        Args:
            df: Input DataFrame

        Returns:
            Tuple[bool, Dict[str, List[str]]]: Validation result and list of issues
        """
        issues = {}
        
        try:
            # Check for required columns
            missing_columns = [col for col, rules in self.validation_rules.items() 
                             if rules['required'] and col not in df.columns]
            if missing_columns:
                issues['missing_columns'] = missing_columns

            # Validate each column
            for col, rules in self.validation_rules.items():
                if col not in df.columns:
                    continue

                col_issues = []
                
                # Check data type
                if rules['type'] == 'float':
                    non_numeric = df[col].apply(lambda x: not pd.api.types.is_numeric_dtype(type(x)))
                    if non_numeric.any():
                        col_issues.append(f"Column contains non-numeric values at rows: {non_numeric[non_numeric].index.tolist()}")

                # Check value range
                if 'range' in rules:
                    min_val, max_val = rules['range']
                    out_of_range = df[col][(df[col] < min_val) | (df[col] > max_val)]
                    if not out_of_range.empty:
                        col_issues.append(f"Values out of range ({min_val}, {max_val}) at rows: {out_of_range.index.tolist()}")

                # Check categorical values
                if 'values' in rules:
                    invalid_values = df[col][~df[col].isin(rules['values'])]
                    if not invalid_values.empty:
                        col_issues.append(f"Invalid values found: {invalid_values.unique().tolist()}")

                if col_issues:
                    issues[col] = col_issues

            # Check for duplicate rows
            duplicates = df.duplicated()
            if duplicates.any():
                issues['duplicates'] = f"Found {duplicates.sum()} duplicate rows"

            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.any():
                issues['missing_values'] = missing_values[missing_values > 0].to_dict()

        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            issues['validation_error'] = str(e)

        return len(issues) == 0, issues

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data based on validation rules.

        Args:
            df: Input DataFrame

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        try:
            # Make a copy to avoid modifying the original
            df = df.copy()

            # Handle missing values
            for col, rules in self.validation_rules.items():
                if col not in df.columns:
                    continue

                if rules['type'] == 'float':
                    # Fill missing numeric values with median
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(df[col].median())

                    # Clip values to valid range if specified
                    if 'range' in rules:
                        min_val, max_val = rules['range']
                        df[col] = df[col].clip(min_val, max_val)

                elif rules['type'] == 'category':
                    # Fill missing categorical values with mode
                    if 'values' in rules:
                        df[col] = df[col].fillna(df[col].mode()[0])
                        # Replace invalid values with the most common valid value
                        valid_mask = df[col].isin(rules['values'])
                        if not valid_mask.all():
                            valid_values = df[col][valid_mask]
                            if not valid_values.empty:
                                most_common = valid_values.mode()[0]
                                df.loc[~valid_mask, col] = most_common

            # Remove duplicate rows
            df = df.drop_duplicates()

            return df

        except Exception as e:
            logger.error(f"Data cleaning error: {str(e)}")
            raise ValueError(f"Error cleaning data: {str(e)}")

    def generate_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate a data quality report.

        Args:
            df: Input DataFrame

        Returns:
            Dict: Report containing data quality metrics
        """
        try:
            report = {
                'row_count': len(df),
                'column_count': len(df.columns),
                'missing_values': df.isnull().sum().to_dict(),
                'duplicate_rows': df.duplicated().sum(),
                'column_stats': {}
            }

            for col in df.columns:
                if col in self.validation_rules:
                    rules = self.validation_rules[col]
                    stats = {
                        'unique_values': df[col].nunique(),
                        'missing_percentage': (df[col].isnull().sum() / len(df)) * 100
                    }

                    if rules['type'] == 'float':
                        stats.update({
                            'mean': df[col].mean(),
                            'median': df[col].median(),
                            'std': df[col].std(),
                            'min': df[col].min(),
                            'max': df[col].max()
                        })
                    elif rules['type'] == 'category':
                        stats['value_counts'] = df[col].value_counts().to_dict()

                    report['column_stats'][col] = stats

            return report

        except Exception as e:
            logger.error(f"Report generation error: {str(e)}")
            raise ValueError(f"Error generating report: {str(e)}") 