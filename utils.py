"""
Utility functions for the link crawler application.
"""
import pandas as pd
import logging
from typing import List, Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def read_csv(file_path: str, url_column: str) -> pd.DataFrame:
    """
    Read a CSV file and validate that it contains the required URL column.
    
    Args:
        file_path: Path to the CSV file
        url_column: Name of the column containing URLs
        
    Returns:
        DataFrame containing the CSV data
    """
    try:
        df = pd.read_csv(file_path)
        
        # Check if URL column exists
        if url_column not in df.columns:
            raise ValueError(f"Column '{url_column}' not found in CSV file. Available columns: {', '.join(df.columns)}")
        
        logger.info(f"Successfully loaded CSV with {len(df)} rows")
        return df
    
    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}")
        raise

def save_csv(df: pd.DataFrame, output_path: str) -> None:
    """
    Save DataFrame to a CSV file.
    
    Args:
        df: DataFrame to save
        output_path: Path where the CSV will be saved
    """
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved results to {output_path}")
    
    except Exception as e:
        logger.error(f"Error saving CSV file: {str(e)}")
        raise

def clean_text(text: str) -> str:
    """
    Clean and normalize text for better processing.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove non-printable characters
    text = ''.join(c for c in text if c.isprintable() or c in ['\n', '\t'])
    
    return text
