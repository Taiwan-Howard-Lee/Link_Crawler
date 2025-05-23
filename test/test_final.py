"""
Test script for processing a small subset of the final.csv file.
"""
import asyncio
import os
import sys
import pandas as pd
import argparse
from dotenv import load_dotenv

# Add the parent directory to the path so we can import from main.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import process_csv
from api_key_rotator import ApiKeyRotator
# No need to import AsyncWebCrawler directly as we're using process_csv

# Load environment variables
load_dotenv()

# Initialize the API key rotator
api_key_rotator = ApiKeyRotator(cooldown_period=30)

async def test_final_csv(num_workers: int = 1, save_interval: int = 5, max_retries: int = 3):
    """
    Test processing a small subset of final.csv.

    Args:
        num_workers: Number of parallel workers (1-30)
        save_interval: How often to save progress (every N URLs processed)
        max_retries: Maximum number of retries for crawling URLs
    """
    print("Testing crawler on a small subset of final.csv...")

    # Input and output files
    input_file = "final.csv"
    output_file = "results/final_test_results.csv"

    # Ensure results directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Column names
    url_column = "Site"
    summary_column = "Resumo"

    # Process the first 6 rows
    start_index = 0
    end_index = 5

    # Validate number of workers
    num_workers = max(1, min(30, num_workers))

    # Validate save interval
    save_interval = max(1, save_interval)

    # Validate max retries
    max_retries = max(1, max_retries)

    print(f"Processing rows {start_index} to {end_index} from {input_file}")
    print(f"Using {num_workers} parallel worker{'s' if num_workers > 1 else ''}")
    print(f"Save interval: {save_interval}, Max retries: {max_retries}")
    print(f"Results will be saved to {output_file}")

    # Process the CSV
    await process_csv(input_file, output_file, url_column, summary_column,
                     start_index, end_index, num_workers, save_interval, max_retries)

    # Print the results
    df = pd.read_csv(output_file)
    print("\nResults:")
    for index, row in df.iloc[start_index:end_index+1].iterrows():
        print(f"\nStartup: {row['Nome da Startup']}")
        print(f"URL: {row[url_column]}")
        print(f"Summary: {row[summary_column]}")

def main():
    """Parse command line arguments and run the test."""
    parser = argparse.ArgumentParser(description="Test the link crawler on a small subset of final.csv.")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (1-30, default: 1)")
    parser.add_argument("--save_interval", type=int, default=5, help="How often to save progress (every N URLs processed, default: 5)")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries for crawling URLs (default: 3)")

    args = parser.parse_args()

    # Run the test
    asyncio.run(test_final_csv(args.workers, args.save_interval, args.max_retries))

if __name__ == "__main__":
    main()
