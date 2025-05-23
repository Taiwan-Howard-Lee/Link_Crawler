# Link Crawler

A tool that crawls URLs from a CSV file, gets content summaries from Gemini API, and adds the summaries back to the CSV.

## Features

- Reads URLs from a specified column in a CSV file
- Uses crawl4ai's AsyncWebCrawler for efficient web crawling
- Extracts clean, structured content in markdown format
- Sends content to Gemini API for summarization (3 lines, ~60 words)
- **API Key Rotation**: Rotates between multiple Gemini API keys to avoid rate limits
- **API Key Exhaustion Handling**: Gracefully stops processing when all API keys are exhausted
- **API Key Recovery**: Automatically checks if exhausted API keys have recovered after a configurable period
- **Parallel Processing**: Supports processing multiple URLs concurrently (1-30 workers)
- **Retry Mechanism**: Automatically retries failed URL crawling attempts
- **Configurable Save Interval**: Control how often progress is saved
- Adds summaries to a new column in the CSV
- Handles errors gracefully and provides logging
- Supports processing specific ranges of rows

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Gemini API keys:
   ```
   GEMINI_API_KEY1=your_first_api_key_here
   GEMINI_API_KEY2=your_second_api_key_here
   GEMINI_API_KEY3=your_third_api_key_here
   ```
   You can add one, two, or three API keys. The system will rotate between them to avoid rate limits.

   When all API keys are exhausted (e.g., due to rate limits or usage quotas), the tool will:
   - Save the current progress to the output file
   - Log a clear error message
   - Gracefully stop processing

   This allows you to resume processing later when the API keys are available again.

## Project Structure

- `main.py`: Main script for processing CSV files
- `api_key_rotator.py`: Handles API key rotation to avoid rate limits
- `test/`: Contains test scripts
  - `test.py`: Tests the crawler on a single URL
  - `test_final.py`: Tests processing a small subset of final.csv
- `results/`: Output directory for all results
- `.env`: Configuration file for API keys

## Usage

Basic usage (processes final.csv by default):

```bash
python main.py
```

This will read URLs from the "Site" column in final.csv and save the results to "results/final_with_summaries.csv".

Advanced usage with parallel processing:

```bash
python main.py --input_file final.csv --output_file results/custom_results.csv --url_column Site --summary_column Resumo --start_index 0 --end_index 100 --workers 10
```

This will process URLs using 10 parallel workers, significantly speeding up the crawling process.

Running tests:

```bash
# Test on a single URL
python test/test.py

# Test on a small subset of final.csv
python test/test_final.py
```

### Arguments

- `--input_file`: Path to the input CSV file containing URLs (default: 'final.csv')
- `--output_file`: Path to save the output CSV file (default: 'results/input_file_with_summaries.csv')
- `--url_column`: Name of the column containing URLs (default: 'Site')
- `--summary_column`: Name of the column to store summaries (default: 'Resumo')
- `--start_index`: Start processing from this row index (default: 0)
- `--end_index`: End processing at this row index (default: -1, process all rows)
- `--workers`: Number of parallel workers for processing URLs (1-30, default: 1)
- `--save_interval`: How often to save progress (every N URLs processed, default: 5)
- `--max_retries`: Maximum number of retries for crawling URLs (default: 3)

## Example

Input CSV:
```
id,url,title
1,https://example.com,Example Domain
2,https://wikipedia.org,Wikipedia
```

Output CSV:
```
id,url,title,summary
1,https://example.com,Example Domain,"Example Domain is a reserved domain for documentation purposes. It's used in examples without risk of collision with actual websites. The domain is maintained by IANA and has been reserved since 1999."
2,https://wikipedia.org,Wikipedia,"Wikipedia is a free online encyclopedia created and edited by volunteers around the world. It's available in multiple languages and covers a vast range of topics. The site is operated by the Wikimedia Foundation and is one of the most visited websites globally."
```

## How It Works

1. The script reads a CSV file containing URLs
2. For each URL, it uses crawl4ai's AsyncWebCrawler to fetch and extract content
3. The content is processed and sent to Gemini API for summarization
4. The summary is added to a new column in the CSV
5. The updated CSV is saved periodically and at the end of processing

## Requirements

- Python 3.7+
- crawl4ai >= 0.6.0
- pandas
- google-generativeai
- python-dotenv
- tqdm
- asyncio
