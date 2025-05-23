"""
Link Crawler: Crawls URLs from a CSV file, gets content summaries from Gemini API,
and adds the summaries back to the CSV.

This script uses crawl4ai for efficient web crawling and Gemini API for summarization.
"""
import os
import asyncio
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from tqdm import tqdm
import logging
import argparse
from typing import List, Dict, Optional, Any, Tuple
from api_key_rotator import ApiKeyRotator
from concurrent.futures import ThreadPoolExecutor
import threading

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize the API key rotator
api_key_rotator = ApiKeyRotator(cooldown_period=30)  # 30 seconds cooldown between using the same key

def setup_gemini_model():
    """
    Set up and return the Gemini model for text generation.
    Uses the API key rotator to get a fresh key.

    Returns:
        A Gemini model instance, or None if all keys are exhausted
    """
    # Get a model with a fresh API key using gemini-2.0-flash as specified by the user
    model = api_key_rotator.get_gemini_model('gemini-2.0-flash')

    if model is None:
        logger.error("All API keys are exhausted. Cannot set up Gemini model.")

    return model

async def get_summary_from_gemini(content: str, model) -> str:
    """
    Get a summary of the content from Gemini API.

    Args:
        content: The web page content to summarize
        model: The Gemini model to use

    Returns:
        A summary of the content (60 words, 3 lines)
    """
    # Check if model is None (all API keys exhausted)
    if model is None:
        logger.error("Cannot generate summary: all API keys are exhausted")
        return "API keys exhausted"

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            if not content or len(content.strip()) < 50:
                return "Not enough information"

            prompt = f"""
            Summarize the following web page content in exactly 3 lines with a total of about 60 words.
            Focus on the main points and key information. The web page is about Brazil startups.

            If there is not enough meaningful information to create a good summary, respond with exactly "Not enough information".

            Content:
            {content[:10000]}  # Limiting content to avoid token limits
            """

            response = model.generate_content(prompt)
            summary = response.text.strip()

            return summary

        except Exception as e:
            error_message = str(e).lower()
            retry_count += 1

            # Check if it's a rate limit error
            if "rate limit" in error_message or "quota" in error_message:
                logger.warning(f"Rate limit reached. Rotating API key and retrying ({retry_count}/{max_retries})...")

                # Get a fresh model with a new API key
                model = api_key_rotator.get_gemini_model('gemini-2.0-flash')

                # If model is None, all API keys are exhausted
                if model is None:
                    logger.error("All API keys are exhausted during retry")
                    return "API keys exhausted"

                # If it's not the last retry, continue to the next iteration
                if retry_count < max_retries:
                    continue

            # For other errors or if we've exhausted retries
            logger.error(f"Error getting summary from Gemini: {str(e)}")
            return "Error generating summary"

async def process_url(url: str, crawler: AsyncWebCrawler, model: Any, max_retries: int = 3) -> str:
    """
    Process a single URL: crawl it and get a summary.

    Args:
        url: The URL to process
        crawler: The AsyncWebCrawler instance
        model: The Gemini model
        max_retries: Maximum number of retries for crawling (default: 3)

    Returns:
        A summary of the URL content
    """
    # Check if model is None (all API keys exhausted)
    if model is None:
        logger.error(f"Cannot process URL {url}: all API keys are exhausted")
        return "API keys exhausted"

    # Check if URL is valid
    if not url or pd.isna(url) or url == "":
        return "No URL provided"

    # Ensure URL has a scheme (http:// or https://)
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    # Use minimal configuration for crawl4ai 0.6.2
    config = CrawlerRunConfig()

    # Initialize retry counter
    retry_count = 0

    while retry_count < max_retries:
        try:
            logger.info(f"Crawling URL: {url} (attempt {retry_count + 1}/{max_retries})")

            # Crawl the URL
            result = await crawler.arun(url=url, config=config)

            # Check if crawl was successful
            if hasattr(result, 'success') and not result.success:
                error_msg = getattr(result, 'error', 'Unknown error')
                logger.warning(f"Failed to crawl URL: {url}, Error: {error_msg}")

                # Increment retry counter and try again if not at max retries
                retry_count += 1
                if retry_count < max_retries:
                    logger.info(f"Retrying URL: {url} (attempt {retry_count + 1}/{max_retries})")
                    await asyncio.sleep(1)  # Wait a second before retrying
                    continue
                else:
                    return "Invalid link"

            # In crawl4ai 0.6.2, content is accessed differently
            # Try to get markdown content first, then fall back to text
            content = ""
            if hasattr(result, 'markdown') and result.markdown:
                content = result.markdown
            elif hasattr(result, 'text') and result.text:
                content = result.text
            elif hasattr(result, 'content') and result.content:
                content = result.content

            if not content:
                logger.warning(f"No content found for URL: {url}")

                # Increment retry counter and try again if not at max retries
                retry_count += 1
                if retry_count < max_retries:
                    logger.info(f"Retrying URL: {url} (attempt {retry_count + 1}/{max_retries})")
                    await asyncio.sleep(1)  # Wait a second before retrying
                    continue
                else:
                    return "Invalid link"

            # Check if content is too short (less than 100 characters)
            if len(content.strip()) < 100:
                logger.warning(f"Content too short for URL: {url} (length: {len(content)} chars)")
                # We'll still send it to Gemini and let it decide if there's enough to summarize

            logger.info(f"Successfully extracted content from URL: {url} (length: {len(content)} chars)")

            # Get summary from Gemini
            summary = await get_summary_from_gemini(content, model)
            return summary

        except Exception as e:
            logger.error(f"Exception while crawling URL {url}: {str(e)}")

            # Increment retry counter and try again if not at max retries
            retry_count += 1
            if retry_count < max_retries:
                logger.info(f"Retrying URL: {url} (attempt {retry_count + 1}/{max_retries})")
                await asyncio.sleep(1)  # Wait a second before retrying
                continue
            else:
                return "Invalid link"

    # This should not be reached, but just in case
    return "Invalid link"

async def process_url_batch(urls_data: List[Tuple[int, str]], crawler: AsyncWebCrawler, model: Any,
                      url_column: str, summary_column: str, df: pd.DataFrame,
                      lock: threading.Lock) -> List[Tuple[int, str, str]]:
    """
    Process a batch of URLs in parallel.

    Args:
        urls_data: List of tuples containing (index, url)
        crawler: The AsyncWebCrawler instance
        model: The Gemini model
        url_column: Name of the column containing URLs
        summary_column: Name of the column to store summaries
        df: The DataFrame to update
        lock: Lock for thread-safe DataFrame updates

    Returns:
        List of tuples containing (index, url, summary)
    """
    results = []

    for index, url in urls_data:
        # Skip empty URLs
        if not url or pd.isna(url) or url == "":
            with lock:
                df.at[index, summary_column] = "No URL provided"
            results.append((index, url, "No URL provided"))
            continue

        # Skip already processed URLs (if rerunning)
        if summary_column in df.columns and df.at[index, summary_column] and not pd.isna(df.at[index, summary_column]):
            logger.info(f"Skipping already processed URL: {url}")
            results.append((index, url, df.at[index, summary_column]))
            continue

        logger.info(f"Processing URL: {url}")

        # Process the URL
        summary = await process_url(url, crawler, model)

        # Update the DataFrame with the summary
        with lock:
            df.at[index, summary_column] = summary

        results.append((index, url, summary))

        # Check if API keys are exhausted
        if summary == "API keys exhausted":
            logger.error("All API keys are exhausted in batch processing.")
            break

    return results

async def process_urls_in_parallel(urls_data: List[Tuple[int, str]], crawler: AsyncWebCrawler, model: Any,
                                  url_column: str, summary_column: str, df: pd.DataFrame,
                                  num_workers: int) -> bool:
    """
    Process URLs in parallel using multiple workers.

    Args:
        urls_data: List of tuples containing (index, url)
        crawler: The AsyncWebCrawler instance
        model: The Gemini model
        url_column: Name of the column containing URLs
        summary_column: Name of the column to store summaries
        df: The DataFrame to update
        num_workers: Number of parallel workers

    Returns:
        True if processing completed successfully, False if all API keys were exhausted
    """
    # Create a lock for thread-safe DataFrame updates
    lock = threading.Lock()

    # Split the URLs into batches for each worker
    batch_size = max(1, len(urls_data) // num_workers)
    batches = [urls_data[i:i + batch_size] for i in range(0, len(urls_data), batch_size)]

    logger.info(f"Processing {len(urls_data)} URLs with {num_workers} workers ({len(batches)} batches)")

    # Process batches in parallel
    tasks = []
    for batch in batches:
        task = asyncio.create_task(process_url_batch(batch, crawler, model, url_column, summary_column, df, lock))
        tasks.append(task)

    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)

    # Flatten the results
    all_results = [item for sublist in results for item in sublist]

    # Check if any batch encountered API key exhaustion
    for _, _, summary in all_results:
        if summary == "API keys exhausted":
            logger.error("API key exhaustion detected in parallel processing. Stopping.")
            return False

    return True

async def process_csv(input_file: str, output_file: str, url_column: str, summary_column: str,
                     start_index: int = 0, end_index: int = -1, num_workers: int = 1,
                     save_interval: int = 5, max_retries: int = 3) -> None:
    """
    Process a CSV file containing URLs, crawl each URL, get a summary from Gemini,
    and add the summaries to a new column in the CSV.

    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the output CSV file
        url_column: Name of the column containing URLs
        summary_column: Name of the column to store summaries
        start_index: Start processing from this row index
        end_index: End processing at this row index (-1 means process all rows)
        num_workers: Number of parallel workers (1-30)
        save_interval: How often to save progress (every N URLs processed)
        max_retries: Maximum number of retries for crawling URLs
    """
    try:
        # Validate number of workers
        num_workers = max(1, min(30, num_workers))
        logger.info(f"Using {num_workers} parallel worker{'s' if num_workers > 1 else ''}")

        # Read the CSV file
        df = pd.read_csv(input_file)

        # Check if URL column exists
        if url_column not in df.columns:
            raise ValueError(f"Column '{url_column}' not found in CSV file. Available columns: {', '.join(df.columns)}")

        # Add summary column if it doesn't exist
        if summary_column not in df.columns:
            df[summary_column] = ""

        # Set end_index to the last row if it's -1
        if end_index == -1 or end_index >= len(df):
            end_index = len(df) - 1

        # Validate indices
        if start_index < 0 or start_index >= len(df):
            raise ValueError(f"Invalid start_index: {start_index}. Must be between 0 and {len(df)-1}")
        if end_index < start_index:
            raise ValueError(f"Invalid end_index: {end_index}. Must be greater than or equal to start_index: {start_index}")

        # Get the subset of rows to process
        rows_to_process = df.iloc[start_index:end_index+1]

        logger.info(f"Processing rows from index {start_index} to {end_index} (total: {len(rows_to_process)} rows)")

        # Initialize Gemini model
        model = setup_gemini_model()

        # Check if all API keys are already exhausted
        if model is None:
            logger.error("All API keys are exhausted. Cannot process CSV file.")
            # Save the current state before exiting
            df.to_csv(output_file, index=False)
            logger.info(f"Saved current state to {output_file}")
            return

        # Prepare data for parallel processing
        urls_data = [(index, row[url_column]) for index, row in rows_to_process.iterrows()]

        # Initialize the crawler
        async with AsyncWebCrawler() as crawler:
            # If only one worker, process sequentially
            if num_workers == 1:
                # Create a progress bar with more information
                progress_bar = tqdm(
                    total=len(urls_data),
                    desc="Processing URLs",
                    unit="url",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                )

                processed_count = 0
                for i, (index, url) in enumerate(urls_data):
                    # Skip empty URLs
                    if not url or pd.isna(url) or url == "":
                        df.at[index, summary_column] = "No URL provided"
                        progress_bar.update(1)
                        processed_count += 1
                        continue

                    # Skip already processed URLs (if rerunning)
                    if summary_column in df.columns and df.at[index, summary_column] and not pd.isna(df.at[index, summary_column]):
                        logger.info(f"Skipping already processed URL: {url}")
                        progress_bar.update(1)
                        processed_count += 1
                        continue

                    # Update progress bar description with current URL
                    progress_bar.set_description(f"Processing: {url[:30]}..." if len(url) > 30 else f"Processing: {url}")

                    logger.info(f"Processing URL {i+1}/{len(urls_data)}: {url}")

                    # Process the URL with retries
                    summary = await process_url(url, crawler, model, max_retries)
                    df.at[index, summary_column] = summary

                    # Update progress bar
                    progress_bar.update(1)
                    processed_count += 1

                    # Check if API keys are exhausted
                    if summary == "API keys exhausted":
                        logger.error("All API keys are exhausted. Stopping processing.")
                        # Save progress before stopping
                        df.to_csv(output_file, index=False)
                        logger.info(f"Progress saved after processing {processed_count} URLs. Stopping due to API key exhaustion.")
                        progress_bar.close()
                        return

                    # Save progress periodically based on configurable interval
                    if processed_count > 0 and processed_count % save_interval == 0:
                        df.to_csv(output_file, index=False)
                        logger.info(f"Progress saved after processing {processed_count} URLs")

                # Close the progress bar
                progress_bar.close()
            else:
                # Process URLs in parallel
                success = await process_urls_in_parallel(
                    urls_data, crawler, model, url_column, summary_column, df, num_workers
                )

                if not success:
                    logger.error("Parallel processing stopped due to API key exhaustion.")
                    df.to_csv(output_file, index=False)
                    logger.info(f"Progress saved to {output_file} after API key exhaustion.")
                    return

        # Save the final results
        df.to_csv(output_file, index=False)
        logger.info(f"Successfully processed {len(rows_to_process)} URLs and saved results to {output_file}")

    except Exception as e:
        logger.error(f"Error processing CSV file: {str(e)}")
        # Save current progress in case of error
        try:
            df.to_csv(output_file, index=False)
            logger.info(f"Saved current progress to {output_file} after error")
        except:
            logger.error("Could not save progress after error")
        raise

async def async_main(input_file: str, output_file: str, url_column: str, summary_column: str,
                 start_index: int = 0, end_index: int = -1, num_workers: int = 1,
                 save_interval: int = 5, max_retries: int = 3) -> None:
    """
    Async main function to process the CSV file.

    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the output CSV file
        url_column: Name of the column containing URLs
        summary_column: Name of the column to store summaries
        start_index: Start processing from this row index
        end_index: End processing at this row index (-1 means process all rows)
        num_workers: Number of parallel workers (1-30)
        save_interval: How often to save progress (every N URLs processed)
        max_retries: Maximum number of retries for crawling URLs
    """
    await process_csv(input_file, output_file, url_column, summary_column,
                     start_index, end_index, num_workers, save_interval, max_retries)

def main():
    """
    Main function to parse arguments and run the crawler.
    """
    parser = argparse.ArgumentParser(description="Crawl URLs from a CSV file and add summaries from Gemini API.")
    parser.add_argument("--input_file", default="final.csv", help="Path to the input CSV file containing URLs (default: final.csv)")
    parser.add_argument("--output_file", help="Path to save the output CSV file (default: results/input_file_with_summaries.csv)")
    parser.add_argument("--url_column", default="Site", help="Name of the column containing URLs (default: 'Site')")
    parser.add_argument("--summary_column", default="Resumo", help="Name of the column to store summaries (default: 'Resumo')")
    parser.add_argument("--start_index", type=int, default=0, help="Start processing from this row index (default: 0)")
    parser.add_argument("--end_index", type=int, default=-1, help="End processing at this row index (default: -1, process all rows)")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (1-30, default: 1)")
    parser.add_argument("--save_interval", type=int, default=5, help="How often to save progress (every N URLs processed, default: 5)")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries for crawling URLs (default: 3)")

    args = parser.parse_args()

    # Set default output file if not provided
    if not args.output_file:
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        args.output_file = f"results/{base_name}_with_summaries.csv"

    # Ensure results directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Validate number of workers
    num_workers = max(1, min(30, args.workers))
    if num_workers != args.workers:
        logger.warning(f"Adjusted number of workers to {num_workers} (valid range: 1-30)")

    # Validate save interval
    save_interval = max(1, args.save_interval)
    if save_interval != args.save_interval:
        logger.warning(f"Adjusted save interval to {save_interval} (must be at least 1)")

    # Validate max retries
    max_retries = max(1, args.max_retries)
    if max_retries != args.max_retries:
        logger.warning(f"Adjusted max retries to {max_retries} (must be at least 1)")

    # Run the async main function
    asyncio.run(async_main(
        args.input_file, args.output_file, args.url_column, args.summary_column,
        args.start_index, args.end_index, num_workers, save_interval, max_retries
    ))

if __name__ == "__main__":
    main()
