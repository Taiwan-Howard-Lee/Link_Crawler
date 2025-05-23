"""
Test script for the link crawler.
This script tests the crawler on a single URL to verify it works correctly.
"""
import asyncio
import os
import sys
from dotenv import load_dotenv
# No need to import genai directly as we're using the ApiKeyRotator
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

# Add the parent directory to the path so we can import from main.py if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api_key_rotator import ApiKeyRotator

# Load environment variables
load_dotenv()

# Initialize the API key rotator
api_key_rotator = ApiKeyRotator(cooldown_period=30)

async def test_crawler():
    """Test the crawler on a single URL."""
    print("Testing crawler on a single URL...")

    # Output file for the test result
    output_file = "results/test_single_url_result.md"

    # Ensure results directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Initialize the crawler
    async with AsyncWebCrawler() as crawler:
        # Use minimal configuration for crawl4ai 0.6.2
        config = CrawlerRunConfig()

        # Test URL
        url = "https://example.com"

        print(f"Crawling URL: {url}")

        # Crawl the URL
        result = await crawler.arun(url=url, config=config)

        # Print the results
        print(f"\nURL: {url}")
        print(f"Success: {result.success}")

        if result.success:
            print(f"\nTitle: {result.metadata.get('title', 'No title')}")
            print(f"Description: {result.metadata.get('description', 'No description')}")

            print("\nMarkdown Content (first 500 chars):")
            if result.markdown:
                print(result.markdown[:500] + "...")
            else:
                print("No markdown content")

            # Set up Gemini model with API key rotation using gemini-2.0-flash
            model = api_key_rotator.get_gemini_model('gemini-2.0-flash')

            # Get content using compatible approach for crawl4ai 0.6.2
            content = ""
            if hasattr(result, 'markdown') and result.markdown:
                content = result.markdown
                print("Using markdown content")
            elif hasattr(result, 'text') and result.text:
                content = result.text
                print("Using text content")
            elif hasattr(result, 'content') and result.content:
                content = result.content
                print("Using raw content")

            if content:
                prompt = f"""
                Summarize the following web page content in exactly 3 lines with a total of about 60 words.
                Focus on the main points and key information.

                Content:
                {content[:10000]}
                """

                response = model.generate_content(prompt)
                summary = response.text.strip()

                print("\nGemini Summary:")
                print(summary)

                # Save the result to a file
                with open(output_file, "w") as f:
                    f.write(f"# Test Result for {url}\n\n")
                    f.write(f"## Metadata\n\n")
                    f.write(f"- Title: {result.metadata.get('title', 'No title')}\n")
                    f.write(f"- Description: {result.metadata.get('description', 'No description')}\n\n")
                    f.write(f"## Summary\n\n{summary}\n\n")
                    f.write(f"## Content\n\n{result.markdown[:1000]}...\n")

                print(f"\nTest result saved to {output_file}")
            else:
                print("\nNo content to summarize")
        else:
            print(f"Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(test_crawler())
