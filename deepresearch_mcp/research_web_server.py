import tempfile
import json
import os
from typing import List, Union, Any

import PyPDF2
import openai
import requests
from PIL import Image
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from mcp.server.fastmcp import FastMCP

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"<path_to>\tesseract.exe"

print("DEBUG: Initializing FastMCP...")
mcp = FastMCP("DeepResearch Web Search")
print("DEBUG: FastMCP initialized.")

openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    print("ERROR: OPENAI_API_KEY environment variable not set in server process!")

print("DEBUG: OpenAI API Key set. (or attempted to be set)")


@mcp.tool(name="web_search")
def web_search(query: str) -> str:
    ddgs = DDGS()
    results = ddgs.text(query)
    if not results:
        return "[]"

    articles = []
    for r in results:
        title = r.get("title", "No title")
        summary = r.get("body") or r.get("snippet") or r.get("description", "")
        url = r.get("href") or r.get("url") or r.get("link", "")

        articles.append({
            "title": title,
            "summary": summary,
            "url": url
        })

    return json.dumps(articles[:5])


@mcp.tool(
    name="extract_urls_from_json",
    description="Extracts a list of URLs from either a JSON-string or a list of dicts."
)
def extract_urls_from_json(
    json_input: Union[str, List[dict]]) -> List[str]:
    """
    Args:
      json_input: either a JSON string representing a list of dicts,
                  or an actual Python list of dicts each containing a 'url' key.
    Returns:
      A list of extracted URLs.
    """
    # 1) Normalize to a Python list of dicts
    if isinstance(json_input, str):
        try:
            data = json.loads(json_input)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string provided: {e}") from e
    elif isinstance(json_input, list):
        data = json_input
    else:
        raise ValueError(f"Unsupported type for extract_urls_from_json: {type(json_input)}")

    # 2) Extract URLs
    urls: List[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "url" in item:
                urls.append(item["url"])
        return urls

    raise ValueError("Expected a list of objects with 'url' keys.")


@mcp.tool(name="extract_text_from_pdf", description="Extracts text from a PDF file.")
def extract_text_from_pdf(file_path: str, page_limit: int = 1) -> str:
    """
    Extracts text from a specified PDF file.
    Args:
        file_path (str): The absolute or relative path to the PDF file.
        page_limit (int): Max number of pages to process (0 for all pages).
    Returns:
        str: The extracted text content, or an error message.
    """
    if not os.path.exists(file_path):
        return f"Error: PDF file not found at '{file_path}'."
    if not file_path.lower().endswith('.pdf'):
        return f"Error: Provided file '{file_path}' is not a PDF."

    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text_content = []
            num_pages = len(reader.pages)

            pages_to_process = num_pages
            if page_limit > 0:
                pages_to_process = min(num_pages, page_limit)

            for i in range(pages_to_process):
                page = reader.pages[i]
                text_content.append(page.extract_text() or "")  # .extract_text() can return None

            return "\n".join(text_content)
    except PyPDF2.errors.PdfReadError as e:
        return f"Error reading PDF file (possibly corrupted or encrypted): {e}"
    except Exception as e:
        return f"An unexpected error occurred while processing PDF: {e}"


@mcp.tool(name="extract_text_from_url",
          description="Extracts the full, clean main text content from a specific web page URL. "
                      "Use this after a 'web_search' to read the detailed content of a promising link, "
                      "rather than just the snippet. It helps in deep analysis of specific articles.")
def extract_text_from_url(url: str) -> str:
    """
    Downloads content from a URL and extracts the main readable text.
    Prioritizes common content tags like <article> or <main> for better relevance.
    Args:
        url (str): The URL of the web page.
    Returns:
        str: The extracted text, or an error message.
    """
    try:
        headers = {'User-Agent': 'DeepResearchBot/1.0 (+http://your-research-project.com/)'}
        response = requests.get(url, headers=headers, timeout=15)  # Increased timeout slightly
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script, style, and other non-content tags
        for script_or_style in soup(["script", "style", "header", "footer", "nav", "form", "aside"]):
            script_or_style.extract()

            # --- START OF IMPROVED CONTENT EXTRACTION ---
        # Prioritize common tags that usually contain main article content
        main_content_tags = [
            'article',  # HTML5 article tag
            'main',  # HTML5 main content tag
            'div.entry-content',  # Common for blog posts/articles
            'div.post-content',  # Another common class for post content
            'div#main-content',  # Common ID for main content
            'div.content',  # Generic content div, less specific but sometimes applies
        ]

        content_element = None
        for tag_selector in main_content_tags:
            if tag_selector.startswith('div.'):
                # Handle class selector
                class_name = tag_selector.split('.')[1]
                content_element = soup.find('div', class_=class_name)
            elif tag_selector.startswith('div#'):
                # Handle ID selector
                id_name = tag_selector.split('#')[1]
                content_element = soup.find('div', id=id_name)
            else:
                # Handle direct tag names like 'article', 'main'
                content_element = soup.find(tag_selector)

            if content_element:
                break  # Found a suitable content element, stop searching

        # If a specific content element was found, extract text from it
        if content_element:
            text = content_element.get_text(separator='\n')
        else:
            # Fallback to extracting text from the entire body if no specific content tag is found
            text = soup.get_text(separator='\n')
        # --- END OF IMPROVED CONTENT EXTRACTION ---

        # Clean up whitespace and empty lines
        lines = (line.strip() for line in text.splitlines())
        text = '\n'.join(line for line in lines if line)

        if len(text) < 50:  # Add a warning for very short extracted content
            return f"Warning: Extracted very little text from {url}. It might be behind a paywall, not primarily text, or parsing failed. Extracted snippet: {text[:200]}"

        return text

    except requests.exceptions.MissingSchema:
        return f"Error: Invalid URL format. Did you include http:// or https://? - {url}"
    except requests.exceptions.ConnectionError:
        return f"Error: Could not connect to URL '{url}'. Check internet connection or URL validity."
    except requests.exceptions.Timeout:
        return f"Error: Request to URL '{url}' timed out."
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL '{url}': {e}"
    except Exception as e:
        return f"An unexpected error occurred while parsing HTML from '{url}': {e}"


@mcp.tool(name="perform_ocr", description="Performs OCR on an image file to extract text.")
def perform_ocr(image_path: str, lang: str = 'eng') -> str:
    """
    Performs Optical Character Recognition (OCR) on an image file.
    Args:
        image_path (str): The absolute or relative path to the image file (e.g., .png, .jpg, .tiff).
        lang (str): The language of the text in the image (e.g., 'eng', 'spa', 'fra').
    Returns:
        str: The extracted text, or an error message.
    """
    if not os.path.exists(image_path):
        return f"Error: Image file not found at '{image_path}'."

    # Check if Tesseract command is set or available in PATH
    try:
        pytesseract.get_tesseract_version()  # This will raise an error if tesseract is not found
    except pytesseract.TesseractNotFoundError:
        return ("Error: Tesseract-OCR engine not found. Please install it and ensure it's in your PATH, "
                "or set pytesseract.pytesseract.tesseract_cmd.")
    except Exception as e:
        return f"Error checking Tesseract: {e}"

    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang=lang)

        if not text.strip():
            return ("OCR performed, but no readable text was extracted. Image might be low quality "
                    "or contain no text.")

        return text
    except FileNotFoundError:
        return f"Error: Image file not found at '{image_path}'."
    except Exception as e:
        return f"An error occurred during OCR: {e}"


@mcp.tool(name="summarize_sources",
          description="Summarizes key takeaways from a list of extracted text documents or pages. Use this to synthesize information after gathering content from multiple sources.")
def summarize_sources(texts: List[str]) -> str:
    """
    Summarizes key ideas from a list of provided text documents.
    Args:
        texts (List[str]): A list of strings, where each string is the content of a document.
    Returns:
        str: A summary of the provided texts.
    """
    if not texts:
        return "No text provided to summarize."

    # Concatenate texts, ensuring context separation
    full_text_for_summary = "\n\n--- Document Separator ---\n\n".join(texts)

    prompt = (f"Please summarize the key ideas and main points from the following documents. Synthesize information, "
              f"avoid redundancy, and be concise:\n\n{full_text_for_summary}")

    try:
        completion = openai.chat.completions.create(
            model="gpt-4o",  # Using gpt-4o for summarization
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error summarizing sources: {e}"


@mcp.tool(
    name="download_and_summarize_pdf",
    description="Downloads a PDF from a URL, extracts its text, then returns a short summary."
)
def download_and_summarize_pdf(url: str, page_limit: int = 0) -> str:
    """
    1) Downloads the PDF at `url` to a temp file.
    2) Extracts text from up to `page_limit` pages (0 = all).
    3) Uses the existing `summarize_sources` logic to summarize that text.
    """
    # 1) download
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        return f"Error downloading PDF: {e}"

    # 2) write to temp file
    suffix = ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(resp.content)
        tmp_path = tmp.name

    text = extract_text_from_pdf(tmp_path, page_limit=page_limit)
    os.unlink(tmp_path)  # clean up

    # 4) summarize
    # if text is long, wrap it in a list for summarize_sources
    summary = summarize_sources([text])
    return summary


if __name__ == "__main__":
    # Initialize and run the server
    print("Starting FastMCP server...")
    try:
        mcp.run(transport='stdio')
        web_search = mcp.tools.web_search
        extract_urls_from_json = mcp.tools.extract_urls_from_json
        extract_text_from_pdf = mcp.tools.extract_text_from_pdf
        extract_text_from_url = mcp.tools.extract_text_from_url
        perform_ocr = mcp.tools.perform_ocr
        summarize_sources = mcp.tools.summarize_sources
        download_and_summarize_pdf = mcp.tools.download_and_summarize_pdf
    except Exception as e:
        print(f"Error starting server: {e}")
