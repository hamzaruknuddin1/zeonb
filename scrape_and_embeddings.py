import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import openai
import numpy as np
import faiss
import tiktoken
import time
import logging
import os
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

def scrape_website_advanced(url, keywords=None, use_selenium=False):
    """
    Scrape a website and extract relevant sections based on keywords.
    
    Args:
        url (str): Website URL to scrape
        keywords (list): List of keywords to look for in content
        use_selenium (bool): Whether to use Selenium for JavaScript-rendered content
        
    Returns:
        list: List of relevant text sections from the website
    """
    if keywords is None:
        keywords = [
            "about", "services", "pricing", "packages", "plans", "rates", "offers",
            "features", "products", "solutions", "contact", "team", "testimonials",
            "portfolio", "projects", "clients", "partners", "faq", "support", "help",
            "terms", "policy", "privacy", "refund", "development", "design", 
            "marketing", "optimization", "branding", "consulting", "training",
            "subscription", "case study", "demo", "careers", "jobs", "pricing table",
            "discount", "special offer", "overview", "our story", "mission",
            "vision", "values", "resources", "tools", "blog", "updates", "news",
            "events", "newsletter", "tutorials", "how it works", "integration"
        ]

    text_data = []
    
    try:
        if use_selenium:
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
            
            try:
                logging.info(f"Using Selenium to scrape: {url}")
                driver.get(url)
                time.sleep(5)  # Increased wait time for JavaScript content
                page_source = driver.page_source
                soup = BeautifulSoup(page_source, "html.parser")
            finally:
                driver.quit()
        else:
            logging.info(f"Using Requests to scrape: {url}")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            soup = BeautifulSoup(response.content, "html.parser")

        # Extract text from various HTML elements
        relevant_sections = []
        elements = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "div", "span", "article"])
        
        for element in elements:
            # Skip hidden elements or those with specific classes
            if element.parent.get('style') and 'display: none' in element.parent.get('style').lower():
                continue
                
            text = element.get_text(strip=True)
            if text and len(text) > 20:  # Filter out very short texts
                # Check if text contains any keywords or is part of main content
                if any(keyword.lower() in text.lower() for keyword in keywords):
                    cleaned_text = ' '.join(text.split())  # Normalize whitespace
                    if cleaned_text not in relevant_sections:  # Avoid duplicates
                        relevant_sections.append(cleaned_text)

        if not relevant_sections:
            logging.warning(f"No relevant content found on {url}")
            return []

        text_data.extend(relevant_sections)
        logging.info(f"Successfully scraped {len(text_data)} sections from {url}")
        return text_data

    except Exception as e:
        logging.error(f"Error scraping {url}: {str(e)}")
        return []

def create_chunks_with_token_limit(text_data, model="text-embedding-3-large", max_tokens=800):
    """
    Create chunks of text while ensuring they stay within the token limit.
    
    Args:
        text_data (list): List of text sections to chunk
        model (str): Model name for tokenization
        max_tokens (int): Maximum tokens per chunk
        
    Returns:
        list: List of text chunks within token limit
    """
    if not text_data:
        logging.error("No text data provided for chunking")
        return []

    try:
        encoding = tiktoken.encoding_for_model(model)
        chunks = []
        current_chunk = ""
        current_tokens = 0

        for text in text_data:
            if not isinstance(text, str):
                logging.warning(f"Skipping non-string text: {type(text)}")
                continue
            
            if not text.strip():
                continue

            # Encode the text and split into smaller pieces if necessary
            tokens = encoding.encode(text)
            
            if len(tokens) > max_tokens:
                # Split long texts into smaller chunks
                for i in range(0, len(tokens), max_tokens):
                    chunk_tokens = tokens[i:i + max_tokens]
                    chunk_text = encoding.decode(chunk_tokens).strip()
                    if chunk_text:
                        chunks.append(chunk_text)
            else:
                chunks.append(text)

        logging.info(f"Created {len(chunks)} chunks")
        return chunks

    except Exception as e:
        logging.error(f"Error during chunking: {str(e)}")
        return []

def create_embeddings(chunks, model="text-embedding-3-large"):
    """
    Generate embeddings for text chunks using OpenAI API.
    
    Args:
        chunks (list): List of text chunks
        model (str): OpenAI embedding model name
        
    Returns:
        numpy.ndarray: Array of embeddings
    """
    if not chunks:
        logging.error("No chunks provided for embedding creation")
        raise ValueError("No chunks provided for embedding creation")

    embeddings = []
    retry_count = 3
    
    for i, chunk in enumerate(chunks):
        for attempt in range(retry_count):
            try:
                if not isinstance(chunk, str) or not chunk.strip():
                    logging.warning(f"Skipping invalid chunk at index {i}")
                    continue

                logging.info(f"Creating embedding for chunk {i+1}/{len(chunks)}")
                response = openai.Embedding.create(
                    input=chunk,
                    model=model
                )
                embeddings.append(response["data"][0]["embedding"])
                break  # Break the retry loop if successful
                
            except Exception as e:
                if attempt == retry_count - 1:  # Last attempt
                    logging.error(f"Failed to create embedding for chunk {i} after {retry_count} attempts: {str(e)}")
                else:
                    logging.warning(f"Attempt {attempt + 1} failed for chunk {i}, retrying...")
                    time.sleep(1)  # Wait before retrying

    if not embeddings:
        raise ValueError("No valid embeddings were created. Check chunks and API configuration.")
    
    logging.info(f"Successfully created {len(embeddings)} embeddings")
    return np.array(embeddings)

def store_embeddings_faiss(embeddings):
    """
    Store embeddings in a FAISS index.
    
    Args:
        embeddings (numpy.ndarray): Array of embeddings
        
    Returns:
        faiss.Index: FAISS index containing the embeddings
    """
    if embeddings.size == 0:
        raise ValueError("No embeddings available to store in FAISS index.")
        
    try:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        logging.info(f"Successfully created FAISS index with {index.ntotal} vectors")
        return index
    except Exception as e:
        logging.error(f"Error creating FAISS index: {str(e)}")
        raise

def retrieve_context(query, index, chunks, model="text-embedding-3-large", top_k=3):
    """
    Retrieve the most relevant context for a given query.
    
    Args:
        query (str): User's question or query
        index (faiss.Index): FAISS index containing embeddings
        chunks (list): List of text chunks corresponding to embeddings
        model (str): OpenAI embedding model name
        top_k (int): Number of relevant chunks to retrieve
        
    Returns:
        str: Concatenated relevant context
    """
    try:
        query_embedding = openai.Embedding.create(
            input=query,
            model=model
        )["data"][0]["embedding"]

        distances, indices = index.search(np.array([query_embedding]), top_k)
        relevant_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
        
        if not relevant_chunks:
            logging.warning("No relevant context found for query")
            return ""
            
        return " ".join(relevant_chunks)
        
    except Exception as e:
        logging.error(f"Error retrieving context: {str(e)}")
        return ""

# Optional: Test function
def test_scraping_and_embeddings(url):
    """
    Test the entire pipeline for a given URL.
    
    Args:
        url (str): Website URL to test
    """
    try:
        # Test scraping
        content = scrape_website_advanced(url, use_selenium=True)
        if not content:
            logging.error("Scraping test failed: No content retrieved")
            return
        
        # Test chunking
        chunks = create_chunks_with_token_limit(content)
        if not chunks:
            logging.error("Chunking test failed: No chunks created")
            return
        
        # Test embeddings
        embeddings = create_embeddings(chunks)
        if embeddings is None:
            logging.error("Embedding test failed: No embeddings created")
            return
        
        # Test FAISS index
        index = store_embeddings_faiss(embeddings)
        if index is None:
            logging.error("FAISS index test failed: Index not created")
            return
        
        # Test retrieval
        test_query = "What services do you offer?"
        context = retrieve_context(test_query, index, chunks)
        if not context:
            logging.error("Context retrieval test failed: No context retrieved")
            return
            
        logging.info("All tests completed successfully!")
        
    except Exception as e:
        logging.error(f"Test failed with error: {str(e)}")