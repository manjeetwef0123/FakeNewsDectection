import requests
from bs4 import BeautifulSoup
from model import clean_text

def extract_text_from_url(url):
    try:
        response = requests.get(url)
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        text_elements = soup.find_all(text=True)
        page_text = ' '.join(text.strip() for text in text_elements if text.strip())
        return clean_text(page_text)
    except Exception as e:
        print("An error occurred while extracting text:", e)
        return None

# Example URL for testing
# url = "https://www.geeksforgeeks.org/python-programming-language/?ref=ghm"

# # Extract text from the URL
# extracted_text = extract_text_from_url(url)





