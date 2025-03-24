import requests
import re
import os
import json
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
import warnings
import logging
from requests.packages.urllib3.exceptions import InsecureRequestWarning

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress InsecureRequestWarning for verify=False
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

class ConfluenceExtractor:
    """
    A simplified class to extract content from any Confluence URL
    without requiring deep knowledge of Confluence structures.
    """
    
    def __init__(self, username=None, api_token=None):
        """
        Initialize the ConfluenceExtractor with optional authentication details.
        
        Args:
            username (str, optional): Username for Basic Auth
            api_token (str, optional): API token or password for Basic Auth
        """
        self.session = requests.Session()
        
        # Set basic authentication if credentials are provided
        if username and api_token:
            self.session.auth = (username, api_token)
        
        # Set headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })

    def _extract_base_url(self, url):
        """
        Extract the base URL from a Confluence page URL.
        
        Args:
            url (str): The Confluence page URL
            
        Returns:
            str: The base URL
        """
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        # Check if it's a standard wiki path
        if parsed_url.path.startswith('/wiki'):
            base_url += '/wiki'
            
        return base_url

    def _extract_page_id(self, url):
        """
        Extract the page ID from any Confluence URL.
        
        Args:
            url (str): The Confluence page URL
            
        Returns:
            str: The page ID or None if not found
        """
        # Parse the URL
        parsed_url = urlparse(url)
        
        # Try to find the page ID in the URL path (cloud pattern)
        page_id_match = re.search(r'/pages/(\d+)', parsed_url.path)
        if page_id_match:
            return page_id_match.group(1)
        
        # Try to find the page ID in query parameters (server pattern)
        query_params = parse_qs(parsed_url.query)
        if 'pageId' in query_params:
            return query_params['pageId'][0]
        
        logger.warning(f"Could not extract page ID from URL: {url}")
        return None

    def _get_api_url(self, base_url):
        """
        Determine the API URL based on the base URL.
        
        Args:
            base_url (str): The base URL of the Confluence instance
            
        Returns:
            str: The API URL
        """
        return f"{base_url}/rest/api"

    def get_page_content(self, url, username=None, api_token=None):
        """
        Get all text content from a Confluence page using just the URL.
        Handles authentication automatically.
        
        Args:
            url (str): The Confluence page URL
            username (str, optional): Override the default username
            api_token (str, optional): Override the default API token
            
        Returns:
            dict: A dictionary with all the extracted content
        """
        # Override session auth if new credentials provided
        if username and api_token:
            self.session.auth = (username, api_token)
        
        # Extract base URL and page ID
        base_url = self._extract_base_url(url)
        page_id = self._extract_page_id(url)
        
        if not page_id:
            raise ValueError(f"Could not extract page ID from URL: {url}")
        
        # Get the API URL
        api_url = self._get_api_url(base_url)
        
        # Get page content
        endpoint = f"{api_url}/content/{page_id}?expand=body.storage,children.page"
        response = self.session.get(endpoint, verify=False)
        
        if response.status_code == 200:
            page_data = response.json()
            
            # Extract HTML content
            html_content = page_data['body']['storage']['value']
            
            # Parse content with BeautifulSoup
            content = self._extract_content(html_content)
            
            # Add page metadata
            content['title'] = page_data['title']
            content['url'] = url
            
            # Get child pages if available
            if 'children' in page_data and 'page' in page_data['children'] and 'results' in page_data['children']['page']:
                child_pages = page_data['children']['page']['results']
                content['child_pages'] = [
                    {'title': child['title'], 'id': child['id']} 
                    for child in child_pages
                ]
            
            return content
        else:
            logger.error(f"Failed to get page content. Status code: {response.status_code}")
            if response.status_code == 401:
                raise ValueError("Authentication failed. Please check your username and API token.")
            elif response.status_code == 404:
                raise ValueError(f"Page not found. Please check the URL: {url}")
            else:
                logger.error(f"Response: {response.text}")
                raise ValueError(f"Failed to get page content: {response.text}")

    def _extract_content(self, html_content):
        """
        Extract all useful content from HTML.
        
        Args:
            html_content (str): The HTML content
            
        Returns:
            dict: A dictionary with all extracted content
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract text
        text_content = self._extract_text(soup)
        
        # Extract tables
        tables = self._extract_tables(soup)
        
        # Extract list items
        lists = self._extract_lists(soup)
        
        # Extract code blocks
        code_blocks = self._extract_code_blocks(soup)
        
        return {
            'text': text_content,
            'tables': tables,
            'lists': lists,
            'code_blocks': code_blocks
        }

    def _extract_text(self, soup):
        """
        Extract clean text content.
        
        Args:
            soup (BeautifulSoup): The parsed HTML
            
        Returns:
            str: The cleaned text content
        """
        # Remove script and style elements
        for element in soup(["script", "style"]):
            element.decompose()
            
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text

    def _extract_tables(self, soup):
        """
        Extract tables from HTML.
        
        Args:
            soup (BeautifulSoup): The parsed HTML
            
        Returns:
            list: A list of tables as pandas DataFrames
        """
        tables = []
        
        for table in soup.find_all('table'):
            data = []
            headers = []
            
            # Extract headers
            headers_row = table.find('thead')
            if headers_row:
                headers = [header.get_text().strip() for header in headers_row.find_all('th')]
            
            # If no headers found in thead, try the first row
            if not headers:
                first_row = table.find('tr')
                if first_row:
                    headers = [header.get_text().strip() for header in first_row.find_all(['th', 'td'])]
            
            # Extract data rows
            for row in table.find_all('tr')[1:] if headers else table.find_all('tr'):
                row_data = [cell.get_text().strip() for cell in row.find_all(['td', 'th'])]
                if row_data:  # Skip empty rows
                    data.append(row_data)
            
            # Create DataFrame
            if data:
                if headers and len(headers) == len(data[0]):
                    df = pd.DataFrame(data, columns=headers)
                else:
                    df = pd.DataFrame(data)
                tables.append(df)
            
        return tables

    def _extract_lists(self, soup):
        """
        Extract list items.
        
        Args:
            soup (BeautifulSoup): The parsed HTML
            
        Returns:
            dict: A dictionary with ordered and unordered lists
        """
        lists = {
            'ordered': [],
            'unordered': []
        }
        
        # Extract ordered lists
        for ol in soup.find_all('ol'):
            items = [li.get_text().strip() for li in ol.find_all('li')]
            if items:
                lists['ordered'].append(items)
        
        # Extract unordered lists
        for ul in soup.find_all('ul'):
            items = [li.get_text().strip() for li in ul.find_all('li')]
            if items:
                lists['unordered'].append(items)
        
        return lists

    def _extract_code_blocks(self, soup):
        """
        Extract code blocks.
        
        Args:
            soup (BeautifulSoup): The parsed HTML
            
        Returns:
            list: A list of code blocks
        """
        code_blocks = []
        
        # Look for code blocks (different Confluence versions use different markup)
        for code in soup.find_all(['pre', 'code', 'div', 'span'], class_=lambda c: c and ('code' in c or 'codeBlock' in c)):
            code_text = code.get_text().strip()
            if code_text:
                code_blocks.append(code_text)
        
        return code_blocks

    def extract_from_url(self, url, username=None, api_token=None, include_child_pages=False, max_depth=1):
        """
        Extract all content from a URL and optionally its child pages.
        
        Args:
            url (str): The Confluence page URL
            username (str, optional): Username for authentication
            api_token (str, optional): API token for authentication
            include_child_pages (bool, optional): Whether to include child pages
            max_depth (int, optional): Maximum depth for child pages
            
        Returns:
            dict: A dictionary with all extracted content
        """
        # Get content from the main page
        content = self.get_page_content(url, username, api_token)
        
        # If including child pages and we haven't reached max depth
        if include_child_pages and max_depth > 0 and 'child_pages' in content:
            # Extract base URL
            base_url = self._extract_base_url(url)
            
            # Process each child page
            for i, child in enumerate(content['child_pages']):
                child_id = child['id']
                
                # Generate child URL based on whether it's cloud or server
                if 'atlassian.net' in url:
                    # Cloud instance
                    child_url = f"{base_url}/pages/{child_id}"
                else:
                    # Server instance
                    child_url = f"{base_url}/pages/viewpage.action?pageId={child_id}"
                
                # Get child page content
                try:
                    logger.info(f"Processing child page {i+1}/{len(content['child_pages'])}: {child['title']}")
                    child_content = self.extract_from_url(
                        child_url, 
                        username, 
                        api_token,
                        include_child_pages=True, 
                        max_depth=max_depth-1
                    )
                    content['child_pages'][i]['content'] = child_content
                except Exception as e:
                    logger.error(f"Error processing child page {child['title']}: {e}")
                    content['child_pages'][i]['error'] = str(e)
        
        return content

    def extract_text_only(self, url, username=None, api_token=None, include_child_pages=False, max_depth=1):
        """
        Extract only the text content from a URL and optionally its child pages.
        This is a simplified method that returns just the text without tables, lists, etc.
        
        Args:
            url (str): The Confluence page URL
            username (str, optional): Username for authentication
            api_token (str, optional): API token for authentication
            include_child_pages (bool, optional): Whether to include child pages
            max_depth (int, optional): Maximum depth for child pages
            
        Returns:
            str: The extracted text content
        """
        # Get full content
        content = self.extract_from_url(url, username, api_token, include_child_pages, max_depth)
        
        # Start with the main page text
        all_text = f"# {content['title']}\n\n{content['text']}\n\n"
        
        # Add child page text if available
        if include_child_pages and 'child_pages' in content:
            for child in content['child_pages']:
                if 'content' in child:
                    all_text += f"\n\n## {child['title']}\n\n{child['content']['text']}\n\n"
        
        return all_text


def read_config_file(config_path):
    """
    Read the configuration file and return username and token.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        tuple: (username, token)
    """
    try:
        with open(config_path, 'r') as file:
            config_data = file.read().strip().split('\n')
        
        config = {}
        for line in config_data:
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
        
        username = config.get('username')
        token = config.get('token') or config.get('password')  # Accept either token or password
        
        return username, token
    except Exception as e:
        logger.error(f"Error reading config file: {e}")
        return None, None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract content from any Confluence URL')
    parser.add_argument('url', help='Confluence page URL')
    parser.add_argument('--output', '-o', help='Output file to save the content')
    parser.add_argument('--children', '-c', action='store_true', help='Include child pages')
    parser.add_argument('--depth', '-d', type=int, default=1, help='Maximum depth for child pages')
    parser.add_argument('--text-only', '-x', action='store_true', help='Extract only text content')
    parser.add_argument('--config', default='config.txt', help='Path to configuration file with credentials (default: config.txt)')
    
    args = parser.parse_args()
    
    try:
        # Read credentials from config file
        username, token = read_config_file(args.config)
        
        if not username or not token:
            print(f"Warning: Credentials not found in {args.config}")
            print("Create a config.txt file with the following content:")
            print("username=your_email@example.com")
            print("token=your_api_token_or_password")
            
            # Ask if user wants to continue without authentication
            response = input("Continue without authentication? (y/n): ")
            if response.lower() != 'y':
                print("Exiting.")
                return
        
        # Create extractor
        extractor = ConfluenceExtractor()
        
        # Get content based on options
        if args.text_only:
            content = extractor.extract_text_only(
                args.url, 
                username, 
                token, 
                args.children, 
                args.depth
            )
            print(content)
            
            # Save to file if specified
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Content saved to {args.output}")
        else:
            content = extractor.extract_from_url(
                args.url, 
                username, 
                token, 
                args.children, 
                args.depth
            )
            
            # Print basic info
            print(f"Title: {content['title']}")
            print(f"URL: {content['url']}")
            print(f"Text length: {len(content['text'])} characters")
            print(f"Tables: {len(content['tables'])}")
            
            if 'child_pages' in content:
                print(f"Child pages: {len(content['child_pages'])}")
            
            # Save to file if specified
            if args.output:
                import json
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(content, f, indent=2)
                print(f"Content saved to {args.output}")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
