import requests
import re
import os
import json
import base64
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse, unquote
import warnings
import logging
import time
from requests.packages.urllib3.exceptions import InsecureRequestWarning

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress InsecureRequestWarning for verify=False
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

class ConfluenceExtractor:
    """
    A user-friendly class to extract content from any Confluence URL
    without requiring knowledge of Confluence structures.
    """
    
    def __init__(self):
        """Initialize the ConfluenceExtractor."""
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
    def _load_credentials(self, config_path='config.txt'):
        """
        Load credentials from config file.
        
        Args:
            config_path (str): Path to configuration file
            
        Returns:
            tuple: (username, password)
        """
        try:
            if not os.path.exists(config_path):
                logger.warning(f"Config file not found: {config_path}")
                return None, None
                
            with open(config_path, 'r') as file:
                config_data = file.read().strip().split('\n')
            
            config = {}
            for line in config_data:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
            
            username = config.get('username')
            token = config.get('token') or config.get('password')
            
            return username, token
        except Exception as e:
            logger.error(f"Error reading config file: {e}")
            return None, None
    
    def _setup_auth(self, config_path='config.txt'):
        """
        Set up authentication from config file.
        
        Args:
            config_path (str): Path to configuration file
        """
        username, token = self._load_credentials(config_path)
        if username and token:
            self.session.auth = (username, token)
            logger.info(f"Authentication set up for user: {username}")
        else:
            logger.warning("No authentication credentials found or loaded")
    
    def extract_content(self, url, config_path='config.txt', output_dir=None, 
                       include_children=False, max_depth=1, extract_images=True, 
                       save_tables=True, save_code=True):
        """
        Extract all content from a Confluence page and save it locally.
        
        Args:
            url (str): The Confluence page URL
            config_path (str): Path to configuration file with credentials
            output_dir (str): Directory to save content (if None, creates based on page title)
            include_children (bool): Whether to extract child pages
            max_depth (int): Maximum depth for child pages
            extract_images (bool): Whether to extract and save images
            save_tables (bool): Whether to save tables as CSV files
            save_code (bool): Whether to save code blocks as separate files
            
        Returns:
            dict: Content extraction results
        """
        # Set up authentication
        self._setup_auth(config_path)
        
        # Extract content
        content = self._get_page_content(url)
        
        if not content:
            logger.error(f"Failed to extract content from {url}")
            return None
        
        # Create output directory if not specified
        if not output_dir:
            # Extract space key and title from URL for folder name
            parsed_url = urlparse(url)
            path_parts = parsed_url.path.split('/')
            
            # Look for 'display' in the URL path
            if 'display' in path_parts:
                display_index = path_parts.index('display')
                if display_index + 2 < len(path_parts):
                    space_key = path_parts[display_index + 1]
                    page_title = unquote(path_parts[display_index + 2].replace('+', ' '))
                    output_dir = f"{space_key}_{page_title}"
                else:
                    output_dir = "confluence_content"
            else:
                output_dir = "confluence_content"
            
            # Make sure the directory name is valid
            output_dir = "".join(c for c in output_dir if c.isalnum() or c in "_ -").rstrip()
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving content to directory: {output_dir}")
        
        # Save content
        self._save_content(content, output_dir, extract_images, save_tables, save_code)
        
        # Extract child pages if requested
        if include_children and 'child_pages' in content and content['child_pages'] and max_depth > 0:
            child_results = []
            for child in content['child_pages']:
                child_url = child.get('url')
                if not child_url and 'id' in child:
                    # Construct URL if not available
                    base_url = self._extract_base_url(url)
                    if 'atlassian.net' in url:
                        child_url = f"{base_url}/pages/{child['id']}"
                    else:
                        child_url = f"{base_url}/pages/viewpage.action?pageId={child['id']}"
                
                if child_url:
                    logger.info(f"Processing child page: {child['title']}")
                    child_dir = os.path.join(output_dir, "children", self._sanitize_filename(child['title']))
                    child_result = self.extract_content(
                        child_url, 
                        config_path=config_path,
                        output_dir=child_dir,
                        include_children=include_children,
                        max_depth=max_depth-1,
                        extract_images=extract_images,
                        save_tables=save_tables,
                        save_code=save_code
                    )
                    if child_result:
                        child_results.append(child_result)
            
            content['extracted_children'] = child_results
        
        return {
            'title': content.get('title', 'Unknown'),
            'url': url,
            'output_directory': output_dir,
            'has_children': 'child_pages' in content and bool(content['child_pages']),
            'children_extracted': include_children and max_depth > 0
        }
    
    def _get_page_content(self, url):
        """
        Get content from a Confluence page using multiple methods.
        
        Args:
            url (str): The Confluence page URL
            
        Returns:
            dict: Page content or None if failed
        """
        logger.info(f"Extracting content from: {url}")
        
        # Try different methods to get the content
        content = self._try_api_by_url_pattern(url)
        
        if not content:
            content = self._try_direct_html(url)
        
        return content
    
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
        
        # Check if it includes the context path
        path_parts = parsed_url.path.split('/')
        if len(path_parts) > 1 and path_parts[1] in ['wiki', 'confluence', 'ETCB']:
            base_url += f"/{path_parts[1]}"
            # Check if there's another context path
            if len(path_parts) > 2 and path_parts[2] == 'confluence':
                base_url += "/confluence"
            
        return base_url
    
    def _try_api_by_url_pattern(self, url):
        """
        Try to extract content by recognizing the URL pattern and using the appropriate API.
        
        Args:
            url (str): The Confluence page URL
            
        Returns:
            dict: Content or None if failed
        """
        try:
            base_url = self._extract_base_url(url)
            api_url = f"{base_url}/rest/api"
            
            # Handle display pattern URLs
            parsed_url = urlparse(url)
            path_parts = parsed_url.path.split('/')
            
            # Look for 'display' in the URL path
            if 'display' in path_parts:
                display_index = path_parts.index('display')
                if display_index + 2 < len(path_parts):
                    space_key = path_parts[display_index + 1]
                    page_title = unquote(path_parts[display_index + 2].replace('+', ' '))
                    
                    logger.info(f"Detected display URL pattern. Space: {space_key}, Title: {page_title}")
                    
                    # Try to get content by space key and title
                    endpoint = f"{api_url}/content"
                    params = {
                        'spaceKey': space_key,
                        'title': page_title,
                        'expand': 'body.storage,children.page'
                    }
                    
                    response = self.session.get(endpoint, params=params, verify=False)
                    
                    if response.status_code == 200:
                        results = response.json()
                        if results.get('results') and len(results['results']) > 0:
                            page_data = results['results'][0]
                            
                            # Extract HTML content
                            if 'body' in page_data and 'storage' in page_data['body']:
                                html_content = page_data['body']['storage']['value']
                                
                                # Parse content
                                content = self._parse_html_content(html_content)
                                
                                # Add metadata
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
            
            return None
        except Exception as e:
            logger.error(f"Error using API to extract content: {e}")
            return None
    
    def _try_direct_html(self, url):
        """
        Try to extract content by directly requesting the page HTML.
        
        Args:
            url (str): The Confluence page URL
            
        Returns:
            dict: Content or None if failed
        """
        try:
            logger.info(f"Trying direct HTML extraction from: {url}")
            
            # Make direct request to the page
            response = self.session.get(url, verify=False)
            
            if response.status_code == 200:
                # Parse the HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Try to find the title
                title_elem = soup.find('title')
                title = title_elem.text.strip() if title_elem else "Unknown Title"
                
                # Try to find the main content
                content_div = None
                
                # Try different selectors for the main content area
                for selector in [
                    'div#main-content', 
                    'div#content', 
                    'div.wiki-content',
                    'div.confluenceContent', 
                    'div.content-container'
                ]:
                    content_div = soup.select_one(selector)
                    if content_div:
                        break
                
                if not content_div:
                    # Last resort: try to find any large div that might be the content
                    largest_div = None
                    max_size = 0
                    
                    for div in soup.find_all('div'):
                        div_content = div.get_text()
                        if len(div_content) > max_size:
                            max_size = len(div_content)
                            largest_div = div
                    
                    content_div = largest_div
                
                if content_div:
                    # Parse the content
                    content = self._parse_html_content(str(content_div))
                    
                    # Add metadata
                    content['title'] = title
                    content['url'] = url
                    
                    # Try to find child pages
                    child_links = []
                    
                    # Look for navigation elements that might contain child pages
                    for nav in soup.select('.children, .child-pages, .pagetree'):
                        for link in nav.find_all('a'):
                            href = link.get('href')
                            if href and not href.startswith(('#', 'javascript:')):
                                # Make relative URLs absolute
                                if not href.startswith(('http://', 'https://')):
                                    base_url = self._extract_base_url(url)
                                    if href.startswith('/'):
                                        href = f"{base_url}{href}"
                                    else:
                                        href = f"{base_url}/{href}"
                                
                                child_links.append({
                                    'title': link.text.strip(),
                                    'url': href
                                })
                    
                    if child_links:
                        content['child_pages'] = child_links
                    
                    return content
            
            logger.warning(f"Failed to extract content directly. Status code: {response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Error extracting content directly: {e}")
            return None
    
    def _parse_html_content(self, html_content):
        """
        Parse HTML content to extract text, tables, code blocks, and images.
        
        Args:
            html_content (str): HTML content
            
        Returns:
            dict: Parsed content
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract text
        text_content = self._extract_text(soup)
        
        # Extract tables
        tables = self._extract_tables(soup)
        
        # Extract images 
        images = self._extract_images(soup)
        
        # Extract code blocks
        code_blocks = self._extract_code_blocks(soup)
        
        # Extract list items
        lists = self._extract_lists(soup)
        
        return {
            'text': text_content,
            'tables': tables,
            'images': images,
            'code_blocks': code_blocks,
            'lists': lists,
            'html': html_content  # Keep the original HTML for reference
        }
    
    def _extract_text(self, soup):
        """
        Extract clean text content with preserved structure.
        
        Args:
            soup (BeautifulSoup): Parsed HTML
            
        Returns:
            str: Clean text with preserved structure
        """
        # Clone the soup to avoid modifying the original
        soup_clone = BeautifulSoup(str(soup), 'html.parser')
        
        # Remove script, style, and other non-content elements
        for element in soup_clone(['script', 'style', 'noscript', 'svg', 'iframe', 'form']):
            element.decompose()
        
        # Process headers to maintain hierarchy
        structured_text = []
        
        # Extract title if available
        title_elem = soup_clone.find(['h1', 'title'], class_=lambda x: x and ('title' in x or 'pageTitle' in x))
        if title_elem:
            structured_text.append(f"# {title_elem.get_text().strip()}\n")
        
        # Track tables and images for reference
        table_count = 0
        image_count = 0
        
        # Process the content with structure preservation
        for element in soup_clone.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'pre', 'ul', 'ol', 'table', 'blockquote', 'img']):
            # Skip if the element is inside another container we're already processing
            if element.parent.name in ['div', 'blockquote'] and element.parent in soup_clone.find_all(['div', 'blockquote']):
                continue
                
            # Skip empty elements and navigation elements
            if not element.get_text().strip() and element.name != 'img':
                continue
                
            if any(cls in element.get('class', []) for cls in ['pagetree', 'navitem', 'breadcrumb']):
                continue
                
            # Process based on element type
            if element.name.startswith('h') and len(element.name) == 2:
                level = int(element.name[1])
                prefix = '#' * level
                structured_text.append(f"\n{prefix} {element.get_text().strip()}\n")
            
            elif element.name == 'p':
                text = element.get_text().strip()
                if text:
                    structured_text.append(f"{text}\n")
            
            elif element.name == 'div' and element.get('class') and any(cls in element.get('class') for cls in ['code', 'codeBlock']):
                code = element.get_text().strip()
                if code:
                    structured_text.append(f"\n```\n{code}\n```\n")
            
            elif element.name == 'pre':
                code = element.get_text().strip()
                if code:
                    structured_text.append(f"\n```\n{code}\n```\n")
            
            elif element.name == 'ul':
                items = []
                for li in element.find_all('li', recursive=False):
                    items.append(f"- {li.get_text().strip()}")
                if items:
                    structured_text.append("\n" + "\n".join(items) + "\n")
            
            elif element.name == 'ol':
                items = []
                for i, li in enumerate(element.find_all('li', recursive=False), 1):
                    items.append(f"{i}. {li.get_text().strip()}")
                if items:
                    structured_text.append("\n" + "\n".join(items) + "\n")
            
            elif element.name == 'blockquote':
                quote = element.get_text().strip().replace('\n', '\n> ')
                if quote:
                    structured_text.append(f"\n> {quote}\n")
            
            elif element.name == 'table':
                table_count += 1
                structured_text.append(f"\n[TABLE {table_count}: See tables/table_{table_count}.csv or tables/table_{table_count}.html]\n")
            
            elif element.name == 'img':
                # Skip small icons and emoticons
                src = element.get('src', '')
                if src and not any(skip in src.lower() for skip in ['emoticon', 'icon', 'spacer', 'blank.gif']):
                    image_count += 1
                    alt_text = element.get('alt', '').strip() or element.get('title', '').strip() or f'image_{image_count}'
                    structured_text.append(f"\n[IMAGE: {alt_text} - See images/image_{image_count}.png]\n")
            
            elif element.name == 'div':
                # For general divs, only add their content if they have substance and aren't just containers
                text = element.get_text().strip()
                if text and len(text) > 10 and not any(cls in element.get('class', []) for cls in ['panel', 'container', 'wrapper']):
                    structured_text.append(f"{text}\n")
        
        # Join all parts and clean up
        text = "\n".join(structured_text)
        
        # Clean up the text - remove excessive newlines, etc.
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Replace common Unicode issues
        text = text.replace('\u00a0', ' ')  # Replace non-breaking spaces
        text = text.replace('\u2019', "'")  # Replace smart quotes
        text = text.replace('\u201c', '"')  # Replace smart quotes
        text = text.replace('\u201d', '"')  # Replace smart quotes
        text = text.replace('\u2013', '-')  # Replace en dash
        text = text.replace('\u2014', '--')  # Replace em dash
        
        return text
    
    def _extract_tables(self, soup):
        """
        Extract tables from HTML with improved accuracy.
        
        Args:
            soup (BeautifulSoup): Parsed HTML
            
        Returns:
            list: List of extracted tables as pandas DataFrames with accurate content
        """
        tables = []
        
        for table in soup.find_all('table'):
            # Skip navigation and layout tables
            if any(cls in table.get('class', []) for cls in ['pagetree', 'navtable', 'layout']):
                continue
            
            # First, check if this is a confluence table with CSS classes
            is_confluence_table = any(cls in table.get('class', []) for cls in ['confluenceTable', 'data-table', 'grid'])
            
            # Prepare data structure
            all_data = []
            headers = []
            
            # Get headers from thead if available
            thead = table.find('thead')
            if thead:
                header_row = thead.find('tr')
                if header_row:
                    headers = []
                    for cell in header_row.find_all(['th', 'td']):
                        # Get all text inside the cell, preserve structure
                        cell_text = cell.get_text().strip()
                        # Replace newlines and multiple spaces
                        cell_text = re.sub(r'\s+', ' ', cell_text)
                        headers.append(cell_text)
            
            # If no headers in thead, check if the first row has th elements
            if not headers:
                first_row = table.find('tr')
                if first_row:
                    potential_headers = []
                    all_header_cells = True
                    
                    for cell in first_row.find_all(['th', 'td']):
                        cell_text = cell.get_text().strip()
                        cell_text = re.sub(r'\s+', ' ', cell_text)
                        potential_headers.append(cell_text)
                        
                        # Check if this is actually a header cell
                        if cell.name != 'th' and not any(cls in cell.get('class', []) for cls in ['header', 'confluenceTh']):
                            all_header_cells = False
                    
                    # If all cells are header-like or it's a confluence table, use them as headers
                    if all_header_cells or (is_confluence_table and potential_headers):
                        headers = potential_headers
            
            # Get table body rows
            if table.find('tbody'):
                rows = table.find('tbody').find_all('tr')
            else:
                rows = table.find_all('tr')
                # Skip first row if we're using it as headers
                if headers and rows and not thead:
                    rows = rows[1:]
            
            # Process each row
            for row in rows:
                row_data = []
                for cell in row.find_all(['td', 'th']):
                    # Handle colspan and rowspan
                    colspan = int(cell.get('colspan', 1))
                    
                    # Get cell text, preserve links
                    links = []
                    for a in cell.find_all('a'):
                        href = a.get('href', '')
                        if href and not href.startswith('#'):
                            if a.get_text().strip():
                                links.append(f"{a.get_text().strip()} [{href}]")
                    
                    # Get just the text
                    cell_text = cell.get_text().strip()
                    cell_text = re.sub(r'\s+', ' ', cell_text)
                    
                    # Add links if available and not already in the text
                    if links and not any(link in cell_text for link in links):
                        cell_text += " " + " ".join(links)
                    
                    # Add cell to row data, respecting colspan
                    for _ in range(colspan):
                        row_data.append(cell_text)
                
                if row_data:  # Skip empty rows
                    all_data.append(row_data)
            
            # Create DataFrame
            if all_data:
                # Ensure data is consistent
                max_cols = max(len(row) for row in all_data)
                
                # Ensure all rows have the same number of columns
                for row in all_data:
                    while len(row) < max_cols:
                        row.append("")
                
                # If headers are available and match column count
                if headers and len(headers) == max_cols:
                    df = pd.DataFrame(all_data, columns=headers)
                elif headers:
                    # Headers don't match column count, extend or truncate
                    while len(headers) < max_cols:
                        headers.append(f"Column {len(headers) + 1}")
                    headers = headers[:max_cols]
                    df = pd.DataFrame(all_data, columns=headers)
                else:
                    # No headers, use default column names
                    df = pd.DataFrame(all_data, columns=[f"Column {i+1}" for i in range(max_cols)])
                
                tables.append(df)
        
        return tables
    
    def _extract_images(self, soup):
        """
        Extract images from HTML with improved image type detection.
        
        Args:
            soup (BeautifulSoup): Parsed HTML
            
        Returns:
            list: List of image information
        """
        images = []
        
        for img in soup.find_all('img'):
            src = img.get('src', '')
            
            # Skip emoticons, icons, and spacers
            if any(skip in src.lower() for skip in ['emoticon', 'icon', 'spacer', 'blank.gif']):
                continue
                
            alt = img.get('alt', '').strip()
            title = img.get('title', '').strip() or alt
            
            if src:
                image_info = {
                    'src': src,
                    'alt': alt,
                    'title': title
                }
                
                if src.startswith('data:image'):
                    # Handle data URLs
                    image_info['type'] = 'data-url'
                    
                    # Try to extract format from data URL
                    data_url_pattern = re.compile(r'data:image/(?P<format>.*?)[;,]')
                    match = data_url_pattern.match(src)
                    if match:
                        image_info['format'] = match.group('format')
                else:
                    # Handle regular URLs
                    image_info['type'] = 'url'
                    
                    # Extract format from URL
                    if '.' in src.split('/')[-1]:
                        ext = src.split('.')[-1].lower().split('?')[0]
                        if ext in ['png', 'jpg', 'jpeg', 'gif', 'svg', 'webp', 'bmp', 'tiff']:
                            image_info['format'] = ext
                    
                    # Make relative URLs absolute
                    if src.startswith('/'):
                        image_info['type'] = 'relative'
                
                # Check for specific confluence image classes or patterns
                parent = img.parent
                if parent and parent.name == 'div' and parent.get('class'):
                    # Check for Confluence image container
                    if any(cls in parent.get('class') for cls in ['confluence-embedded-image', 'image-container']):
                        # See if there's a high-res version available
                        if parent.get('data-image-src'):
                            image_info['src'] = parent.get('data-image-src')
                            image_info['type'] = 'url'
                            image_info['high_res'] = True
                            
                # Get image dimensions if available
                width = img.get('width')
                height = img.get('height')
                if width and height:
                    image_info['width'] = width
                    image_info['height'] = height
                
                images.append(image_info)
        
        return images
    
    def _extract_code_blocks(self, soup):
        """
        Extract code blocks from HTML.
        
        Args:
            soup (BeautifulSoup): Parsed HTML
            
        Returns:
            list: List of code blocks with language information
        """
        code_blocks = []
        
        # Look for different types of code blocks
        for code in soup.select('pre, code, div.code, div.codeBlock, div.codeContent, span.code'):
            code_text = code.get_text().strip()
            if not code_text:
                continue
                
            # Try to determine the language
            language = None
            
            # Check for language classes
            for cls in code.get('class', []):
                if cls.startswith(('language-', 'brush:', 'lang-')):
                    language = cls.split('-', 1)[1] if '-' in cls else cls.split(':', 1)[1] if ':' in cls else None
                    break
            
            # Look for parent div that might have language info
            if not language and code.parent:
                for cls in code.parent.get('class', []):
                    if cls.startswith(('language-', 'brush:', 'lang-')):
                        language = cls.split('-', 1)[1] if '-' in cls else cls.split(':', 1)[1] if ':' in cls else None
                        break
            
            code_blocks.append({
                'code': code_text,
                'language': language
            })
        
        return code_blocks
    
    def _extract_lists(self, soup):
        """
        Extract lists from HTML.
        
        Args:
            soup (BeautifulSoup): Parsed HTML
            
        Returns:
            dict: Dictionary of ordered and unordered lists
        """
        result = {
            'ordered': [],
            'unordered': []
        }
        
        # Extract ordered lists
        ordered_lists = soup.find_all('ol')
        for ordered_list in ordered_lists:
            list_items = ordered_list.find_all('li', recursive=False)
            items = [item.get_text().strip() for item in list_items]
            if items:
                result['ordered'].append(items)
        
        # Extract unordered lists
        unordered_lists = soup.find_all('ul')
        for unordered_list in unordered_lists:
            list_items = unordered_list.find_all('li', recursive=False)
            items = [item.get_text().strip() for item in list_items]
            if items:
                result['unordered'].append(items)
        
        return result
    
    def _save_content(self, content, output_dir, extract_images=True, save_tables=True, save_code=True):
        """
        Save extracted content to disk with improved formatting and structure.
        
        Args:
            content (dict): Extracted content
            output_dir (str): Output directory
            extract_images (bool): Whether to extract and save images
            save_tables (bool): Whether to save tables as CSV files
            save_code (bool): Whether to save code blocks as separate files
        """
        # Make sure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save text content to a file - use both plain text and markdown
        if 'text' in content:
            with open(os.path.join(output_dir, 'content.txt'), 'w', encoding='utf-8') as f:
                f.write(content['text'])
                
            with open(os.path.join(output_dir, 'content.md'), 'w', encoding='utf-8') as f:
                f.write(content['text'])
            
            logger.info(f"Saved text content to {os.path.join(output_dir, 'content.txt')} and {os.path.join(output_dir, 'content.md')}")
        
        # Save original HTML
        if 'html' in content:
            with open(os.path.join(output_dir, 'original.html'), 'w', encoding='utf-8') as f:
                f.write(content['html'])
            
            logger.info(f"Saved original HTML to {os.path.join(output_dir, 'original.html')}")
        
        # Save tables
        if save_tables and 'tables' in content and content['tables']:
            tables_dir = os.path.join(output_dir, 'tables')
            os.makedirs(tables_dir, exist_ok=True)
            
            for i, table in enumerate(content['tables']):
                table_file = os.path.join(tables_dir, f'table_{i+1}.csv')
                
                try:
                    # Save to CSV with proper encoding
                    table.to_csv(table_file, index=False, encoding='utf-8-sig')
                    
                    # Also save as HTML and Excel for better formatting
                    html_file = os.path.join(tables_dir, f'table_{i+1}.html')
                    excel_file = os.path.join(tables_dir, f'table_{i+1}.xlsx')
                    
                    # Save HTML version
                    with open(html_file, 'w', encoding='utf-8') as f:
                        f.write(table.to_html(index=False, na_rep='', border=1))
                    
                    # Save Excel version if pandas has ExcelWriter
                    try:
                        table.to_excel(excel_file, index=False, engine='openpyxl')
                    except Exception as e:
                        logger.warning(f"Could not save Excel version of table: {e}")
                        
                except Exception as e:
                    logger.error(f"Error saving table {i+1}: {e}")
            
            logger.info(f"Saved {len(content['tables'])} tables to {tables_dir}")
        
        # Save code blocks
        if save_code and 'code_blocks' in content and content['code_blocks']:
            code_dir = os.path.join(output_dir, 'code')
            os.makedirs(code_dir, exist_ok=True)
            
            for i, code_block in enumerate(content['code_blocks']):
                language = code_block.get('language', 'txt')
                extension = self._get_extension_for_language(language)
                
                code_file = os.path.join(code_dir, f'code_{i+1}{extension}')
                with open(code_file, 'w', encoding='utf-8') as f:
                    f.write(code_block['code'])
            
            logger.info(f"Saved {len(content['code_blocks'])} code blocks to {code_dir}")
        
        # Save images
        if extract_images and 'images' in content and content['images']:
            images_dir = os.path.join(output_dir, 'images')
            os.makedirs(images_dir, exist_ok=True)
            
            # Get the base URL for resolving relative image URLs
            base_url = None
            if 'url' in content:
                base_url = self._extract_base_url(content['url'])
            
            for i, image in enumerate(content['images']):
                try:
                    # Create a more descriptive filename
                    image_name = image.get('title', '') or image.get('alt', '') or f'image_{i+1}'
                    image_name = self._sanitize_filename(image_name)
                    if not image_name or len(image_name) < 3:
                        image_name = f'image_{i+1}'
                    
                    # Default extension and file path
                    default_ext = '.png'  # Default to PNG instead of JPG
                    image_file = os.path.join(images_dir, f"{image_name}{default_ext}")
                    
                    if image.get('type') == 'data-url':
                        # Handle data URLs
                        data_url = image['src']
                        data_url_pattern = re.compile(r'data:image/(?P<format>.*?);base64,(?P<data>.*)')
                        match = data_url_pattern.match(data_url)
                        
                        if match:
                            image_format = match.group('format')
                            image_data = match.group('data')
                            
                            # Update the file extension
                            image_file = os.path.join(images_dir, f"{image_name}.{image_format}")
                            
                            # Save the image
                            with open(image_file, 'wb') as f:
                                f.write(base64.b64decode(image_data))
                    else:
                        # Handle URL images
                        image_url = image['src']
                        
                        # Resolve relative URLs
                        if image.get('type') == 'relative' and base_url:
                            if image_url.startswith('/'):
                                image_url = f"{base_url}{image_url}"
                            else:
                                image_url = f"{base_url}/{image_url}"
                        
                        # Handle Confluence specific URLs
                        if '/attachments/' in image_url and '?effects=border-simple' in image_url:
                            # Remove Confluence image effects to get original image
                            image_url = image_url.split('?effects=')[0]
                        
                        # Download the image
                        img_response = self.session.get(image_url, verify=False)
                        
                        if img_response.status_code == 200:
                            # Determine the file extension from Content-Type or URL
                            content_type = img_response.headers.get('Content-Type', '')
                            
                            if 'image/' in content_type:
                                ext = content_type.split('/')[-1].split(';')[0].lower()
                                if ext == 'jpeg':
                                    ext = 'jpg'
                                if ext in ['png', 'jpg', 'gif', 'svg', 'webp']:
                                    image_file = os.path.join(images_dir, f"{image_name}.{ext}")
                            elif 'format' in image:
                                image_file = os.path.join(images_dir, f"{image_name}.{image['format']}")
                            elif '.' in image_url.split('/')[-1]:
                                url_ext = image_url.split('.')[-1].lower().split('?')[0]
                                if url_ext in ['png', 'jpg', 'jpeg', 'gif', 'svg', 'webp']:
                                    if url_ext == 'jpeg':
                                        url_ext = 'jpg'
                                    image_file = os.path.join(images_dir, f"{image_name}.{url_ext}")
                            
                            # Save the image
                            with open(image_file, 'wb') as f:
                                f.write(img_response.content)
                except Exception as e:
                    logger.error(f"Error saving image {i+1}: {e}")
            
            logger.info(f"Saved images to {images_dir}")
        
        # Create a metadata file
        metadata = {
            'title': content.get('title', 'Unknown'),
            'url': content.get('url', ''),
            'extraction_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'content_stats': {
                'tables': len(content.get('tables', [])),
                'code_blocks': len(content.get('code_blocks', [])),
                'images': len(content.get('images', [])),
                'text_length': len(content.get('text', '')),
                'lists': {
                    'ordered': len(content.get('lists', {}).get('ordered', [])),
                    'unordered': len(content.get('lists', {}).get('unordered', []))
                }
            }
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)file = os.path.join(images_dir, f'image_{i+1}.{image_format}')
                            
                            # Save the image
                            with open(image_file, 'wb') as f:
                                f.write(base64.b64decode(image_data))
                    else:
                        # Handle URL images
                        image_url = image['src']
                        
                        # Resolve relative URLs
                        if image['type'] == 'relative' and base_url:
                            if image_url.startswith('/'):
                                image_url = f"{base_url}{image_url}"
                            else:
                                image_url = f"{base_url}/{image_url}"
                        
                        # Download the image
                        img_response = self.session.get(image_url, verify=False)
                        
                        if img_response.status_code == 200:
                            # Try to determine the file extension from the Content-Type
                            content_type = img_response.headers.get('Content-Type', '')
                            if 'image/' in content_type:
                                ext = content_type.split('/')[-1].split(';')[0]
                                image_file = os.path.join(images_dir, f'image_{i+1}.{ext}')
                            
                            # Save the image
                            with open(image_file, 'wb') as f:
                                f.write(img_response.content)
                except Exception as e:
                    logger.error(f"Error saving image {i+1}: {e}")
            
            logger.info(f"Saved images to {images_dir}")
        
        # Create a metadata file
        metadata = {
            'title': content.get('title', 'Unknown'),
            'url': content.get('url', ''),
            'extraction_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'content_stats': {
                'tables': len(content.get('tables', [])),
                'code_blocks': len(content.get('code_blocks', [])),
                'images': len(content.get('images', [])),
                'text_length': len(content.get('text', '')),
            }
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    def _get_extension_for_language(self, language):
        """
        Get the appropriate file extension for a programming language.
        
        Args:
            language (str): Programming language
            
        Returns:
            str: File extension
        """
        if not language:
            return '.txt'
            
        language = language.lower()
        
        extensions = {
            'python': '.py',
            'java': '.java',
            'javascript': '.js',
            'typescript': '.ts',
            'html': '.html',
            'css': '.css',
            'c': '.c',
            'cpp': '.cpp',
            'c++': '.cpp',
            'csharp': '.cs',
            'c#': '.cs',
            'ruby': '.rb',
            'php': '.php',
            'swift': '.swift',
            'kotlin': '.kt',
            'go': '.go',
            'rust': '.rs',
            'scala': '.scala',
            'sql': '.sql',
            'bash': '.sh',
            'shell': '.sh',
            'powershell': '.ps1',
            'xml': '.xml',
            'json': '.json',
            'yaml': '.yaml',
            'yml': '.yml',
            'markdown': '.md',
            'r': '.r'
        }
        
        return extensions.get(language, '.txt')
    
    def _sanitize_filename(self, filename):
        """
        Create a valid filename from a string.
        
        Args:
            filename (str): String to sanitize
            
        Returns:
            str: Sanitized filename
        """
        # Remove invalid characters
        sanitized = re.sub(r'[\\/*?:"<>|]', '', filename)
        # Replace spaces with underscores
        sanitized = sanitized.replace(' ', '_')
        # Ensure it's not just whitespace
        if not sanitized or sanitized.isspace():
            sanitized = "unnamed"
        return sanitized


def extract_confluence_content(url, output_dir=None, config_path='config.txt', 
                             include_children=False, max_depth=1, extract_images=True, 
                             save_tables=True, save_code=True):
    """
    Main function to extract content from a Confluence page.
    
    Args:
        url (str): Confluence page URL
        output_dir (str): Directory to save content
        config_path (str): Path to configuration file
        include_children (bool): Whether to extract child pages
        max_depth (int or str): Maximum depth for child pages (use "all" for all possible children)
        extract_images (bool): Whether to extract images
        save_tables (bool): Whether to save tables
        save_code (bool): Whether to save code blocks
        
    Returns:
        dict: Results of the extraction
    """
    extractor = ConfluenceExtractor()
    
    # Handle "all" max_depth by setting it to a very large number
    if max_depth == "all":
        max_depth = 999  # Effectively unlimited depth
    
    try:
        result = extractor.extract_content(
            url=url,
            config_path=config_path,
            output_dir=output_dir,
            include_children=include_children,
            max_depth=max_depth,
            extract_images=extract_images,
            save_tables=save_tables,
            save_code=save_code
        )
        
        
        if result:
            logger.info(f"Successfully extracted content from {url}")
            logger.info(f"Content saved to {result['output_directory']}")
            return result
        else:
            logger.error(f"Failed to extract content from {url}")
            return None
    except Exception as e:
        logger.error(f"Error extracting content: {e}")
        return None


def main():
    """
    Command-line interface for the Confluence content extractor.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract content from any Confluence page')
    parser.add_argument('url', help='URL of the Confluence page')
    parser.add_argument('--output', '-o', help='Output directory (default: auto-generated)')
    parser.add_argument('--config', '-c', default='config.txt', help='Path to config file (default: config.txt)')
    parser.add_argument('--children', '-r', action='store_true', help='Extract child pages')
    parser.add_argument('--depth', '-d', default=1, help='Maximum depth for child pages (default: 1, use "all" for unlimited)')
    parser.add_argument('--no-images', action='store_true', help='Skip image extraction')
    parser.add_argument('--no-tables', action='store_true', help='Skip table extraction')
    parser.add_argument('--no-code', action='store_true', help='Skip code block extraction')
    
    args = parser.parse_args()
    
    # Convert depth to int if it's not "all"
    if args.depth != "all":
        try:
            args.depth = int(args.depth)
        except ValueError:
            print(f"Warning: Invalid depth value '{args.depth}'. Using default depth of 1.")
            args.depth = 1
    
    result = extract_confluence_content(
        url=args.url,
        output_dir=args.output,
        config_path=args.config,
        include_children=args.children,
        max_depth=args.depth,
        extract_images=not args.no_images,
        save_tables=not args.no_tables,
        save_code=not args.no_code
    )
    
    if result:
        print(f"\nSuccess! Content has been extracted to: {result['output_directory']}")
        if result['has_children'] and not args.children:
            print("\nNote: This page has child pages that were not extracted.")
            print("Use the --children option to extract child pages.")
    else:
        print("\nFailed to extract content. Check the logs for details.")


if __name__ == "__main__":
    main()
