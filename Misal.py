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
                
                # Here is the key change: process the child content the same way
                child_content = self._get_page_content(child_url)
                if child_content:
                    # Ensure child content has a title (important for header formatting)
                    if 'title' not in child_content or not child_content['title']:
                        child_content['title'] = child['title']
                    
                    # Create child directory
                    os.makedirs(child_dir, exist_ok=True)
                    
                    # Save the child content with the same processing as parent
                    self._save_content(child_content, child_dir, extract_images, save_tables, save_code)
                    
                    # Process deeper children if needed
                    if max_depth > 1 and 'child_pages' in child_content and child_content['child_pages']:
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
                    else:
                        child_results.append({
                            'title': child_content.get('title', 'Unknown'),
                            'url': child_url,
                            'output_directory': child_dir,
                            'has_children': False,
                            'children_extracted': False
                        })
        
        content['extracted_children'] = child_results
    
    return {
        'title': content.get('title', 'Unknown'),
        'url': url,
        'output_directory': output_dir,
        'has_children': 'child_pages' in content and bool(content['child_pages']),
        'children_extracted': include_children and max_depth > 0
    }
