# Examples of how to use the Confluence Content Extractor

# 1. Basic usage - Just provide the URL
python confluence_extractor.py https://confluence.example.com/display/SPACE/Page+Title

# 2. Extract child pages as well
python confluence_extractor.py https://confluence.example.com/display/SPACE/Page+Title --children

# 3. Control how deep to go with child pages
python confluence_extractor.py https://confluence.example.com/display/SPACE/Page+Title --children --depth 3

# 4. Extract ALL child pages (unlimited depth)
python confluence_extractor.py https://confluence.example.com/display/SPACE/Page+Title --children --depth all

# 5. Specify a custom output directory
python confluence_extractor.py https://confluence.example.com/display/SPACE/Page+Title --output my_content

# 6. Skip extracting images if you only need text
python confluence_extractor.py https://confluence.example.com/display/SPACE/Page+Title --no-images

# 7. Use a custom config file location
python confluence_extractor.py https://confluence.example.com/display/SPACE/Page+Title --config my_credentials.txt

# 8. Get everything - child pages, images, tables, code blocks
python confluence_extractor.py https://confluence.example.com/display/SPACE/Page+Title --children --depth all

# 9. Extract only text content
python confluence_extractor.py https://confluence.example.com/display/SPACE/Page+Title --no-images --no-tables --no-code

# Advanced example: Extract content from a deep hierarchy
python confluence_extractor.py https://confluence.example.com/display/SPACE/Root+Page --children --depth all --output full_documentation
