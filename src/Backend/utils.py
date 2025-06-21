# utils.py

"""
Utility Functions Module

This module contains caching functions and common helper utilities.
"""

import os
import json
import hashlib
import logging
from datetime import datetime
import re
from typing import Optional
import argparse
from azure.storage.blob import BlobServiceClient

CONTAINER_NAME = "myfiles"
UPLOAD_DIR = "uploads"
STORAGE_CONNECTION_STRING = os.getenv("STORAGE_CONNECTION_STRING") 

logger = logging.getLogger(__name__)

def extract_json(response):
    """Extracts JSON part from a CrewOutput or string response."""
    if not isinstance(response, str):
        try:
            response = response.to_json()
        except AttributeError:
            response = str(response)

    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    
    if json_match:
        try:
            structured_output = json.loads(json_match.group(0))
            return structured_output
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None
    else:
        print("No valid JSON found in the response.")
        return None

#############################################################################
# BLOB UTILITIES
#############################################################################

def download_blob(blob_name: str, download_path: str):
    try:
        local_path = f"/home/azureuser/Backend/{download_path}"
        blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            data = blob_client.download_blob()
            f.write(data.readall())

        logger.info(f"? Downloaded blob '{blob_name}' to '{download_path}'")
        return True
    except Exception as e:
        logger.error(f"? Failed to download blob '{blob_name}': {e}")
        return False

def delete_blob(blob_name):
    blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)

    #blob_client.delete_blob()
    file_path = os.path.join(UPLOAD_DIR, blob_name)
    os.remove(file_path)
    logger.info(f"?? Deleted blob and local copy: {blob_name}")

#############################################################################
# COMMAND ARGUMENTS UTILITIES
#############################################################################

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process medical images from Azure Blob Storage')
    parser.add_argument('blob_name', help='Name of the blob file to process')
    return parser.parse_args()

#############################################################################
# CACHING UTILITIES
#############################################################################

def get_cache_key(condition: str, source: str) -> str:
    """Generate a cache key for a condition and source"""
    cache_string = f"{condition}_{source}".lower()
    return hashlib.md5(cache_string.encode()).hexdigest()

def get_cached_data(condition: str, source: str) -> Optional[str]:
    """Retrieve cached data if it exists and is not expired (24 hours)"""
    cache_key = get_cache_key(condition, source)
    cache_file = f"cache/{cache_key}.json"
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check if cache is expired (24 hours = 86400 seconds)
            cache_time = datetime.fromisoformat(cache_data['timestamp'])
            if (datetime.now() - cache_time).total_seconds() < 86400:
                logger.info(f"Using cached data for {condition} from {source}")
                return cache_data['data']
            else:
                logger.info(f"Cache expired for {condition} from {source}")
                os.remove(cache_file)
        except Exception as e:
            logger.error(f"Error reading cache: {e}")
    
    return None

def cache_data(condition: str, source: str, data: str):
    """Cache data with timestamp for 24-hour expiry"""
    cache_key = get_cache_key(condition, source)
    cache_file = f"cache/{cache_key}.json"
    
    cache_content = {
        'condition': condition,
        'source': source,
        'data': data,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        with open(cache_file, 'w') as f:
            json.dump(cache_content, f)
        logger.info(f"Cached data for {condition} from {source}")
    except Exception as e:
        logger.error(f"Error caching data: {e}")