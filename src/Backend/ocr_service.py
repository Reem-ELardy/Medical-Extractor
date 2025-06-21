# ocr_service.py

"""
OCR Service Module

This module handles text extraction from images using EasyOCR.
"""

import os
import logging
from crewai.tools import tool
from config import reader

logger = logging.getLogger(__name__)

#############################################################################
# OCR TOOLS
#############################################################################

@tool
def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image using EasyOCR.
    
    Args:
        image_path (str): Path to the image file to extract text from
    
    Returns:
        str: Extracted text content from the image
    
    Raises:
        FileNotFoundError: If the image file does not exist
        Exception: If there is an error during image processing or text extraction
    """
    logger.info(f"Starting text extraction from image: {image_path}")
    
    try:

        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            raise FileNotFoundError(f"[Errno 2] No such file or directory: '{image_path}'")

        # Use EasyOCR to extract text
        results = reader.readtext(image_path)
        logger.debug(f"Found {len(results)} text blocks in the image")
        
        # Combine the detected text
        extracted_text = ""
        extracted_text = " ".join([result[1] for result in results])
        
        # Log the extracted text length
        logger.info(f"Text extraction complete. Extracted {len(extracted_text)} characters")
        return extracted_text.strip()
    
    except FileNotFoundError as e:
        logger.error(f"Error extracting text from image: {str(e)}")
        raise

    except Exception as e:
        logger.error(f"Error during image text extraction: {str(e)}", exc_info=True)
        raise Exception(f"Failed to extract text from image: {str(e)}")