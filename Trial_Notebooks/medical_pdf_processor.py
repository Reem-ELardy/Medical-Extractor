# medical_pdf_processor.py

"""
Medical PDF Processor Module

This module implements a CrewAI-based system for extracting structured data from medical PDFs.
It uses a pipeline of specialized agents to extract, validate, format, and provide recommendations
based on medical report content.

The main components are:
1. Text extraction using EasyOCR
2. Data validation for medical values and terminology
3. Structured formatting into standardized JSON
4. Enhanced medical recommendations with web scraping
5. User feedback processing

New Features:
- Real-time treatment guideline scraping from Mayo Clinic and Medline-Plus
- Intelligent condition extraction for web searching
- Multi-layer fallback system for reliability
- Day-long caching for scraped data
- Source attribution for recommendations

Dependencies:
- crewai: For agent orchestration
- easyocr: For OCR text extraction
- requests: For Mayo Clinic scraping
- beautifulsoup4: For HTML parsing
- OpenAI API: For GPT-4 capabilities
"""

import os
import json
import re
import logging
import hashlib
import time
import random
from typing import Dict, Any, Union, List, Optional, Tuple
import tempfile
import numpy as np
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass
import inflect

# Third-party imports
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
import easyocr
from crewai.tools import tool
from PIL import Image
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import google.generativeai as genai

# Ensure the logs directory exists
os.makedirs('logs', exist_ok=True)
os.makedirs("extracted", exist_ok=True)
os.makedirs("cache", exist_ok=True)

# Get current date for the log filename
current_date = datetime.now().strftime('%Y-%m-%d')
log_filename = f"logs/medical_pdf_processor_{current_date}.log"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"Starting new log session in file: {log_filename}")

try:
    from medical_rag import MedicalRAG  # Import from same directory
    logger.info("Medical RAG import successful")
except ImportError as e:
    logger.error(f"Failed to import Medical RAG: {e}")
    logger.warning("RAG functionality will not be available")
    MedicalRAG = None

# Load environment variables from .env file
def load_environment():
    """Load environment variables from .env file"""
    possible_paths = [
        ".env",           # Current directory
        "../.env",        # Parent directory
        "../../.env"      # Grandparent directory
    ]
    
    env_loaded = False
    for env_path in possible_paths:
        if os.path.exists(env_path):
            load_dotenv(dotenv_path=env_path)
            if os.getenv("GEMINI_API_KEY"):
                logger.info(f"Loaded environment variables from {env_path}")
                env_loaded = True
                break
    
    if not env_loaded:
        logger.warning("No .env file found with GEMINI_API_KEY")
        logger.info("Please ensure you have a .env file with GEMINI_API_KEY set")
    
    return env_loaded
#############################################################################
# ENVIRONMENT AND LLM SETUP
#############################################################################

# Initialize environment
env_loaded = load_environment()

# Initialize LLM with API key from environment
def initialize_llm():
    """Initialize the LLM with API key from environment"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    return LLM(
        model='gemini/gemini-2.0-flash',
        provider="google",
        api_key=api_key,
        temperature=0
    )

# Initialize LLM
try:
    llm = initialize_llm()
    logger.info("LLM initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    raise

# Initialize EasyOCR reader
try:
    logger.info("Initializing EasyOCR reader for English language...")
    reader = easyocr.Reader(['en'])
    logger.info("EasyOCR reader initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize EasyOCR: {e}")
    raise

# Initialize Medical RAG system with proper path handling
def initialize_medical_rag():
    """Initialize Medical RAG system with robust path handling"""
    possible_paths = [
        "./embeddings",           # Current directory
        "../embeddings",          # Parent directory  
        "./chroma_db",           # Alternative name
        "../chroma_db"           # Alternative parent
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                rag = MedicalRAG(chroma_db_path=path)
                if rag.is_initialized:
                    logger.info(f"Medical RAG initialized with path: {path}")
                    return rag
            except Exception as e:
                logger.debug(f"Failed to initialize RAG with path {path}: {e}")
                continue
    
    logger.warning("Could not initialize Medical RAG with any path")
    return None

try:
    medical_rag = initialize_medical_rag()
    if medical_rag and medical_rag.is_initialized:
        logger.info("Medical RAG system loaded successfully")
        rag_info = medical_rag.get_collection_stats()
        logger.info(f"RAG Collection Info: {rag_info}")
    else:
        logger.warning("Medical RAG system not available")
        medical_rag = None
except Exception as e:
    logger.warning(f"Medical RAG initialization failed: {e}")
    medical_rag = None

    
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

#############################################################################
# TOOL DEFINITIONS
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

@tool
def validate_medical_values(data: str) -> str:
    """
    Validate medical numerical values and flag questionable terms (RAG-ready).
    
    Args:
        data (str): Medical text containing clinical values and terminology.
    
    Returns:
        str: JSON with validation status, identified issues, and candidates for RAG review.
    """
    logger.info("Starting medical data validation")

    validation_results = {
        "valid": True,
        "issues": [],
        "corrected_data": data,
        "rag_candidates": []
    }

    try:
        # Blood Pressure
        bp_match = re.search(r"Blood Pressure: (\d+)/(\d+)", data)
        if bp_match:
            systolic = int(bp_match.group(1))
            diastolic = int(bp_match.group(2))
            logger.debug(f"Blood pressure: {systolic}/{diastolic}")
            if systolic > 180 or systolic < 80 or diastolic > 120 or diastolic < 50:
                issue = f"Blood Pressure {systolic}/{diastolic} mmHg outside normal range"
                validation_results["issues"].append(issue)
                validation_results["valid"] = False

        # Glucose
        glucose_match = re.search(r"Glucose: (\d+)", data)
        if glucose_match:
            glucose = int(glucose_match.group(1))
            logger.debug(f"Glucose level: {glucose} mg/dL")
            if glucose < 40 or glucose > 300:
                issue = f"Glucose level {glucose} mg/dL outside expected range"
                validation_results["issues"].append(issue)
                validation_results["valid"] = False

        # Heart Rate
        hr_match = re.search(r"Heart Rate: (\d+)", data)
        if hr_match:
            hr = int(hr_match.group(1))
            if hr < 40 or hr > 150:
                issue = f"Heart rate {hr} bpm is abnormal"
                validation_results["issues"].append(issue)
                validation_results["valid"] = False

        # Temperature
        temp_match = re.search(r"Temperature: ([0-9.]+)", data)
        if temp_match:
            temp = float(temp_match.group(1))
            if temp < 35.0 or temp > 40.0:
                issue = f"Temperature {temp}°C outside normal body temperature range"
                validation_results["issues"].append(issue)
                validation_results["valid"] = False

        # --- Medical Term Validation via RAG ---
        if medical_rag and medical_rag.is_initialized:
            logger.info("Performing UMLS medical term validation")
            logger.debug(f"RAG system available: {medical_rag is not None}")
            logger.debug(f"RAG system initialized: {medical_rag.is_initialized if medical_rag else False}")
            try:
                # Extract and validate medical terms using the global RAG instance
                rag_validation = medical_rag.validate_medical_text(data)
                
                # Add this validation:
                if not isinstance(rag_validation, dict) or 'total_terms' not in rag_validation:
                    logger.error(f"Invalid RAG validation result: {type(rag_validation)}")
                    rag_validation = {
                        "total_terms": 0,
                        "validated_terms": 0,
                        "unknown_terms": 0,
                        "confidence_average": 0.0,
                        "term_validations": []
                    }
                # Add RAG validation results to the main validation
                validation_results["rag_validation"] = rag_validation
                validation_results["medical_terms_found"] = rag_validation["total_terms"]
                validation_results["medical_terms_validated"] = rag_validation["validated_terms"]
                
                # Process individual term validations
                for term_validation in rag_validation["term_validations"]:
                    if not term_validation["found"]:
                        # Add unknown terms to RAG candidates
                        validation_results["rag_candidates"].append({
                            "term": term_validation['term'],
                            "suggestions": term_validation['suggestions'],
                            "confidence": term_validation['confidence'],
                            "category": term_validation.get('category', 'unknown')
                        })
                        
                        # Add to issues if confidence is very low
                        if term_validation['confidence'] < 0.3:
                            issue = f"Unrecognized medical term: '{term_validation['term']}'"
                            if term_validation.get('suggestions'):
                                suggestions = ', '.join(term_validation['suggestions'][:2])
                                issue += f" - Similar terms: {suggestions}"
                            validation_results["issues"].append(issue)
                            validation_results["valid"] = False
                
                # Add overall validation metrics
                validation_results["rag_metrics"] = {
                    "total_terms": rag_validation['total_terms'],
                    "validated_terms": rag_validation['validated_terms'],
                    "unknown_terms": rag_validation['unknown_terms'],
                    "confidence_average": rag_validation['confidence_average'],
                    "semantic_types": rag_validation.get('semantic_types', [])
                }
                
                # Determine overall validation based on RAG results
                if rag_validation["unknown_terms"] > rag_validation["validated_terms"]:
                    validation_results["valid"] = False
                    validation_results["issues"].append(
                        f"More unknown medical terms ({rag_validation['unknown_terms']}) than validated ones ({rag_validation['validated_terms']})"
                    )
                
                logger.info(f"RAG Validation Complete: {rag_validation['validated_terms']}/{rag_validation['total_terms']} terms validated")
                
            except Exception as rag_error:
                logger.error(f"RAG validation error: {str(rag_error)}")
                validation_results["rag_error"] = str(rag_error)
                validation_results["rag_available"] = False
        else:
            logger.warning("UMLS RAG not available, using basic term extraction")
            validation_results["rag_available"] = False

        # You can use regex or a chunker to identify terms to validate
        suspect_terms = re.findall(r"(Diagnosis: .+?)(?:\n|$)", data)
        if suspect_terms:
            validation_results["rag_candidates"].extend([term.strip() for term in suspect_terms])

        # Log result
        if validation_results["valid"]:
            logger.info("Validation complete - No critical issues")
        else:
            logger.info(f"Validation found {len(validation_results['issues'])} issues")

        return json.dumps(validation_results)

    except Exception as e:
        logger.error(f"Validation error: {str(e)}", exc_info=True)
        validation_results["valid"] = False
        validation_results["issues"].append(f"Validation error: {str(e)}")
        return json.dumps(validation_results)

@tool
def format_to_json(extracted_text: str) -> str:
    """
    Transform unstructured medical text into a structured JSON format with specific sections.
    
    Args:
        extracted_text (str): The unstructured medical text extracted from a document
        
    Returns:
        str: JSON string with the structured medical data
    """
    logger.info("Starting formatting of extracted text to JSON")
    logger.debug(f"Text length: {len(extracted_text)} characters")
    
    # Prompt for the LLM to structure the data
    prompt = f"""
    You are a highly specialized medical data extraction expert with clinical experience. Transform the following unstructured medical text into a structured JSON format with extreme precision and attention to detail.

    Follow these strict section-wise instructions:

    1. Patient Information:
        - Extract only the following fields: Name, Age, and Gender.
        - Combine them in a single string in this exact format: "Name: <value>, Age: <value>, Gender: <value>"
        - Skip any missing field (e.g., if Age is missing, just include Name and Gender).
        - DO NOT include any other values like Patient ID, SSN, Floor, or Consultant.
        - DO NOT translate keys; use exactly "Name", "Age", and "Gender" as shown.
        - Use comma-separated key-value pairs, in the order: Name, Age, Gender.
        - Example outputs:
            - "Name: John Smith, Age: 45, Gender: Male"
            - "Name: Mary Jane, Gender: Female"
            - "Name: Tom Lee"
            
    2. Date of Issue:
        - Extract the exact date when the report was created.
        - Use standard format (DD.MM.YYYY or as presented in the document).
        - If multiple dates exist, prioritize report creation or discharge date over admission date.
        - If no date is found, output an empty string.
        - Accept synonymous labels like "Date of Report", "Report Date", or "Issued on".

    3. Type of Report:
        - Identify the overall document type based on its clinical purpose and structure.
        - Valid types include (but are not limited to): "Hospital Discharge Summary", "Radiology Report", "Consultation Note", "Laboratory Report", "Operative Report", "Pathology Report", "Medical Certificate", "Referral Letter", "Follow-Up Note".
        - Focus on recognizing high-level report categories.
        - DO NOT list:
            - Individual tests or procedures (e.g., "CT Scan", "MRI")
            - Body parts (e.g., "Brain Report")
        - If the report contains test names that are clearly used as report titles (e.g., "Echocardiography Report", "MRI Report", etc.), include them exactly as stated.
        - If clearly stated, extract the type directly.
        - If uncertain, infer based on structure and clinical context.
        - If no type can be determined, return an empty string.

    4. Medical Problem (DETAILED):
        - Extract only confirmed diagnoses or clearly stated medical conditions.
        - Use professional medical terminology, as if writing in a clinical chart.
        - DO NOT include:
            - Lab values, vital signs, or tests.
            - Symptoms unless confirmed as diagnosis.
            - Unconfirmed impressions or future considerations.
        - Return multiple conditions if mentioned, concisely.

    5. Simplified Explanation:
        - Translate the Medical Problem into plain English a patient can understand.
        - Avoid jargon or clinical abbreviations unless widely known (e.g., "heart attack").
        - Keep it simple, accurate, and empathetic.
        - Clearly explain what the problem is and what it means for the patient.
        - DO NOT provide treatment plans, risks, or prognosis unless clearly stated in the document.
        - Target a reading level of a high school student (~9th grade).
        - If unclear, return an empty string.

    Important Notes:
    - If a section cannot be determined, leave it as an **empty string** rather than guessing.
    - Do not hallucinate or infer information not in the text.
    - Output ONLY the following JSON with no additional commentary or explanation.

    Unstructured Medical Text:
    {extracted_text}

    Output ONLY the following JSON with no additional text:
    {{
        "Patient Information": "string in format: Name: <value>, Age: <value>, Gender: <value>",
        "Date of Issue": "string",
        "Type of Report": "string",
        "Medical Problem": "detailed string with all clinical findings",
        "Simplified Explanation": "string"
    }}
    """

    
    try:
        # Use the LLM to structure the data
        logger.debug("Sending text to LLM for structuring")
        response = llm.call(prompt) 
        logger.debug("Received response from LLM")
        
        # Extract JSON from the response
        logger.debug("Attempting to extract JSON from LLM response")
        json_match = re.search(r'({.*})', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            # Parse and format the JSON
            structured_data = json.loads(json_str)
            logger.info("Successfully structured the data into JSON format")
            
            # Log some details about the structured data
            logger.debug(f"Structured data has {len(structured_data)} fields")
            for key in structured_data:
                value_length = len(str(structured_data[key]))
                logger.debug(f"Field '{key}' has {value_length} characters")
                
            return json.dumps(structured_data, indent=2)
        else:
            error_msg = "No valid JSON found in LLM response"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})
    
    except Exception as e:
        logger.error(f"Error during formatting to JSON: {str(e)}", exc_info=True)
        return json.dumps({"error": str(e)})

@tool
def process_user_feedback(structured_data: str, user_feedback: str, extracted_text: str = "") -> str:
    """
    Process user feedback to update the structured medical data.
    
    Args:
        structured_data (str): JSON string of the current structured data
        user_feedback (str): User's feedback or correction request
        extracted_text (str): Original extracted text to reference for new fields
    
    Returns:
        str: Updated JSON string with the modifications
    """
    logger.info("Processing user feedback for structured data")
    
    # Parse the structured data if it's a string
    if isinstance(structured_data, str):
        try:
            data = json.loads(structured_data)
            logger.debug("Successfully parsed structured data from JSON string")
        except json.JSONDecodeError as e:
            logger.error(f"Could not parse the structured data: {e}", exc_info=True)
            return json.dumps({"error": "Could not parse the structured data"})
    else:
        data = structured_data
        logger.debug("Using provided structured data object")
    
    
    # Enhanced prompt for the LLM to process the feedback
    prompt = f"""
    You are a medical data processing specialist. Your task is to update structured medical data based on user feedback.

    CURRENT STRUCTURED DATA:
    {json.dumps(data, indent=2)}

    ORIGINAL EXTRACTED TEXT (for reference when adding new fields):
    {extracted_text}

    USER FEEDBACK:
    "{user_feedback}"

    INSTRUCTIONS:
    Analyze the user's feedback and determine what action to take:

    1. **ADDING NEW FIELDS**: If user requests to add information (e.g., "add Blood Pressure", "add Temperature", "add vital signs"):
    - Search the original extracted text for the requested information
    - Extract the exact values/information found
    - Add as new field(s) to the JSON structure  
    - If not found in original text, set value to "Not found in original report"

    2. **CORRECTING EXISTING FIELDS**: If user points out errors (e.g., "age should be 45", "name is wrong"):
    - Update the specific field with the corrected information
    - Keep all other fields unchanged

    3. **MODIFYING PRESENTATION**: If user requests format changes (e.g., "make explanation simpler", "use different wording"):
    - Modify the specified field according to the request
    - Maintain medical accuracy while improving presentation

    4. **GENERAL REQUESTS**: If user asks for broader changes (e.g., "add more details", "include lab results"):
    - Look for relevant information in the original extracted text
    - Add appropriate fields based on what's available
    - Be conservative - only add information that clearly exists in the source

    IMPORTANT RULES:
    - Base all new information strictly on the original extracted text
    - Never invent or assume information not present in the source
    - If information is not available, clearly state "Not found in original report"
    - Focus precisely on what the user specifically requested
    - Maintain the existing JSON structure and only modify what's needed
    - Preserve all existing fields unless specifically asked to change them

    OUTPUT: Return ONLY the updated JSON structure with no additional text or explanation.
    """

    try:
        # Use the LLM to process the feedback
        logger.debug("Sending feedback to LLM for processing")
        response = llm.call(prompt) 
        logger.debug("Received response from LLM")
        
        # Extract the updated JSON from the response
        logger.debug("Attempting to extract JSON from LLM response")
        json_match = re.search(r'({.*})', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            # Parse and format the JSON
            updated_data = json.loads(json_str)
            logger.info("Successfully updated the structured data based on feedback")
            
            # Log changes made to the data
            for key in updated_data:
                if key not in data:
                    logger.debug(f"Added new field: '{key}'")
                elif data[key] != updated_data[key]:
                    logger.debug(f"Modified field: '{key}'")
            
            return json.dumps(updated_data, indent=2)
        else:
            logger.warning("No JSON pattern found in LLM response for feedback")
            # If no JSON pattern found, return the original with an error
            data["feedback_processing_error"] = "Could not process feedback"
            logger.info("Returning original data with error note")
            return json.dumps(data, indent=2)
    
    except Exception as e:
        logger.error(f"Error processing user feedback: {str(e)}", exc_info=True)
        data["feedback_processing_error"] = f"Error processing feedback: {str(e)}"
        return json.dumps(data, indent=2)

@tool
def extract_searchable_conditions(medical_problem: str) -> str:
    """
    Extract searchable medical conditions from complex medical text using a generalizable prompt.
    
    Args:
        medical_problem (str): Raw medical report text
    
    Returns:
        str: JSON array of simplified, web-searchable medical condition terms (e.g., Mayo Clinic style)
    """
    logger.info("Extracting searchable conditions from medical report")

    prompt = f"""
    You are a medical domain expert trained to convert complex clinical findings into
    simplified, patient-friendly terms that users can easily search online (like on Mayo Clinic or Medline-Plus).

    INPUT MEDICAL REPORT:
    "{medical_problem}"

    TASK:
    - Extract only key medical conditions, diagnoses, or significant findings
    - Convert them into layperson-friendly, web-searchable terms that would match the names of diseases or conditions as found on websites like Mayo Clinic or MedlinePlus
    - Avoid technical modifiers (e.g. anatomical locations, severity unless critical)
    - Avoid repetition or rare synonyms
    - Output max 5 key conditions

    OUTPUT:
    Return only a JSON array of simplified medical condition terms.

    Example:
    INPUT: "Maxillo-ethmoidal and frontal sinusitis, Nasal septum deviation, Hypertrophied inferior nasal turbinates, Allergic rhinitis"
    OUTPUT: ["Chronic sinusitis", "Deviated septum", "Nasal polyps", "Hay fever"]
    """

    try:
        logger.debug("Sending prompt to LLM...")
        response = llm.call(prompt)
        logger.debug(f"Received response: {response}")
        return response  # Already in JSON format
    except Exception as e:
        logger.error(f"Error extracting searchable conditions: {e}")
        return json.dumps([])

@tool
def scrape_mayo_treatments(conditions: List[str]) -> str:
    """
    Scrape treatment information from Mayo Clinic for given medical conditions
    using the exact same logic as the MayoClinicScraper class.
    
    Args:
        conditions (List[str]): List of medical conditions to search for treatments
    
    Returns:
        str: JSON string with treatment information from Mayo Clinic
    """
    if isinstance(conditions, str):
        try:
            conditions = json.loads(conditions)
        except json.JSONDecodeError:
            conditions = [conditions]
    
    results = {}
    base_url = "https://www.mayoclinic.org"
    
    # Set up session with proper headers
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    })
    
    def clean_query_for_url(query: str) -> str:
        """
        Clean and standardize query for URL formation
        Directly implements MayoClinicScraper._clean_query_for_url logic
        """
        # Convert to lowercase
        clean_query = query.lower()
        
        # Remove common punctuation but preserve meaningful separators temporarily
        punctuation_to_remove = "'\".,()[]{}!?;:"
        for punct in punctuation_to_remove:
            clean_query = clean_query.replace(punct, '')
        
        # Replace underscores and multiple spaces with single spaces
        clean_query = re.sub(r'[_]+', ' ', clean_query)
        clean_query = re.sub(r'\s+', ' ', clean_query)
        
        # Replace spaces with hyphens
        clean_query = clean_query.replace(' ', '-')
        
        # Remove multiple consecutive hyphens
        clean_query = re.sub(r'-+', '-', clean_query)
        
        # Remove leading/trailing hyphens
        clean_query = clean_query.strip('-')
        
        return clean_query
    
    def check_url_exists(url: str) -> bool:
        """
        Check if a URL exists with better error handling
        Directly implements MayoClinicScraper._check_url_exists logic
        """
        try:
            response = session.head(url, timeout=8, allow_redirects=True)
            
            # Accept both 200 and 3xx redirects as valid
            if response.status_code in [200, 301, 302, 303, 307, 308]:
                return True
            else:
                return False
                
        except Exception:
            return False
    
    def get_page(url: str) -> Optional[BeautifulSoup]:
        """
        Fetch and parse a web page with error handling
        Directly implements MayoClinicScraper._get_page logic
        """
        try:
            # Add random delay to be respectful
            delay = random.uniform(1, 3)
            time.sleep(delay)
            
            response = session.get(url, timeout=10)
            response.raise_for_status()
            
            return BeautifulSoup(response.content, 'html.parser')
            
        except Exception:
            return None
    
    def discover_condition_subpages(base_url: str, condition_pattern: str) -> List[str]:
        """
        Directly implements MayoClinicScraper._discover_condition_subpages logic
        """
        subpage_urls = []
        
        # Only look for diagnosis-treatment page
        target_subpage = "diagnosis-treatment"
        
        # First, get the base page to find actual subpage links
        soup = get_page(base_url)
        if soup:
            # Look for links to diagnosis-treatment page in the navigation or content
            all_links = soup.find_all('a', href=True)
            
            for link in all_links:
                href = link.get('href')
                if href:
                    # Convert relative URLs to absolute
                    if href.startswith('/'):
                        full_url = urljoin(base_url, href)
                    elif href.startswith('http'):
                        full_url = href
                    else:
                        continue
                    
                    # Check if this link is the diagnosis-treatment page we're looking for
                    if f"/diseases-conditions/{condition_pattern}/{target_subpage}/" in full_url:
                        if full_url not in subpage_urls:
                            subpage_urls.append(full_url)
                            break  # Found it, no need to continue
        
        # If scraping didn't find the diagnosis-treatment page, try simple pattern
        if not subpage_urls:
            test_url = f"{base_url}/diseases-conditions/{condition_pattern}/{target_subpage}"
            if check_url_exists(test_url):
                subpage_urls.append(test_url)
        
        # Return the diagnosis-treatment page or base URL as fallback
        return subpage_urls if subpage_urls else ([base_url] if check_url_exists(base_url) else [])
    
    def search_conditions(query: str) -> List[str]:
        """
        Directly implements MayoClinicScraper.search_conditions logic
        """
        search_urls = []
        
        # Clean the query for URL formation
        clean_query = clean_query_for_url(query)
    
        # Basic URL patterns to try
        base_patterns = [
            clean_query,
            clean_query.replace('-', ''),
            clean_query.replace('-', ' ').replace(' ', ''),
            f"chronic-{clean_query}",
            f"acute-{clean_query}"
        ]
        
        # For each base pattern, try to find the main condition page first
        for pattern in base_patterns:
            condition_url = f"{base_url}/diseases-conditions/{pattern}"
            
            # Check if the base condition page exists
            if check_url_exists(condition_url):
                # Now find the diagnosis-treatment page specifically
                condition_urls = discover_condition_subpages(condition_url, pattern)
                search_urls.extend(condition_urls)
                
                if search_urls:
                    break  # Found working URLs, no need to try more patterns
        
        # If no direct patterns worked, try alternative discovery methods
        if not search_urls:
            # Extract key medical terms and search for them
            words = query.lower().replace('-', ' ').split()
            
            # Common medical term transformations
            transformations = {
                'deviation': 'deviated',
                'enlargement': 'enlarged', 
                'inflammation': 'inflamed',
                'infection': 'infected',
                'deficiency': 'deficient',
                'insufficiency': 'insufficient',
                'dysfunction': 'dysfunctional',
                'syndrome': 'disease',
                'disorder': 'condition',
                'palsy': 'paralysis',
            }
            
            # Transform words and create new patterns
            transformed_words = []
            for word in words:
                if word in transformations:
                    transformed_words.append(transformations[word])
                else:
                    transformed_words.append(word)
            
            # Try different combinations
            test_patterns = [
                # Transformed version
                '-'.join(transformed_words),
                # First few words only
                '-'.join(words[:2]) if len(words) > 1 else words[0],
                # Reverse order
                '-'.join(reversed(words)) if len(words) > 1 else words[0],
                # Just the longest word (often the main medical term)
                max(words, key=len) if len(words) > 1 else None,
                # Last word only (often a key term)
                words[-1] if len(words) > 1 else words[0],
            ]
            
            test_patterns = [p for p in test_patterns if p]  # Remove None values
            
            for pattern in test_patterns:
                condition_url = f"{base_url}/diseases-conditions/{pattern}"
                if check_url_exists(condition_url):
                    return discover_condition_subpages(condition_url, pattern)
        
        return search_urls
    
    def extract_treatments(soup: BeautifulSoup) -> List[str]:
        """
        Extract treatment information from Mayo Clinic page, 
        directly implementing MayoClinicScraper._extract_treatments logic
        """
        treatments = []
        
        # First, look specifically for h2 with "Treatment" text
        treatment_h2 = None
        for h2 in soup.find_all('h2'):
            if re.match(r'^\s*Treatment\s*$', h2.get_text().strip(), re.I):
                treatment_h2 = h2
                break
        
        if treatment_h2:
            # Get content following the Treatment h2
            current = treatment_h2.next_sibling
            
            while current:
                if hasattr(current, 'name'):
                    # Stop when we hit another h2 (next major section)
                    if current.name == 'h2':
                        break
                    
                    # Process h3 headers (treatment subsections)
                    elif current.name == 'h3':
                        h3_text = current.get_text().strip()
                        
                        # Add the h3 header as a treatment category
                        if h3_text and len(h3_text) > 3:
                            treatments.append(f"**{h3_text}**")
                        
                        # Get content following this h3 until next h3 or h2
                        h3_current = current.next_sibling
                        while h3_current:
                            if hasattr(h3_current, 'name'):
                                if h3_current.name in ['h2', 'h3']:
                                    break  # Stop at next heading
                                elif h3_current.name in ['ul', 'ol']:
                                    # Extract list items under this h3
                                    items = h3_current.find_all('li')
                                    for item in items:
                                        treatment = item.get_text().strip()
                                        if treatment and len(treatment) > 10:
                                            treatments.append(f"  • {treatment}")
                                elif h3_current.name == 'p':
                                    # Extract paragraph content under this h3
                                    text = h3_current.get_text().strip()
                                    if text and len(text) > 20:
                                        treatments.append(f"  {text}")
                                elif h3_current.name == 'div':
                                    # Look inside divs for nested content
                                    div_lists = h3_current.find_all(['ul', 'ol'])
                                    for ul in div_lists:
                                        items = ul.find_all('li')
                                        for item in items:
                                            treatment = item.get_text().strip()
                                            if treatment and len(treatment) > 10:
                                                treatments.append(f"  • {treatment}")
                                    
                                    # Also check paragraphs in the div
                                    div_paras = h3_current.find_all('p')
                                    for para in div_paras:
                                        text = para.get_text().strip()
                                        if text and len(text) > 20:
                                            treatments.append(f"  {text}")
                            
                            h3_current = h3_current.next_sibling
                    
                    # Process direct content under the main h2 (before any h3s)
                    elif current.name in ['ul', 'ol']:
                        # Extract list items directly under h2
                        items = current.find_all('li')
                        for item in items:
                            treatment = item.get_text().strip()
                            if treatment and len(treatment) > 10:
                                treatments.append(treatment)
                    elif current.name == 'p':
                        # Extract paragraph content directly under h2
                        text = current.get_text().strip()
                        if text and len(text) > 20:
                            treatments.append(text)
                    elif current.name == 'div':
                        # Look inside divs for nested content directly under h2
                        div_lists = current.find_all(['ul', 'ol'])
                        for ul in div_lists:
                            items = ul.find_all('li')
                            for item in items:
                                treatment = item.get_text().strip()
                                if treatment and len(treatment) > 10:
                                    treatments.append(treatment)
                        
                        # Also check paragraphs in the div
                        div_paras = current.find_all('p')
                        for para in div_paras:
                            text = para.get_text().strip()
                            if text and len(text) > 20:
                                treatments.append(text)
                
                current = current.next_sibling
        
        # If no h2 'Treatment' found, try other treatment headings as fallback
        if not treatments:
            treatment_headings = soup.find_all(['h2', 'h3', 'h4'], 
                                             string=re.compile(r'treatment|therapy|management|care|medication', re.I))
            
            for heading in treatment_headings:
                # Get content following the heading
                current = heading.next_sibling
                found_items = 0
                
                while current and found_items < 10:
                    if hasattr(current, 'name'):
                        if current.name in ['ul', 'ol']:
                            items = current.find_all('li')
                            for item in items:
                                treatment = item.get_text().strip()
                                if treatment and len(treatment) > 10:
                                    treatments.append(treatment)
                                    found_items += 1
                        elif current.name == 'p':
                            text = current.get_text().strip()
                            if text and len(text) > 20:
                                treatments.append(text)
                                found_items += 1
                        elif current.name in ['h2', 'h3', 'h4']:
                            break  # Stop at next heading
                    
                    current = current.next_sibling
                
                if treatments:
                    break  # Found treatments, no need to check other headings
        
        # If still no treatments found, try enhanced selectors as final fallback
        if not treatments:
            treatment_selectors = [
                '[class*="treatment" i] ul li',
                '[class*="treatment" i] ol li',
                '[class*="therapy" i] ul li',
                '[class*="management" i] ul li',
                '[id*="treatment" i] ul li',
                '[id*="treatment" i] ol li',
                '.content ul li',
                '.page-content ul li', 
                '.main-content ul li',
            ]
            
            for selector in treatment_selectors:
                try:
                    elements = soup.select(selector)
                    for elem in elements:
                        treatment = elem.get_text().strip()
                        if treatment and len(treatment) > 10:
                            treatments.append(treatment)
                    
                    if treatments:
                        break
                except Exception:
                    continue
        
        # Remove duplicates while preserving order
        unique_treatments = list(dict.fromkeys(treatments))
        
        return unique_treatments[:25]  # Limit to 25 items
    
    # Process each condition
    for condition in conditions:
        if not condition or not condition.strip():
            continue
            
        try:
            # Use the search_conditions method to find the correct URL
            found_urls = search_conditions(condition)
            
            if found_urls:
                # Get the first URL (preferably diagnosis-treatment)
                condition_url = found_urls[0]
                
                # Fetch the page
                soup = get_page(condition_url)
                if soup:
                    # Extract treatments
                    treatments = extract_treatments(soup)
                    
                    # Join treatments into a string
                    treatment_text = '\n\n'.join(treatments)
                    
                    # Store results
                    results[condition] = {
                        "name": condition,
                        "url": condition_url,
                        "source": "Mayo Clinic",
                        "treatment": treatment_text if treatment_text else "No treatment information found"
                    }
                else:
                    results[condition] = {
                        "name": condition,
                        "url": condition_url,
                        "source": "Mayo Clinic",
                        "error": "Could not fetch page content"
                    }
            else:
                results[condition] = {
                    "name": condition,
                    "error": "No Mayo Clinic page found",
                    "source": "Mayo Clinic"
                }
            
            # Add delay between conditions
            time.sleep(random.uniform(1.5, 3))
            
        except Exception as e:
            results[condition] = {
                "name": condition,
                "error": str(e),
                "source": "Mayo Clinic"
            }
    
    return json.dumps(results, indent=2)

@tool
def scrape_medlineplus_treatments(conditions: List[str]) -> str:
    """
    Scrape treatment information from MedlinePlus for given medical conditions.

    Args:
        conditions (List[str]): List of condition names to search for.

    Returns:
        str: JSON string with treatment information.
    """
    results = {}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    def clean_condition_for_url(condition: str) -> str:
        # Remove anything in parentheses and extra characters
        condition = re.sub(r"\(.*?\)", "", condition)  # remove (UTI) or similar
        condition = condition.lower()
        condition = condition.replace("'", "").replace("'", "").replace("–", "").replace("-", "")
        condition = condition.replace(" ", "")
        return condition.strip()

    def pluralize_slug(condition: str) -> str:
        # Create inflect engine
        p = inflect.engine()
        words = condition.lower().split()
        if words:
            words[-1] = p.plural(words[-1])
        return ''.join(words)

    def singularize_slug(condition: str) -> str:
        # Create inflect engine
        p = inflect.engine()
        words = condition.lower().split()
        if words:
            singular = p.singular_noun(words[-1])
            words[-1] = singular if singular else words[-1]
        return ''.join(words)

    for condition in conditions:
        try:
            time.sleep(1)
            # ⿡ Try original slug
            slug = clean_condition_for_url(condition)
            url = f"https://medlineplus.gov/{slug}.html"
            resp = requests.get(url, headers=headers, timeout=10)

            # ⿢ Try adding s to slug
            if resp.status_code != 200:
                fallback = f"https://medlineplus.gov/{slug}s.html"
                resp = requests.get(fallback, headers=headers, timeout=10)
                url = fallback if resp.status_code == 200 else url

            # ⿣ Try pluralized slug
            if resp.status_code != 200:
                plural_slug = pluralize_slug(condition)
                fallback = f"https://medlineplus.gov/{plural_slug}.html"
                resp = requests.get(fallback, headers=headers, timeout=10)
                url = fallback if resp.status_code == 200 else url

            # ⿤ Try singularized slug
            if resp.status_code != 200:
                singular_slug = singularize_slug(condition)
                fallback = f"https://medlineplus.gov/{singular_slug}.html"
                resp = requests.get(fallback, headers=headers, timeout=10)
                url = fallback if resp.status_code == 200 else url

            # Check if we got a successful response
            if resp.status_code != 200:
                results[condition] = {"error": f"No page found at {url}"}
                continue

            soup = BeautifulSoup(resp.content, 'html.parser')
            treatment_text = ""

            # Step 1: Look for treatment-related headings and paragraphs after them
            headers_list = soup.find_all(["h2", "h3"])
            for header in headers_list:
                header_text = header.get_text(strip=True).lower()
                if any(kw in header_text for kw in ["summary","treat", "manage", "care", "therapy", "medicine"]):
                    content = []
                    for sibling in header.find_next_siblings():
                        if sibling.name in ["p", "ul", "ol", "li"]:
                            content.append(sibling.get_text(strip=True))
                        elif sibling.name in ["h2", "h3"]:
                            break
                    if content:
                        treatment_text = " ".join(content)
                        break

            # Step 2: Enhanced fallback – look in summary section and early content
            if not treatment_text:
                # Try different summary approaches
                summary_found = False
                
                # Approach 1: Look for summary headers
                for header in headers_list:
                    header_text = header.get_text(strip=True).lower()
                    if any(kw in header_text for kw in ["summary", "overview", "about", "what is"]):
                        summary_parts = []
                        for sibling in header.find_next_siblings():
                            if sibling.name in ["p", "ul", "ol", "li"]:
                                summary_parts.append(sibling.get_text(strip=True))
                            elif sibling.name in ["h2", "h3"]:
                                break
                        if summary_parts:
                            summary_text = " ".join(summary_parts)
                            # Look for treatment keywords with more flexibility
                            treatment_keywords = ['treat', 'treatment', 'therapy', 'medicine', 'medication', 'manage', 'care']
                            if any(keyword in summary_text.lower() for keyword in treatment_keywords):
                                treatment_text = summary_text[:3000]
                                summary_found = True
                                break
                
                # Approach 2: If no summary header found, check first few paragraphs
                if not summary_found:
                    main_content = soup.find('main') or soup.find('div', class_='content') or soup
                    first_paragraphs = main_content.find_all('p')[:4]  # First 4 paragraphs
                    
                    for p in first_paragraphs:
                        p_text = p.get_text(strip=True)
                        if len(p_text) > 100:  # Skip very short paragraphs
                            treatment_keywords = ['treat', 'treatment', 'therapy', 'medicine', 'medication', 'manage']
                            if any(keyword in p_text.lower() for keyword in treatment_keywords):
                                treatment_text = p_text
                                break
                
                # Approach 3: Look for any paragraph with treatment content
                if not treatment_text:
                    all_paragraphs = soup.find_all('p')[:10]  # Check first 10 paragraphs
                    for p in all_paragraphs:
                        p_text = p.get_text(strip=True)
                        if len(p_text) > 80 and 'treat' in p_text.lower():
                            treatment_text = p_text
                            break

            # Step 3: Store result
            if treatment_text:
                results[condition] = {
                    "name": condition,
                    "url": url,
                    "source": "MedlinePlus",
                    "treatment": treatment_text.strip()
                }
            else:
                results[condition] = {"error": "No treatment info found"}

        except Exception as e:
            logging.exception(f"Error scraping {condition}")
            results[condition] = {"error": str(e)}

    return json.dumps(results, indent=2)

@tool
def generate_web_enhanced_recommendations(patient_data: str, mayo_data: str = "", medlineplus_data: str = "") -> str:
    """
    Generate enhanced medical recommendations combining patient data with scraped treatment guidelines.
    
    Args:
        patient_data (str): Patient's structured medical data
        mayo_data (str): Treatment data scraped from Mayo Clinic
        medlineplus_data (str): Treatment data scraped from Medline-Plus
        
    Returns:
        str: JSON string with enhanced recommendations including source attribution
    """
    logger.info("Generating web-enhanced medical recommendations")
    
    # Parse inputs
    try:
        if isinstance(patient_data, str):
            patient_info = json.loads(patient_data)
        else:
            patient_info = patient_data
    except json.JSONDecodeError:
        logger.error("Could not parse patient data")
        return json.dumps({"error": "Could not parse patient data"})
    
    try:
        mayo_treatments = json.loads(mayo_data) if mayo_data else {}
    except json.JSONDecodeError:
        mayo_treatments = {}
    
    try:
        medlineplus_treatments = json.loads(medlineplus_data) if medlineplus_data else {}
    except json.JSONDecodeError:
        medlineplus_treatments = {}
    
    logger.debug(f"Processing patient: {patient_info.get('Patient Information', 'Unknown')}")
    logger.debug(f"Mayo treatments available: {len(mayo_treatments)}")
    logger.debug(f"MedLinePlus treatments available: {len(medlineplus_treatments)}")
    
    # Create enhanced prompt with web data
    prompt = f"""
    You are a clinically responsible AI medical advisor with access to the latest treatment guidelines. Generate evidence-based, patient-friendly recommendations.

    PATIENT INFORMATION:
    {json.dumps(patient_info, indent=2)}

    CURRENT MAYO CLINIC TREATMENT GUIDELINES:
    {json.dumps(mayo_treatments, indent=2)}

    CURRENT MedLinePlus TREATMENT PROTOCOLS:
    {json.dumps(medlineplus_treatments, indent=2)}

    INSTRUCTIONS:
    Generate 2-3 comprehensive recommendations that:

    1. **INTEGRATE CURRENT GUIDELINES**: Use the latest treatment information from Mayo Clinic and MedLinePlus to ensure recommendations are current and evidence-based.

    2. **PERSONALIZE FOR PATIENT**: Tailor recommendations specifically to this patient's conditions and circumstances.

    3. **PRIORITIZE SAFETY**: Focus on safe, non-invasive recommendations that complement professional medical care.

    4. **PROVIDE SOURCE ATTRIBUTION**: Clearly indicate when recommendations are based on current medical guidelines vs. general medical knowledge.

    5. **ENSURE ACTIONABILITY**: Each recommendation should be specific and actionable for the patient.

    For each recommendation, provide:
    - **recommendation**: Clear, specific action the patient should take
    - **explanation**: Why this is important based on their condition and current medical guidelines
    - **lifestyle_modifications**: Practical daily life tips to support the recommendation
    - **source**: Whether based on "Mayo Clinic Guidelines", "Medline-Plus Protocols", "Combined Guidelines", or "General Medical Knowledge"

    FOCUS AREAS (if applicable):
    - Medication adherence and management
    - Lifestyle modifications for condition management
    - When to seek immediate medical attention
    - Preventive care and monitoring
    - Patient education and self-management

    TARGET AUDIENCE: Patient and family members (9th-grade reading level)

    OUTPUT FORMAT:
    {{
        "recommendations": [
            {{
                "recommendation": "specific actionable advice",
                "explanation": "why this helps based on current guidelines and patient condition",
                "lifestyle_modifications": "practical daily tips",
                "source": "Mayo Clinic Guidelines / Medline-Plus Protocols / Combined Guidelines / General Medical Knowledge"
            }}
        ],
        "data_source": "web_enhanced",
        "sources_used": ["Mayo Clinic", "Medline-Plus"],
        "disclaimer": "These recommendations are based on current medical guidelines but should not replace professional medical advice."
    }}
    """
    
    try:
        logger.debug("Sending enhanced prompt to LLM")
        response = llm.call(prompt)
        logger.debug("Received enhanced recommendations from LLM")
        
        # Extract JSON from response
        json_match = re.search(r'({.*})', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            recommendations = json.loads(json_str)
            
            # Add metadata about sources used
            sources_used = []
            if mayo_treatments:
                sources_used.append("Mayo Clinic")
            if medlineplus_treatments:
                sources_used.append("Medline-Plus")
            
            recommendations["sources_used"] = sources_used
            recommendations["data_source"] = "web_enhanced"
            recommendations["fallback_used"] = False
            
            logger.info(f"Successfully generated {len(recommendations.get('recommendations', []))} web-enhanced recommendations")
            return json.dumps(recommendations, indent=2)
        else:
            logger.warning("No JSON found in enhanced recommendations response")
            return json.dumps({
                "error": "Could not parse enhanced recommendations",
                "fallback_used": True
            })
    
    except Exception as e:
        logger.error(f"Error generating web-enhanced recommendations: {str(e)}", exc_info=True)
        return json.dumps({
            "error": f"Failed to generate enhanced recommendations: {str(e)}",
            "fallback_used": True
        })

@tool
def generate_fallback_recommendations(patient_data: str) -> str:
    """
    Generate standard LLM recommendations when web scraping fails.
    
    Args:
        patient_data (str): Patient's structured medical data
        
    Returns:
        str: JSON string with standard recommendations and fallback indication
    """
    logger.info("Generating fallback recommendations (LLM-only)")
    
    try:
        if isinstance(patient_data, str):
            patient_info = json.dumps(patient_data)
        else:
            patient_info = patient_data
    except json.JSONDecodeError:
        logger.error("Could not parse patient data for fallback")
        return json.dumps({"error": "Could not parse patient data"})
    
    prompt = f"""
    You are a clinically responsible AI medical advisor. Generate safe, evidence-based recommendations for this patient.

    PATIENT INFORMATION:
    {json.dumps(patient_info, indent=2)}

    NOTE: Current web-based treatment guidelines are not available, so base recommendations on established medical knowledge.

    INSTRUCTIONS:
    Generate 3-5 comprehensive recommendations that:
    1. Focus on safe, general medical advice
    2. Emphasize the importance of professional medical consultation
    3. Provide practical lifestyle and management tips
    4. Include when to seek immediate medical attention

    For each recommendation:
    - **recommendation**: Clear, specific action
    - **explanation**: Why this is important for their condition
    - **lifestyle_modifications**: Practical daily tips

    OUTPUT FORMAT:
    {{
        "recommendations": [
            {{
                "recommendation": "specific actionable advice",
                "explanation": "why this helps based on medical knowledge",
                "lifestyle_modifications": "practical daily tips"
            }}
        ],
        "data_source": "llm_fallback",
        "disclaimer": "These recommendations are based on general medical knowledge. Current treatment guidelines were not available. Please consult your healthcare provider for the most up-to-date treatment options."
    }}
    """
    
    try:
        logger.debug("Sending fallback prompt to LLM")
        response = llm.call(prompt)
        logger.debug("Received fallback recommendations from LLM")
        
        json_match = re.search(r'({.*})', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            recommendations = json.loads(json_str)
            recommendations["fallback_used"] = True
            recommendations["data_source"] = "llm_fallback"
            
            logger.info(f"Successfully generated {len(recommendations.get('recommendations', []))} fallback recommendations")
            return json.dumps(recommendations, indent=2)
        else:
            logger.error("Could not parse fallback recommendations")
            return json.dumps({
                "recommendations": [
                    {
                        "recommendation": "Consult with your healthcare provider for personalized treatment advice",
                        "explanation": "Professional medical evaluation is essential for proper diagnosis and treatment planning",
                        "lifestyle_modifications": "Follow any existing treatment plans and maintain regular medical appointments"
                    }
                ],
                "data_source": "emergency_fallback",
                "fallback_used": True,
                "disclaimer": "Automated recommendation generation failed. Please seek professional medical advice."
            })
    
    except Exception as e:
        logger.error(f"Error generating fallback recommendations: {str(e)}", exc_info=True)
        return json.dumps({
            "recommendations": [
                {
                    "recommendation": "Please consult with your healthcare provider",
                    "explanation": "Unable to generate automated recommendations due to technical issues",
                    "lifestyle_modifications": "Continue any existing treatment and seek professional medical guidance"
                }
            ],
            "error": str(e),
            "data_source": "error_fallback",
            "fallback_used": True
        })
        
#############################################################################
# AGENT DEFINITIONS
#############################################################################

logger.info("Defining CrewAI agents")

# Validation Agent - Responsible for data validation and extraction
validation_agent = Agent(
    role="Medical Data Validator",
    goal="Ensure all extracted medical data is accurate and valid",
    backstory="""You are a specialized medical data validator with extensive knowledge of 
    medical terminology and standard ranges for clinical measurements. Your job is to carefully 
    review extracted medical data to identify and correct errors, ensuring the highest level 
    of accuracy for downstream processing.""",
    verbose=True,
    allow_delegation=False,
    tools=[validate_medical_values, extract_text_from_image],
    llm=llm
)
logger.debug("Validation Agent defined")

# Formatting Agent - Responsible for structuring data into standardized formats
formatting_agent = Agent(
    role="Medical Data Formatter",
    goal="Structure validated medical data into a standardized JSON format",
    backstory="""You are a data formatting expert specializing in medical information. 
    Your expertise lies in transforming unstructured medical text into well-organized, 
    structured formats that can be easily processed by computer systems. You have a deep
    understanding of medical data schemas and JSON structuring.""",
    verbose=True,
    allow_delegation=False,
    tools=[format_to_json, process_user_feedback],
    llm=llm
)
logger.debug("Formatting Agent defined")

# Enhanced Doctor Agent - NEW with web scraping capabilities
enhanced_doctor_agent = Agent(
    role="AI-Enhanced Medical Advisor with Real-Time Research",
    goal="Provide current, evidence-based recommendations using patient data and latest treatment guidelines from reputable medical sources",
    backstory="""You are an advanced medical advisor who combines patient-specific data 
    with the latest treatment guidelines from reputable medical sources like Mayo Clinic and Medline-Plus. 
    You excel at synthesizing current medical knowledge with real-time treatment protocols to provide 
    the most up-to-date, evidence-based recommendations. You always prioritize patient safety and 
    clearly attribute your sources.""",
    verbose=True,
    allow_delegation=False,
    tools=[
        extract_searchable_conditions,
        scrape_mayo_treatments,
        scrape_medlineplus_treatments,
        generate_web_enhanced_recommendations,
        generate_fallback_recommendations
    ],
    llm=llm
)
logger.debug("Enhanced Doctor Agent defined")

#############################################################################
# TASK DEFINITIONS
#############################################################################

logger.info("Defining CrewAI tasks")

# Task 1: Extract text from PDF
extraction_task = Task(
    description="""
    Extract all text content from the provided medical image using EasyOCR technology.
    Ensure that all relevant sections, including patient information, diagnoses, lab results,
    and recommendations are captured accurately.
    
    Input: {image_path}
    Output: Extracted text content from the image
    """,
    expected_output="The complete extracted text from the medical image",
    agent=validation_agent,
    tools=[extract_text_from_image],
    async_execution=False,
    output_file="extracted/extracted_text.txt"
)
logger.debug("Extraction Task defined")

# Task 2: Validate the extracted data
validation_task = Task(
    description="""
    Carefully review the extracted medical text data and validate all numerical values,
    medical terminology, and content structure. Identify any errors, inconsistencies, or
    missing information. For each issue found, provide a clear explanation and a suggested
    correction.
    
    Input: Extracted text from the PDF
    Output: Validation report with issues identified and corrected text
    
    Focus on:
    1. Numerical data validation (e.g., blood pressure, glucose levels)
    2. Medical terminology verification
    3. Completeness of critical information
    """,
    expected_output="A validation report detailing any issues found and corrections made",
    agent=validation_agent,
    tools=[validate_medical_values],
    async_execution=False,
    context=[extraction_task]
)
logger.debug("Validation Task defined")

# Task 3: Format the validated data into structured JSON
formatting_task = Task(
    description="""
    Transform the validated medical text data into a structured JSON format with these specific sections:
    1. Patient Information
    2. Date of Issue
    3. Type of Report (Heart, Brain, Skin, Bones, etc.)
    4. Medical Problem (technical terms for doctors)
    5. Simplified Explanation of the Medical Problem (for non-experts)
    
    Input: Validated text from the previous task
    Output: JSON structure with the specified sections
    
    Ensure the JSON is properly formatted and all sections are populated with the relevant information.
    """,
    expected_output="Structured JSON representation of the medical data",
    agent=formatting_agent,
    tools=[format_to_json],
    async_execution=False,
    context=[validation_task]
)
logger.debug("Formatting Task defined")

# Task 4: Process user feedback and update the structured data
feedback_task = Task(
    description="""
    Process user feedback to update or correct the structured medical data.
    You have access to both the current structured data and the original extracted text.
    
    Input:
    1. {structured_data}
    2. {user_feedback}
    3. {extracted_text}
    
    Output: Updated JSON structure with the user's modifications applied
    
    The user may want to:
    1. Correct inaccurate information (e.g., wrong age, name, or diagnosis)
    2. Add missing information from the original text
    3. Add new fields like "Blood Pressure" if they exist in the extracted text
    4. Request clarification or simplification of certain sections
    
    For new field requests, search the original extracted text for the requested information.
    Make appropriate changes to the structured data based on the feedback.
    """,
    expected_output="Updated structured JSON with user feedback incorporated",
    agent=formatting_agent,
    tools=[process_user_feedback],
    async_execution=False,
    context=[formatting_task]
)
logger.debug("Feedback Task defined")

# Task 5: Generate recommendations based on the structured data
enhanced_recommendation_task = Task(
    description="""
    Generate comprehensive, up-to-date medical recommendations using a multi-step process:

    1. **EXTRACT CONDITIONS**: From the structured medical data, identify specific searchable medical conditions
    2. **RESEARCH TREATMENTS**: Scrape current treatment guidelines from Mayo Clinic and Medline-Plus for each condition
    3. **ANALYZE & COMBINE**: Integrate web research with patient-specific data using LLM analysis
    4. **GENERATE RECOMMENDATIONS**: Create evidence-based, current recommendations with source attribution
    5. **FALLBACK HANDLING**: If web research fails, use LLM knowledge with clear indication

    Input: {structured_data}
    
    Process Flow:
    - Extract searchable conditions from medical problems
    - Attempt Mayo Clinic scraping (primary source)
    - Attempt Medline-Plus scraping (secondary source)  
    - Generate enhanced recommendations if web data available
    - Use LLM fallback if scraping fails completely

    Focus on:
    - Current treatment protocols and guidelines
    - Patient-specific actionable advice
    - Safety and evidence-based recommendations
    - Clear source attribution
    - Practical lifestyle modifications
    """,
    expected_output="Enhanced recommendations with current medical guidelines, source attribution, and fallback handling",
    agent=enhanced_doctor_agent,
    tools=[
        extract_searchable_conditions,
        scrape_mayo_treatments, 
        scrape_medlineplus_treatments,
        generate_web_enhanced_recommendations,
        generate_fallback_recommendations
    ],
    async_execution=False,
    context=[feedback_task]
)
logger.debug("Recommendation Task defined")

#############################################################################
# CREW DEFINITION AND HELPER FUNCTIONS
#############################################################################

logger.info("Creating CrewAI crew with agents and tasks")

# Create the main crew for processing medical PDFs
medical_image_crew = Crew(
    agents=[validation_agent, formatting_agent],
    tasks=[extraction_task, validation_task, formatting_task],
    verbose=True,
    process=Process.sequential,
    cache=True
)
logger.info("Medical PDF Crew created successfully")

def process_image(image_path: str) -> Dict[str, Any]:
    """
    Process a medical image using the CrewAI workflow
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Processed results including extracted text, structured data, and recommendations
        
    Raises:
        Exception: If there is an error during processing
    """
    logger.info(f"Starting full image processing workflow for: {image_path}")
    
    try:
        # Log the start time for performance measurement
        start_time = datetime.now()
        
        # Run the crew
        result = medical_image_crew.kickoff(inputs={"image_path": image_path})
        
        # Calculate and log processing time
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Image processing completed in {duration:.2f} seconds")
        
        # Format the result if needed
        if isinstance(result, str):
            try:
                result = json.loads(result)
                logger.debug("Successfully parsed JSON result from string")
            except json.JSONDecodeError:
                logger.warning("Could not parse result as JSON, returning as string")
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise

def process_feedback(structured_data: Dict[str, Any], user_feedback: str) -> Dict[str, Any]:
    """
    Process user feedback to update structured data
    
    Args:
        structured_data (dict): The current structured data
        user_feedback (str): User's feedback or correction request
        
    Returns:
        dict: Updated structured data
        
    Raises:
        Exception: If there is an error during feedback processing
    """
    logger.info("Processing user feedback as a separate operation")
    
    try:
        # Get the original extracted text from session state or file
        extracted_text = ""
        if os.path.exists("extracted/extracted_text.txt"):
            with open("extracted/extracted_text.txt", "r") as f:
                extracted_text = f.read()
            logger.debug("Loaded extracted text from file")
        
        # Create a mini-crew for processing feedback
        feedback_crew = Crew(
            agents=[formatting_agent],
            tasks=[feedback_task],
            verbose=True,
            process=Process.sequential
        )
        logger.debug("Feedback crew created")
        
        # Run the crew with extracted text included
        result = feedback_crew.kickoff(
            inputs={
                "structured_data": json.dumps(structured_data),
                "user_feedback" : str(user_feedback),
                "extracted_text": str(extracted_text)
            }
        )
        logger.debug("Feedback crew completed processing")
        
        # Format the result
        if isinstance(result, str):
            try:
                result = json.loads(result)
                logger.debug("Successfully parsed JSON result from string")
            except json.JSONDecodeError:
                logger.warning("Could not parse feedback result as JSON")
                structured_data["feedback_error"] = "Could not process feedback"
                return structured_data
        
        logger.info("Feedback processing completed successfully")
        return result
    
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}", exc_info=True)
        structured_data["feedback_error"] = f"Error processing feedback: {str(e)}"
        return structured_data

def generate_recommendations(structured_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate enhanced medical recommendations with web scraping
    
    Args:
        structured_data (dict): The verified structured data
        
    Returns:
        dict: Enhanced medical recommendations with source attribution
        
    Raises:
        Exception: If there is an error during recommendation generation
    """
    logger.info("Generating enhanced recommendations with web scraping")
    
    try:
        recommendation_crew = Crew(
            agents=[enhanced_doctor_agent],
            tasks=[enhanced_recommendation_task],
            verbose=True,
            process=Process.sequential
        )
        logger.debug("Enhanced recommendation crew created")
        
        result = recommendation_crew.kickoff(
            inputs={"structured_data": json.dumps(structured_data)}
        )
        logger.debug("Enhanced recommendation crew completed processing")
        
        if isinstance(result, str):
            try:
                result = json.loads(result)
                logger.debug("Successfully parsed JSON result from string")
            except json.JSONDecodeError:
                logger.warning("Could not parse enhanced recommendations result as JSON")
                return {"recommendations_error": "Could not generate enhanced recommendations"}
        
        logger.info("Enhanced recommendation generation completed successfully")
        return result
    
    except Exception as e:
        logger.error(f"Error generating enhanced recommendations: {str(e)}", exc_info=True)
        return {"recommendations_error": f"Error generating enhanced recommendations: {str(e)}"}


# If this module is run directly
if __name__ == "__main__":
    logger.info("medical_pdf_processor.py module loaded")

    