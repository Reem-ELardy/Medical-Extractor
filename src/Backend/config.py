# config.py

"""
Configuration and Environment Setup Module

This module handles environment variable loading, LLM initialization,
EasyOCR setup, Medical RAG initialization, and logging configuration.
"""

import os
import logging
import easyocr
from datetime import datetime
from dotenv import load_dotenv
from crewai import LLM

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

try:
    from medical_rag import MedicalRAG  # Import from same directory
    logger.info("Medical RAG import successful")
except ImportError as e:
    logger.error(f"Failed to import Medical RAG: {e}")
    logger.warning("RAG functionality will not be available")
    MedicalRAG = None
    
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