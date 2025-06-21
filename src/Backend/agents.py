# agents.py

"""
CrewAI Agents Module

This module defines all the specialized agents used in the medical PDF processing workflow.
"""

import logging
from crewai import Agent
from config import llm
from ocr_service import extract_text_from_image
from validation_service import validate_medical_values
from formatting_service import format_to_json, process_user_feedback
from web_scraping_service import extract_searchable_conditions, scrape_mayo_treatments, scrape_medlineplus_treatments
from recommendation_service import generate_web_enhanced_recommendations, generate_fallback_recommendations

logger = logging.getLogger(__name__)

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