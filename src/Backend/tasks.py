# tasks.py

"""
CrewAI Tasks Module

This module defines all the tasks used in the medical PDF processing workflow.
"""

import logging
from crewai import Task
from agents import validation_agent, formatting_agent, enhanced_doctor_agent
import json
from ocr_service import extract_text_from_image
from validation_service import validate_medical_values
from formatting_service import format_to_json, process_user_feedback
from web_scraping_service import extract_searchable_conditions, scrape_mayo_treatments, scrape_medlineplus_treatments
from recommendation_service import generate_web_enhanced_recommendations, generate_fallback_recommendations

logger = logging.getLogger(__name__)

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

enhanced_recommendation_task = Task(
    description="""
    Generate comprehensive, up-to-date medical recommendations using a multi-step process:

    1. **EXTRACT CONDITIONS**: From the structured medical data, identify specific searchable medical conditions
    2. **RESEARCH TREATMENTS**: Scrape current treatment guidelines from Mayo Clinic and WebMD for each condition
    3. **ANALYZE & COMBINE**: Integrate web research with patient-specific data using LLM analysis
    4. **GENERATE RECOMMENDATIONS**: Create evidence-based, current recommendations with source attribution
    5. **FALLBACK HANDLING**: If web research fails, use LLM knowledge with clear indication

    Input: {structured_data}
    
    Process Flow:
    - Extract searchable conditions from medical problems
    - Attempt Mayo Clinic scraping (primary source)
    - Attempt WebMD scraping (secondary source)  
    - Generate enhanced recommendations if web data available
    - Use LLM fallback if scraping fails completely

    Focus on:
    - Current treatment protocols and guidelines
    - Patient-specific actionable advice
    - Safety and evidence-based recommendations
    - Clear source attribution
    - Practical lifestyle modifications

    ERROR HANDLING:
    - Validate all inputs using enhanced parsing
    - Implement retry logic for transient failures
    - Provide graceful fallbacks for each step
    - Always return structured, usable output
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
    async_execution=False
)
logger.debug("Recommendation Task defined")