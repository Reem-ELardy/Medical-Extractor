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
4. Medical recommendations generation
5. User feedback processing

Dependencies:
- crewai: For agent orchestration
- easyocr: For OCR text extraction
- langchain: For LLM interactions
- OpenAI API: For GPT-4 capabilities
"""

import os
import json
import re
import logging
from typing import Dict, Any, Union, List, Optional
import tempfile
import numpy as np
from datetime import datetime


# Third-party imports
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
# from langchain.tools import tool
# from langchain_groq import ChatGroq
import easyocr
from crewai.tools import tool
from PIL import Image
import google.generativeai as genai
# from google import genai

# Ensure the logs directory exists
os.makedirs('logs', exist_ok=True)
os.makedirs("extracted", exist_ok=True)
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/medical_pdf_processor.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Load environment variables
# Try several possible locations for the .env file
possible_paths = [
    ".env",                         # Current directory
    "../.env",                      # Parent directory
    os.path.expanduser("~/.env")    # Home directory
]

env_loaded = False
for path in possible_paths:
    if os.path.exists(path):
        load_dotenv(dotenv_path=path)
        if os.getenv("GEMINI_API_KEY"):
            logger.info(f"Loaded environment variables from {path}")
            env_loaded = True
            break

if not env_loaded:
    logger.error("Could not find .env file with GEMINI_API_KEY")
    raise FileNotFoundError("No .env file with GEMINI_API_KEY found")
# Check if the API key is set
def initialize_llm():
    return LLM(
        model='gemini/gemini-2.0-flash',
        provider="google",
        api_key=os.getenv("GEMINI_API_KEY"),
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

# def extract_json(response):
#     """Extracts JSON part from the response."""
#     # Regular expression to match valid JSON block
#     json_match = re.search(r'\{.*\}', response, re.DOTALL)
    
#     if json_match:
#         try:
#             # Parse the matched JSON string
#             structured_output = json.loads(json_match.group(0))
#             return structured_output
#         except json.JSONDecodeError as e:
#             print(f"Error decoding JSON: {e}")
#             return None
#     else:
#         print("No valid JSON found in the response.")
#         return None

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

        # Load the image
        # img = Image.open(image_path)
        # img_np = np.array(img)
        
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
    Validate numerical values in medical data to ensure they are within acceptable ranges.
    
    Args:
        data (str): Medical text data containing values to validate
        
    Returns:
        str: JSON string with validation results, issues found, and corrected data
    """
    logger.info("Starting medical data validation")
    
    # Initialize validation results
    validation_results = {
        "valid": True,
        "issues": [],
        "corrected_data": data
    }
    
    try:
        # Example validation for blood pressure
        if "Blood Pressure" in data:
            logger.debug("Checking blood pressure values")
            bp_match = re.search(r"Blood Pressure: (\d+)/(\d+)", data)
            if bp_match:
                systolic = int(bp_match.group(1))
                diastolic = int(bp_match.group(2))
                logger.debug(f"Found blood pressure: {systolic}/{diastolic} mmHg")
                
                # Normal range: Systolic 90-140, Diastolic 60-90
                if systolic > 180 or systolic < 80 or diastolic > 120 or diastolic < 50:
                    validation_results["valid"] = False
                    issue = f"Blood pressure reading {systolic}/{diastolic} outside normal range"
                    validation_results["issues"].append(issue)
                    logger.warning(issue)
        
        # Example validation for glucose level
        if "Glucose" in data:
            logger.debug("Checking glucose values")
            glucose_match = re.search(r"Glucose: (\d+)", data)
            if glucose_match:
                glucose = int(glucose_match.group(1))
                logger.debug(f"Found glucose level: {glucose} mg/dL")
                
                # Normal range: 70-140 mg/dL
                if glucose > 300 or glucose < 40:
                    validation_results["valid"] = False
                    issue = f"Glucose level {glucose} mg/dL outside normal range"
                    validation_results["issues"].append(issue)
                    logger.warning(issue)
        
        # Log validation results
        if validation_results["valid"]:
            logger.info("Validation complete - No issues found")
        else:
            logger.info(f"Validation complete - Found {len(validation_results['issues'])} issues")
            
        return json.dumps(validation_results)
    
    except Exception as e:
        logger.error(f"Error during medical data validation: {str(e)}", exc_info=True)
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
    You are an advanced medical assistant. Transform the following unstructured medical text into a structured JSON format with these sections:
    1. Patient Information
    2. Date of Issue
    3. Type of Report (e.g., CT scan, MRI, virtual colonoscopy) based on the type of procedure mentioned
    4. Medical Problem (using clinical terminology, as a doctor would describe it to another doctor)
    5. Simplified Explanation of the Medical Problem (for non-experts)

    Unstructured Medical Text:
    {extracted_text}

    Output only the following JSON:
    {{
        "Patient Information": "string",
        "Date of Issue": "string",
        "Type of Report": "string",
        "Medical Problem": "string",
        "Simplified Explanation": "string"
    }}
    """
    
    try:
        # Use the LLM to structure the data
        logger.debug("Sending text to LLM for structuring")
        # response = llm.invoke(prompt)
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
def process_user_feedback(structured_data: str, user_feedback: str) -> str:
    """
    Process user feedback to update the structured medical data.
    
    Args:
        structured_data (str): JSON string of the current structured data
        user_feedback (str): User's feedback or correction request
    
    Returns:
        str: Updated JSON string with the modifications
    """
    logger.info("Processing user feedback for structured data")
    logger.debug(f"User feedback: {user_feedback}")
    
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
    
    # Prompt for the LLM to process the feedback
    prompt = f"""
    I have the following structured medical data:
    {json.dumps(data, indent=2)}
    
    The user has provided this feedback or correction:
    "{user_feedback}"
    
    Please update the structured data based on this feedback. The user may be:
    1. Correcting an existing field (e.g., fixing an incorrect age, name, or diagnosis)
    2. Requesting to add a new field that wasn't extracted
    3. Asking to modify how a field is presented
    
    Return only the updated JSON structure with no additional text.
    """
    
    try:
        # Use the LLM to process the feedback
        logger.debug("Sending feedback to LLM for processing")
        # response = llm.invoke(prompt)
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
def generate_medical_recommendations(structured_data: Union[str, dict]) -> str:
    """
    Generate patient-friendly recommendations based on the structured medical data.

    Args:
        structured_data (Dict[str, Any]): Structured medical data as a Python dictionary
        
    Returns:
        str: JSON string with patient-friendly recommendations
    """
    logger.info("Generating medical recommendations based on structured data")
    
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
    
    logger.debug(f"Received data: {data}")

    # Create a prompt for the LLM to generate recommendations
    prompt = f"""
    Based on the following structured medical data, provide patient-friendly recommendations:
    
    Patient Information: {data.get('Patient Information', 'Unknown')}
    Medical Problem: {data.get('Medical Problem', 'Unknown')}
    Simplified Explanation: {data.get('Simplified Explanation', 'Unknown')}
    
    Generate 3-5 practical recommendations that would be helpful for this patient.
    Each recommendation should include:
    1. A clear action item
    2. A brief explanation of why it's important
    3. Any relevant lifestyle modifications
    
    Format the recommendations as a JSON array of objects with "recommendation" and "explanation" fields.
    """
    
    try:
        # Use the LLM to generate recommendations
        logger.debug("Sending request to LLM for recommendations")
        # response = llm.invoke(prompt)
        response = llm.call(prompt) 
        logger.debug("Received response from LLM")
        
        # Extract recommendations from the response
        logger.debug("Attempting to extract JSON from LLM response")
        json_match = re.search(r'({.*}|\[.*\])', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            # Parse and format the recommendations
            recommendations = json.loads(json_str)
            
            # Structure as a recommendations object if it's a plain array
            if isinstance(recommendations, list):
                recommendations = {"recommendations": recommendations}
                logger.debug("Wrapped recommendations array in object")

            logger.info(f"Successfully generated {len(recommendations.get('recommendations', []))} recommendations")
            return json.dumps(recommendations, indent=2)
        else:
            logger.warning("No JSON pattern found in LLM response for recommendations")
            # If no JSON pattern found, return a default response
            default_recs = {"recommendations": [
                {"recommendation": "Please consult with your doctor for personalized advice", 
                 "explanation": "We couldn't generate specific recommendations based on the available information."}
            ]}

            logger.info("Returning default recommendations due to parsing failure")
            return json.dumps(default_recs)
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}", exc_info=True)
        return json.dumps({"error": f"Failed to generate recommendations: {str(e)}"})

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
print("here")

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

# Doctor Agent - Responsible for medical recommendations
doctor_agent = Agent(
    role="Medical Advisor",
    goal="Provide patient-friendly interpretations and recommendations based on medical data",
    backstory="""You are a licensed physician with excellent communication skills and a 
    talent for explaining complex medical concepts in simple terms. Your role is to review 
    structured medical data and generate helpful, actionable advice that patients can 
    understand and follow. You focus on translating medical jargon into practical guidance.""",
    verbose=True,
    allow_delegation=False,
    tools=[generate_medical_recommendations],
    llm=llm
)
logger.debug("Doctor Agent defined")

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
    
    Input:
    1. Current structured JSON data
    2. User feedback or correction request
    
    Output: Updated JSON structure with the user's modifications applied
    
    The user may want to:
    1. Correct inaccurate information (e.g., wrong age, name, or diagnosis)
    2. Add missing information
    3. Request clarification or simplification of certain sections
    
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
recommendation_task = Task(
    description="""
    Based on the structured medical data, provide patient-friendly interpretations and
    actionable recommendations. Your output should:
    
    1. Explain medical terms in simple language
    2. Highlight key findings from the report
    3. Provide practical advice based on the diagnoses and lab results
    4. Suggest lifestyle modifications if appropriate
    
    Input: {structured_data}
    Output: JSON with patient-friendly recommendations
    
    Focus on being helpful, clear, and encouraging in your recommendations.
    """,
    expected_output="Patient-friendly recommendations and explanations",
    agent=doctor_agent,
    tools=[generate_medical_recommendations],
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
    # memory=True,
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
        # Create a mini-crew for processing feedback
        feedback_crew = Crew(
            agents=[formatting_agent],
            tasks=[feedback_task],
            verbose=True,
            process=Process.sequential
        )
        logger.debug("Feedback crew created")
        
        # Run the crew
        result = feedback_crew.kickoff(
            inputs={
                "structured_data": json.dumps(structured_data),
                "user_feedback": user_feedback
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
    Generate medical recommendations based on structured data
    
    Args:
        structured_data (dict): The verified structured data
        
    Returns:
        dict: Medical recommendations
        
    Raises:
        Exception: If there is an error during recommendation generation
    """
    logger.info("Generating recommendations as a separate operation")
    
    try:
        # Create a mini-crew for generating recommendations
        recommendation_crew = Crew(
            agents=[doctor_agent],
            tasks=[recommendation_task],
            verbose=True,
            process=Process.sequential
        )
        logger.debug("Recommendation crew created")
        
        # Run the crew
        result = recommendation_crew.kickoff(
            inputs={"structured_data": json.dumps(structured_data)}
        )
        logger.debug("Recommendation crew completed processing")
        
        # Format the result
        if isinstance(result, str):
            try:
                result = json.loads(result)
                logger.debug("Successfully parsed JSON result from string")
            except json.JSONDecodeError:
                logger.warning("Could not parse recommendations result as JSON")
                return {"recommendations_error": "Could not generate recommendations"}
        
        logger.info("Recommendation generation completed successfully")
        return result
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}", exc_info=True)
        return {"recommendations_error": f"Error generating recommendations: {str(e)}"}

# If this module is run directly
if __name__ == "__main__":
    logger.info("medical_pdf_processor.py module loaded")