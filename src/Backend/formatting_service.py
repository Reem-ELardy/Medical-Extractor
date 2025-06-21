# formatting_service.py

"""
Formatting Service Module

This module handles data formatting and structuring into standardized JSON format,
as well as processing user feedback to update structured data.
"""

import json
import re
import logging
from crewai.tools import tool
from config import llm

logger = logging.getLogger(__name__)

#############################################################################
# FORMATTING TOOLS
#############################################################################

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