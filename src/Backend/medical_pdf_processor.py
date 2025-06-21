# medical_pdf_processor.py

"""
Medical PDF Processor - Main Entry Point

This module provides the main workflow orchestration and entry points for the
medical PDF processing system. It contains the main functions that external
code uses to interact with the system.

Main Functions:
- process_image(): Process a medical image using the CrewAI workflow
- process_feedback(): Process user feedback to update structured data
- generate_recommendations(): Generate enhanced medical recommendations
"""

import os
import json
import logging
from typing import Dict, Any
from datetime import datetime
from crewai import Crew, Process
from agents import validation_agent, formatting_agent, enhanced_doctor_agent
from tasks import extraction_task, validation_task, formatting_task, feedback_task, enhanced_recommendation_task

logger = logging.getLogger(__name__)

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