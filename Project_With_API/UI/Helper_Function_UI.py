import streamlit as st
import tempfile
import os
import json
import logging
from datetime import datetime
import uuid
import re
from azure.storage.blob import BlobServiceClient


from UI_api import(
    logger,
    generate_recommendations_api,
    refresh
)

BLOB_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=projinput;AccountKey=RxKwdWCdBEJBasVvkBw3o/zR2jTc5KcXOqs6GZsKcbCwrKLL2SppTGJ3K0rrZoz40huNt+sGKHlb+ASt+F7wWA==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "myfiles"

def upload_image_to_blob(file):
    blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    blob_client = container_client.get_blob_client(file.name)
    blob_client.upload_blob(file, overwrite=True)
    return file.name 

def save_uploaded_file(uploaded_file):
    """
    Save an uploaded file to a temporary location
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        str: Path to the saved temporary file
        
    Raises:
        Exception: If there is an error saving the file
    """
    try:
        # Create a temporary file with the same extension
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        logger.info(f"Successfully saved uploaded file to temporary path: {tmp_path}")
        return tmp_path
    
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}", exc_info=True)
        raise

UPLOAD_DIR = "uploads"

def save_uploaded_file_persistently(uploaded_file):
    """
    Save uploaded file to a persistent directory and return its path.
    """
    try:
        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR)
        
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        logger.info(f"Saved uploaded file to: {file_path}")
        return file_path

    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}", exc_info=True)
        raise


def cleanup_temp_file(file_path):
    """
    Delete a temporary file
    
    Args:
        file_path (str): Path to the file to delete
    """
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.info(f"Successfully deleted temporary file: {file_path}")
    except Exception as e:
        logger.error(f"Error deleting temporary file {file_path}: {str(e)}", exc_info=True)

def generate_session_id():
    """
    Generate a unique session ID for tracking user sessions
    
    Returns:
        str: Unique session ID
    """
    return str(uuid.uuid4())

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


def generate_recommendation():
    updated_data = st.session_state.structured_data
    try:
        logger.debug("Starting recommendation generation")
        recommendations_result = generate_recommendations_api(updated_data)
        st.session_state.recommendations = recommendations_result
        logger.info("Successfully generated recommendations")
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}", exc_info=True)
        st.error(f"Error generating recommendations: {str(e)}")
        st.session_state.recommendations = {"error": "Failed to generate recommendations"}


def download_results():
    results = {
        "session_id": st.session_state.session_id,
        "generated_at": datetime.now().isoformat(),
        "medical_data": st.session_state.structured_data,
        "recommendations": st.session_state.recommendations,
    }
    st.download_button(
        label="Download JSON",
        data=json.dumps(results, indent=2),
        file_name=f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

def reset_session():
    for key in ['structured_data', 'recommendations', 'pdf_path', 'user_feedback']:
        st.session_state[key] = None
    refresh()
    st.session_state.current_step = 'upload'
    st.rerun()

def safe_parse_string_to_dict(value):
    """
    Tries to parse a string into a dictionary using multiple strategies.
    1. Try extract_json()
    2. Fallback to manual parsing with ': '
    Returns a dict if successful, otherwise None.
    """

    # Step 1: Try extract_json
    try:
        json_like = extract_json(value)
        if isinstance(json_like, dict):
            return json_like
    except Exception:
        pass

    # Step 2: Try manually splitting key-value pairs
    try:
        if ": " in value:
            return dict(item.strip().split(": ", 1) for item in value.split(",") if ": " in item)
    except Exception:
        pass

    return None
