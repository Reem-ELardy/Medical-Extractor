import streamlit as st
import requests
import time
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UI_logger")

API_URL = "http://20.250.164.42:8080"
# API_URL = "http://localhost:8000"
START_VM_URL = "https://trigerringvm.azurewebsites.net/api/TriggerToStartVM?code=s9zPCxHKg1MSm6PpLfQJfJlPAg80iEPgyRCEUXNtKSrzAzFuT7U0Xg=="


def start_vm():
    try:
        response = requests.get(START_VM_URL)
        try:
            return response.json().get("message", "No message received")
        except ValueError:
            return response.text or "No response content"
    except Exception as e:
        return f"Failed to trigger VM start: {e}"
    
def check_server_health():
    """
    Check if FastAPI server is running
    """
    try:
        response = call_fastapi_endpoint("health", method="GET")
        return response.get("status") == "healthy"
    except:
        return False
    
def Start():
    if check_server_health():
        st.success("FastAPI server is already running. No need to start VM.")
    else:
        start_msg = start_vm()
        st.success(start_msg)


def call_fastapi_endpoint(endpoint: str, method: str = "POST", **kwargs):
    """
    Generic function to call FastAPI endpoints
    """
    url = f"{API_URL}/{endpoint.lstrip('/')}"
    
    try:
        if method.upper() == "POST":
            response = requests.post(url, **kwargs, timeout=300)  # 5 minute timeout
        elif method.upper() == "GET":
            response = requests.get(url, **kwargs, timeout=60)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling FastAPI endpoint {endpoint}: {str(e)}")
        raise Exception(f"API call failed: {str(e)}")

def upload_and_process_image(uploaded_file):
    """
    Upload file to FastAPI server and process it
    """
    logger.info(f"Uploading and processing file: {uploaded_file.name}")
    
    # Prepare file for upload
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    
    # Call FastAPI endpoint
    response = call_fastapi_endpoint("upload-image", files=files)
    
    if not response.get("success", False):
        raise Exception(response.get("error", "Unknown error occurred"))
    
    return response.get("data")


def get_process_results(blob_name):
    while True:
        try:
            res = requests.get(f"{API_URL}/get_result/{blob_name}")
            if res.status_code == 200:
                print(res.json().get("data"))
                print(res.json)
                return res.json().get("data")
            else:
                # Optionally log the 404 response
                print(f"Waiting for result... (Status: {res.status_code})")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching result: {e}")
        
        time.sleep(10)  # Wait 2 seconds before retrying


def process_user_feedback_api(structured_data: Dict[str, Any], user_feedback: str):
    """
    Send feedback to FastAPI server for processing
    """
    logger.info("Sending feedback to FastAPI server")
    
    payload = {
        "structured_data": structured_data,
        "user_feedback": user_feedback
    }
    
    response = call_fastapi_endpoint("process-feedback", json=payload)
    
    if not response.get("success", False):
        raise Exception(response.get("error", "Unknown error occurred"))
    
    return response.get("updated_data")

def generate_recommendations_api(structured_data: Dict[str, Any]):
    """
    Generate recommendations via FastAPI server
    """
    logger.info("Requesting recommendations from FastAPI server")
    
    payload = {"structured_data": structured_data}
    
    response = call_fastapi_endpoint("generate-recommendations", json=payload)
    
    if not response.get("success", False):
        raise Exception(response.get("error", "Unknown error occurred"))
    
    return response.get("recommendations")


def refresh():
    response = call_fastapi_endpoint("refresh")
    return response

