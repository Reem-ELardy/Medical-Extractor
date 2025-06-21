# backend_api.py (FastAPI server)
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import json
import os
import tempfile
import shutil
import logging
from datetime import datetime
from fastapi import Body


# Import your existing medical processor module
from medical_pdf_processor import (
    process_image, 
    process_feedback, 
    generate_recommendations,
    logger,
    extract_json
)

app = FastAPI()

last_message = ""
results_storage = {}

# Pydantic models for request/response
class ProcessImageResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None

class FeedbackRequest(BaseModel):
    structured_data: Dict[str, Any]
    user_feedback: str

class FeedbackResponse(BaseModel):
    success: bool
    updated_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class RecommendationRequest(BaseModel):
    structured_data: Dict[str, Any]

class RecommendationResponse(BaseModel):
    success: bool
    recommendations: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# In-memory storage for temporary files (use Redis/database in production)
temp_storage = {}

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Medical PDF Processing API", 
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/upload-image", response_model=ProcessImageResponse)
async def upload_and_process_image(file: UploadFile = File(...)):
    """
    Upload an image file and process it to extract medical data
    """
    logger.info(f"Received image upload request: {file.filename}")
    
    # Validate file type
    allowed_types = ["image/jpeg", "image/jpg", "image/png"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"File type {file.content_type} not supported. Allowed types: {allowed_types}"
        )
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            # Copy uploaded file content to temporary file
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        logger.info(f"Saved uploaded file to temporary path: {temp_file_path}")
        
        # Process the image using your existing crew
        result = process_image(temp_file_path)

        # Safely extract structured data
        if isinstance(result.raw, str):
            structured_data = extract_json(result.raw)
        elif isinstance(result.raw, dict):
            structured_data = result.raw
        else:
            raise ValueError("Unsupported type for result.raw")
        
        logger.info(structured_data)

        # Clean up temp file
        os.unlink(temp_file_path)

        # Optional: handle edge case if still string
        if isinstance(structured_data, str):
            try:
                structured_data = extract_json(structured_data)
            except json.JSONDecodeError:
                logger.warning("Could not parse structured_data as JSON")
                structured_data = {"raw_result": structured_data}

        # Pass actual dict
        return ProcessImageResponse(
            success=True,
            data=structured_data,
        )
    
    except Exception as e:
        logger.error(f"Error processing uploaded image: {str(e)}", exc_info=True)
        
        # Clean up temporary file if it exists
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        
        return ProcessImageResponse(
            success=False,
            error=str(e)
        )

@app.post("/process-feedback", response_model=FeedbackResponse)
async def process_user_feedback(request: FeedbackRequest):
    """
    Process user feedback to update structured medical data
    """
    logger.info("Received feedback processing request")
    
    try:
        data = process_feedback(
            request.structured_data,
            request.user_feedback
        )

        # Safely extract structured data
        if isinstance(data.raw, str):
            updated_data = extract_json(data.raw)
        elif isinstance(data.raw, dict):
            updated_data = data.raw
        else:
            raise ValueError("Unsupported type for result.raw")
        
        logger.info("Feedback processing completed successfully")
        return FeedbackResponse(
            success=True,
            updated_data=updated_data
        )
    
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}", exc_info=True)
        return FeedbackResponse(
            success=False,
            error=str(e)
        )

@app.post("/generate-recommendations", response_model=RecommendationResponse)
async def get_medical_recommendations(request: RecommendationRequest):
    """
    Generate medical recommendations based on structured data
    """
    logger.info("Received recommendation generation request")
    
    try:
        recommendation = generate_recommendations(request.structured_data)
        
        # Safely extract structured data
        if isinstance(recommendation.raw, str):
            recommendations = extract_json(recommendation.raw)
        elif isinstance(recommendation.raw, dict):
            recommendations = recommendation.raw
        else:
            raise ValueError("Unsupported type for result.raw")

        logger.info("Recommendation generation completed successfully")
        return RecommendationResponse(
            success=True,
            recommendations=recommendations
        )
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}", exc_info=True)
        return RecommendationResponse(
            success=False,
            error=str(e)
        )


# Additional utility endpoints
@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "services": {
            "ocr": "available",
            "llm": "available",
            "crew": "available"
        }
    }

@app.get("/stats")
async def get_stats():
    """Get processing statistics"""
    return {
        "temp_files_count": len(temp_storage),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/save_result")
async def save_result(payload: dict = Body(...)):
    filename = payload.get("filename")
    data = payload.get("data")

    if not filename or not data:
        raise HTTPException(status_code=400, detail="Missing filename or data")

    global results_storage
    results_storage[filename] = data
    logger.info(results_storage)
    return {"success": True}

@app.get("/get_result/{filename}")
async def get_result(filename: str):
    global results_storage
    logger.info(results_storage)
    if filename in results_storage:
        return {"data": results_storage[filename]}
    else:
        raise HTTPException(status_code=404, detail="Result not found")
    
@app.post("/refresh")
def refresh():
    global results_storage
    results_storage = {}
    return {"success": True}
