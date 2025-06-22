from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from config import logger

class AgentExecutionResult(BaseModel):
    """Structured result from agent execution"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    retry_count: int = 0

# class ConditionExtractionInput(BaseModel):
#     """Structured input for condition extraction tool"""
#     medical_problem: str = Field(
#         ..., 
#         description="Raw medical report text to extract conditions from",
#         min_length=10,
#         max_length=10000
#     )
#     max_conditions: int = Field(
#         default=5,
#         description="Maximum number of conditions to extract",
#         ge=1,
#         le=10
#     )
    
#     @validator('medical_problem')
#     def validate_medical_text(cls, v):
#         if not v or not v.strip():
#             raise ValueError("Medical problem text cannot be empty")
#         return v.strip()

# class ScrapingInput(BaseModel):
#     """Structured input for web scraping tools"""
#     conditions: List[str] = Field(
#         ...,
#         description="List of medical conditions to search for",
#         min_items=1,
#         max_items=10
#     )
#     timeout: int = Field(
#         default=10,
#         description="Request timeout in seconds",
#         ge=5,
#         le=30
#     )
#     max_retries: int = Field(
#         default=3,
#         description="Maximum number of retry attempts",
#         ge=1,
#         le=5
#     )
    
#     @validator('conditions')
#     def validate_conditions(cls, v):
#         if not v:
#             raise ValueError("At least one condition must be provided")
#         # Clean and validate each condition
#         cleaned = []
#         for condition in v:
#             if condition and condition.strip():
#                 cleaned.append(condition.strip()[:100])  # Limit length
#         if not cleaned:
#             raise ValueError("No valid conditions provided")
#         return cleaned

# class RecommendationInput(BaseModel):
#     """Structured input for recommendation generation"""
#     patient_data: Dict[str, Any] = Field(
#         ...,
#         description="Structured patient medical data"
#     )
#     mayo_data: Optional[str] = Field(
#         default="",
#         description="Mayo Clinic treatment data (JSON string)"
#     )
#     medlineplus_data: Optional[str] = Field(
#         default="",
#         description="WebMD treatment data (JSON string)"
#     )
#     recommendation_count: int = Field(
#         default=3,
#         description="Number of recommendations to generate",
#         ge=1,
#         le=10
#     )
    
#     @validator('patient_data')
#     def validate_patient_data(cls, v):
#         if not v:
#             raise ValueError("Patient data cannot be empty")
#         # Ensure required fields exist
#         required_fields = ['Patient Information', 'Primary Concern']
#         for field in required_fields:
#             if field not in v:
#                 logger.warning(f"Missing recommended field: {field}")
#         return v