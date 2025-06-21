# __init__.py

"""
Medical PDF Processor Package

This package provides functionality for processing medical PDFs and images
to extract structured data and generate medical recommendations.

Main entry points:
- process_image: Extract and structure data from medical images
- process_feedback: Update structured data based on user feedback  
- generate_recommendations: Generate medical recommendations with web research
"""

from .medical_pdf_processor import process_image, process_feedback, generate_recommendations

__version__ = "1.0.0"
__author__ = "Medical PDF Processing Team"

__all__ = [
    "process_image",
    "process_feedback", 
    "generate_recommendations"
]