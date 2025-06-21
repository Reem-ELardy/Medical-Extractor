# validation_service.py

"""
Validation Service Module

This module handles medical data validation including numerical values
and medical terminology validation using RAG integration.
"""

import json
import re
import logging
from crewai.tools import tool
from config import medical_rag

logger = logging.getLogger(__name__)

#############################################################################
# VALIDATION TOOLS
#############################################################################

@tool
def validate_medical_values(data: str) -> str:
    """
    Validate medical numerical values and flag questionable terms (RAG-ready).
    
    Args:
        data (str): Medical text containing clinical values and terminology.
    
    Returns:
        str: JSON with validation status, identified issues, and candidates for RAG review.
    """
    logger.info("Starting medical data validation")

    validation_results = {
        "valid": True,
        "issues": [],
        "corrected_data": data,
        "rag_candidates": []
    }

    try:
        # Blood Pressure
        bp_match = re.search(r"Blood Pressure: (\d+)/(\d+)", data)
        if bp_match:
            systolic = int(bp_match.group(1))
            diastolic = int(bp_match.group(2))
            logger.debug(f"Blood pressure: {systolic}/{diastolic}")
            if systolic > 180 or systolic < 80 or diastolic > 120 or diastolic < 50:
                issue = f"Blood Pressure {systolic}/{diastolic} mmHg outside normal range"
                validation_results["issues"].append(issue)
                validation_results["valid"] = False

        # Glucose
        glucose_match = re.search(r"Glucose: (\d+)", data)
        if glucose_match:
            glucose = int(glucose_match.group(1))
            logger.debug(f"Glucose level: {glucose} mg/dL")
            if glucose < 40 or glucose > 300:
                issue = f"Glucose level {glucose} mg/dL outside expected range"
                validation_results["issues"].append(issue)
                validation_results["valid"] = False

        # Heart Rate
        hr_match = re.search(r"Heart Rate: (\d+)", data)
        if hr_match:
            hr = int(hr_match.group(1))
            if hr < 40 or hr > 150:
                issue = f"Heart rate {hr} bpm is abnormal"
                validation_results["issues"].append(issue)
                validation_results["valid"] = False

        # Temperature
        temp_match = re.search(r"Temperature: ([0-9.]+)", data)
        if temp_match:
            temp = float(temp_match.group(1))
            if temp < 35.0 or temp > 40.0:
                issue = f"Temperature {temp}°C outside normal body temperature range"
                validation_results["issues"].append(issue)
                validation_results["valid"] = False

        # --- Medical Term Validation via RAG ---
        if medical_rag and medical_rag.is_initialized:
            logger.info("Performing UMLS medical term validation")
            logger.debug(f"RAG system available: {medical_rag is not None}")
            logger.debug(f"RAG system initialized: {medical_rag.is_initialized if medical_rag else False}")
            try:
                # Extract and validate medical terms using the global RAG instance
                rag_validation = medical_rag.validate_medical_text(data)
                
                # Add this validation:
                if not isinstance(rag_validation, dict) or 'total_terms' not in rag_validation:
                    logger.error(f"Invalid RAG validation result: {type(rag_validation)}")
                    rag_validation = {
                        "total_terms": 0,
                        "validated_terms": 0,
                        "unknown_terms": 0,
                        "confidence_average": 0.0,
                        "term_validations": []
                    }
                # Add RAG validation results to the main validation
                validation_results["rag_validation"] = rag_validation
                validation_results["medical_terms_found"] = rag_validation["total_terms"]
                validation_results["medical_terms_validated"] = rag_validation["validated_terms"]
                
                # Process individual term validations
                for term_validation in rag_validation["term_validations"]:
                    if not term_validation["found"]:
                        # Add unknown terms to RAG candidates
                        validation_results["rag_candidates"].append({
                            "term": term_validation['term'],
                            "suggestions": term_validation['suggestions'],
                            "confidence": term_validation['confidence'],
                            "category": term_validation.get('category', 'unknown')
                        })
                        
                        # Add to issues if confidence is very low
                        if term_validation['confidence'] < 0.3:
                            issue = f"Unrecognized medical term: '{term_validation['term']}'"
                            if term_validation.get('suggestions'):
                                suggestions = ', '.join(term_validation['suggestions'][:2])
                                issue += f" - Similar terms: {suggestions}"
                            validation_results["issues"].append(issue)
                            validation_results["valid"] = False
                
                # Add overall validation metrics
                validation_results["rag_metrics"] = {
                    "total_terms": rag_validation['total_terms'],
                    "validated_terms": rag_validation['validated_terms'],
                    "unknown_terms": rag_validation['unknown_terms'],
                    "confidence_average": rag_validation['confidence_average'],
                    "semantic_types": rag_validation.get('semantic_types', [])
                }
                
                # Determine overall validation based on RAG results
                if rag_validation["unknown_terms"] > rag_validation["validated_terms"]:
                    validation_results["valid"] = False
                    validation_results["issues"].append(
                        f"More unknown medical terms ({rag_validation['unknown_terms']}) than validated ones ({rag_validation['validated_terms']})"
                    )
                
                logger.info(f"RAG Validation Complete: {rag_validation['validated_terms']}/{rag_validation['total_terms']} terms validated")
                
            except Exception as rag_error:
                logger.error(f"RAG validation error: {str(rag_error)}")
                validation_results["rag_error"] = str(rag_error)
                validation_results["rag_available"] = False
        else:
            logger.warning("UMLS RAG not available, using basic term extraction")
            validation_results["rag_available"] = False

        # You can use regex or a chunker to identify terms to validate
        suspect_terms = re.findall(r"(Diagnosis: .+?)(?:\n|$)", data)
        if suspect_terms:
            validation_results["rag_candidates"].extend([term.strip() for term in suspect_terms])

        # Log result
        if validation_results["valid"]:
            logger.info("Validation complete - No critical issues")
        else:
            logger.info(f"Validation found {len(validation_results['issues'])} issues")

        return json.dumps(validation_results)

    except Exception as e:
        logger.error(f"Validation error: {str(e)}", exc_info=True)
        validation_results["valid"] = False
        validation_results["issues"].append(f"Validation error: {str(e)}")
        return json.dumps(validation_results)