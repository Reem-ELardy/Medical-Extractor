# recommendation_service.py

"""
Recommendation Service Module

This module handles generating medical recommendations, both web-enhanced
(using scraped data) and fallback recommendations (LLM-only).
"""

import json
import re
import logging
from crewai.tools import tool
from config import llm

logger = logging.getLogger(__name__)

#############################################################################
# RECOMMENDATION TOOLS
#############################################################################

@tool
def generate_web_enhanced_recommendations(patient_data: str, mayo_data: str = "", medlineplus_data: str = "") -> str:
    """
    Generate enhanced medical recommendations combining patient data with scraped treatment guidelines.
    
    Args:
        patient_data (str): Patient's structured medical data
        mayo_data (str): Treatment data scraped from Mayo Clinic
        medlineplus_data (str): Treatment data scraped from Medline-Plus
        
    Returns:
        str: JSON string with enhanced recommendations including source attribution
    """
    logger.info("Generating web-enhanced medical recommendations")
    
    # Parse inputs
    try:
        if isinstance(patient_data, str):
            patient_info = json.loads(patient_data)
        else:
            patient_info = patient_data
    except json.JSONDecodeError:
        logger.error("Could not parse patient data")
        return json.dumps({"error": "Could not parse patient data"})
    
    try:
        mayo_treatments = json.loads(mayo_data) if mayo_data else {}
    except json.JSONDecodeError:
        mayo_treatments = {}
    
    try:
        webmd_treatments = json.loads(medlineplus_data) if medlineplus_data else {}
    except json.JSONDecodeError:
        webmd_treatments = {}
    
    logger.debug(f"Processing patient: {patient_info.get('Patient Information', 'Unknown')}")
    logger.debug(f"Mayo treatments available: {len(mayo_treatments)}")
    logger.debug(f"WebMD treatments available: {len(webmd_treatments)}")
    
    # Create enhanced prompt with web data
    prompt = f"""
    You are a clinically responsible AI medical advisor with access to the latest treatment guidelines. Generate evidence-based, patient-friendly recommendations.

    PATIENT INFORMATION:
    {json.dumps(patient_info, indent=2)}

    CURRENT MAYO CLINIC TREATMENT GUIDELINES:
    {json.dumps(mayo_treatments, indent=2)}

    CURRENT WEBMD TREATMENT PROTOCOLS:
    {json.dumps(webmd_treatments, indent=2)}

    INSTRUCTIONS:
    Generate 2-3 comprehensive recommendations that:

    1. **INTEGRATE CURRENT GUIDELINES**: Use the latest treatment information from Mayo Clinic and WebMD to ensure recommendations are current and evidence-based.

    2. **PERSONALIZE FOR PATIENT**: Tailor recommendations specifically to this patient's conditions and circumstances.

    3. **PRIORITIZE SAFETY**: Focus on safe, non-invasive recommendations that complement professional medical care.

    4. **PROVIDE SOURCE ATTRIBUTION**: Clearly indicate when recommendations are based on current medical guidelines vs. general medical knowledge.

    5. **ENSURE ACTIONABILITY**: Each recommendation should be specific and actionable for the patient.

    For each recommendation, provide:
    - **recommendation**: Clear, specific action the patient should take
    - **explanation**: Why this is important based on their condition and current medical guidelines
    - **lifestyle_modifications**: Practical daily life tips to support the recommendation
    - **source**: Whether based on "Mayo Clinic Guidelines", "Medline-Plus Protocols", "Combined Guidelines", or "General Medical Knowledge"

    FOCUS AREAS (if applicable):
    - Medication adherence and management
    - Lifestyle modifications for condition management
    - When to seek immediate medical attention
    - Preventive care and monitoring
    - Patient education and self-management

    TARGET AUDIENCE: Patient and family members (9th-grade reading level)

    OUTPUT FORMAT:
    {{
        "recommendations": [
            {{
                "recommendation": "specific actionable advice",
                "explanation": "why this helps based on current guidelines and patient condition",
                "lifestyle_modifications": "practical daily tips",
                "source": "Mayo Clinic Guidelines / Medline-Plus Protocols / Combined Guidelines / General Medical Knowledge"
            }}
        ],
        "data_source": "web_enhanced",
        "sources_used": ["Mayo Clinic", "Medline-Plus"],
        "disclaimer": "These recommendations are based on current medical guidelines but should not replace professional medical advice."
    }}
    """
    
    try:
        logger.debug("Sending enhanced prompt to LLM")
        response = llm.call(prompt)
        logger.debug("Received enhanced recommendations from LLM")
        
        # Extract JSON from response
        json_match = re.search(r'({.*})', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            recommendations = json.loads(json_str)
            
            # Add metadata about sources used
            sources_used = []
            if mayo_treatments:
                sources_used.append("Mayo Clinic")
            if webmd_treatments:
                sources_used.append("Medline-Plus")
            
            recommendations["sources_used"] = sources_used
            recommendations["data_source"] = "web_enhanced"
            recommendations["fallback_used"] = False
            
            logger.info(f"Successfully generated {len(recommendations.get('recommendations', []))} web-enhanced recommendations")
            return json.dumps(recommendations, indent=2)
        else:
            logger.warning("No JSON found in enhanced recommendations response")
            return json.dumps({
                "error": "Could not parse enhanced recommendations",
                "fallback_used": True
            })
    
    except Exception as e:
        logger.error(f"Error generating web-enhanced recommendations: {str(e)}", exc_info=True)
        return json.dumps({
            "error": f"Failed to generate enhanced recommendations: {str(e)}",
            "fallback_used": True
        })

@tool
def generate_fallback_recommendations(patient_data: str) -> str:
    """
    Generate standard LLM recommendations when web scraping fails.
    
    Args:
        patient_data (str): Patient's structured medical data
        
    Returns:
        str: JSON string with standard recommendations and fallback indication
    """
    logger.info("Generating fallback recommendations (LLM-only)")
    
    try:
        if isinstance(patient_data, str):
            patient_info = json.dumps(patient_data)
        else:
            patient_info = patient_data
    except json.JSONDecodeError:
        logger.error("Could not parse patient data for fallback")
        return json.dumps({"error": "Could not parse patient data"})
    
    prompt = f"""
    You are a clinically responsible AI medical advisor. Generate safe, evidence-based recommendations for this patient.

    PATIENT INFORMATION:
    {json.dumps(patient_info, indent=2)}

    NOTE: Current web-based treatment guidelines are not available, so base recommendations on established medical knowledge.

    INSTRUCTIONS:
    Generate 3-5 comprehensive recommendations that:
    1. Focus on safe, general medical advice
    2. Emphasize the importance of professional medical consultation
    3. Provide practical lifestyle and management tips
    4. Include when to seek immediate medical attention

    For each recommendation:
    - **recommendation**: Clear, specific action
    - **explanation**: Why this is important for their condition
    - **lifestyle_modifications**: Practical daily tips

    OUTPUT FORMAT:
    {{
        "recommendations": [
            {{
                "recommendation": "specific actionable advice",
                "explanation": "why this helps based on medical knowledge",
                "lifestyle_modifications": "practical daily tips"
            }}
        ],
        "data_source": "llm_fallback",
        "disclaimer": "These recommendations are based on general medical knowledge. Current treatment guidelines were not available. Please consult your healthcare provider for the most up-to-date treatment options."
    }}
    """
    
    try:
        logger.debug("Sending fallback prompt to LLM")
        response = llm.call(prompt)
        logger.debug("Received fallback recommendations from LLM")
        
        json_match = re.search(r'({.*})', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            recommendations = json.loads(json_str)
            recommendations["fallback_used"] = True
            recommendations["data_source"] = "llm_fallback"
            
            logger.info(f"Successfully generated {len(recommendations.get('recommendations', []))} fallback recommendations")
            return json.dumps(recommendations, indent=2)
        else:
            logger.error("Could not parse fallback recommendations")
            return json.dumps({
                "recommendations": [
                    {
                        "recommendation": "Consult with your healthcare provider for personalized treatment advice",
                        "explanation": "Professional medical evaluation is essential for proper diagnosis and treatment planning",
                        "lifestyle_modifications": "Follow any existing treatment plans and maintain regular medical appointments"
                    }
                ],
                "data_source": "emergency_fallback",
                "fallback_used": True,
                "disclaimer": "Automated recommendation generation failed. Please seek professional medical advice."
            })
    
    except Exception as e:
        logger.error(f"Error generating fallback recommendations: {str(e)}", exc_info=True)
        return json.dumps({
            "recommendations": [
                {
                    "recommendation": "Please consult with your healthcare provider",
                    "explanation": "Unable to generate automated recommendations due to technical issues",
                    "lifestyle_modifications": "Continue any existing treatment and seek professional medical guidance"
                }
            ],
            "error": str(e),
            "data_source": "error_fallback",
            "fallback_used": True
        })