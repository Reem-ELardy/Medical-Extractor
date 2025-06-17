# app.py

"""
Medical PDF Data Extraction System - Streamlit Application

This module provides a web-based user interface for the medical PDF extraction system.
It allows users to upload PDFs, extract structured data, review and modify the extracted
information, and receive patient-friendly recommendations.

The application follows a three-step workflow:
1. Upload: User uploads a medical PDF
2. Verify: User reviews and modifies the extracted data
3. Recommend: System generates and displays recommendations

Features:
- PDF upload and processing
- Manual verification and correction of extracted data
- Custom field addition
- Medical recommendations generation
- Feedback mechanism for data refinement
"""

import streamlit as st
import tempfile
import os
import json
import logging
from datetime import datetime
import uuid
import re

# Import the medical PDF processor functions - using the crew-based approach
from medical_pdf_processor import (
    process_image,                # Main crew-based processing function
    process_feedback,          # Feedback processing using formatting agent
    generate_recommendations   # Recommendations using doctor agent
)

# Ensure the logs directory exists
os.makedirs('logs', exist_ok=True)
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/streamlit_app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

#############################################################################
# HELPER FUNCTIONS
#############################################################################

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


#############################################################################
# APPLICATION SETUP
#############################################################################

# Set page configuration
st.set_page_config(
    page_title="Medical PDF Data Extraction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application title and description
st.title("Medical PDF Data Extraction System")
st.write("Upload a medical PDF to extract structured information and receive recommendations")

# Initialize or access session variables
if 'session_id' not in st.session_state:
    st.session_state.session_id = generate_session_id()
    logger.info(f"New session started with ID: {st.session_state.session_id}")

# Initialize session state variables if they don't exist
if 'current_step' not in st.session_state:
    st.session_state.current_step = 'upload'  # Possible values: 'upload', 'verify', 'recommend', 'feedback'
    logger.debug("Initialized current_step to 'upload'")

if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = None

if 'structured_data' not in st.session_state:
    st.session_state.structured_data = None

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None

if 'pdf_path' not in st.session_state:
    st.session_state.pdf_path = None

if "user_feedback" not in st.session_state:
    st.session_state.user_feedback = None

# Sidebar navigation
with st.sidebar:
    st.header("Navigation")
    st.write("Current step: " + st.session_state.current_step.capitalize())
    
    # Display progress
    step_mapping = {"upload": 0, "Processing": 1, "verify": 2, "feedback": 3, "recommend": 4}
    current_step_num = step_mapping.get(st.session_state.current_step, 1)
    
    st.progress(current_step_num / 4)
    
    # Session information
    st.subheader("Session Info")
    st.write(f"Session ID: {st.session_state.session_id[:8]}...")
    st.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

#############################################################################
# UPLOAD STEP
#############################################################################

if st.session_state.current_step == 'upload':
    logger.debug("Rendering upload step UI")
    
    # File uploader - accepting PDFs as the original design intended
    uploaded_file = st.file_uploader("Choose an Image file", type=["jpeg", "jpg", "png"])
    
    if uploaded_file is not None:
        logger.info(f"User uploaded file: {uploaded_file.name}")
        
        # Display file details
        st.success(f"File uploaded: {uploaded_file.name}")
        
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.1f} KB",
            "File type": uploaded_file.type
        }
        
        st.write("File Details:")
        for key, value in file_details.items():
            st.write(f"- {key}: {value}")

        # Process button
        if st.button("Process PDF"):
            logger.info(f"Starting PDF processing for {uploaded_file.name}")
            
            # Show a spinner during processing
            with st.spinner("Processing PDF... This may take several minutes."):
                try:
                    # Save the uploaded file
                    image_path = save_uploaded_file_persistently(uploaded_file)
                    st.session_state.image_path = image_path
                    
                    # Use the CrewAI processing pipeline
                    logger.debug("Starting full CrewAI processing")
                    # The process_pdf function returns a complete result with all needed data
                    st.session_state.current_step = 'Processing'
                    result = process_image(image_path)
                    logger.info("CrewAI processing completed successfully")
                    
                    # Extract the information we need from the result
                    # The structured data should be included in the result
                    if isinstance(result, str):
                        result = json.loads(result)
                        print("here")
                    
                    # Extract text will have been processed by the crew
                    # Store any extracted text if available in the results
                    if "extracted_text" in result:
                        st.session_state.extracted_text = result["extracted_text"]
                    else:
                        # If not explicitly included, use the structured data as a fallback
                        st.session_state.extracted_text = json.dumps(result.model_dump(), indent=2)  

                    st.session_state.structured_data = result
                    # Move to verification step
                    st.session_state.current_step = 'verify'
                    logger.info("Moving to verification step")
                    st.rerun()
                
                except Exception as e:
                    logger.error(f"Error processing the PDF: {str(e)}", exc_info=True)
                    st.error(f"Error processing the PDF: {str(e)}")
                    
                    # Display more detailed error information
                    with st.expander("Error Details"):
                        st.write(str(e))
                        st.write("Please try again with a different PDF file.")

#############################################################################
# VERIFICATION STEP
#############################################################################

elif st.session_state.current_step == 'verify':
    logger.debug("Rendering verification step UI")
    
    st.subheader("Verify Extracted Information")
    st.write("Please review the extracted information and make any necessary corrections.")
    
    # Use columns for better layout
    col1, col2 = st.columns([2, 1])
    structured_data = extract_json(st.session_state.structured_data)
    
    with col1:
        # Display and allow editing of each field
        # patient_info = st.text_area(
        #     "Patient Information", 
        #     value=structured_data["Patient Information"],
        #     help="Including name, age, gender, and ID"
        # )

        patient_info = structured_data["Patient Information"]
        patient_info = dict(item.strip().split(": ", 1) for item in patient_info.split(",") if ": " in item)


        st.subheader("Patient Information")

        col3, col4 = st.columns([0.01, 0.99])  # Adjust ratio for more/less indentation
        with col4:
            name = st.text_input("Name", value=patient_info.get("Name", ""))
            if "Age" in patient_info:
                age = st.text_input("Age", value=patient_info.get("Age", ""))
            if "Gender" in patient_info:
                gender = st.text_input("Gender", value=patient_info.get("Gender", ""))
        
        date_of_issue = st.text_area(
            label = "Date of Issue",
            value=structured_data["Date of Issue"],
            help="When the report was created"
        )
        
        report_type = st.text_area(
            "Type of Report",
            value=structured_data["Type of Report"],
            help="E.g., Heart, Brain, Skin, etc.",
            disabled= True
        )
        
        medical_problem = st.text_area(
            "Medical Problem (Technical)",
            value=structured_data["Medical Problem"],
            height=150,
            help="Technical description using medical terminology",
            disabled= True
        )
        
        simplified_explanation = st.text_area(
            "Simplified Explanation",
            value=structured_data["Simplified Explanation"],
            height=150,
            help="Patient-friendly explanation of the medical problem",
            disabled= True
        )
    
    with col2:
        st.info("📋 Verification Guidelines")
        st.write("""
        - Check that patient details are correct
        - Verify the date format (YYYY-MM-DD)
        - Ensure the report type accurately reflects the content
        - Technical medical terms should be in the 'Medical Problem' field
        - Make sure the simplified explanation is clear and non-technical
        """)
        
        # # Preview of original text if available
        # if st.session_state.extracted_text:
        #     with st.expander("View Original Extracted Text"):
        #         extracted_text = st.session_state.extracted_text
        #         if isinstance(extracted_text, str):
        #             st.text(extracted_text[:2000] + "..." if len(extracted_text) > 2000 else extracted_text)
        #         else:
        #             st.text("Extracted text not available in string format")
    
    # Buttons for navigation
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("← Back to Upload", help="Return to the upload step"):
            logger.info("User chose to go back to upload step")
            st.session_state.current_step = 'upload'
            st.rerun()
    
    with col2:
        if st.button("Generate Recommendations →", help="Proceed to recommendations based on verified data"):
            logger.info("User verified data and requested recommendations")
            
            # Update the structured data with edited values
            updated_data = {
                "Patient Information": patient_info,
                "Date of Issue": date_of_issue,
                "Type of Report": report_type,
                "Medical Problem": medical_problem,
                "Simplified Explanation": simplified_explanation
            }
            
            # Save the updated data
            st.session_state.structured_data = updated_data
            logger.debug("Updated structured data saved to session state")
            
            # Generate recommendations using the crew-based approach
            with st.spinner("Generating recommendations..."):
                try:
                    logger.debug("Starting recommendation generation")
                    recommendations_result = generate_recommendations(updated_data)
                    st.session_state.recommendations = recommendations_result
                    logger.info("Successfully generated recommendations")
                except Exception as e:
                    logger.error(f"Error generating recommendations: {str(e)}", exc_info=True)
                    st.error(f"Error generating recommendations: {str(e)}")
                    st.session_state.recommendations = {"error": "Failed to generate recommendations"}
            
            # Move to recommendations step
            logger.info("Moving to recommendations step")
            st.session_state.current_step = 'recommend'
            st.rerun()

    # Feedback mechanism
    st.header("Provide Feedback")
    st.write("Request modifications or additional information to improve the results.")

    st.session_state.user_feedback = st.text_area(
        "What would you like to modify or add?",
        value=st.session_state.user_feedback,
        placeholder="Example: 'Please add more details about medication dosages' or 'The age in the patient information is incorrect, it should be 45'",
        key="feedback_input"
    )
    
    if st.button("Apply Feedback", help="Process feedback and update results"):
        if not st.session_state.user_feedback.strip():
            st.warning("Please enter your feedback before submitting.")
        else:
            logger.info(f"Processing user feedback: {st.session_state.user_feedback[:50]}...")
            
            with st.spinner("Processing your feedback..."):
                try:
                    verified_data = {
                        "Patient Information": patient_info,
                        "Date of Issue": date_of_issue,
                        "Type of Report": report_type,
                        "Medical Problem": medical_problem,
                        "Simplified Explanation": simplified_explanation
                    }
                    
                    logger.info(f"User Feedback in UI is: **************************************{st.session_state.user_feedback}************************************************")

                    # Process feedback using the crew-based approach
                    updated_data = process_feedback(
                        verified_data,
                        st.session_state.user_feedback
                    )

                    if isinstance(updated_data, str):
                        updated_data = json.loads(updated_data)
                    
                    # Extract text will have been processed by the crew
                    # Store any extracted text if available in the results
                    if "extracted_text" in updated_data:
                        st.session_state.extracted_text = updated_data["extracted_text"]
                    else:
                        # If not explicitly included, use the structured data as a fallback
                        st.session_state.extracted_text = json.dumps(updated_data.model_dump(), indent=2)  

                    # st.session_state.structured_data = updated_data
                    
                    # Update the session state
                    st.session_state.structured_data = extract_json(updated_data)
                    logger.info("Successfully updated data based on feedback")
                    
                    st.session_state.user_feedback = ""
                    st.success("Feedback applied successfully!")
                except Exception as e:
                    logger.error(f"Error processing feedback: {str(e)}", exc_info=True)
                    st.error(f"Error processing feedback: {str(e)}")

            st.session_state.current_step = 'feedback'
            st.rerun()

#############################################################################
# FEEDBACK LOOP STEP
#############################################################################

elif st.session_state.current_step == 'feedback':
    logger.debug("Rendering recommendations step UI")

    st.header("Medical Information")
    
    # Display the verified information
    col1, col2 = st.columns(2)
    
    with col1:
        patient_info = st.session_state.structured_data["Patient Information"]

        st.subheader("Patient Information")

        st.write(f"**Name:** {patient_info.get('Name', 'N/A')}")
        if "Age" in patient_info:
            st.write(f"**Age:** {patient_info.get('Age', 'N/A')}")
        if "Gender" in patient_info:
            st.write(f"**Gender:** {patient_info.get('Gender', 'N/A')}")

        
        st.subheader("Date of Issue")
        st.write(st.session_state.structured_data["Date of Issue"])
        
        st.subheader("Type of Report")
        st.write(st.session_state.structured_data["Type of Report"])
    
    with col2:
        st.subheader("Medical Problem (Technical)")
        st.write(st.session_state.structured_data["Medical Problem"])
        
        st.subheader("Simplified Explanation")
        st.write(st.session_state.structured_data["Simplified Explanation"])
    
    # Display any custom fields
    for key, value in st.session_state.structured_data.items():
        if key not in ["Patient Information", "Date of Issue", "Type of Report", 
                    "Medical Problem", "Simplified Explanation"]:
            st.subheader(key)
            st.write(value)
    
    # Actions
    st.subheader("Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Button to start over
        if st.button("Start Over", help="Reset and upload a new PDF"):
            logger.info("User chose to start over")
            
            # Clean up temp file if it exists
            if st.session_state.pdf_path:
                cleanup_temp_file(st.session_state.pdf_path)
            
            # Reset session state
            st.session_state.current_step = 'upload'
            st.session_state.extracted_text = None
            st.session_state.structured_data = None
            st.session_state.recommendations = None
            st.session_state.pdf_path = None
            st.session_state.user_feedback = None
            
            # Generate a new session ID
            st.session_state.session_id = generate_session_id()
            logger.info(f"Generated new session ID: {st.session_state.session_id}")
            
            st.rerun()
    
    with col2:
        # Download results button
        if st.button("Download Results", help="Save structured data and recommendations as JSON"):
            logger.info("User requested to download results")
            
            # Prepare the complete results
            complete_results = {
                "session_id": st.session_state.session_id,
                "generated_at": datetime.now().isoformat(),
                "medical_data": extract_json(json.dumps(st.session_state.structured_data)),
                "recommendations": extract_json(st.session_state.recommendations)
            }
            
            # Convert to JSON
            results_json = json.dumps(complete_results, indent=2)
            
            # Offer download
            st.download_button(
                label="Download JSON",
                data=results_json,
                file_name=f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    with col3:
        if st.button("Generate Recommendations →", help="Proceed to recommendations based on verified data"):
            logger.info("User verified data and requested recommendations")
            
            # Update the structured data with edited values
            updated_data = extract_json(st.session_state.structured_data)
            updated_data = json.dumps(updated_data, indent=2)
            
            # Save the updated data
            logger.debug("Updated structured data saved to session state")
            
            # Generate recommendations using the crew-based approach
            with st.spinner("Generating recommendations..."):
                try:
                    logger.debug("Starting recommendation generation")
                    recommendations_result = generate_recommendations(updated_data)
                    st.session_state.recommendations = recommendations_result
                    logger.info("Successfully generated recommendations")
                except Exception as e:
                    logger.error(f"Error generating recommendations: {str(e)}", exc_info=True)
                    st.error(f"Error generating recommendations: {str(e)}")
                    st.session_state.recommendations = {"error": "Failed to generate recommendations"}
            
            # Move to recommendations step
            logger.info("Moving to recommendations step")
            st.session_state.current_step = 'recommend'
            st.rerun()

#############################################################################
# RECOMMENDATIONS STEP
#############################################################################

elif st.session_state.current_step == 'recommend':
    logger.debug("Rendering recommendations step UI")
    
    # Use tabs for better organization
    tab1, tab2 = st.tabs(["Recommendations", "Medical Information"])
    # if st.session_state["feedback_input"] != '':
    #     st.session_state["feedback_input"] = ''
    
    with tab1:
        st.header("Medical Recommendations")

        print(st.session_state.recommendations)
        recomendations = extract_json(st.session_state.recommendations)
        print(recomendations)
        
        # Display recommendations
        if recomendations and "recommendations" in recomendations:
            for i, rec in enumerate(recomendations["recommendations"]):
                with st.container():
                    st.subheader(f"Recommendation {i+1}")
                    st.markdown(f"**Action:** {rec.get('recommendation', '')}")
                    st.markdown(f"**Why it's important:** {rec.get('explanation', '')}")
                    st.markdown(f"**Lifestyle Tip:** {rec.get('lifestyle_modifications', '')}")
                    st.markdown("---")
        elif recomendations and "error" in recomendations:
            st.error(f"Error: {recomendations['error']}")
        else:
            st.warning("No recommendations available")
            
        # Disclaimer
        st.info("⚠️ These recommendations are computer-generated and should not replace professional medical advice. Always consult with your healthcare provider before making medical decisions.")
    
    with tab2:
        st.header("Medical Information")
        
        # Display the verified information
        col1, col2 = st.columns(2)
        # st.session_state.structured_data = json.load(st.session_state.structured_data.model_dump(), indent=2)
        
        with col1:
            patient_info = st.session_state.structured_data["Patient Information"]

            st.subheader("Patient Information")
            st.write(f"**Name:** {patient_info.get('Name', 'N/A')}")
            if "Age" in patient_info:
                st.write(f"**Age:** {patient_info.get('Age', 'N/A')}")
            if "Gender" in patient_info:
                st.write(f"**Gender:** {patient_info.get('Gender', 'N/A')}")

            
            st.subheader("Date of Issue")
            st.write(st.session_state.structured_data["Date of Issue"])
            
            st.subheader("Type of Report")
            st.write(st.session_state.structured_data["Type of Report"])
        
        with col2:
            st.subheader("Medical Problem (Technical)")
            st.write(st.session_state.structured_data["Medical Problem"])
            
            st.subheader("Simplified Explanation")
            st.write(st.session_state.structured_data["Simplified Explanation"])
        
        # Display any custom fields
        for key, value in st.session_state.structured_data.items():
            if key not in ["Patient Information", "Date of Issue", "Type of Report", 
                        "Medical Problem", "Simplified Explanation"]:
                st.subheader(key)
                st.write(value)
    
    # Actions
    st.subheader("Actions")
    col1, col2 = st.columns(2)
    
    with col1:
        # Button to start over
        if st.button("Start Over", help="Reset and upload a new PDF"):
            logger.info("User chose to start over")
            
            # Clean up temp file if it exists
            if st.session_state.pdf_path:
                cleanup_temp_file(st.session_state.pdf_path)
            
            # Reset session state
            st.session_state.current_step = 'upload'
            st.session_state.extracted_text = None
            st.session_state.structured_data = None
            st.session_state.recommendations = None
            st.session_state.pdf_path = None
            st.session_state.user_feedback = None
            
            # Generate a new session ID
            st.session_state.session_id = generate_session_id()
            logger.info(f"Generated new session ID: {st.session_state.session_id}")
            
            st.rerun()
    
    with col2:
        # Download results button
        if st.button("Download Results", help="Save structured data and recommendations as JSON"):
            logger.info("User requested to download results")
            
            # Prepare the complete results
            complete_results = {
                "session_id": st.session_state.session_id,
                "generated_at": datetime.now().isoformat(),
                "medical_data": extract_json(json.dumps(st.session_state.structured_data)),
                "recommendations": extract_json(st.session_state.recommendations)
            }
            
            # Convert to JSON
            results_json = json.dumps(complete_results, indent=2)
            
            # Offer download
            st.download_button(
                label="Download JSON",
                data=results_json,
                file_name=f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Feedback mechanism
    st.header("Provide Feedback")
    st.write("Request modifications or additional information to improve the results.")


    st.session_state.user_feedback = st.text_area(
        "What would you like to modify or add?",
        value=st.session_state.user_feedback,
        placeholder="Example: 'Please add more details about medication dosages' or 'The age in the patient information is incorrect, it should be 45'",
        key="feedback_input"
    )
    
    if st.button("Apply Feedback", help="Process feedback and update results"):
        if not st.session_state.user_feedback.strip():
            st.warning("Please enter your feedback before submitting.")
        else:
            logger.info(f"Processing user feedback: {st.session_state.user_feedback[:50]}...")
            
            with st.spinner("Processing your feedback..."):
                try:
                    logger.info(f"User Feedback in UI is: **************************************{st.session_state.user_feedback}************************************************")

                    # Process feedback using the crew-based approach
                    updated_data = process_feedback(
                        st.session_state.structured_data,
                        st.session_state.user_feedback
                    )

                    if isinstance(updated_data, str):
                        updated_data = json.loads(updated_data)
                    
                    # Extract text will have been processed by the crew
                    # Store any extracted text if available in the results
                    if "extracted_text" in updated_data:
                        st.session_state.extracted_text = updated_data["extracted_text"]
                    else:
                        # If not explicitly included, use the structured data as a fallback
                        st.session_state.extracted_text = json.dumps(updated_data.model_dump(), indent=2)  

                    # st.session_state.structured_data = updated_data
                    
                    # Update the session state
                    st.session_state.structured_data = extract_json(updated_data)
                    logger.info("Successfully updated data based on feedback")

                    print(st.session_state.structured_data)

                    data = json.dumps(updated_data.model_dump(), indent=2)
                    
                    # Regenerate recommendations
                    recommendations_result = generate_recommendations(data)
                    st.session_state.recommendations = recommendations_result
                    logger.info("Successfully regenerated recommendations")
                    
                    st.session_state.user_feedback = ""
                    st.success("Feedback applied successfully!")
                    st.rerun()
                except Exception as e:
                    logger.error(f"Error processing feedback: {str(e)}", exc_info=True)
                    st.error(f"Error processing feedback: {str(e)}")

# Cleanup on session end
if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
    try:
        # Register cleanup function to be called when app reruns
        def cleanup():
            cleanup_temp_file(st.session_state.pdf_path)
        
        st.session_state.on_change = cleanup
    except:
        pass  # Fail silently if cleanup registration fails

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 0.8em;">
        Medical PDF Data Extraction System © 2025 | Powered by CrewAI and EasyOCR
    </div>
    """, 
    unsafe_allow_html=True
)