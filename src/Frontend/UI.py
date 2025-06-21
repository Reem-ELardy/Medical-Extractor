import streamlit as st
import os
from datetime import datetime

# Import the medical PDF processor functions - using the crew-based approach
from UI_api import(
    Start,
    refresh,
    logger
)

from Helper_Function_UI import (
    cleanup_temp_file,
    generate_recommendation,
    reset_session,
    download_results
)

from UI_Components import (
    render_upload,
    render_verification,
    render_feedback,
    render_recommendations,
    init_session_state,
    feedback_field
)

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

with st.spinner("Starting VM..."):
    Start()

# Application title and description
st.title("Medical PDF Data Extraction System")
st.write("Upload a medical PDF to extract structured information and receive recommendations")

init_session_state()

# Sidebar navigation
with st.sidebar:
    st.header("Navigation")
    st.write("Current step: " + st.session_state.current_step.capitalize())
    
    # Display progress
    step_mapping = {"upload": 0, "verify": 1, "feedback": 2, "recommend": 3}
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
    render_upload()
    
#############################################################################
# VERIFICATION STEP
#############################################################################

elif st.session_state.current_step == 'verify':
    logger.debug("Rendering verification step UI")
    
    render_verification()
    
    # Buttons for navigation
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("← Back to Upload", help="Return to the upload step"):
            logger.info("User chose to go back to upload step")
            st.session_state.current_step = 'upload'
            refresh()
            st.rerun()
    
    with col2:
        if st.button("Generate Recommendations →", help="Proceed to recommendations based on verified data"):
            logger.info("User verified data and requested recommendations")
            
            # Save the updated data
            logger.debug("Updated structured data saved to session state")
            
            # Generate recommendations using the crew-based approach
            with st.spinner("Generating recommendations..."):
                generate_recommendation()
            
            # Move to recommendations step
            logger.info("Moving to recommendations step")
            st.session_state.current_step = 'recommend'
            st.rerun()

    # Feedback mechanism
    st.header("Provide Feedback")
    st.write("Request modifications or additional information to improve the results.")

    feedback_field(False)

#############################################################################
# FEEDBACK LOOP STEP
#############################################################################

elif st.session_state.current_step == 'feedback':
    logger.debug("Rendering recommendations step UI")

    st.header("Medical Information")
    
    render_feedback()
    
    # Actions
    st.subheader("Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Button to start over
        if st.button("Start Over", help="Reset and upload a new PDF"):
            reset_session()
    
    with col2:
        # Download results button
        if st.button("Download Results", help="Save structured data and recommendations as JSON"):
            logger.info("User requested to download results")
            
            download_results()

    with col3:
        if st.button("Generate Recommendations →", help="Proceed to recommendations based on verified data"):
            logger.info("User verified data and requested recommendations")
            
            # Update the structured data with edited values
            updated_data = st.session_state.structured_data
            
            # Save the updated data
            logger.debug("Updated structured data saved to session state")
            
            # Generate recommendations using the crew-based approach
            with st.spinner("Generating recommendations..."):
                generate_recommendation()
            
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
    
    with tab1:
        render_recommendations()
            
        # Disclaimer
        st.info("⚠️ These recommendations are computer-generated and should not replace professional medical advice. Always consult with your healthcare provider before making medical decisions.")
    
    with tab2:
        st.header("Medical Information")
        
        render_feedback()
    
    # Actions
    st.subheader("Actions")
    col1, col2 = st.columns(2)
    
    with col1:
        # Button to start over
        if st.button("Start Over", help="Reset and upload a new PDF"):
            reset_session()
    
    with col2:
        # Download results button
        if st.button("Download Results", help="Save structured data and recommendations as JSON"):
            logger.info("User requested to download results")
            
            download_results()
    
    # Feedback mechanism
    st.header("Provide Feedback")
    st.write("Request modifications or additional information to improve the results.")

    feedback_field(True)

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
