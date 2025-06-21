import streamlit as st
import json
from datetime import datetime
import uuid
from UI_api import(
    process_user_feedback_api,
    generate_recommendations_api,
    get_process_results,
    refresh,
    logger
)

from Helper_Function_UI import (
    extract_json,
    safe_parse_string_to_dict,
    upload_image_to_blob
)

def init_session_state():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'upload'
    for var in ['extracted_text', 'structured_data', 'recommendations', 'pdf_path', 'user_feedback']:
        if var not in st.session_state:
            st.session_state[var] = None

def render_upload():
    uploaded_file = st.file_uploader("Choose an Image file", type=["jpeg", "jpg", "png"])
    if uploaded_file:
        blob_name = upload_image_to_blob(uploaded_file)

        logger.info(f"Uploaded {blob_name} to Blob Storage.")

        st.success(f"File uploaded: {uploaded_file.name}")
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.1f} KB",
            "File type": uploaded_file.type
        }
        
        st.write("File Details:")
        for key, value in file_details.items():
            st.write(f"- {key}: {value}")
            
        if st.button("Process PDF"):
            with st.spinner("Processing..."):
                try:
                    logger.debug("Starting full CrewAI processing")
                    result = get_process_results(uploaded_file.name)
                    st.session_state.structured_data = result
                    logger.info("CrewAI processing completed successfully")
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


def render_verification():
    st.subheader("Verify Extracted Information")
    st.write("Please review the extracted information and make any necessary corrections.")

    col1, col2 = st.columns([2, 1])

    structured_data = st.session_state.structured_data
    verified_data = {}
    with col1:
         for key, value in structured_data.copy().items():
            if isinstance(value, str) and ": " in value:
                value = safe_parse_string_to_dict(value)

            if isinstance(value, dict):
                col3, col4 = st.columns([0.01, 0.99])  # Adjust ratio for more/less indentation
                with col4:
                    updated = {}
                    for subkey, subval in value.items():
                        updated[subkey] = st.text_input(f"{subkey}", value=subval)
                    verified_data[key] = updated

            elif value == "":
                continue

            else:
                if len(value) > 80 or "\n" in value:
                    verified_data[key] = st.text_area(key, value=value, height=100)
                else:
                    verified_data[key] = st.text_input(key, value=value)

    with col2:
        st.info("📋 Verification Guidelines")
        st.write("""
        - Check that patient details are correct
        - Verify the date format (YYYY-MM-DD)
        - Ensure the report type accurately reflects the content
        - Technical medical terms should be in the 'Medical Problem' field
        - Make sure the simplified explanation is clear and non-technical
        """)
    
    st.session_state.structured_data = verified_data


def feedback_field(with_recommendation):
    user_feedback = st.text_area(
        "What would you like to modify or add?",
        value=st.session_state.user_feedback,
        placeholder="Example: 'Please add more details about medication dosages' or 'The age in the patient information is incorrect, it should be 45'",
        key="feedback_input"
    )
    
    if st.button("Apply Feedback", help="Process feedback and update results"):
        if not user_feedback.strip():
            st.warning("Please enter your feedback before submitting.")
        else:
            logger.info(f"Processing user feedback: {user_feedback[:50]}...")
            
            with st.spinner("Processing your feedback..."):
                try:
                    # Process feedback using the crew-based approach
                    updated_data = process_user_feedback_api(
                        st.session_state.structured_data,
                        user_feedback
                    )

                    st.session_state.structured_data = updated_data
                    
                    # Update the session state
                    logger.info("Successfully updated data based on feedback")
                    
                    if with_recommendation:
                        # Regenerate recommendations
                        recommendations_result = generate_recommendations_api(updated_data)
                        st.session_state.recommendations = recommendations_result
                        logger.info("Successfully regenerated recommendations")
                    
                    st.session_state.user_feedback = None
                    st.success("Feedback applied successfully!")
                except Exception as e:
                    logger.error(f"Error processing feedback: {str(e)}", exc_info=True)
                    st.error(f"Error processing feedback: {str(e)}")

            if not with_recommendation:
                st.session_state.current_step = 'feedback'        
            st.rerun()


def render_feedback():
    main_fields = {k: v for k, v in st.session_state.structured_data.copy().items() if v != "" and v != {}}

    # Split into two nearly equal parts
    field_items = list(main_fields.items())
    mid_index = (len(field_items) + 1) // 2
    left_fields = field_items[:mid_index]
    right_fields = field_items[mid_index:]

    # Display using columns
    col1, col2 = st.columns(2)

    with col1:
        for field, value in left_fields:
            st.subheader(field)
            if isinstance(value, str) and ": " in value:
                value = safe_parse_string_to_dict(value)
            
            if isinstance(value, dict):
                for subkey, subval in value.items():
                    st.write(f"**{subkey}:** {subval}")
            else:
                st.write(value)

    with col2:
        for field, value in right_fields:
            st.subheader(field)
            if isinstance(value, str) and ": " in value:
                value = safe_parse_string_to_dict(value)            
            
            if isinstance(value, dict):
                for subkey, subval in value.items():
                    st.write(f"**{subkey}:** {subval}")
            else:
                st.write(value)


def render_recommendations():
    st.header("Medical Recommendations")
    recs = st.session_state.recommendations

    if "recommendations" in recs:
        for i, rec in enumerate(recs["recommendations"]):
            with st.container():
                st.subheader(f"Recommendation {i+1}")
                st.markdown(f"**Action:** {rec.get('recommendation', '')}")
                if rec.get('explanation'):
                    st.markdown(f"**Why it's important:** {rec.get('explanation', '')}")
                if rec.get('lifestyle_modifications'):
                    st.markdown(f"**Lifestyle Tip:** {rec.get('lifestyle_modifications', '')}")
                if rec.get('source'):
                    st.markdown(f"**Source:** {rec.get('source', '')}")
                st.markdown("---")
    else:
        st.warning("No recommendations available")
