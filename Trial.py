import streamlit as st
import nbformat
from pathlib import Path
from io import BytesIO
import numpy as np
from PIL import Image
import json
import pandas as pd

# Function to load the notebook and extract necessary function definitions
def load_notebook_functions(notebook_path):
    """Load and run code from the Jupyter notebook to expose functions."""
    with open(notebook_path) as f:
        notebook_content = nbformat.read(f, as_version=4)

    # Execute each code cell in the notebook
    for cell in notebook_content.cells:
        if cell.cell_type == 'code':
            exec(cell.source, globals())

# Load the notebook functions (ensure your notebook file is in the same directory)
notebook_path = Path("WorkFlowWith.ipynb")
load_notebook_functions(notebook_path)

# Streamlit UI
st.title("Medical Report Processor with LLM")
tab1, tab2 = st.tabs(["Upload and Process", "View LLM Output"])
# tab1 = st.tabs(["Upload and Process"])


# Store extracted text and structured data for later use in the second tab
extracted_text = ""

# Initialize variables
llm_output = None
llm_response = None


with tab1:
    st.header("Upload a Medical Report")
    uploaded_file = st.file_uploader("Choose an image", type=["jpeg", "jpg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert image to NumPy array (to pass to the OCR function)
        image_cv = np.array(image)

        # Call OCR function
        extracted_text = perform_ocr(image_cv)

        if extracted_text:
            # Call LLM function
            llm_response = perform_llm(extracted_text)

            if llm_response:
                try:
                    # Try to extract JSON from the LLM response
                    llm_output = extract_json(llm_response)
                    st.success("LLM processing complete! You can now navigate to the 'LLM Output' tab.")
                except json.JSONDecodeError as e:
                    st.error("Failed to parse LLM response. Please check the LLM output.")
                    st.text_area("Raw LLM Response", llm_response, height=300)
            else:
                st.error("Failed to get a response from the LLM.")

with tab2:
    if llm_output:
        st.header("LLM Output")
        st.subheader("Structured Data")

        # Display Patient Information in a Table
        if "Patient Information" in llm_output:
            st.write("### Patient Information")
            st.info(llm_output["Patient Information"])

        # Display Date of Issue
        if "Date of Issue" in llm_output:
            st.write("### Date of Issue")
            st.info(llm_output["Date of Issue"])

        # Display Type of Report
        if "Type of Report" in llm_output:
            st.write("### Type of Report")
            st.success(llm_output["Type of Report"])

        # Display Medical Problem (if exists)
        if "Medical Problem" in llm_output:
            st.write("### Medical Problem (Technical)")
            st.error(llm_output["Medical Problem"])

        # Display Simplified Explanation (if exists)
        if "Simplified Explanation" in llm_output:
            st.write("### Simplified Explanation (For Non-Experts)")
            st.warning(llm_output["Simplified Explanation"])
    else:
        st.warning("No LLM output available. Please process a medical report in the 'Upload and Process' tab first.")
