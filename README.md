# Medical PDF Extraction System

A comprehensive tool for extracting structured data from medical PDFs using OCR, LLMs, and AI agents. This system processes unstructured medical documents into standardized JSON formats and provides patient-friendly recommendations based on the extracted information.

---

## Overview

This project implements an intelligent PDF extraction framework that systematically converts unstructured medical data from clinical reports and notes into a structured format. It uses a multi-agent workflow powered by CrewAI to handle different aspects of the extraction process:

- **Validation Agent**: Ensures data accuracy and validates medical terminology.
- **Formatting Agent**: Structures extracted data into standardized JSON.
- **Doctor Agent**: Provides medical recommendations based on the structured data.

---

## Installation Guide

Follow these steps carefully to ensure proper installation and dependency management:

### 1. Create and Activate a Virtual Environment
```bash
# Create a new virtual environment
python -m venv env

# Activate the environment
# On macOS/Linux:
source env/bin/activate
# On Windows:
env\Scripts\activate
```

### 2. Install CrewAI First (for Compatibility)
```bash
pip install crewai
```

### 3. Install Core Dependencies
```bash
pip install streamlit langchain-openai openai python-dotenv pdf2image pillow
```

### 4. Install Google Generative AI
```bash
pip install google-generativeai
```

### 5. Install NumPy with Specific Version
```bash
pip install numpy==1.26.0
```

### 6. Install EasyOCR
EasyOCR should be installed after NumPy to ensure compatibility:
```bash
pip install easyocr
```

### 7. Install PDF Processing Dependencies
#### For Ubuntu/Debian:
```bash
sudo apt-get install poppler-utils
```
#### For macOS:
```bash
brew install poppler
```
#### For Windows:
1. Download Poppler from [Poppler Windows Releases](https://github.com/oschwartz10612/poppler-windows/releases).
2. Extract to a known location (e.g., `C:\Program Files\poppler`).
3. Add the `bin` directory to your PATH environment variable.

### 8. Install Any Additional Requirements
```bash
pip install torch torchvision opencv-python-headless
```

### 9. Generate Requirements File (Optional)
```bash
pip freeze > requirements.txt
```

---

## Configuration

### API Keys
Create a `.env` file in the project root directory with the following content:
```
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

---

## Project Structure

```
Medical-Extractor/
├── UI.py                       # Streamlit user interface
├── medical_pdf_processor.py    # Core processing logic & CrewAI implementation
├── requirements.txt            # Frozen dependencies
├── .env                        # API keys configuration
└── logs/                       # Log files directory
```

---

## Usage Instructions

### Running the Application
```bash
# Make sure your virtual environment is activated
source env/bin/activate  # macOS/Linux
# OR
env\Scripts\activate     # Windows

# Run the Streamlit application
streamlit run UI.py
```

### Using the Application
1. **Upload PDF**: Upload a medical PDF document through the interface.
2. **Verify Extraction**: Review and edit the automatically extracted information.
3. **Generate Recommendations**: Receive patient-friendly explanations and recommendations.
4. **Provide Feedback**: Request modifications or additional information as needed.

---

## Architecture

The system utilizes a sequential workflow through several agents:

1. **PDF Upload & Processing**: PDF document is processed using EasyOCR.
2. **Data Validation**: Extracted data is validated for medical accuracy.
3. **Structured Formatting**: Validated data is organized into a standardized JSON format.
4. **Recommendations Generation**: Structured data is analyzed to provide useful patient recommendations.
5. **Feedback Loop**: User can provide feedback to refine the extraction and analysis.

---

## Troubleshooting

### NumPy Not Available Error
If you encounter a "NumPy is not available" error:
```bash
pip uninstall -y numpy easyocr
pip install numpy==1.26.0
pip install easyocr
```

### PDF Processing Issues
If PDF extraction fails:
1. Verify Poppler is installed correctly.
2. Check that `pdf2image` can find Poppler:
    ```python
    from pdf2image import convert_from_path
    images = convert_from_path(pdf_path, poppler_path=r'C:\Program Files\poppler\bin')  # Adjust path as needed
    ```

### CrewAI Validation Errors
If you encounter Pydantic validation errors with CrewAI tools:
1. Make sure you're using the latest CrewAI syntax.
2. Define tools using CrewAI's `Tool` class:
    ```python
    from crewai.tools import Tool
    extract_tool = Tool(name="extract_text", func=extract_text_function)
    ```

### Google GenAI Import Error
If you encounter "cannot import name 'genai' from 'google'" error:
```bash
pip install google-generativeai
```
Then update your imports:
```python
import google.generativeai as genai
```

---
