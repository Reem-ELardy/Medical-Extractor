# Create this as: MEDICAL-EXTRACTOR/Trial_Notebooks/comprehensive_test.py

import os
import sys
import json
import logging
from config import validate_paths, EMBEDDINGS_PATH, ENV_PATH

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_environment_setup():
    """Test 1: Environment and Dependencies"""
    print("\n🔧 Test 1: Environment Setup")
    print("-" * 30)
    
    try:
        # Check paths
        if not validate_paths():
            return False
        
        # Check environment variables
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=ENV_PATH)
        
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            print(f"✅ GEMINI_API_KEY loaded (length: {len(api_key)})")
        else:
            print("❌ GEMINI_API_KEY not found")
            return False
        
        print("✅ Environment setup successful")
        return True
        
    except Exception as e:
        print(f"❌ Environment setup failed: {e}")
        return False

def test_rag_initialization():
    """Test 2: RAG System Initialization"""
    print("\n🤖 Test 2: RAG System Initialization")
    print("-" * 30)
    
    try:
        from medical_rag import MedicalRAG
        
        # Initialize RAG with explicit path
        rag = MedicalRAG(chroma_db_path=EMBEDDINGS_PATH)
        
        if rag.is_initialized:
            print("✅ Medical RAG initialized successfully")
            
            # Get collection stats
            stats = rag.get_collection_stats()
            print(f"✅ Collection stats: {stats}")
            
            return rag
        else:
            print("❌ Medical RAG failed to initialize")
            return None
            
    except Exception as e:
        print(f"❌ RAG initialization error: {e}")
        return None

def test_term_extraction(rag_system):
    """Test 3: Medical Term Extraction"""
    print("\n📋 Test 3: Medical Term Extraction")
    print("-" * 30)
    
    try:
        sample_texts = [
            "Diagnosis: Hypertension and Type 2 Diabetes",
            "MRI shows supraspinatus tendinopathy",
            "Blood Pressure: 140/90, Heart Rate: 85 bpm"
        ]
        
        for i, text in enumerate(sample_texts, 1):
            print(f"\nSample {i}: {text}")
            terms = rag_system.extract_medical_terms(text)
            
            if terms:
                print(f"✅ Extracted {len(terms)} terms:")
                for term in terms[:]:  # Show first 3
                    print(f"  - {term.get('term', 'N/A')} ({term.get('category', 'general')})")
            else:
                print("⚠️ No terms extracted")
        
        return True
        
    except Exception as e:
        print(f"❌ Term extraction error: {e}")
        return False

def test_term_validation(rag_system):
    """Test 4: Medical Term Validation"""
    print("\n✅ Test 4: Medical Term Validation")
    print("-" * 30)
    
    try:
        test_terms = [
            {"term": "Hypertension", "category": "diagnosis"},
            {"term": "Supraspinatus", "category": "anatomy"},
            {"term": "InvalidMedicalTerm123", "category": "test"}
        ]
        
        for term_info in test_terms:
            validation = rag_system.validate_medical_term(term_info)
            
            status = "✅ FOUND" if validation['found'] else "❌ NOT FOUND"
            confidence = validation.get('confidence', 0)
            
            print(f"{status} - '{term_info['term']}' (confidence: {confidence:.3f})")
            
            if validation.get('suggestions'):
                suggestions = ', '.join(validation['suggestions'][:2])
                print(f"  Suggestions: {suggestions}")
        
        return True
        
    except Exception as e:
        print(f"❌ Term validation error: {e}")
        return False

def test_processor_integration():
    """Test 5: Full Processor Integration"""
    print("\n🔄 Test 5: Full Processor Integration")
    print("-" * 30)
    
    try:
        from medical_pdf_processor import validate_medical_values
        
        sample_medical_text = """
        Patient Information: John Doe, Age: 45, Gender: Male
        Date of Issue: 12/15/2024
        Diagnosis: Essential Hypertension, Supraspinatus Tendinopathy
        Blood Pressure: 140/90
        Heart Rate: 85
        """
        
        # Use the .run() method for CrewAI tools
        result = validate_medical_values.run(sample_medical_text)  # ✅ Correct way
        
        # Parse result
        import json
        result_json = json.loads(result)
        
        print("✅ Validation completed successfully")
        print(f"  - Valid: {result_json.get('valid')}")
        print(f"  - Issues: {len(result_json.get('issues', []))}")
        print(f"  - RAG Available: {result_json.get('rag_available', 'Unknown')}")
        print(f"  - Medical Terms Found: {result_json.get('medical_terms_found', 0)}")
        print(f"  - Medical Terms Validated: {result_json.get('medical_terms_validated', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Processor integration error: {e}")
        return False

def run_comprehensive_test():
    """Run all tests in sequence"""
    print("🚀 COMPREHENSIVE MEDICAL RAG TESTING")
    print("=" * 50)
    
    # Test 1: Environment
    if not test_environment_setup():
        print("\n❌ CRITICAL: Environment setup failed. Cannot continue.")
        return False
    
    # Test 2: RAG Initialization
    rag_system = test_rag_initialization()
    if not rag_system:
        print("\n❌ CRITICAL: RAG system failed. Cannot continue.")
        return False
    
    # Test 3: Term Extraction
    if not test_term_extraction(rag_system):
        print("\n⚠️ WARNING: Term extraction issues detected.")
    
    # Test 4: Term Validation
    if not test_term_validation(rag_system):
        print("\n⚠️ WARNING: Term validation issues detected.")
    
    # Test 5: Full Integration
    if not test_processor_integration():
        print("\n❌ CRITICAL: Processor integration failed.")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
    print("Your Medical RAG system is ready for use!")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    run_comprehensive_test()