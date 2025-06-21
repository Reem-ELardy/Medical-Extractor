"""
Medical RAG System Integration
Connects UMLS knowledge base with medical validation
"""

import chromadb
import json
import logging
import re
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Load .env from parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(parent_dir, ".env")
load_dotenv(dotenv_path=env_path)

# Define embeddings path
DEFAULT_EMBEDDINGS_PATH = os.path.join(parent_dir, "embeddings")
logger = logging.getLogger(__name__)

class MedicalRAG:
    def __init__(self, chroma_db_path: str = None):
        """Initialize the Medical RAG system"""
        # Use configured path if none provided
        if chroma_db_path is None:
            chroma_db_path = DEFAULT_EMBEDDINGS_PATH
        
        self.chroma_db_path = chroma_db_path
        self.client = None
        self.collection = None
        self.encoder = None
        self.is_initialized = False
        
        # Common medical term variations with semantic mappings
        self.term_variations = {
            'heart attack': {
                'variants': ['myocardial infarction', 'mi', 'acute coronary syndrome', 'ami', 'stemi', 'nstemi'],
                'semantic_type': 'Disease or Syndrome',
                'confidence_boost': 0.1
            },
            'high blood pressure': {
                'variants': ['hypertension', 'htn', 'elevated blood pressure', 'essential hypertension'],
                'semantic_type': 'Disease or Syndrome',
                'confidence_boost': 0.1
            },
            'diabetes': {
                'variants': ['diabetes mellitus', 'dm', 'type 2 diabetes', 'type 1 diabetes', 't2dm', 't1dm'],
                'semantic_type': 'Disease or Syndrome',
                'confidence_boost': 0.1
            },
            'mri': {
                'variants': ['magnetic resonance imaging', 'm.r.i', 'm.r.i.', 'mr imaging'],
                'semantic_type': 'Diagnostic Procedure',
                'confidence_boost': 0.05
            },
            'axial': {
                'variants': ['ax', 'axial plane', 'transverse', 'transverse plane'],
                'semantic_type': 'Spatial Concept',
                'confidence_boost': 0.05
            },
            'sagittal': {
                'variants': ['sag', 'sagittal plane', 'median plane'],
                'semantic_type': 'Spatial Concept',
                'confidence_boost': 0.05
            },
            'coronal': {
                'variants': ['cor', 'coronal plane', 'frontal', 'frontal plane'],
                'semantic_type': 'Spatial Concept',
                'confidence_boost': 0.05
            }
        }
        
        # Context-based confidence thresholds with semantic type consideration
        self.context_thresholds = {
            'imaging': {
                'base': 0.25,
                'semantic_types': {
                    'Diagnostic Procedure': 0.2,
                    'Spatial Concept': 0.2,
                    'Body Location or Region': 0.25
                }
            },
            'anatomy': {
                'base': 0.3,
                'semantic_types': {
                    'Body Location or Region': 0.25,
                    'Body Part, Organ, or Organ Component': 0.25,
                    'Tissue': 0.3
                }
            },
            'pathology': {
                'base': 0.35,
                'semantic_types': {
                    'Disease or Syndrome': 0.3,
                    'Finding': 0.3,
                    'Sign or Symptom': 0.35
                }
            }
        }
        
        try:
            self._initialize_components()
            logger.info("Medical RAG system initialized successfully")
            self.is_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize Medical RAG: {e}")
            self.is_initialized = False

    def _initialize_components(self):
        """Initialize ChromaDB and encoder"""
        try:
            # Check if ChromaDB directory exists
            if not os.path.exists(self.chroma_db_path):
                logger.error(f"ChromaDB directory not found: {self.chroma_db_path}")
                raise FileNotFoundError(f"ChromaDB directory not found: {self.chroma_db_path}")
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=self.chroma_db_path)
            
            # List existing collections and log them
            existing_collections = self.client.list_collections()
            collection_names = [c.name for c in existing_collections]
            logger.info(f"Available collections: {collection_names}")
            
            # Try to get the first available collection (since you likely have only one)
            if existing_collections:
                self.collection = existing_collections[0]  # Get first collection
                logger.info(f"Connected to collection: {self.collection.name}")
            else:
                raise Exception("No collections found in ChromaDB")
            
            # Initialize sentence transformer
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Initialized sentence transformer")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise

    def _initialize_basic_terms(self):
        """Initialize the collection with basic medical terms"""
        basic_terms = [
            {
                'term': 'myocardial infarction',
                'metadata': {
                    'semantic_type': 'Disease or Syndrome',
                    'confidence_boost': 0.1,
                    'category': 'pathology'
                }
            },
            {
                'term': 'hypertension',
                'metadata': {
                    'semantic_type': 'Disease or Syndrome',
                    'confidence_boost': 0.1,
                    'category': 'pathology'
                }
            },
            {
                'term': 'diabetes mellitus',
                'metadata': {
                    'semantic_type': 'Disease or Syndrome',
                    'confidence_boost': 0.1,
                    'category': 'pathology'
                }
            },
            {
                'term': 'magnetic resonance imaging',
                'metadata': {
                    'semantic_type': 'Diagnostic Procedure',
                    'confidence_boost': 0.05,
                    'category': 'imaging'
                }
            },
            {
                'term': 'axial plane',
                'metadata': {
                    'semantic_type': 'Spatial Concept',
                    'confidence_boost': 0.05,
                    'category': 'anatomy'
                }
            },
            {
                'term': 'sagittal plane',
                'metadata': {
                    'semantic_type': 'Spatial Concept',
                    'confidence_boost': 0.05,
                    'category': 'anatomy'
                }
            },
            {
                'term': 'coronal plane',
                'metadata': {
                    'semantic_type': 'Spatial Concept',
                    'confidence_boost': 0.05,
                    'category': 'anatomy'
                }
            }
        ]
        
        # Add terms to collection
        for term_info in basic_terms:
            try:
                self.collection.add(
                    documents=[term_info['term']],
                    metadatas=[term_info['metadata']],
                    ids=[f"term_{len(self.collection.get()['ids']) + 1}"]
                )
            except Exception as e:
                logger.error(f"Error adding term '{term_info['term']}': {e}")
        
        logger.info(f"Initialized collection with {len(basic_terms)} basic medical terms")

    def normalize_medical_term(self, term: str) -> Tuple[str, Dict[str, Any]]:
        """Normalize medical terms to standard form and return metadata"""
        if not term or not isinstance(term, str):
            return '', {'semantic_type': None, 'confidence_boost': 0.0}
        
        # Basic cleaning
        term = term.lower().strip()
        
        # Remove special characters but preserve hyphens and spaces
        term = re.sub(r'[^\w\s-]', '', term)
        
        # Handle common OCR errors
        ocr_corrections = {
            'fatient': 'patient',
            'medicai': 'medical',
            'diagosis': 'diagnosis',
            'symtoms': 'symptoms',
            'prescibed': 'prescribed'
        }
        term = ocr_corrections.get(term, term)
        
        # Initialize metadata
        metadata = {
            'semantic_type': None,
            'confidence_boost': 0.0,
            'original_term': term
        }
        
        # Check for variations
        for standard, info in self.term_variations.items():
            if term in info['variants'] or term == standard:
                metadata['semantic_type'] = info['semantic_type']
                metadata['confidence_boost'] = info['confidence_boost']
                return standard, metadata
        
        # Handle compound terms
        if ' ' in term:
            parts = term.split()
            # Check if any part matches a standard term
            for part in parts:
                for standard, info in self.term_variations.items():
                    if part in info['variants'] or part == standard:
                        metadata['semantic_type'] = info['semantic_type']
                        metadata['confidence_boost'] = info['confidence_boost'] * 0.8  # Reduced boost for partial matches
                        return standard, metadata
        
        # Handle common medical suffixes
        suffixes = {
            'itis': 'inflammation',
            'osis': 'condition',
            'pathy': 'disease',
            'ectomy': 'removal',
            'scopy': 'examination',
            'graphy': 'imaging'
        }
        
        for suffix, meaning in suffixes.items():
            if term.endswith(suffix):
                base_term = term[:-len(suffix)]
                for standard, info in self.term_variations.items():
                    if base_term in info['variants'] or base_term == standard:
                        metadata['semantic_type'] = info['semantic_type']
                        metadata['confidence_boost'] = info['confidence_boost'] * 0.9
                        return standard, metadata
        
        return term, metadata

    def get_context_threshold(self, term: str, context: str, semantic_type: str = None) -> float:
        """Get appropriate confidence threshold based on term context and semantic type"""
        if not context:
            return 0.3  # Default threshold for no context
        
        context = context.lower()
        term = term.lower()
        
        # Define context keywords with weights
        context_keywords = {
            'imaging': {
                'primary': ['mri', 'ct', 'scan', 'imaging', 'radiology', 'x-ray', 'ultrasound'],
                'secondary': ['findings', 'results', 'report', 'study', 'examination'],
                'weight': 1.0
            },
            'anatomy': {
                'primary': ['anatomy', 'structure', 'organ', 'tissue', 'region', 'area'],
                'secondary': ['location', 'position', 'site', 'part'],
                'weight': 1.0
            },
            'pathology': {
                'primary': ['disease', 'condition', 'diagnosis', 'pathology', 'disorder'],
                'secondary': ['symptoms', 'signs', 'complaints', 'problems'],
                'weight': 1.0
            }
        }
        
        # Calculate context scores
        context_scores = {}
        for category, info in context_keywords.items():
            primary_matches = sum(1 for word in info['primary'] if word in context)
            secondary_matches = sum(1 for word in info['secondary'] if word in context)
            context_scores[category] = (primary_matches * 2 + secondary_matches) * info['weight']
        
        # Determine dominant context
        if context_scores:
            dominant_context = max(context_scores.items(), key=lambda x: x[1])[0]
            context_strength = context_scores[dominant_context]
        else:
            return 0.3  # Default threshold if no context matches
        
        # Get base threshold for context
        base_threshold = self.context_thresholds[dominant_context]['base']
        
        # Adjust threshold based on context strength
        if context_strength > 3:
            base_threshold *= 0.9  # Lower threshold for strong context
        elif context_strength < 2:
            base_threshold *= 1.1  # Higher threshold for weak context
        
        # Adjust threshold based on semantic type if available
        if semantic_type and semantic_type in self.context_thresholds[dominant_context]['semantic_types']:
            type_threshold = self.context_thresholds[dominant_context]['semantic_types'][semantic_type]
            # Weight the threshold based on context strength
            return (type_threshold * 0.7 + base_threshold * 0.3) if context_strength > 2 else type_threshold
        
        # Ensure threshold is within reasonable bounds
        return max(0.1, min(0.5, base_threshold))

    def extract_medical_terms(self, text: str) -> List[Dict[str, Any]]:
        """Extract potential medical terms from text with metadata"""
        medical_terms = []
        
        # Enhanced patterns for medical reports
        patterns = {
            # Diagnoses and findings
            'diagnoses': r'(?:Diagnosis|Dx|Impression|Condition|OPINION):\s*([^.\n:]+)',
            'findings': r'(?:FINDINGS|Finding):\s*([^:]+?)(?=OPINION|$)',
            
            # Anatomical structures
            'anatomy': r'\b(supraspinatus|infraspinatus|subscapularis|teres|glenoid|labrum|acromio-clavicular|glenohumeral|biceps|tendon|joint|bone|muscle|ligament|cartilage|meniscus|rotator\s+cuff)\b',
            
            # Pathological conditions
            'pathology': r'\b(tendinopathy|tendinitis|tear|rupture|effusion|infiltrative|thickened|elevated|atrophy|hypertrophy|degeneration|inflammation|impingement|bursitis)\b',
            
            # Imaging findings descriptors
            'imaging_descriptors': r'\b(signal|intensity|enhancement|consolidation|edema|hemorrhage|mass|lesion|nodule|cyst)\b',
            
            # Procedures and imaging
            'procedures': r'\b(MRI|CT|X-ray|ultrasound|arthrography|arthroscopy|biopsy|scan)\b',
            
            # Medical measurements and values
            'measurements': r'\b\d+(?:\.\d+)?\s*(?:mm|cm|ml|mL|degrees?|%)\b',
            
            # Common medical prefixes/suffixes
            'medical_terms': r'\b\w*(?:itis|osis|pathy|ectomy|scopy|graphy|plasty)\b',
            
            # Specific MRI sequences and technical terms
            'mri_technical': r'\b(T1|T2|STIR|FLAIR|sagittal|coronal|axial|WI|weighted|sequence)\b',
            
            # Additional patterns for better coverage
            'imaging_planes': r'\b(axial|sagittal|coronal|transverse)\b',
            'mri_sequences': r'\b(T1|T2|FLAIR|STIR|DWI|ADC)\b',
            'anatomical_regions': r'\b(cerebral|cerebellar|brainstem|ventricular|tentorial)\b',
            'common_conditions': r'\b(hypertension|diabetes|heart attack|stroke|cancer)\b'
        }
        
        for category, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if category in ['anatomy', 'pathology', 'imaging_descriptors', 'procedures', 'medical_terms', 'mri_technical', 'imaging_planes', 'mri_sequences', 'anatomical_regions', 'common_conditions']:
                    term, metadata = self.normalize_medical_term(match.strip())
                    medical_terms.append({
                        'term': term,
                        'category': category,
                        'metadata': metadata
                    })
                else:
                    terms = [term.strip() for term in re.split(r'[,;:]', match) if term.strip()]
                    for term in terms:
                        normalized_term, metadata = self.normalize_medical_term(term)
                        medical_terms.append({
                            'term': normalized_term,
                            'category': category,
                            'metadata': metadata
                        })
        
        # Remove duplicates while preserving metadata
        unique_terms = []
        seen_terms = set()
        for term_info in medical_terms:
            if term_info['term'] not in seen_terms:
                seen_terms.add(term_info['term'])
                unique_terms.append(term_info)
        
        # Filter out very short terms and common non-medical words
        stop_words = {'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'is', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall'}
        
        filtered_terms = [
            term_info for term_info in unique_terms 
            if len(term_info['term']) > 2 and len(term_info['term']) < 50 and term_info['term'] not in stop_words
        ]
        
        return filtered_terms

    def validate_medical_term(self, term_info: Dict[str, Any], context: str = "") -> Dict[str, Any]:
        """Validate a medical term against UMLS knowledge base with context awareness"""
        if not self.is_initialized:
            return {
                'term': term_info['term'],
                'found': False,
                'confidence': 0.0,
                'suggestions': [],
                'error': 'RAG system not initialized'
            }
        # Ensure term_info has required structure
        if not isinstance(term_info, dict) or 'term' not in term_info:
            return {
                'term': str(term_info) if not isinstance(term_info, dict) else term_info.get('term', 'unknown'),
                'found': False,
                'confidence': 0.0,
                'suggestions': [],
                'error': 'Invalid term_info structure'
            }
        try:
            term = term_info['term']
            metadata = term_info.get('metadata', {})  # Use get() with default empty dict
            
            # Get context-appropriate threshold
            context_threshold = self.get_context_threshold(
                term, 
                context, 
                semantic_type=metadata.get('semantic_type')  # Use get() for safe access
            )
            
            # Apply confidence boost from term variations
            context_threshold -= metadata.get('confidence_boost', 0.0)  # Use get() with default
            
            # Query ChromaDB
            results = self.collection.query(
                query_texts=[term],
                n_results=5,
                include=['documents', 'metadatas', 'distances']
            )
            
            validation_result = {
                'term': term,
                'found': False,
                'confidence': 0.0,
                'suggestions': [],
                'umls_matches': [],
                'semantic_type': metadata.get('semantic_type'),  # Use get() for safe access
                'context_threshold': context_threshold,
                'category': term_info.get('category', 'unknown')  # Use get() with default
            }
            
            if results['distances'][0]:
                best_distance = results['distances'][0][0]
                confidence = 1 - best_distance
                
                if confidence >= context_threshold:
                    validation_result['found'] = True
                    validation_result['confidence'] = confidence
                    
                    # Extract UMLS information
                    for i, (doc, metadata, distance) in enumerate(zip(
                        results['documents'][0],
                        results['metadatas'][0] if results['metadatas'][0] else [{}] * len(results['documents'][0]), 
                        results['distances'][0]
                    )):
                        umls_match = {
                            'term': doc,
                            'cui': metadata.get('cui', '') if metadata else '',
                            'semantic_type': metadata.get('semantic_type', '') if metadata else '',
                            'confidence': 1 - distance
                        }
                        validation_result['umls_matches'].append(umls_match)
                
                # Always provide suggestions
                validation_result['suggestions'] = results['documents'][0][:3]
                if results['metadatas'][0] and results['metadatas'][0][0]:
                    validation_result['semantic_type'] = results['metadatas'][0][0].get('semantic_type', 'Unknown')
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating term '{term_info['term']}': {e}")
            return {
                'term': term_info['term'],
                'found': False,
                'confidence': 0.0,
                'suggestions': [],
                'error': str(e)
            }

    def validate_medical_text(self, text: str) -> Dict[str, Any]:
        """Validate entire medical text with context awareness"""
        medical_terms = self.extract_medical_terms(text)
        
        validation_summary = {
            'total_terms': len(medical_terms),
            'validated_terms': 0,
            'unknown_terms': 0,
            'confidence_average': 0.0,
            'term_validations': [],
            'extracted_terms': [term_info['term'] for term_info in medical_terms],
            'semantic_types': set()
        }
        
        total_confidence = 0.0
        
        for term_info in medical_terms:
            validation = self.validate_medical_term(term_info, context=text)
            validation_summary['term_validations'].append(validation)
            
            if validation['found']:
                validation_summary['validated_terms'] += 1
                total_confidence += validation['confidence']
                if validation['semantic_type']:
                    validation_summary['semantic_types'].add(validation['semantic_type'])
            else:
                validation_summary['unknown_terms'] += 1
        
        if validation_summary['validated_terms'] > 0:
            validation_summary['confidence_average'] = total_confidence / validation_summary['validated_terms']
        
        validation_summary['semantic_types'] = list(validation_summary['semantic_types'])
        
        return validation_summary

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the UMLS collection"""
        if not self.is_initialized:
            return {"error": "RAG system not initialized"}
            
        try:
            count = self.collection.count()
            return {
                "total_terms": count,
                "collection_name": self.collection.name,
                "is_operational": True
            }
        except Exception as e:
            return {"error": str(e), "is_operational": False}