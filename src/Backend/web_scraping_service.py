# web_scraping_service.py

"""
Web Scraping Service Module

This module handles scraping treatment information from Mayo Clinic and MedlinePlus,
as well as extracting searchable medical conditions from complex medical text.
"""

import json
import re
import time
import random
import logging
from typing import List, Optional
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from crewai.tools import tool
from config import llm
import inflect

logger = logging.getLogger(__name__)

#############################################################################
# WEB SCRAPING TOOLS
#############################################################################

@tool
def extract_searchable_conditions(medical_problem: str) -> str:
    """
    Extract searchable medical conditions from complex medical text using a generalizable prompt.
    
    Args:
        medical_problem (str): Raw medical report text
    
    Returns:
        str: JSON array of simplified, web-searchable medical condition terms (e.g., Mayo Clinic style)
    """
    logger.info("Extracting searchable conditions from medical report")

    prompt = f"""
    You are a medical domain expert trained to convert complex clinical findings into
    simplified, patient-friendly terms that users can easily search online (like on Mayo Clinic or Medline-Plus).

    INPUT MEDICAL REPORT:
    "{medical_problem}"

    TASK:
    - Extract only key medical conditions, diagnoses, or significant findings
    - Convert them into layperson-friendly, web-searchable terms that would match the names of diseases or conditions as found on websites like Mayo Clinic or MedlinePlus
    - Avoid technical modifiers (e.g. anatomical locations, severity unless critical)
    - Avoid repetition or rare synonyms
    - Output max 5 key conditions

    OUTPUT:
    Return only a JSON array of simplified medical condition terms.

    Example:
    INPUT: "Maxillo-ethmoidal and frontal sinusitis, Nasal septum deviation, Hypertrophied inferior nasal turbinates, Allergic rhinitis"
    OUTPUT: ["Chronic sinusitis", "Deviated septum", "Nasal polyps", "Hay fever"]
    """

    try:
        logger.debug("Sending prompt to LLM...")
        response = llm.call(prompt)
        logger.debug(f"Received response: {response}")
        return response  # Already in JSON format
    except Exception as e:
        logger.error(f"Error extracting searchable conditions: {e}")
        return json.dumps([])

@tool
def scrape_mayo_treatments(conditions: List[str]) -> str:
    """
    Scrape treatment information from Mayo Clinic for given medical conditions
    using the exact same logic as the MayoClinicScraper class.
    
    Args:
        conditions (List[str]): List of medical conditions to search for treatments
    
    Returns:
        str: JSON string with treatment information from Mayo Clinic
    """
    if isinstance(conditions, str):
        try:
            conditions = json.loads(conditions)
        except json.JSONDecodeError:
            conditions = [conditions]
    
    results = {}
    base_url = "https://www.mayoclinic.org"
    
    # Set up session with proper headers
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    })
    
    def clean_query_for_url(query: str) -> str:
        """
        Clean and standardize query for URL formation
        Directly implements MayoClinicScraper._clean_query_for_url logic
        """
        # Convert to lowercase
        clean_query = query.lower()
        
        # Remove common punctuation but preserve meaningful separators temporarily
        punctuation_to_remove = "'\".,()[]{}!?;:"
        for punct in punctuation_to_remove:
            clean_query = clean_query.replace(punct, '')
        
        # Replace underscores and multiple spaces with single spaces
        clean_query = re.sub(r'[_]+', ' ', clean_query)
        clean_query = re.sub(r'\s+', ' ', clean_query)
        
        # Replace spaces with hyphens
        clean_query = clean_query.replace(' ', '-')
        
        # Remove multiple consecutive hyphens
        clean_query = re.sub(r'-+', '-', clean_query)
        
        # Remove leading/trailing hyphens
        clean_query = clean_query.strip('-')
        
        return clean_query
    
    def check_url_exists(url: str) -> bool:
        """
        Check if a URL exists with better error handling
        Directly implements MayoClinicScraper._check_url_exists logic
        """
        try:
            response = session.head(url, timeout=8, allow_redirects=True)
            
            # Accept both 200 and 3xx redirects as valid
            if response.status_code in [200, 301, 302, 303, 307, 308]:
                return True
            else:
                return False
                
        except Exception:
            return False
    
    def get_page(url: str) -> Optional[BeautifulSoup]:
        """
        Fetch and parse a web page with error handling
        Directly implements MayoClinicScraper._get_page logic
        """
        try:
            # Add random delay to be respectful
            delay = random.uniform(1, 3)
            time.sleep(delay)
            
            response = session.get(url, timeout=10)
            response.raise_for_status()
            
            return BeautifulSoup(response.content, 'html.parser')
            
        except Exception:
            return None
    
    def discover_condition_subpages(base_url: str, condition_pattern: str) -> List[str]:
        """
        Directly implements MayoClinicScraper._discover_condition_subpages logic
        """
        subpage_urls = []
        
        # Only look for diagnosis-treatment page
        target_subpage = "diagnosis-treatment"
        
        # First, get the base page to find actual subpage links
        soup = get_page(base_url)
        if soup:
            # Look for links to diagnosis-treatment page in the navigation or content
            all_links = soup.find_all('a', href=True)
            
            for link in all_links:
                href = link.get('href')
                if href:
                    # Convert relative URLs to absolute
                    if href.startswith('/'):
                        full_url = urljoin(base_url, href)
                    elif href.startswith('http'):
                        full_url = href
                    else:
                        continue
                    
                    # Check if this link is the diagnosis-treatment page we're looking for
                    if f"/diseases-conditions/{condition_pattern}/{target_subpage}/" in full_url:
                        if full_url not in subpage_urls:
                            subpage_urls.append(full_url)
                            break  # Found it, no need to continue
        
        # If scraping didn't find the diagnosis-treatment page, try simple pattern
        if not subpage_urls:
            test_url = f"{base_url}/diseases-conditions/{condition_pattern}/{target_subpage}"
            if check_url_exists(test_url):
                subpage_urls.append(test_url)
        
        # Return the diagnosis-treatment page or base URL as fallback
        return subpage_urls if subpage_urls else ([base_url] if check_url_exists(base_url) else [])
    
    def search_conditions(query: str) -> List[str]:
        """
        Directly implements MayoClinicScraper.search_conditions logic
        """
        search_urls = []
        
        # Clean the query for URL formation
        clean_query = clean_query_for_url(query)
    
        # Basic URL patterns to try
        base_patterns = [
            clean_query,
            clean_query.replace('-', ''),
            clean_query.replace('-', ' ').replace(' ', ''),
            f"chronic-{clean_query}",
            f"acute-{clean_query}"
        ]
        
        # For each base pattern, try to find the main condition page first
        for pattern in base_patterns:
            condition_url = f"{base_url}/diseases-conditions/{pattern}"
            
            # Check if the base condition page exists
            if check_url_exists(condition_url):
                # Now find the diagnosis-treatment page specifically
                condition_urls = discover_condition_subpages(condition_url, pattern)
                search_urls.extend(condition_urls)
                
                if search_urls:
                    break  # Found working URLs, no need to try more patterns
        
        # If no direct patterns worked, try alternative discovery methods
        if not search_urls:
            # Extract key medical terms and search for them
            words = query.lower().replace('-', ' ').split()
            
            # Common medical term transformations
            transformations = {
                'deviation': 'deviated',
                'enlargement': 'enlarged', 
                'inflammation': 'inflamed',
                'infection': 'infected',
                'deficiency': 'deficient',
                'insufficiency': 'insufficient',
                'dysfunction': 'dysfunctional',
                'syndrome': 'disease',
                'disorder': 'condition',
                'palsy': 'paralysis',
            }
            
            # Transform words and create new patterns
            transformed_words = []
            for word in words:
                if word in transformations:
                    transformed_words.append(transformations[word])
                else:
                    transformed_words.append(word)
            
            # Try different combinations
            test_patterns = [
                # Transformed version
                '-'.join(transformed_words),
                # First few words only
                '-'.join(words[:2]) if len(words) > 1 else words[0],
                # Reverse order
                '-'.join(reversed(words)) if len(words) > 1 else words[0],
                # Just the longest word (often the main medical term)
                max(words, key=len) if len(words) > 1 else None,
                # Last word only (often a key term)
                words[-1] if len(words) > 1 else words[0],
            ]
            
            test_patterns = [p for p in test_patterns if p]  # Remove None values
            
            for pattern in test_patterns:
                condition_url = f"{base_url}/diseases-conditions/{pattern}"
                if check_url_exists(condition_url):
                    return discover_condition_subpages(condition_url, pattern)
        
        return search_urls
    
    def extract_treatments(soup: BeautifulSoup) -> List[str]:
        """
        Extract treatment information from Mayo Clinic page, 
        directly implementing MayoClinicScraper._extract_treatments logic
        """
        treatments = []
        
        # First, look specifically for h2 with "Treatment" text
        treatment_h2 = None
        for h2 in soup.find_all('h2'):
            if re.match(r'^\s*Treatment\s*$', h2.get_text().strip(), re.I):
                treatment_h2 = h2
                break
        
        if treatment_h2:
            # Get content following the Treatment h2
            current = treatment_h2.next_sibling
            
            while current:
                if hasattr(current, 'name'):
                    # Stop when we hit another h2 (next major section)
                    if current.name == 'h2':
                        break
                    
                    # Process h3 headers (treatment subsections)
                    elif current.name == 'h3':
                        h3_text = current.get_text().strip()
                        
                        # Add the h3 header as a treatment category
                        if h3_text and len(h3_text) > 3:
                            treatments.append(f"**{h3_text}**")
                        
                        # Get content following this h3 until next h3 or h2
                        h3_current = current.next_sibling
                        while h3_current:
                            if hasattr(h3_current, 'name'):
                                if h3_current.name in ['h2', 'h3']:
                                    break  # Stop at next heading
                                elif h3_current.name in ['ul', 'ol']:
                                    # Extract list items under this h3
                                    items = h3_current.find_all('li')
                                    for item in items:
                                        treatment = item.get_text().strip()
                                        if treatment and len(treatment) > 10:
                                            treatments.append(f"  • {treatment}")
                                elif h3_current.name == 'p':
                                    # Extract paragraph content under this h3
                                    text = h3_current.get_text().strip()
                                    if text and len(text) > 20:
                                        treatments.append(f"  {text}")
                                elif h3_current.name == 'div':
                                    # Look inside divs for nested content
                                    div_lists = h3_current.find_all(['ul', 'ol'])
                                    for ul in div_lists:
                                        items = ul.find_all('li')
                                        for item in items:
                                            treatment = item.get_text().strip()
                                            if treatment and len(treatment) > 10:
                                                treatments.append(f"  • {treatment}")
                                    
                                    # Also check paragraphs in the div
                                    div_paras = h3_current.find_all('p')
                                    for para in div_paras:
                                        text = para.get_text().strip()
                                        if text and len(text) > 20:
                                            treatments.append(f"  {text}")
                            
                            h3_current = h3_current.next_sibling
                    
                    # Process direct content under the main h2 (before any h3s)
                    elif current.name in ['ul', 'ol']:
                        # Extract list items directly under h2
                        items = current.find_all('li')
                        for item in items:
                            treatment = item.get_text().strip()
                            if treatment and len(treatment) > 10:
                                treatments.append(treatment)
                    elif current.name == 'p':
                        # Extract paragraph content directly under h2
                        text = current.get_text().strip()
                        if text and len(text) > 20:
                            treatments.append(text)
                    elif current.name == 'div':
                        # Look inside divs for nested content directly under h2
                        div_lists = current.find_all(['ul', 'ol'])
                        for ul in div_lists:
                            items = ul.find_all('li')
                            for item in items:
                                treatment = item.get_text().strip()
                                if treatment and len(treatment) > 10:
                                    treatments.append(treatment)
                        
                        # Also check paragraphs in the div
                        div_paras = current.find_all('p')
                        for para in div_paras:
                            text = para.get_text().strip()
                            if text and len(text) > 20:
                                treatments.append(text)
                
                current = current.next_sibling
        
        # If no h2 'Treatment' found, try other treatment headings as fallback
        if not treatments:
            treatment_headings = soup.find_all(['h2', 'h3', 'h4'], 
            string=re.compile(r'treatment|therapy|management|care|medication', re.I))
            
            for heading in treatment_headings:
                # Get content following the heading
                current = heading.next_sibling
                found_items = 0
                
                while current and found_items < 10:
                    if hasattr(current, 'name'):
                        if current.name in ['ul', 'ol']:
                            items = current.find_all('li')
                            for item in items:
                                treatment = item.get_text().strip()
                                if treatment and len(treatment) > 10:
                                    treatments.append(treatment)
                                    found_items += 1
                        elif current.name == 'p':
                            text = current.get_text().strip()
                            if text and len(text) > 20:
                                treatments.append(text)
                                found_items += 1
                        elif current.name in ['h2', 'h3', 'h4']:
                            break  # Stop at next heading
                    
                    current = current.next_sibling
                
                if treatments:
                    break  # Found treatments, no need to check other headings
        
        # If still no treatments found, try enhanced selectors as final fallback
        if not treatments:
            treatment_selectors = [
                '[class*="treatment" i] ul li',
                '[class*="treatment" i] ol li',
                '[class*="therapy" i] ul li',
                '[class*="management" i] ul li',
                '[id*="treatment" i] ul li',
                '[id*="treatment" i] ol li',
                '.content ul li',
                '.page-content ul li', 
                '.main-content ul li',
            ]
            
            for selector in treatment_selectors:
                try:
                    elements = soup.select(selector)
                    for elem in elements:
                        treatment = elem.get_text().strip()
                        if treatment and len(treatment) > 10:
                            treatments.append(treatment)
                    
                    if treatments:
                        break
                except Exception:
                    continue
        
        # Remove duplicates while preserving order
        unique_treatments = list(dict.fromkeys(treatments))
        
        return unique_treatments[:25]  # Limit to 25 items
    
    # Process each condition
    for condition in conditions:
        if not condition or not condition.strip():
            continue
            
        try:
            # Use the search_conditions method to find the correct URL
            found_urls = search_conditions(condition)
            
            if found_urls:
                # Get the first URL (preferably diagnosis-treatment)
                condition_url = found_urls[0]
                
                # Fetch the page
                soup = get_page(condition_url)
                if soup:
                    # Extract treatments
                    treatments = extract_treatments(soup)
                    
                    # Join treatments into a string
                    treatment_text = '\n\n'.join(treatments)
                    
                    # Store results
                    results[condition] = {
                        "name": condition,
                        "url": condition_url,
                        "source": "Mayo Clinic",
                        "treatment": treatment_text if treatment_text else "No treatment information found"
                    }
                else:
                    results[condition] = {
                        "name": condition,
                        "url": condition_url,
                        "source": "Mayo Clinic",
                        "error": "Could not fetch page content"
                    }
            else:
                results[condition] = {
                    "name": condition,
                    "error": "No Mayo Clinic page found",
                    "source": "Mayo Clinic"
                }
            
            # Add delay between conditions
            time.sleep(random.uniform(1.5, 3))
            
        except Exception as e:
            results[condition] = {
                "name": condition,
                "error": str(e),
                "source": "Mayo Clinic"
            }
    
    return json.dumps(results, indent=2)

@tool
def scrape_medlineplus_treatments(conditions: List[str]) -> str:
    """
    Scrape treatment information from MedlinePlus for given medical conditions.

    Args:
        conditions (List[str]): List of condition names to search for.

    Returns:
        str: JSON string with treatment information.
    """
    results = {}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    def clean_condition_for_url(condition: str) -> str:
        # Remove anything in parentheses and extra characters
        condition = re.sub(r"\(.*?\)", "", condition)  # remove (UTI) or similar
        condition = condition.lower()
        condition = condition.replace("'", "").replace("'", "").replace("–", "").replace("-", "")
        condition = condition.replace(" ", "")
        return condition.strip()

    def pluralize_slug(condition: str) -> str:
        # Create inflect engine
        p = inflect.engine()
        words = condition.lower().split()
        if words:
            words[-1] = p.plural(words[-1])
        return ''.join(words)

    def singularize_slug(condition: str) -> str:
        # Create inflect engine
        p = inflect.engine()
        words = condition.lower().split()
        if words:
            singular = p.singular_noun(words[-1])
            words[-1] = singular if singular else words[-1]
        return ''.join(words)

    for condition in conditions:
        try:
            time.sleep(1)
            # ⿡ Try original slug
            slug = clean_condition_for_url(condition)
            url = f"https://medlineplus.gov/{slug}.html"
            resp = requests.get(url, headers=headers, timeout=10)

            # ⿢ Try adding s to slug
            if resp.status_code != 200:
                fallback = f"https://medlineplus.gov/{slug}s.html"
                resp = requests.get(fallback, headers=headers, timeout=10)
                url = fallback if resp.status_code == 200 else url

            # ⿣ Try pluralized slug
            if resp.status_code != 200:
                plural_slug = pluralize_slug(condition)
                fallback = f"https://medlineplus.gov/{plural_slug}.html"
                resp = requests.get(fallback, headers=headers, timeout=10)
                url = fallback if resp.status_code == 200 else url

            # ⿤ Try singularized slug
            if resp.status_code != 200:
                singular_slug = singularize_slug(condition)
                fallback = f"https://medlineplus.gov/{singular_slug}.html"
                resp = requests.get(fallback, headers=headers, timeout=10)
                url = fallback if resp.status_code == 200 else url

            # Check if we got a successful response
            if resp.status_code != 200:
                results[condition] = {"error": f"No page found at {url}"}
                continue

            soup = BeautifulSoup(resp.content, 'html.parser')
            treatment_text = ""

            # Step 1: Look for treatment-related headings and paragraphs after them
            headers_list = soup.find_all(["h2", "h3"])
            for header in headers_list:
                header_text = header.get_text(strip=True).lower()
                if any(kw in header_text for kw in ["summary","treat", "manage", "care", "therapy", "medicine"]):
                    content = []
                    for sibling in header.find_next_siblings():
                        if sibling.name in ["p", "ul", "ol", "li"]:
                            content.append(sibling.get_text(strip=True))
                        elif sibling.name in ["h2", "h3"]:
                            break
                    if content:
                        treatment_text = " ".join(content)
                        break

            # Step 2: Enhanced fallback – look in summary section and early content
            if not treatment_text:
                # Try different summary approaches
                summary_found = False
                
                # Approach 1: Look for summary headers
                for header in headers_list:
                    header_text = header.get_text(strip=True).lower()
                    if any(kw in header_text for kw in ["summary", "overview", "about", "what is"]):
                        summary_parts = []
                        for sibling in header.find_next_siblings():
                            if sibling.name in ["p", "ul", "ol", "li"]:
                                summary_parts.append(sibling.get_text(strip=True))
                            elif sibling.name in ["h2", "h3"]:
                                break
                        if summary_parts:
                            summary_text = " ".join(summary_parts)
                            # Look for treatment keywords with more flexibility
                            treatment_keywords = ['treat', 'treatment', 'therapy', 'medicine', 'medication', 'manage', 'care']
                            if any(keyword in summary_text.lower() for keyword in treatment_keywords):
                                treatment_text = summary_text[:3000]
                                summary_found = True
                                break
                
                # Approach 2: If no summary header found, check first few paragraphs
                if not summary_found:
                    main_content = soup.find('main') or soup.find('div', class_='content') or soup
                    first_paragraphs = main_content.find_all('p')[:4]  # First 4 paragraphs
                    
                    for p in first_paragraphs:
                        p_text = p.get_text(strip=True)
                        if len(p_text) > 100:  # Skip very short paragraphs
                            treatment_keywords = ['treat', 'treatment', 'therapy', 'medicine', 'medication', 'manage']
                            if any(keyword in p_text.lower() for keyword in treatment_keywords):
                                treatment_text = p_text
                                break
                
                # Approach 3: Look for any paragraph with treatment content
                if not treatment_text:
                    all_paragraphs = soup.find_all('p')[:10]  # Check first 10 paragraphs
                    for p in all_paragraphs:
                        p_text = p.get_text(strip=True)
                        if len(p_text) > 80 and 'treat' in p_text.lower():
                            treatment_text = p_text
                            break

            # Step 3: Store result
            if treatment_text:
                results[condition] = {
                    "name": condition,
                    "url": url,
                    "source": "MedlinePlus",
                    "treatment": treatment_text.strip()
                }
            else:
                results[condition] = {"error": "No treatment info found"}

        except Exception as e:
            logging.exception(f"Error scraping {condition}")
            results[condition] = {"error": str(e)}

    return json.dumps(results, indent=2)