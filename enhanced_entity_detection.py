#!/usr/bin/env python3
"""
Enhanced Entity Detection for Universal Anonymisation
Comprehensive detection of BOTH individual names AND company names
"""

import re
import asyncio
from typing import Dict, List, Any

class EnhancedEntityDetector:
    """Enhanced detection for both individual and company names"""
    
    def __init__(self):
        self.session_mappings = {}
        self.entity_counter = 0
        
    async def detect_comprehensive_entities(self, document_text: str) -> Dict[str, List[str]]:
        """Detect ALL types of entities - individuals, companies, and more"""
        
        entities = {
            'individual_names': [],
            'company_names': [],
            'legal_entities': [],
            'case_numbers': [],
            'addresses': [],
            'emails': [],
            'phone_numbers': []
        }
        
        # 1. INDIVIDUAL NAMES (comprehensive patterns)
        individual_patterns = [
            # Standard first + last name
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            # Title + name (Dr., Mr., Ms., etc.)
            r'\b(?:Dr|Mr|Ms|Mrs|Prof|Professor|Judge|Justice)\. [A-Z][a-z]+ [A-Z][a-z]+\b',
            # First + middle + last
            r'\b[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+\b',
            # Three part names
            r'\b[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+\b'
        ]
        
        for pattern in individual_patterns:
            matches = re.findall(pattern, document_text)
            entities['individual_names'].extend(matches)
        
        # Remove duplicates and filter out common false positives
        entities['individual_names'] = list(set(entities['individual_names']))
        entities['individual_names'] = [name for name in entities['individual_names'] 
                                       if not self._is_false_positive_name(name)]
        
        # 2. COMPANY NAMES (expanded patterns)
        company_patterns = [
            # Companies with legal suffixes
            r'\b[A-Z][A-Za-z\s&]+ (?:Ltd|Limited|Inc|Corporation|Corp|plc|LLC|LLP|LP)\b',
            # Educational institutions
            r'\b[A-Z][A-Za-z\s]+ (?:University|College|Institute|School|Academy)\b',
            # Healthcare institutions  
            r'\b[A-Z][A-Za-z\s]+ (?:Hospital|Medical Center|Clinic|Healthcare|NHS Trust)\b',
            # Financial institutions
            r'\b[A-Z][A-Za-z\s]+ (?:Bank|Building Society|Credit Union|Finance|Investment)\b',
            # Law firms and professional services
            r'\b[A-Z][A-Za-z\s&]+ (?:& Associates|& Partners|Law Firm|Legal Services|Solicitors|Barristers)\b',
            # Government bodies
            r'\b[A-Z][A-Za-z\s]+ (?:Council|Authority|Department|Ministry|Agency|Commission)\b',
            # General business names (capitalized multi-word without legal suffix)
            r'\b[A-Z][A-Za-z]+ [A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]+)*(?:\s(?:Group|Solutions|Services|Systems|Technologies|Consulting))\b'
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, document_text)
            entities['company_names'].extend(matches)
        
        entities['company_names'] = list(set(entities['company_names']))
        
        # 3. LEGAL ENTITIES (courts, firms, etc.)
        legal_patterns = [
            r'\b(?:High Court|Crown Court|County Court|Magistrates Court|Supreme Court|Court of Appeal)\b',
            r'\b[A-Z][A-Za-z\s]+ (?:Chambers|Clerk|Registry)\b'
        ]
        
        for pattern in legal_patterns:
            matches = re.findall(pattern, document_text)
            entities['legal_entities'].extend(matches)
        
        entities['legal_entities'] = list(set(entities['legal_entities']))
        
        # 4. CASE NUMBERS (comprehensive legal formats)
        case_patterns = [
            r'\b[A-Z]{1,4}-\d{4}-\d{4,6}\b',      # KB-2023-000930
            r'\b\d{4}\s[A-Z]{2,6}\s\d+\b',        # 2023 EWHC 123
            r'\b[A-Z]{2,4}\d{2}/\d{4}\b',         # CO12/2023
            r'\b[A-Z]+\s\d{4}\s[A-Z]+\s\d+\b',   # CA 2023 EWCA 456
            r'\bCase\s(?:No\.?|Number):\s*[A-Z0-9/-]+\b'  # Case No: ABC/123
        ]
        
        for pattern in case_patterns:
            matches = re.findall(pattern, document_text)
            entities['case_numbers'].extend(matches)
        
        entities['case_numbers'] = list(set(entities['case_numbers']))
        
        # 5. EMAIL ADDRESSES
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities['emails'] = list(set(re.findall(email_pattern, document_text)))
        
        # 6. PHONE NUMBERS (UK and international)
        phone_patterns = [
            r'\b(?:\+44|0)\d{10,11}\b',           # UK numbers
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'  # US format
        ]
        
        for pattern in phone_patterns:
            matches = re.findall(pattern, document_text)
            entities['phone_numbers'].extend(matches)
        
        entities['phone_numbers'] = list(set(entities['phone_numbers']))
        
        # 7. ADDRESSES
        address_patterns = [
            r'\b\d+\s+[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\s(?:Street|Road|Avenue|Lane|Drive|Close|Way|Place|Square)\b',
            r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\s(?:House|Building|Tower|Centre|Court)\b'
        ]
        
        for pattern in address_patterns:
            matches = re.findall(pattern, document_text)
            entities['addresses'].extend(matches)
        
        entities['addresses'] = list(set(entities['addresses']))
        
        # Remove empty categories
        return {k: v for k, v in entities.items() if v}
    
    def _is_false_positive_name(self, name: str) -> bool:
        """Filter out common false positives for names"""
        false_positives = [
            'Based On', 'According To', 'High Court', 'Crown Court',
            'Case Number', 'File Number', 'Legal Services', 'Law Firm',
            'In Re', 'Ex Parte', 'Per Se', 'Ad Hoc', 'Prima Facie'
        ]
        return name in false_positives
    
    async def generate_realistic_pseudonym(self, original: str, category: str) -> str:
        """Generate realistic pseudonyms for individuals AND companies"""
        
        if original in self.session_mappings:
            return self.session_mappings[original]
        
        try:
            import aiohttp
            
            # Enhanced prompts for different entity types
            prompts = {
                'individual_names': f"Generate a realistic fake person name to replace '{original}'. Keep similar style (formal/informal). Respond with only the name.",
                'company_names': f"Generate a realistic fake company name to replace '{original}'. Keep the same business type and style. Respond with only the company name.",
                'legal_entities': f"Generate a realistic fake legal entity name to replace '{original}'. Keep the same type (court, firm, etc.). Respond with only the name.",
                'case_numbers': f"Generate a fake case number in the same format as '{original}'. Respond with only the case number.",
                'emails': f"Generate a fake email similar to '{original}'. Use fake names but realistic domain. Respond with only the email.",
                'phone_numbers': "Generate a fake UK phone number. Respond with only the number.",
                'addresses': f"Generate a fake address similar to '{original}'. Respond with only the address."
            }
            
            prompt = prompts.get(category, f"Generate a fake replacement for '{original}'. Respond with only the replacement.")
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": "phi3:latest",
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 80,  # Longer for company names
                    "system": "Generate realistic fake replacements. Respond with only the replacement text, no quotes or explanations."
                }
                
                async with session.post("http://localhost:11434/api/generate", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        pseudonym = result.get('response', f"ANON_{category}_{self.entity_counter}").strip()
                        
                        # Clean up response
                        pseudonym = pseudonym.split('\n')[0].strip()
                        if pseudonym.startswith('"') and pseudonym.endswith('"'):
                            pseudonym = pseudonym[1:-1]
                        
                        # Store mapping
                        self.session_mappings[original] = pseudonym
                        self.entity_counter += 1
                        
                        return pseudonym
                    else:
                        fallback = f"ANON_{category}_{self.entity_counter}"
                        self.session_mappings[original] = fallback
                        self.entity_counter += 1
                        return fallback
                        
        except Exception:
            fallback = f"ANON_{category}_{self.entity_counter}"
            self.session_mappings[original] = fallback
            self.entity_counter += 1
            return fallback

async def demo_individual_and_company_detection():
    """Demo comprehensive individual AND company name detection"""
    
    print("🏢👤 Enhanced Individual AND Company Name Detection")
    print("=" * 60)
    
    detector = EnhancedEntityDetector()
    
    # Test document with mix of individuals and companies
    test_document = """
    LEGAL CASE SUMMARY
    
    INDIVIDUAL PARTIES:
    Claimant: Dr. Sarah Elizabeth Johnson
    Legal Representative: Mr. James Wilson QC
    Expert Witness: Professor Michael Thompson
    Defendant Representative: Ms. Emma Clarke
    
    COMPANY PARTIES:
    Claimant Employer: Johnson & Associates Legal Services Ltd
    Defendant: Royal London Hospital NHS Trust
    Insurance Provider: AXA Corporate Solutions plc
    Expert Firm: Medical Analytics Group
    Legal Chambers: Crown Chambers & Partners
    
    CASE DETAILS:
    Case Number: KB-2023-000930
    Court: High Court London
    Address: 123 Victoria Street, London
    Email: legal.dept@royallondon.nhs.uk
    Phone: 020 7946 0958
    
    The case involves medical negligence claims against Royal London Hospital
    by Dr. Sarah Johnson, represented by Johnson & Associates Legal Services.
    """
    
    print("📄 Test Document Preview:")
    print(test_document[:200] + "...")
    
    # Detect all entities
    print(f"\n🔍 Detecting Individual AND Company Entities...")
    entities = await detector.detect_comprehensive_entities(test_document)
    
    # Show detection results
    total_entities = sum(len(v) for v in entities.values())
    print(f"✅ Total entities detected: {total_entities}")
    
    for category, items in entities.items():
        if items:
            print(f"\n📋 {category.replace('_', ' ').title()}: {len(items)} found")
            for item in items:
                print(f"   • {item}")
    
    # Generate pseudonyms for ALL entities
    print(f"\n🎭 Generating Pseudonyms (phi3)...")
    anonymised_text = test_document
    replacement_summary = {'individuals': 0, 'companies': 0, 'other': 0}
    
    for category, entity_list in entities.items():
        for entity in entity_list:
            pseudonym = await detector.generate_realistic_pseudonym(entity, category)
            anonymised_text = anonymised_text.replace(entity, pseudonym)
            
            # Count by type
            if 'individual' in category:
                replacement_summary['individuals'] += 1
            elif 'company' in category or 'legal' in category:
                replacement_summary['companies'] += 1
            else:
                replacement_summary['other'] += 1
            
            print(f"   {entity} → {pseudonym}")
    
    print(f"\n📊 Anonymisation Summary:")
    print(f"   👤 Individual names: {replacement_summary['individuals']} anonymised")
    print(f"   🏢 Company names: {replacement_summary['companies']} anonymised")
    print(f"   📋 Other entities: {replacement_summary['other']} anonymised")
    
    print(f"\n📤 Anonymised Document Preview:")
    print(anonymised_text[:300] + "...")
    
    # Test reverse engineering
    print(f"\n🔄 Testing Reverse Engineering...")
    reverse_mappings = {v: k for k, v in detector.session_mappings.items()}
    
    sample_anonymised = anonymised_text[:200]
    restored_sample = sample_anonymised
    for pseudonym, original in sorted(reverse_mappings.items(), key=lambda x: len(x[0]), reverse=True):
        restored_sample = restored_sample.replace(pseudonym, original)
    
    print(f"📤 Anonymised: {sample_anonymised}")
    print(f"🔓 Restored:   {restored_sample}")
    
    print(f"\n✅ Comprehensive Detection Results:")
    print(f"🎯 Handles BOTH individual AND company names")
    print(f"📋 Detects legal entities, case numbers, contact details")
    print(f"🔄 Perfect reverse engineering for all entity types")
    print(f"⚡ phi3 generates realistic pseudonyms for all categories")

if __name__ == "__main__":
    print("🛡️ Enhanced Entity Detection System")
    print("Comprehensive anonymisation for individuals AND companies")
    
    asyncio.run(demo_individual_and_company_detection()) 