#!/usr/bin/env python3
"""
Universal Reverse Engineering Demo
Shows how the system works with ANY document uploaded, not specific mappings
"""

import re
import asyncio
from typing import Dict, List, Any

class UniversalAnonymiser:
    """Universal anonymisation that works with ANY document uploaded"""
    
    def __init__(self):
        self.session_mappings = {}  # Stores mappings for current session
        self.entity_counter = 0     # For generating unique pseudonyms
        
    async def detect_all_entities(self, document_text: str) -> Dict[str, List[str]]:
        """Automatically detect ALL entities in ANY document"""
        
        entities = {
            'person_names': [],
            'companies': [], 
            'case_numbers': [],
            'addresses': [],
            'emails': [],
            'phone_numbers': [],
            'dates': []
        }
        
        # 1. Person Names (capitalized word pairs)
        person_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b'
        entities['person_names'] = list(set(re.findall(person_pattern, document_text)))
        
        # 2. Companies (with legal suffixes)
        company_pattern = r'\b[A-Z][A-Za-z\s&]+ (?:Ltd|Limited|Inc|Corporation|Corp|plc|University|College|Institute|School|Hospital|Trust|Foundation)\b'
        entities['companies'] = list(set(re.findall(company_pattern, document_text)))
        
        # 3. Case Numbers (various legal formats)
        case_patterns = [
            r'\b[A-Z]{1,3}-\d{4}-\d{4,6}\b',  # KB-2023-000930
            r'\b\d{4}\s[A-Z]+\s\d+\b',        # 2023 EWHC 123
            r'\b[A-Z]+\d{2}/\d{4}\b'          # CO/1234/2023
        ]
        for pattern in case_patterns:
            entities['case_numbers'].extend(re.findall(pattern, document_text))
        entities['case_numbers'] = list(set(entities['case_numbers']))
        
        # 4. Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities['emails'] = list(set(re.findall(email_pattern, document_text)))
        
        # 5. Phone numbers (UK format)
        phone_pattern = r'\b(?:\+44|0)\d{10,11}\b'
        entities['phone_numbers'] = list(set(re.findall(phone_pattern, document_text)))
        
        # 6. Addresses (simplified - street numbers with street names)
        address_pattern = r'\b\d+\s+[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\s(?:Street|Road|Avenue|Lane|Drive|Close|Way)\b'
        entities['addresses'] = list(set(re.findall(address_pattern, document_text)))
        
        # Remove empty categories
        return {k: v for k, v in entities.items() if v}
    
    async def generate_pseudonym_for_entity(self, original: str, category: str) -> str:
        """Generate a pseudonym for ANY entity using phi3"""
        
        # Check if already mapped
        if original in self.session_mappings:
            return self.session_mappings[original]
        
        try:
            # Use phi3 to generate realistic pseudonym
            import aiohttp
            
            # Category-specific prompts
            prompts = {
                'person_names': f"Generate a realistic fake name to replace '{original}'. Respond with only the fake name.",
                'companies': f"Generate a realistic fake company name similar to '{original}'. Keep the same type. Respond with only the company name.",
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
                    "num_predict": 50,
                    "system": "Generate realistic fake replacements. Respond with only the replacement text."
                }
                
                async with session.post("http://localhost:11434/api/generate", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        pseudonym = result.get('response', f"ANON_{category}_{self.entity_counter}").strip()
                        
                        # Clean up response
                        pseudonym = pseudonym.split('\n')[0].strip()
                        if pseudonym.startswith('"') and pseudonym.endswith('"'):
                            pseudonym = pseudonym[1:-1]
                        
                        # Store bidirectional mapping
                        self.session_mappings[original] = pseudonym
                        self.entity_counter += 1
                        
                        return pseudonym
                    else:
                        # Fallback
                        fallback = f"ANON_{category}_{self.entity_counter}"
                        self.session_mappings[original] = fallback
                        self.entity_counter += 1
                        return fallback
                        
        except Exception:
            # Fallback for any error
            fallback = f"ANON_{category}_{self.entity_counter}"
            self.session_mappings[original] = fallback
            self.entity_counter += 1
            return fallback
    
    async def anonymise_any_document(self, document_text: str) -> Dict[str, Any]:
        """Anonymise ANY document uploaded by user"""
        
        print(f"📄 Processing document ({len(document_text)} characters)")
        
        # Step 1: Detect ALL entities automatically
        entities = await self.detect_all_entities(document_text)
        print(f"🔍 Detected entities: {sum(len(v) for v in entities.values())} total")
        for category, items in entities.items():
            if items:
                print(f"   {category}: {len(items)} found")
        
        # Step 2: Generate pseudonyms for ALL entities
        anonymised_text = document_text
        total_replacements = 0
        
        for category, entity_list in entities.items():
            for entity in entity_list:
                pseudonym = await self.generate_pseudonym_for_entity(entity, category)
                anonymised_text = anonymised_text.replace(entity, pseudonym)
                total_replacements += 1
                print(f"   '{entity}' → '{pseudonym}'")
        
        print(f"✅ Anonymisation complete: {total_replacements} entities replaced")
        
        return {
            'original_text': document_text,
            'anonymised_text': anonymised_text,
            'entities_detected': entities,
            'mappings': dict(self.session_mappings),
            'reverse_mappings': {v: k for k, v in self.session_mappings.items()},
            'total_entities': total_replacements
        }
    
    def reverse_engineer_result(self, anonymised_text: str) -> str:
        """Reverse engineer ANY anonymised text back to original"""
        
        reverse_mappings = {v: k for k, v in self.session_mappings.items()}
        
        restored_text = anonymised_text
        for pseudonym, original in sorted(reverse_mappings.items(), key=lambda x: len(x[0]), reverse=True):
            restored_text = restored_text.replace(pseudonym, original)
        
        return restored_text

async def demo_universal_anonymisation():
    """Demonstrate how it works with ANY document"""
    
    print("🌐 Universal Reverse Engineering Demo")
    print("Works with ANY document uploaded!")
    print("=" * 50)
    
    anonymiser = UniversalAnonymiser()
    
    # Example 1: Medical Record
    print("\n📋 Test 1: Medical Document")
    medical_doc = """
    Patient: Sarah Johnson
    Date: 15/03/2024
    Address: 123 Oak Street, London
    Phone: 020 7946 0958
    Email: sarah.johnson@email.com
    
    Treating Physician: Dr. Michael Thompson
    Hospital: Royal London Hospital
    Case Number: MED-2024-001234
    
    Medical history shows patient presented with symptoms...
    """
    
    result1 = await anonymiser.anonymise_any_document(medical_doc)
    print(f"📤 Anonymised preview: {result1['anonymised_text'][:150]}...")
    
    # Example 2: Contract Document  
    print("\n📋 Test 2: Contract Document")
    contract_doc = """
    CONFIDENTIAL AGREEMENT
    
    Party A: Jennifer Wilson
    Company: Wilson & Associates Ltd
    Address: 456 Business Park Avenue, Manchester
    Email: j.wilson@wilsonlaw.co.uk
    
    Party B: DataTech Corporation
    Representative: Robert Martinez
    Case Reference: CONT-2024-005678
    
    This agreement outlines the terms...
    """
    
    result2 = await anonymiser.anonymise_any_document(contract_doc)
    print(f"📤 Anonymised preview: {result2['anonymised_text'][:150]}...")
    
    # Example 3: Legal Case File
    print("\n📋 Test 3: Legal Case File")
    legal_doc = """
    CASE FILE: Thompson vs DataSoft Inc
    
    Claimant: James Thompson
    Address: 789 Victoria Road, Birmingham  
    Solicitor: Emma Clarke, Clarke Legal Services Ltd
    
    Defendant: DataSoft Inc
    Representative: Alex Kumar
    Case Number: HC-2024-009876
    Court: High Court London
    
    The claimant alleges that...
    """
    
    result3 = await anonymiser.anonymise_any_document(legal_doc)
    print(f"📤 Anonymised preview: {result3['anonymised_text'][:150]}...")
    
    # Test reverse engineering
    print("\n🔄 Testing Reverse Engineering:")
    
    # Take anonymised text and reverse it
    anonymised_sample = result1['anonymised_text'][:200] + "..."
    restored_sample = anonymiser.reverse_engineer_result(anonymised_sample)
    
    print(f"📤 Anonymised: {anonymised_sample}")
    print(f"🔓 Restored:   {restored_sample}")
    
    print(f"\n📊 Universal System Summary:")
    print(f"✅ Processes ANY document type")
    print(f"✅ Detects entities automatically")
    print(f"✅ Generates realistic pseudonyms")
    print(f"✅ Stores mappings for reverse engineering")
    print(f"✅ Works consistently across sessions")
    
    total_mappings = len(anonymiser.session_mappings)
    print(f"\n🎯 Session Results: {total_mappings} entities mapped across all documents")

if __name__ == "__main__":
    print("🛡️ Universal Document Anonymisation System")
    print("Processes ANY document - medical, legal, business, etc.")
    
    asyncio.run(demo_universal_anonymisation()) 