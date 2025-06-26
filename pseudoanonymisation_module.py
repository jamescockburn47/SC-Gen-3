#!/usr/bin/env python3
"""
Pseudoanonymisation Module for RAG Output
Uses phi3's creative capabilities for privacy protection
"""

import asyncio
import aiohttp
import re
import json
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from pathlib import Path
import pickle

class PseudoAnonymiser:
    """Fast pseudoanonymisation with reversible mappings for cloud workflows"""
    
    def __init__(self, storage_path: str = "anonymisation_storage"):
        # Forward mappings (original → pseudonym)
        self.forward_mappings = {}  # Combined storage for all mappings
        
        # Reverse mappings (pseudonym → original) for reconstruction
        self.reverse_mappings = {}  
        
        # Storage configuration
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.mappings_file = self.storage_path / "mappings.json"
        self.session_file = self.storage_path / "current_session.json"
        
        # Load existing mappings
        self._load_persistent_mappings()
        
        # Session tracking for cloud workflows
        self.current_session_id = None
        self.session_mappings = {}  # Track mappings per session
    
    def _load_persistent_mappings(self):
        """Load existing mappings from storage"""
        try:
            if self.mappings_file.exists():
                with open(self.mappings_file, 'r') as f:
                    data = json.load(f)
                    self.forward_mappings = data.get('forward', {})
                    self.reverse_mappings = data.get('reverse', {})
                    print(f"Loaded {len(self.forward_mappings)} persistent mappings")
        except Exception as e:
            print(f"Warning: Could not load persistent mappings: {e}")
            self.forward_mappings = {}
            self.reverse_mappings = {}
    
    def _save_persistent_mappings(self):
        """Save current mappings to storage"""
        try:
            data = {
                'forward': self.forward_mappings,
                'reverse': self.reverse_mappings,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.mappings_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save persistent mappings: {e}")
    
    def start_session(self, session_id: Optional[str] = None) -> str:
        """Start a new anonymisation session for cloud workflow"""
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session_id = session_id
        self.session_mappings[session_id] = {
            'forward': {},
            'reverse': {},
            'created': datetime.now().isoformat()
        }
        
        return session_id
    
    def export_session_mappings(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Export mappings for a session (for cloud analysis)"""
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id not in self.session_mappings:
            return {'error': f'Session {session_id} not found'}
        
        return {
            'session_id': session_id,
            'mappings': self.session_mappings[session_id]['forward'],
            'reverse_mappings': self.session_mappings[session_id]['reverse'],
            'created': self.session_mappings[session_id]['created']
        }
        
    async def anonymise_rag_output(self, text: str, context_sources: List[Dict]) -> Dict[str, Any]:
        """Anonymise a RAG output while preserving legal structure"""
        
        # Step 1: Extract sensitive patterns
        sensitive_patterns = self._identify_sensitive_data(text)
        
        # Step 2: Generate consistent pseudonyms using phi3
        anonymisation_map = await self._generate_pseudonyms(sensitive_patterns)
        
        # Step 3: Apply anonymisation
        anonymised_text = self._apply_anonymisation(text, anonymisation_map)
        
        # Step 4: Anonymise source references
        anonymised_sources = self._anonymise_sources(context_sources, anonymisation_map)
        
        return {
            'anonymised_text': anonymised_text,
            'anonymised_sources': anonymised_sources,
            'anonymisation_map': anonymisation_map,
            'original_length': len(text),
            'anonymised_length': len(anonymised_text),
            'entities_anonymised': len(anonymisation_map),
            'processing_time': datetime.now().isoformat()
        }
    
    def _identify_sensitive_data(self, text: str) -> Dict[str, List[str]]:
        """Identify potentially sensitive information patterns"""
        
        patterns = {
            'person_names': [],
            'companies': [],
            'case_numbers': [],
            'addresses': [],
            'phone_numbers': [],
            'email_addresses': [],
            'dates': []
        }
        
        # Person names (capitalized words, common name patterns)
        name_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        patterns['person_names'] = list(set(re.findall(name_pattern, text)))
        
        # Case numbers (common legal formats)
        case_pattern = r'\b[A-Z]{1,3}-\d{4}-\d{6}\b|\b\d{4} [A-Z]+ \d+\b'
        patterns['case_numbers'] = list(set(re.findall(case_pattern, text)))
        
        # Company names (with Ltd, Inc, plc, etc.)
        company_pattern = r'\b[A-Z][A-Za-z\s]+ (?:Ltd|Limited|Inc|plc|Corporation|Corp|University|College)\b'
        patterns['companies'] = list(set(re.findall(company_pattern, text)))
        
        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        patterns['email_addresses'] = list(set(re.findall(email_pattern, text)))
        
        # Phone numbers (UK format)
        phone_pattern = r'\b(?:\+44|0)\d{10,11}\b'
        patterns['phone_numbers'] = list(set(re.findall(phone_pattern, text)))
        
        # Remove empty lists
        return {k: v for k, v in patterns.items() if v}
    
    async def _generate_pseudonyms(self, sensitive_patterns: Dict[str, List[str]]) -> Dict[str, str]:
        """Use phi3 to generate realistic pseudonyms"""
        
        anonymisation_map = {}
        
        for category, items in sensitive_patterns.items():
            for item in items:
                if item not in anonymisation_map:
                    pseudonym = await self._generate_single_pseudonym(item, category)
                    anonymisation_map[item] = pseudonym
        
        return anonymisation_map
    
    async def _generate_single_pseudonym(self, original: str, category: str) -> str:
        """Generate a single pseudonym using phi3"""
        
        # Check if we already have a mapping for consistency
        cache_key = f"{category}_{original}"
        if cache_key in self.forward_mappings:
            return self.forward_mappings[cache_key]
        
        # Category-specific prompts for phi3
        prompts = {
            'person_names': f"Generate a realistic fake name to replace '{original}'. Respond with only the fake name, nothing else.",
            'companies': f"Generate a realistic fake company name to replace '{original}'. Keep the same type (university, corporation, etc.). Respond with only the fake name.",
            'case_numbers': f"Generate a realistic fake case number in the same format as '{original}'. Respond with only the fake case number.",
            'email_addresses': f"Generate a fake email address to replace '{original}'. Use fake names but keep realistic domain. Respond with only the email.",
            'phone_numbers': "Generate a fake UK phone number. Respond with only the number.",
            'addresses': f"Generate a fake address to replace '{original}'. Keep it realistic but completely fictional. Respond with only the address."
        }
        
        prompt = prompts.get(category, f"Generate a fake replacement for '{original}'. Respond with only the replacement.")
        
        try:
            timeout = aiohttp.ClientTimeout(total=30)  # Shorter timeout for speed
            async with aiohttp.ClientSession(timeout=timeout) as session:
                payload = {
                    "model": "phi3:latest",
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.7,  # Some creativity for realistic names
                    "top_p": 0.9,
                    "num_predict": 50,  # Short responses only
                    "system": "You are a data anonymisation assistant. Generate realistic fake replacements. Respond with only the replacement text, no explanation."
                }
                
                async with session.post("http://localhost:11434/api/generate", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        pseudonym = result.get('response', f"ANON_{category}_{len(self.name_mappings)}").strip()
                        
                        # Clean up response (phi3 sometimes adds extra text)
                        pseudonym = pseudonym.split('\n')[0].strip()
                        if pseudonym.startswith('"') and pseudonym.endswith('"'):
                            pseudonym = pseudonym[1:-1]
                        
                        # Cache for consistency
                        self.name_mappings[cache_key] = pseudonym
                        return pseudonym
                    else:
                        # Fallback to generic anonymisation
                        return f"ANON_{category}_{len(self.name_mappings)}"
                        
        except Exception as e:
            # Fallback to generic anonymisation
            return f"ANON_{category}_{len(self.name_mappings)}"
    
    def _apply_anonymisation(self, text: str, anonymisation_map: Dict[str, str]) -> str:
        """Apply the anonymisation mappings to the text"""
        
        anonymised_text = text
        
        # Apply replacements (longest first to avoid partial replacements)
        for original, pseudonym in sorted(anonymisation_map.items(), key=lambda x: len(x[0]), reverse=True):
            anonymised_text = anonymised_text.replace(original, pseudonym)
        
        return anonymised_text
    
    def _anonymise_sources(self, sources: List[Dict], anonymisation_map: Dict[str, str]) -> List[Dict]:
        """Anonymise source references"""
        
        anonymised_sources = []
        
        for source in sources:
            anonymised_source = source.copy()
            
            # Anonymise document names
            if 'document' in anonymised_source:
                for original, pseudonym in anonymisation_map.items():
                    anonymised_source['document'] = anonymised_source['document'].replace(original, pseudonym)
            
            # Anonymise text previews
            if 'text_preview' in anonymised_source:
                for original, pseudonym in anonymisation_map.items():
                    anonymised_source['text_preview'] = anonymised_source['text_preview'].replace(original, pseudonym)
            
            anonymised_sources.append(anonymised_source)
        
        return anonymised_sources
    
    def get_anonymisation_summary(self) -> Dict[str, Any]:
        """Get summary of all anonymisations performed"""
        
        return {
            'total_mappings': len(self.name_mappings),
            'person_names': len([k for k in self.name_mappings.keys() if k.startswith('person_names_')]),
            'companies': len([k for k in self.name_mappings.keys() if k.startswith('companies_')]),
            'case_numbers': len([k for k in self.name_mappings.keys() if k.startswith('case_numbers_')]),
            'other_entities': len([k for k in self.name_mappings.keys() if not any(k.startswith(prefix) for prefix in ['person_names_', 'companies_', 'case_numbers_'])]),
            'last_updated': datetime.now().isoformat()
        }

# Global instance for consistent mappings across requests
global_anonymiser = PseudoAnonymiser()

async def anonymise_rag_result(rag_result: Dict[str, Any]) -> Dict[str, Any]:
    """Main function to anonymise a complete RAG result"""
    
    original_answer = rag_result.get('answer', '')
    original_sources = rag_result.get('sources', [])
    
    if not original_answer:
        return rag_result
    
    # Perform anonymisation
    anonymisation_result = await global_anonymiser.anonymise_rag_output(original_answer, original_sources)
    
    # Create anonymised version of the full result
    anonymised_result = rag_result.copy()
    anonymised_result.update({
        'answer': anonymisation_result['anonymised_text'],
        'sources': anonymisation_result['anonymised_sources'],
        'anonymisation_info': {
            'entities_anonymised': anonymisation_result['entities_anonymised'],
            'anonymisation_map_summary': len(anonymisation_result['anonymisation_map']),
            'processing_time': anonymisation_result['processing_time'],
            'original_preserved': False  # Set to True if you want to keep original
        }
    })
    
    return anonymised_result

if __name__ == "__main__":
    # Test the anonymisation system
    print("🔒 Pseudoanonymisation Module Ready")
    print("Uses phi3 for creative, fast anonymisation of sensitive data") 