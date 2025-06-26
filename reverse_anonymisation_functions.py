#!/usr/bin/env python3
"""
Reverse Anonymisation Functions for Cloud Workflows
Complete the bidirectional anonymisation system
"""

import json
import re
from typing import Dict, Any, List, Optional

async def reverse_anonymise_text(text: str, reverse_mappings: Dict[str, str]) -> str:
    """Reverse anonymisation of text using provided mappings"""
    
    reversed_text = text
    
    # Apply reverse mappings (longest pseudonyms first to avoid partial replacements)
    for pseudonym, original in sorted(reverse_mappings.items(), key=lambda x: len(x[0]), reverse=True):
        if pseudonym in reversed_text:
            reversed_text = reversed_text.replace(pseudonym, original)
    
    return reversed_text

async def reverse_anonymise_rag_result(cloud_response: Dict[str, Any], session_mappings: Dict[str, Any]) -> Dict[str, Any]:
    """Reverse anonymise a complete cloud analysis result"""
    
    reverse_mappings = session_mappings.get('reverse_mappings', {})
    
    if not reverse_mappings:
        return {
            **cloud_response,
            'reverse_anonymisation_applied': False,
            'error': 'No reverse mappings available'
        }
    
    # Reverse anonymise the main answer
    original_answer = cloud_response.get('answer', '')
    if original_answer:
        cloud_response['answer'] = await reverse_anonymise_text(original_answer, reverse_mappings)
    
    # Reverse anonymise source references if present
    sources = cloud_response.get('sources', [])
    for source in sources:
        if 'document' in source:
            source['document'] = await reverse_anonymise_text(source['document'], reverse_mappings)
        if 'text_preview' in source:
            source['text_preview'] = await reverse_anonymise_text(source['text_preview'], reverse_mappings)
    
    # Add reverse anonymisation metadata
    cloud_response.update({
        'reverse_anonymisation_applied': True,
        'entities_restored': len(reverse_mappings),
        'session_id': session_mappings.get('session_id', 'unknown')
    })
    
    return cloud_response

def create_cloud_workflow_package(anonymised_result: Dict[str, Any], session_mappings: Dict[str, Any]) -> Dict[str, Any]:
    """Package anonymised data for cloud analysis"""
    
    return {
        'anonymised_data': {
            'answer': anonymised_result.get('answer', ''),
            'sources': anonymised_result.get('sources', []),
            'query_context': anonymised_result.get('debug_info', '')
        },
        'reverse_mappings': session_mappings.get('reverse_mappings', {}),
        'session_metadata': {
            'session_id': session_mappings.get('session_id'),
            'created': session_mappings.get('created'),
            'entities_anonymised': len(session_mappings.get('reverse_mappings', {}))
        },
        'instructions': {
            'privacy_notice': 'This data has been anonymised. Real names replaced with pseudonyms.',
            'analysis_request': 'Please analyze this legal content and return structured insights.',
            'return_format': 'Maintain all anonymised names in your response for accurate reverse mapping.'
        }
    }

def validate_reverse_mapping_integrity(session_mappings: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that reverse mappings are complete and consistent"""
    
    forward = session_mappings.get('forward', {})
    reverse = session_mappings.get('reverse', {})
    
    issues = []
    
    # Check forward → reverse consistency
    for original, pseudonym in forward.items():
        if pseudonym not in reverse:
            issues.append(f"Missing reverse mapping for pseudonym: {pseudonym}")
        elif reverse[pseudonym] != original:
            issues.append(f"Inconsistent mapping: {original} ↔ {pseudonym}")
    
    # Check reverse → forward consistency  
    for pseudonym, original in reverse.items():
        if original not in forward:
            issues.append(f"Missing forward mapping for original: {original}")
    
    return {
        'is_valid': len(issues) == 0,
        'forward_count': len(forward),
        'reverse_count': len(reverse),
        'issues': issues,
        'consistency_score': 1.0 - (len(issues) / max(len(forward) + len(reverse), 1))
    }

# Example cloud workflow functions
async def send_to_cloud_analysis(cloud_package: Dict[str, Any], cloud_endpoint: str = "gpt-4") -> Dict[str, Any]:
    """Send anonymised data to cloud model for analysis"""
    
    # This would integrate with your preferred cloud API
    # For now, return a mock response showing the structure
    
    mock_cloud_response = {
        'answer': f"Cloud analysis complete. The case involves [anonymised parties]. Key findings: [detailed analysis using pseudonyms].",
        'analysis_type': 'legal_document_review',
        'confidence': 0.92,
        'recommendations': [
            'Review contract clause regarding [anonymised entity]',
            'Consider precedent cases similar to [anonymised case reference]'
        ],
        'cloud_model': cloud_endpoint,
        'processing_time': 15.2
    }
    
    return mock_cloud_response

async def complete_cloud_workflow(original_rag_result: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """Complete end-to-end cloud analysis with anonymisation"""
    
    from pseudoanonymisation_module import global_anonymiser
    
    try:
        # Step 1: Start session
        if not session_id:
            session_id = global_anonymiser.start_session()
        
        # Step 2: Anonymise for cloud
        from pseudoanonymisation_module import anonymise_rag_result
        anonymised_result = await anonymise_rag_result(original_rag_result)
        session_mappings = global_anonymiser.export_session_mappings(session_id)
        
        # Step 3: Validate mappings
        validation = validate_reverse_mapping_integrity(session_mappings)
        if not validation['is_valid']:
            return {
                'error': 'Mapping validation failed',
                'issues': validation['issues']
            }
        
        # Step 4: Package for cloud
        cloud_package = create_cloud_workflow_package(anonymised_result, session_mappings)
        
        # Step 5: Send to cloud (mock for now)
        cloud_response = await send_to_cloud_analysis(cloud_package)
        
        # Step 6: Reverse anonymisation
        final_result = await reverse_anonymise_rag_result(cloud_response, session_mappings)
        
        # Step 7: Add workflow metadata
        final_result.update({
            'cloud_workflow_complete': True,
            'session_id': session_id,
            'anonymisation_validation': validation,
            'privacy_protected': True
        })
        
        return final_result
        
    except Exception as e:
        return {
            'error': f'Cloud workflow failed: {str(e)}',
            'session_id': session_id,
            'cloud_workflow_complete': False
        }

if __name__ == "__main__":
    print("🔄 Reverse Anonymisation Functions for Cloud Workflows")
    print("Complete bidirectional privacy protection system") 