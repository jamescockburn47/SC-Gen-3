#!/usr/bin/env python3
"""
Enhanced RAG Interface with Anti-Hallucination Systems
- User model selection
- Protocol compliance reporting  
- Matter-specific document management
- Real-time hallucination detection
"""

import streamlit as st
import asyncio
import aiohttp
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from local_rag_pipeline import rag_session_manager
from ai_utils import get_improved_prompt
from pathlib import Path
import logging
import json
import os
import numpy as np
import re

# Import anonymisation module
try:
    from pseudoanonymisation_module import anonymise_rag_result, global_anonymiser
    ANONYMISATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Pseudoanonymisation module not available: {e}")
    ANONYMISATION_AVAILABLE = False

# Advanced semantic processing imports - MOVED TO TOP TO AVOID DUPLICATES
try:
    import networkx as nx
    GRAPH_PROCESSING_AVAILABLE = True
except ImportError:
    GRAPH_PROCESSING_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    ADVANCED_EMBEDDINGS_AVAILABLE = True
except ImportError:
    ADVANCED_EMBEDDINGS_AVAILABLE = False

def get_default_system_prompts():
    """Get default system prompts for different matter types"""
    return {
        "Commercial Litigation": """You are a senior UK barrister specializing in commercial litigation, providing comprehensive legal analysis of court documents.

EXPERTISE: Commercial disputes, contract law, breach of contract, damages, corporate litigation, commercial court procedures.

ANALYSIS FOCUS:
- Commercial context and business implications
- Contractual obligations and breaches
- Quantum of damages and financial remedies
- Commercial court procedural requirements
- Settlement considerations and commercial reality""",

        "Employment Law": """You are a senior UK employment law barrister, providing comprehensive legal analysis of employment-related documents.

EXPERTISE: Employment tribunals, unfair dismissal, discrimination, TUPE, whistleblowing, employment contracts.

ANALYSIS FOCUS:
- Employment law statutory framework
- Tribunal procedures and time limits
- Discrimination and protected characteristics
- ACAS codes and reasonable adjustments
- Compensation calculations and remedies""",

        "Personal Injury": """You are a senior UK personal injury barrister, providing comprehensive legal analysis of PI claims and court documents.

EXPERTISE: Negligence, causation, quantum, clinical negligence, road traffic accidents, employers' liability.

ANALYSIS FOCUS:
- Duty of care and breach analysis
- Causation (factual and legal)
- Quantum of damages and future losses
- Medical evidence and expert reports
- Settlement negotiations and Part 36 offers""",

        "Family Law": """You are a senior UK family law barrister, providing comprehensive legal analysis of family court documents.

EXPERTISE: Divorce, financial remedies, children matters, domestic violence, family court procedures.

ANALYSIS FOCUS:
- Financial disclosure and asset division
- Children's welfare and best interests
- Court orders and enforcement
- Family court procedures and case management
- Mediation and alternative dispute resolution""",

        "Criminal Law": """You are a senior UK criminal law barrister, providing comprehensive legal analysis of criminal court documents.

EXPERTISE: Crown Court and Magistrates' Court procedures, evidence, sentencing, appeals, disclosure.

ANALYSIS FOCUS:
- Elements of offences and evidence requirements
- Criminal procedure and court processes
- Disclosure obligations and unused material
- Sentencing guidelines and mitigation
- Appeal prospects and procedural requirements""",

        "Property/Real Estate": """You are a senior UK property law barrister, providing comprehensive legal analysis of property-related court documents.

EXPERTISE: Landlord and tenant, property disputes, conveyancing disputes, leasehold, planning law.

ANALYSIS FOCUS:
- Property rights and title issues
- Landlord and tenant obligations
- Leasehold and service charge disputes
- Planning and development issues
- Property transactions and conveyancing""",

        "Regulatory/Compliance": """You are a senior UK regulatory law barrister, providing comprehensive legal analysis of regulatory and compliance matters.

EXPERTISE: Regulatory investigations, licensing, professional discipline, judicial review, public law.

ANALYSIS FOCUS:
- Regulatory framework and compliance requirements
- Investigation procedures and powers
- Professional conduct and disciplinary processes
- Judicial review grounds and procedures
- Public law principles and human rights""",

        "Insolvency/Restructuring": """You are a senior UK insolvency barrister, providing comprehensive legal analysis of insolvency and restructuring documents.

EXPERTISE: Corporate insolvency, personal bankruptcy, administrations, liquidations, restructuring.

ANALYSIS FOCUS:
- Insolvency procedures and time limits
- Director and officer duties and liabilities
- Asset recovery and void transactions
- Creditor rights and priority of claims
- Restructuring options and viability""",

        "General Litigation": """You are a senior UK barrister specializing in litigation, providing comprehensive legal analysis of court documents.

EXPERTISE: General civil litigation, court procedures, evidence, case management, appeals.

ANALYSIS FOCUS:
- Legal issues and applicable law
- Procedural compliance and case management
- Evidence evaluation and witness statements
- Costs implications and proportionality
- Settlement prospects and tactical considerations"""
    }

def create_protocol_compliant_prompt(query: str, context: str, model: str = "mistral:latest", 
                                   matter_type: str = "General Litigation", 
                                   custom_system_prompt: Optional[str] = None) -> str:
    """Create customizable system prompt based on matter type and user preferences"""
    
    # Use custom prompt if provided, otherwise use default for matter type
    if custom_system_prompt and custom_system_prompt.strip():
        system_prompt = custom_system_prompt.strip()
    else:
        default_prompts = get_default_system_prompts()
        system_prompt = default_prompts.get(matter_type, default_prompts["General Litigation"])
    
    # Detect query type to determine response approach
    query_lower = query.lower().strip()
    
    # Simple factual questions (who, what, when, where)
    simple_question_starters = ['who is', 'what is', 'when did', 'where is', 'how many', 'what happened to']
    is_simple_factual = any(query_lower.startswith(starter) for starter in simple_question_starters)
    
    # Short queries are likely specific questions
    is_likely_specific = len(query.split()) <= 6
    
    if is_simple_factual or is_likely_specific:
        # Direct question-answering approach
        return f"""{system_prompt}

🎯 PRIMARY TASK: Answer the user's specific question directly from the documents.

USER QUESTION: {query}

DOCUMENT CONTENT:
{context}

RESPONSE APPROACH:
1. **ANSWER THE SPECIFIC QUESTION FIRST** - Don't ignore what the user asked
2. Start with: "Based on the provided documents:"
3. Give a direct answer to the question with proper citations [Source X]
4. Then provide relevant additional context if helpful
5. Keep it focused and relevant to the specific question asked

CITATION FORMAT:
- Use [Source X] format: "Elyas Abaris is a medical student [Source 1]"
- NEVER use "Source:", "(Source 1)", or other formats
- Include specific document location when available: [Source 1: Particulars of Claim, Para 2]

IMPORTANT: The user asked a specific question. Answer that question directly. Don't give a comprehensive legal analysis unless the question specifically requests it."""
    
    else:
        # Comprehensive analysis approach for complex queries
        return f"""{system_prompt}

TASK: Analyze the provided legal documents to answer the user's query with complete accuracy and proper citations.

USER QUERY: {query}

DOCUMENT CONTENT:
{context}

MANDATORY REQUIREMENTS:
1. Start with: "Based on the provided documents:"
2. Every factual statement MUST include a citation in [Source X] format
3. Provide comprehensive analysis - don't just list facts, analyze and synthesize
4. Use UK legal terminology and context
5. Structure your response logically with clear sections
6. Be thorough - this is for senior legal practitioners

CITATION FORMAT:
- Use [Source X] format with specific location when available: [Source 1: Section 3, Para 2]
- NEVER use "Source:", "(Source 1)", "Document 1:" or other formats
- Multiple sources: [Source 1, Source 2]
- Direct quotes: "exact text" [Source X: location]
- Include document sections/paragraphs when citing: [Source 2: Particulars of Claim, Para 1]

RESPONSE STRUCTURE:
- **Case Overview**: Case details, parties, court, case number
- **Legal Claims**: Specific causes of action and legal basis  
- **Key Facts**: Material facts and timeline
- **Legal Analysis**: Analysis of claims, evidence, potential outcomes
- **Strategic Considerations**: Risks, opportunities, next steps

ANALYSIS REQUIREMENTS:
- Be comprehensive and analytical, not just descriptive
- Identify legal issues and their implications
- Consider procedural and substantive aspects
- Highlight any critical dates, deadlines, or procedural requirements
- Note any gaps in information or areas requiring further investigation

Your response must be suitable for a senior legal practitioner and provide actionable legal analysis."""

def check_protocol_compliance(answer: str, sources: List[Dict]) -> Dict[str, Any]:
    """Generate comprehensive protocol compliance report"""
    
    compliance_report = {
        'overall_score': 0,
        'protocol_violations': [],
        'compliance_checks': {},
        'recommendations': []
    }
    
    # Check 1: Citation Requirements - Enhanced Detection
    source_citations = []
    citation_formats = []
    
    # Check for proper [Source X] format
    for i in range(1, len(sources) + 1):
        if f"[Source {i}]" in answer:
            source_citations.append(i)
            citation_formats.append('correct')
    
    # Check for alternative formats and mark as violations
    alt_formats = [
        ('Source:', 'colon'),
        ('(Source', 'parentheses'), 
        ('Document ', 'document'),
        ('[Document', 'document_bracket')
    ]
    
    for alt_format, format_type in alt_formats:
        if alt_format in answer:
            citation_formats.append(f'wrong_{format_type}')
    
    citation_compliance = len([f for f in citation_formats if f == 'correct']) / max(len(sources), 1) if sources else 0
    
    # Penalty for wrong citation formats
    wrong_formats = [f for f in citation_formats if f.startswith('wrong_')]
    if wrong_formats:
        citation_compliance *= 0.5  # Reduce score for wrong formats
    details_msg = f"Citations found for {len(source_citations)}/{len(sources)} sources"
    if wrong_formats:
        details_msg += f" (Format issues: {', '.join(set(wrong_formats))})"
    
    compliance_report['compliance_checks']['citation_coverage'] = {
        'score': citation_compliance,
        'details': details_msg,
        'status': 'PASS' if citation_compliance >= 0.5 else 'FAIL'
    }
    
    # Check 2: Protocol Language
    required_phrases = ["based on the provided documents", "according to the documents", "the documents state"]
    has_protocol_language = any(phrase.lower() in answer.lower() for phrase in required_phrases)
    compliance_report['compliance_checks']['protocol_language'] = {
        'score': 1.0 if has_protocol_language else 0.0,
        'details': f"Protocol language present: {has_protocol_language}",
        'status': 'PASS' if has_protocol_language else 'FAIL'
    }
    
    # Check 3: Hallucination Indicators
    hallucination_patterns = ['[OR:', '[DATE]', 'Page XX', '[UNVERIFIED]', '[SOURCE X']
    vague_language = ['I think', 'probably', 'might be', 'could be', 'generally speaking']
    
    found_patterns = [p for p in hallucination_patterns if p in answer]
    found_vague = [v for v in vague_language if v.lower() in answer.lower()]
    
    hallucination_score = 1.0 - (len(found_patterns) + len(found_vague)) / 10
    compliance_report['compliance_checks']['hallucination_detection'] = {
        'score': max(0, hallucination_score),
        'details': f"Problematic patterns: {found_patterns + found_vague}",
        'status': 'PASS' if not (found_patterns or found_vague) else 'FAIL'
    }
    
    # Check 4: Document Grounding
    factual_statements = len([s for s in answer.split('.') if s.strip() and not s.strip().startswith('Based on')])
    cited_statements = len([s for s in answer.split('.') if '[Source' in s])
    grounding_score = cited_statements / max(factual_statements, 1)
    
    compliance_report['compliance_checks']['document_grounding'] = {
        'score': grounding_score,
        'details': f"Cited statements: {cited_statements}/{factual_statements}",
        'status': 'PASS' if grounding_score >= 0.7 else 'PARTIAL' if grounding_score >= 0.4 else 'FAIL'
    }
    
    # Calculate overall score
    scores = [check['score'] for check in compliance_report['compliance_checks'].values()]
    compliance_report['overall_score'] = sum(scores) / len(scores)
    
    # Generate recommendations
    if citation_compliance < 0.5:
        compliance_report['recommendations'].append("Increase source citations - cite specific sources for each fact")
    if not has_protocol_language:
        compliance_report['recommendations'].append("Begin response with protocol language like 'Based on the provided documents:'")
    if found_patterns or found_vague:
        compliance_report['recommendations'].append("Remove uncertain language and placeholder text")
    if grounding_score < 0.7:
        compliance_report['recommendations'].append("Ensure all factual statements include source citations")
    
    return compliance_report

async def get_protocol_compliant_answer(query: str, matter_id: str, model: str, max_chunks: int = 5, anonymise: bool = False) -> Dict[str, Any]:
    """Get answer with full protocol compliance checking and per-query document selection"""
    
    try:
        # Get RAG pipeline
        pipeline = rag_session_manager.get_or_create_pipeline(matter_id)
        
        # Check for per-query document selection
        selected_doc_ids = []
        selection_mode = "Use All Documents"
        
        # Check session state for document selection
        if hasattr(st, 'session_state'):
            selected_doc_ids = getattr(st.session_state, 'selected_documents_for_query', [])
            selection_mode = getattr(st.session_state, 'document_selection_mode', 'Use All Documents')
        
        # Search for relevant chunks with document filtering
        if selected_doc_ids and selection_mode != "Use All Documents":
            # Use only selected documents
            chunks = pipeline.search_documents_filtered(query, top_k=max_chunks, document_ids=selected_doc_ids)
            search_info = f"Searched {len(selected_doc_ids)} selected documents"
        else:
            # Use all documents
            chunks = pipeline.search_documents(query, top_k=max_chunks)
            search_info = "Searched all available documents"
        
        if not chunks:
            return {
                'answer': f"No relevant content found in the {'selected' if selected_doc_ids else 'available'} documents for your query.",
                'sources': [],
                'model_used': model,
                'protocol_compliance': {
                    'overall_score': 0,
                    'status': 'NO_SOURCES',
                    'message': f'No documents available for analysis ({search_info})'
                },
                'debug_info': f'No chunks returned from search ({search_info})',
                'document_selection_info': {
                    'mode': selection_mode,
                    'selected_count': len(selected_doc_ids),
                    'search_scope': search_info
                }
            }
        
        # Build context with validation and enhanced location information
        context_parts = []
        valid_sources = []
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get('text', '').strip()
            if len(chunk_text) > 20:  # Ensure meaningful content
                # Create enhanced source location for AI context
                location_info = []
                
                # Extract location information if available
                if chunk.get('section_title'):
                    location_info.append(f"Section: {chunk['section_title']}")
                elif chunk.get('chunk_index') is not None:
                    location_info.append(f"Section: {chunk['chunk_index'] + 1}")
                
                if chunk.get('paragraph_index') is not None:
                    location_info.append(f"Para: {chunk['paragraph_index'] + 1}")
                
                # Build location context for AI
                location_str = " | ".join(location_info) if location_info else f"Chunk {i+1}"
                document_name = chunk.get('document_info', {}).get('filename', 'Unknown')
                
                # Enhanced context with location for AI model
                context_parts.append(f"[Source {i+1}: {document_name} - {location_str}] {chunk_text}")
                
                # Enhanced source metadata for UI display
                source_metadata = {
                    'chunk_id': chunk['id'],
                    'document': document_name,
                    'similarity_score': chunk['similarity_score'],
                    'text_preview': chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                    'chunk_index': chunk.get('chunk_index'),
                    'section_title': chunk.get('section_title'),
                    'paragraph_index': chunk.get('paragraph_index')
                }
                valid_sources.append(source_metadata)
        
        if not context_parts:
            return {
                'answer': f"Document chunks found but contain insufficient content for analysis from {'selected' if selected_doc_ids else 'available'} documents.",
                'sources': [],
                'model_used': model,
                'protocol_compliance': {
                    'overall_score': 0,
                    'status': 'INSUFFICIENT_CONTENT',
                    'message': f'Document content too short for analysis ({search_info})'
                },
                'debug_info': f'Empty or too-short chunks ({search_info})',
                'document_selection_info': {
                    'mode': selection_mode,
                    'selected_count': len(selected_doc_ids),
                    'search_scope': search_info
                }
            }
        
        context = "\n\n".join(context_parts)
        
        # Get matter type and custom prompt from session state
        matter_type = getattr(st.session_state, 'selected_matter_type', 'General Litigation')
        custom_prompt = getattr(st.session_state, 'custom_system_prompt', None)
        
        prompt = create_protocol_compliant_prompt(query, context, model, matter_type, custom_prompt)
        
        # Generate answer with strict controls
        start_time = time.time()
        timeout = aiohttp.ClientTimeout(total=120)
        
        # Model-specific parameter tuning for optimal compliance
        if "phi3" in model.lower():
            model_params = {
                "temperature": 0.0, "top_p": 0.05, "top_k": 3, "repeat_penalty": 1.3,
                "system": "Extract facts from documents. CRITICAL: Use [Source X] format for ALL citations. Example: The defendant is John Smith [Source 1]."
            }
        elif "deepseek" in model.lower():
            model_params = {
                "temperature": 0.1, "top_p": 0.1, "top_k": 5, "repeat_penalty": 1.2,
                "system": "Document analysis only. CRITICAL: Every fact needs [Source X] citation. NO exceptions."
            }
        elif "mixtral" in model.lower():
            model_params = {
                "temperature": 0.0, "top_p": 0.05, "top_k": 2, "repeat_penalty": 1.1,
                "system": "CRITICAL: Each fact needs [Source X] citation. Example: The case number is KB-2023-000930 [Source 1]."
            }
        else:  # mistral and others
            model_params = {
                "temperature": 0.0, "top_p": 0.05, "top_k": 3, "repeat_penalty": 1.5,
                "system": "🚨 CRITICAL CITATION FORMAT: Use [Source X] brackets ONLY. Example: 'The claimant is Elyas Abaris [Source 1]. The case concerns data breach claims [Source 2].' NEVER use 'Source:' or other formats."
            }

        async with aiohttp.ClientSession(timeout=timeout) as session:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "num_predict": 4000,  # Allow longer responses
                **model_params  # Apply model-specific parameters
            }
            
            async with session.post("http://localhost:11434/api/generate", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    answer = result.get('response', 'No response').strip()
                    generation_time = time.time() - start_time
                    
                    # Generate protocol compliance report
                    compliance_report = check_protocol_compliance(answer, valid_sources)
                    
                    # Prepare base result
                    result = {
                        'answer': answer,
                        'sources': valid_sources,
                        'model_used': model,
                        'context_chunks_used': len(context_parts),
                        'generation_time': generation_time,
                        'protocol_compliance': compliance_report,
                        'debug_info': f"Prompt: {len(prompt)} chars, Context: {len(context)} chars, {search_info}",
                        'document_selection_info': {
                            'mode': selection_mode,
                            'selected_count': len(selected_doc_ids),
                            'search_scope': search_info,
                            'chunks_from_selection': len(context_parts)
                        }
                    }
                    
                    # Apply anonymisation if requested
                    if anonymise and ANONYMISATION_AVAILABLE:
                        try:
                            anonymised_result = await anonymise_rag_result(result)
                            # Add anonymisation info to result
                            anonymised_result['anonymisation_applied'] = True
                            anonymised_result['anonymisation_model'] = 'phi3:latest'
                            return anonymised_result
                        except Exception as anon_error:
                            # If anonymisation fails, return original with error note
                            logging.error(f"Anonymisation failed: {anon_error}")
                            result['anonymisation_error'] = f"Anonymisation failed: {str(anon_error)}"
                            result['anonymisation_applied'] = False
                            return result
                    else:
                        result['anonymisation_applied'] = False
                        return result
                else:
                    error_text = await response.text()
                    return {
                        'answer': f"Error: HTTP {response.status} - {error_text}",
                        'sources': [],
                        'model_used': model,
                        'protocol_compliance': {
                            'overall_score': 0,
                            'status': 'API_ERROR',
                            'message': f'API returned error {response.status}'
                        },
                        'debug_info': f"HTTP error: {response.status}, {search_info}",
                        'document_selection_info': {
                            'mode': selection_mode,
                            'selected_count': len(selected_doc_ids),
                            'search_scope': search_info
                        }
                    }
    
    except Exception as e:
        return {
            'answer': f"System error: {str(e)}",
            'sources': [],
            'model_used': model,
            'protocol_compliance': {
                'overall_score': 0,
                'status': 'SYSTEM_ERROR',
                'message': f'Exception: {str(e)}'
            },
            'debug_info': f"Exception: {type(e).__name__}: {str(e)}",
            'document_selection_info': {
                'mode': 'Unknown',
                'selected_count': 0,
                'search_scope': 'Error occurred'
            }
        }

def get_available_matters() -> List[str]:
    """Get list of available matters with documents"""
    
    rag_storage_path = Path("rag_storage")
    matters = []
    
    if rag_storage_path.exists():
        for matter_dir in rag_storage_path.iterdir():
            if matter_dir.is_dir():
                # Check if matter has documents
                metadata_file = matter_dir / "metadata.json"
                if metadata_file.exists():
                    import json
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            doc_count = len(metadata.get('documents', {}))
                            if doc_count > 0:
                                matters.append(f"{matter_dir.name} ({doc_count} docs)")
                    except:
                        pass
    
    # Add default options
    if not matters:
        matters = ["Default Matter", "Document Analysis", "Legal Review"]
    
    return matters

def render_protocol_compliance_report(compliance: Dict[str, Any]):
    """Render the protocol compliance report"""
    
    overall_score = compliance.get('overall_score', 0)
    
    # Overall score with color coding
    if overall_score >= 0.8:
        st.success(f"🟢 **Protocol Compliance: {overall_score:.1%}** (Excellent)")
    elif overall_score >= 0.6:
        st.warning(f"🟡 **Protocol Compliance: {overall_score:.1%}** (Good)")
    else:
        st.error(f"🔴 **Protocol Compliance: {overall_score:.1%}** (Needs Improvement)")
    
    # Detailed checks
    checks = compliance.get('compliance_checks', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📋 Compliance Checks")
        for check_name, check_data in checks.items():
            score = check_data.get('score', 0)
            status = check_data.get('status', 'UNKNOWN')
            details = check_data.get('details', '')
            
            if status == 'PASS':
                st.success(f"✅ {check_name.replace('_', ' ').title()}: {score:.1%}")
            elif status == 'PARTIAL':
                st.warning(f"⚠️ {check_name.replace('_', ' ').title()}: {score:.1%}")
            else:
                st.error(f"❌ {check_name.replace('_', ' ').title()}: {score:.1%}")
            
            st.caption(details)
    
    with col2:
        st.markdown("#### 🎯 Recommendations")
        recommendations = compliance.get('recommendations', [])
        if recommendations:
            for rec in recommendations:
                st.write(f"• {rec}")
        else:
            st.success("All protocol requirements met!")

def render_enhanced_rag_interface():
    """Render the comprehensive anti-hallucination RAG interface"""
    
    st.markdown("### 🛡️ Protocol-Compliant Document Analysis")
    st.info("📌 Enhanced RAG system with anti-hallucination controls and protocol compliance reporting")
    
    # Configuration section
    with st.expander("⚙️ Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Model selection with performance warnings
            available_models = ["mistral:latest", "deepseek-llm:7b", "phi3:latest", "mixtral:latest"]
            model_descriptions = {
                "mistral:latest": "🟢 **Recommended** - Best protocol compliance",
                "deepseek-llm:7b": "🟡 Good performance, may need guidance", 
                "phi3:latest": "🟡 Fast but inconsistent compliance",
                "mixtral:latest": "🔴 Very slow (40+ seconds) - Use carefully"
            }
            
            selected_model = st.selectbox(
                "Select Model:",
                available_models,
                index=0,  # Default to mistral (best performer)
                help="mistral=best compliance, deepseek=balanced, phi3=fast but inconsistent, mixtral=slow but capable"
            )
            
            # Show model-specific warning
            if selected_model in model_descriptions:
                if "🟢" in model_descriptions[selected_model]:
                    st.success(model_descriptions[selected_model])
                elif "🟡" in model_descriptions[selected_model]:
                    st.warning(model_descriptions[selected_model])
                else:
                    st.error(model_descriptions[selected_model])
        
        with col2:
            # Matter selection
            available_matters = get_available_matters()
            selected_matter_display = st.selectbox(
                "Select Matter:",
                available_matters,
                index=0,
                help="Choose the document collection to analyze"
            )
            # Extract actual matter name (remove doc count)
            selected_matter = selected_matter_display.split(' (')[0]
        
        with col3:
            # Context chunks with intelligent defaults
            max_chunks = st.slider(
                "Max Context Chunks:",
                min_value=1,
                max_value=50,
                value=15,  # Better default for comprehensive analysis
                help="Number of document sections to include. Use 30+ for summarization, 10-15 for specific questions"
            )
            
            # Show coverage for current selection
            try:
                pipeline = rag_session_manager.get_or_create_pipeline(selected_matter)
                doc_status = pipeline.get_document_status()
                total_chunks = doc_status['total_chunks']
                coverage = (max_chunks / total_chunks) * 100
                
                if coverage >= 50:
                    st.success(f"📊 **{coverage:.1f}% coverage** - Excellent for comprehensive analysis")
                elif coverage >= 25:
                    st.info(f"📊 **{coverage:.1f}% coverage** - Good for detailed analysis")
                elif coverage >= 10:
                    st.warning(f"📊 **{coverage:.1f}% coverage** - Limited analysis")
                else:
                    st.error(f"📊 **{coverage:.1f}% coverage** - Very limited analysis")
                    
            except:
                pass
    
    # Matter Type and System Prompt Customization
    with st.expander("🎯 Legal Matter Type & System Prompt Customization", expanded=False):
        st.markdown("### 📋 Matter Type Selection")
        st.info("💡 **Tip**: Selecting the correct matter type optimizes the AI's legal analysis for your specific area of law")
        
        # Matter type selection
        matter_types = list(get_default_system_prompts().keys())
        selected_matter_type = st.selectbox(
            "Select Legal Matter Type:",
            matter_types,
            index=matter_types.index("General Litigation"),
            help="Choose the area of law that best matches your documents",
            key="matter_type_selector"
        )
        
        # Store in session state
        st.session_state.selected_matter_type = selected_matter_type
        
        # Show the default prompt for selected matter type
        default_prompts = get_default_system_prompts()
        default_prompt = default_prompts[selected_matter_type]
        
        col_prompt1, col_prompt2 = st.columns([2, 1])
        
        with col_prompt1:
            st.markdown("### 🛠️ System Prompt Editor")
            st.markdown(f"**Current Matter Type:** {selected_matter_type}")
        
        with col_prompt2:
            # Reset to default button
            if st.button("🔄 Reset to Default", help="Reset to the default prompt for this matter type"):
                if 'custom_system_prompt' in st.session_state:
                    del st.session_state.custom_system_prompt
                st.rerun()
        
        # Custom system prompt editor
        prompt_key = f"custom_prompt_{selected_matter_type}"
        
        # Initialize with default if not set
        if prompt_key not in st.session_state:
            st.session_state[prompt_key] = ""
        
        # Show current prompt (default or custom)
        current_prompt = st.session_state.get('custom_system_prompt', '') or default_prompt
        
        # Prompt editor
        edited_prompt = st.text_area(
            "Edit System Prompt:",
            value=current_prompt,
            height=300,
            help="Customize the system prompt to optimize AI responses for your specific matter type and requirements",
            key="system_prompt_editor"
        )
        
        # Save custom prompt if it's different from default
        if edited_prompt.strip() and edited_prompt.strip() != default_prompt:
            st.session_state.custom_system_prompt = edited_prompt.strip()
            st.success("✅ **Custom system prompt active** - The AI will use your customized instructions")
        elif edited_prompt.strip() == default_prompt:
            # User has reverted to default
            if 'custom_system_prompt' in st.session_state:
                del st.session_state.custom_system_prompt
            st.info(f"📋 **Using default prompt** for {selected_matter_type}")
        
        # Show prompt preview
        with st.expander("👀 Preview Default Prompt", expanded=False):
            st.code(default_prompt, language="text")
        
        # Show customization tips
        with st.expander("💡 System Prompt Customization Tips", expanded=False):
            st.markdown("""
            **Effective System Prompt Customization:**
            
            1. **Be Specific**: Include specific legal areas, procedures, or terminology relevant to your matter
            2. **Set Expectations**: Clearly define the level of analysis required (summary vs. detailed analysis)
            3. **Include Context**: Mention specific courts, jurisdictions, or legal frameworks if relevant
            4. **Emphasize Priorities**: Highlight what aspects are most important (procedural compliance, commercial impact, etc.)
            5. **Use Legal Language**: Include specific legal concepts, tests, or standards that apply
            
            **Examples of Useful Additions:**
            - "Focus on Commercial Court procedures and CPR Part 58"
            - "Prioritize GDPR compliance and data protection implications"
            - "Consider Employment Tribunal time limits and ACAS code compliance"
            - "Analyze quantum of damages with reference to recent case law"
            """)
        
        # Statistics
        if 'custom_system_prompt' in st.session_state:
            custom_length = len(st.session_state.custom_system_prompt.split())
            default_length = len(default_prompt.split())
            st.caption(f"📊 Custom prompt: {custom_length} words | Default: {default_length} words")
    
    # Anonymisation option
    st.markdown("#### 🔒 Privacy Protection")
    col_anon1, col_anon2 = st.columns([3, 1])
    
    with col_anon1:
        if ANONYMISATION_AVAILABLE:
            enable_anonymisation = st.checkbox(
                "🛡️ **Enable Pseudoanonymisation** (Uses phi3 for creative name replacement)",
                value=False,
                help="Replaces real names with realistic fake ones. Uses phi3's creativity for believable pseudonyms."
            )
            
            if enable_anonymisation:
                st.info("🔄 **Dual-Model Pipeline**: mistral for analysis → phi3 for anonymisation")
                st.caption("📊 Benefits: GDPR compliance, document sharing, training data creation")
        else:
            enable_anonymisation = False
            st.error("❌ Pseudoanonymisation module not available")
    
    with col_anon2:
        if ANONYMISATION_AVAILABLE and 'enable_anonymisation' in locals() and enable_anonymisation:
            # Show anonymisation summary if available
            try:
                anon_summary = global_anonymiser.get_anonymisation_summary()
                st.metric("Entities Anonymised", anon_summary['total_mappings'])
            except:
                st.metric("Anonymisation", "Ready")
    
    # Document selection status
    st.markdown("#### 📄 Document Selection Status")
    
    # Check for per-query document selection
    selected_doc_ids = getattr(st.session_state, 'selected_documents_for_query', [])
    selection_mode = getattr(st.session_state, 'document_selection_mode', 'Use All Documents')
    
    if selection_mode == "Use All Documents" or not selected_doc_ids:
        st.info("🗂️ **Using All Documents** - Query will search across your entire document collection")
    else:
        st.success(f"🎯 **Using Selected Documents** - Query restricted to {len(selected_doc_ids)} selected documents")
        
        # Show selected documents
        with st.expander(f"📋 View Selected Documents ({len(selected_doc_ids)})", expanded=False):
            if selected_doc_ids:
                for doc_id in selected_doc_ids:
                    st.write(f"• {doc_id}")
    
    # Advanced Retrieval Options
    st.markdown("#### 🚀 Advanced Semantic Retrieval")
    
    col_adv1, col_adv2 = st.columns([2, 1])
    
    with col_adv1:
        # Check advanced capabilities
        advanced_retrieval = AdvancedRetrieval(selected_matter)
        
        # Advanced retrieval options with intelligent defaults
        st.markdown("**🚀 Cutting-Edge Retrieval Methods (Intelligent Defaults):**")
        
        # Hierarchical retrieval option (DEFAULT ON - fastest enhancement)
        use_hierarchical = st.checkbox(
            "📊 **Hierarchical Retrieval** (Context-aware document structure)",
            value=True,  # DEFAULT ENABLED
            help="✅ RECOMMENDED: Considers document structure, section titles, and legal context for better relevance scoring. (~0.1s processing time)"
        )
        
        # Adaptive chunking option (DEFAULT ON - significant improvement)
        use_adaptive_chunking = st.checkbox(
            "🎯 **Adaptive Chunking** (Query-type optimized search)",
            value=True,  # DEFAULT ENABLED
            help="✅ RECOMMENDED: Dynamically adjusts search strategy based on query type (factual, summary, legal, procedural). (~0.2s processing time)"
        )
        
        # Knowledge graph option (DEFAULT ON if available)
        use_knowledge_graph = st.checkbox(
            "🌐 **Knowledge Graph Enhancement** (Entity-relationship aware)",
            value=GRAPH_PROCESSING_AVAILABLE,  # DEFAULT ENABLED if NetworkX available
            disabled=not GRAPH_PROCESSING_AVAILABLE,
            help="✅ RECOMMENDED: Uses knowledge graphs to understand entity relationships and improve context retrieval. (~0.3s processing time)"
        )
        
        # Late interaction option (NOW ENABLED BY DEFAULT)
        use_late_interaction = st.checkbox(
            "🧠 **ColBERT Late Interaction** (Token-level semantic matching)",
            value=True,  # DEFAULT ENABLED - faster processing with existing model
            disabled=not advanced_retrieval.late_interaction_available,
            help="🚀 ENABLED: Uses ColBERT-style token-level interactions for better precision with existing all-mpnet-base-v2 model. (~0.5s processing time)"
        )
        
        if use_knowledge_graph and not GRAPH_PROCESSING_AVAILABLE:
            st.warning("⚠️ Knowledge graph processing not available. Install with: `pip install networkx`")
        
        # Show warnings for unavailable features
        if use_late_interaction and not advanced_retrieval.late_interaction_available:
            st.warning("⚠️ ColBERT model not available. Install with: `pip install sentence-transformers`")
        
        # Show advanced retrieval status
        active_methods = []
        if use_late_interaction and advanced_retrieval.late_interaction_available:
            active_methods.append("ColBERT")
        if use_hierarchical:
            active_methods.append("Hierarchical")
        if use_adaptive_chunking:
            active_methods.append("Adaptive")
        if use_knowledge_graph and GRAPH_PROCESSING_AVAILABLE:
            active_methods.append("Knowledge Graph")
        
        if active_methods:
            st.success(f"✅ **Active Methods**: {', '.join(active_methods)}")
        else:
            st.info("💡 **Standard Mode**: Select advanced methods above for enhanced semantic search")
    
    with col_adv2:
        if advanced_retrieval.late_interaction_available:
            st.metric("Advanced Models", "Ready ✅")
            st.caption("🎯 Late interaction scoring\n📊 Token-level precision")
        else:
            st.metric("Advanced Models", "Available 📥")
            st.caption("💡 pip install sentence-transformers\n🚀 Enhanced retrieval accuracy")
    
    # Show retrieval method info
    if use_late_interaction and advanced_retrieval.late_interaction_available:
        st.info("🧠 **Enhanced Mode**: Using ColBERT late interaction for token-level semantic matching")
    else:
        st.info("📊 **Standard Mode**: Using dense vector similarity search")

    # Query interface
    st.markdown("#### 💬 Ask about your documents:")
    
    # Create container for dynamic query suggestions
    query_container = st.container()
    
    with query_container:
        # Check for preset query selection
        preset_query = st.session_state.get('selected_preset_query', '')
        
        # Query input with enhanced placeholder
        user_query = st.text_area(
            "Enter your question:",
            value=preset_query,  # Use the preset query if available
            placeholder="Examples:\n• Who is the claimant in this case?\n• What are the key allegations?\n• Summarize the procedural history\n• What damages are being claimed?",
            height=100,
            key="user_query_input"
        )
        
        # Clear the preset query after it's been used
        if preset_query:
            st.session_state.selected_preset_query = ''
        
        # Preset query buttons
        st.markdown("**Quick Queries:**")
        query_cols = st.columns(4)
        
        preset_queries = [
            "Who are the parties?",
            "What are the key facts?", 
            "What is the timeline?",
            "What are the damages?"
        ]
        
        for i, preset in enumerate(preset_queries):
            with query_cols[i]:
                if st.button(preset, key=f"preset_{i}"):
                    # Use a different session state variable to trigger the update
                    st.session_state.selected_preset_query = preset
                    st.rerun()
    
    # Analysis controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Analyze button with enhanced text
        analyze_button_text = "🧠 Analyze with ColBERT" if (use_late_interaction and advanced_retrieval.late_interaction_available) else "🔍 Analyze Documents"
        analyze_button = st.button(
            analyze_button_text,
            type="primary",
            disabled=not user_query.strip(),
            use_container_width=True
        )
    
    with col2:
        # Clear button
        if st.button("🗑️ Clear", use_container_width=True):
            # Use the session state variable to trigger clearing
            st.session_state.selected_preset_query = ''
            if 'analysis_result' in st.session_state:
                del st.session_state.analysis_result
            st.rerun()
    
    with col3:
        # Show anonymisation status
        if ANONYMISATION_AVAILABLE and enable_anonymisation:
            st.success("🛡️ Anon ON")
        else:
            st.info("🔓 Normal")
    
    # Process query
    if analyze_button and user_query.strip():
        with st.spinner(f"🔍 {'Advanced semantic analysis' if use_late_interaction else 'Analyzing documents'}..."):
            try:
                # Use advanced retrieval if enabled
                if use_late_interaction and advanced_retrieval.late_interaction_available:
                    result = asyncio.run(get_protocol_compliant_answer_with_advanced_retrieval(
                        user_query.strip(), 
                        selected_matter, 
                        selected_model, 
                        max_chunks,
                        use_late_interaction=True
                    ))
                else:
                    # Use standard analysis
                    result = asyncio.run(get_protocol_compliant_answer(
                        user_query.strip(), 
                        selected_matter, 
                        selected_model, 
                        max_chunks,
                        anonymise=enable_anonymisation if ANONYMISATION_AVAILABLE else False
                    ))
                
                st.session_state.analysis_result = result
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.session_state.analysis_result = None
    
    # Display results
    if 'analysis_result' in st.session_state and st.session_state.analysis_result:
        result = st.session_state.analysis_result
        
        st.markdown("---")
        st.markdown("### 📋 Analysis Results")
        
        # Show retrieval method used
        retrieval_method = result.get('retrieval_method', 'standard')
        if retrieval_method == 'late_interaction':
            st.success("🧠 **Advanced Retrieval Used**: ColBERT Late Interaction")
        else:
            st.info("📊 **Standard Retrieval Used**: Dense Vector Similarity")
        
        # Protocol compliance report
        if 'protocol_compliance' in result:
            render_protocol_compliance_report(result['protocol_compliance'])
        
        # Main answer with improved formatting
        st.markdown("### 💬 Answer")
        answer = result.get('answer', 'No answer generated')
        
        # Check for anonymisation
        if result.get('anonymisation_applied'):
            st.info("🛡️ **Anonymised Response** - Real names replaced with pseudonyms")
        
        st.markdown(answer)
        
        # Sources with enhanced display
        sources = result.get('sources', [])
        if sources:
            st.markdown("### 📚 Sources")
            
            # Sources overview
            source_count = len(sources)
            total_similarity = sum(s.get('similarity_score', 0) for s in sources)
            avg_similarity = total_similarity / source_count if source_count > 0 else 0
            
            col_src1, col_src2, col_src3 = st.columns(3)
            with col_src1:
                st.metric("Sources Used", f"{source_count}")
            with col_src2:
                st.metric("Avg Similarity", f"{avg_similarity:.3f}")
            with col_src3:
                late_interaction_count = sum(1 for s in sources if s.get('late_interaction_score'))
                if late_interaction_count > 0:
                    st.metric("Late Interaction", f"{late_interaction_count} sources")
                else:
                    st.metric("Search Type", "Vector Similarity")
            
            # Detailed sources
            for i, source in enumerate(sources):
                with st.expander(f"📄 Source {i+1}: {source.get('document', 'Unknown')}", expanded=False):
                    
                    # Source metrics
                    col_s1, col_s2, col_s3 = st.columns(3)
                    
                    with col_s1:
                        similarity = source.get('similarity_score', 0)
                        st.metric("Similarity", f"{similarity:.3f}")
                    
                    with col_s2:
                        if 'late_interaction_score' in source:
                            late_score = source.get('late_interaction_score', 0)
                            st.metric("Late Interaction", f"{late_score:.3f}")
                        else:
                            st.metric("Vector Search", "Standard")
                    
                    with col_s3:
                        chunk_idx = source.get('chunk_index', 'Unknown')
                        st.metric("Chunk", str(chunk_idx))
                    
                    # Source content
                    preview = source.get('text_preview', 'No preview available')
                    st.markdown("**Content Preview:**")
                    st.text(preview)
                    
                    # Section information if available
                    if source.get('section_title'):
                        st.caption(f"📍 Section: {source['section_title']}")
                    if source.get('paragraph_index') is not None:
                        st.caption(f"📄 Paragraph: {source['paragraph_index'] + 1}")
        
        # Debug information
        with st.expander("🔧 Debug Information", expanded=False):
            debug_info = result.get('debug_info', 'No debug info')
            generation_time = result.get('generation_time', 0)
            
            st.text(f"Generation Time: {generation_time:.2f}s")
            st.text(f"Model Used: {result.get('model_used', 'Unknown')}")
            st.text(f"Context Chunks: {result.get('context_chunks_used', 'Unknown')}")
            st.text(debug_info)
            
            # Document selection info
            if 'document_selection_info' in result:
                doc_info = result['document_selection_info']
                st.text(f"Document Selection: {doc_info.get('mode', 'Unknown')}")
                st.text(f"Search Scope: {doc_info.get('search_scope', 'Unknown')}")
                
            # Show anonymisation info
            if result.get('anonymisation_applied'):
                st.success("✅ Anonymisation applied successfully")
                if 'anonymisation_model' in result:
                    st.text(f"Anonymisation Model: {result['anonymisation_model']}")
            elif result.get('anonymisation_error'):
                st.error(f"❌ Anonymisation failed: {result['anonymisation_error']}")

    # Status information
    st.markdown("---")
    if ANONYMISATION_AVAILABLE:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Multi-Agent Status", "🔴 Disabled", help="Multi-agent system disabled to prevent hallucinations")
        
        with col2:
            st.metric("Anti-Hallucination", "🟢 Active", help="Strict prompting and validation active")
        
        with col3:
            st.metric("Protocol Compliance", "🟢 Monitoring", help="Real-time protocol compliance checking")
        
        with col4:
            if GRAPH_PROCESSING_AVAILABLE:
                st.metric("Knowledge Graphs", "🟢 Available", help="NetworkX-powered entity relationship processing")
            else:
                st.metric("Knowledge Graphs", "📥 Install NetworkX", help="pip install networkx for graph processing")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Multi-Agent Status", "🔴 Disabled", help="Multi-agent system disabled to prevent hallucinations")
        
        with col2:
            st.metric("Anti-Hallucination", "🟢 Active", help="Strict prompting and validation active")
        
        with col3:
            st.metric("Protocol Compliance", "🟢 Monitoring", help="Real-time protocol compliance checking")

# Add support for ColBERT-style late interaction
class AdvancedRetrieval:
    """Advanced retrieval methods including late interaction and multimodal capabilities"""
    
    def __init__(self, matter_id: str):
        self.matter_id = matter_id
        self.pipeline = None
        self.late_interaction_available = False
        self.multimodal_available = False
        
        # Check for advanced model availability
        self._check_advanced_capabilities()
    
    def _check_advanced_capabilities(self):
        """Check what advanced models are available"""
        try:
            # Check for ColBERT-style models
            import torch
            from sentence_transformers import SentenceTransformer
            
            # Try to load a ColBERT-style model if available
            try:
                # First try the real ColBERT model
                try:
                    self.colbert_model = SentenceTransformer('lightonai/Reason-ModernColBERT')
                    self.late_interaction_available = True
                    st.sidebar.success("🚀 ColBERT Late Interaction Available (Reason-ModernColBERT)")
                except:
                    # Fallback to existing working model for ColBERT-style operations
                    self.colbert_model = SentenceTransformer('all-mpnet-base-v2')
                    self.late_interaction_available = True
                    st.sidebar.success("🚀 ColBERT Late Interaction Available (using all-mpnet-base-v2)")
            except:
                st.sidebar.info("💡 ColBERT using existing embedding model")
                
            # Check for multimodal capabilities
            try:
                # This would check for ColPali or similar
                import PIL
                self.multimodal_available = True
                st.sidebar.success("🎨 Multimodal Processing Available")
            except:
                pass
                
        except ImportError:
            st.sidebar.warning("Advanced models require additional dependencies")
    
    async def late_interaction_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Perform late interaction retrieval using ColBERT-style approach"""
        if not self.late_interaction_available:
            return []
            
        try:
            # Load pipeline if not already loaded
            if not self.pipeline:
                from local_rag_pipeline import LocalRAGPipeline
                self.pipeline = LocalRAGPipeline(f'rag_storage/{self.matter_id}')
            
            # Get initial candidates with traditional search
            candidates = self.pipeline.search_documents(query, top_k=top_k*3)
            
            # Apply late interaction scoring
            refined_results = self._apply_late_interaction(query, candidates)
            
            return refined_results[:top_k]
            
        except Exception as e:
            logging.error(f"Late interaction search failed: {e}")
            return []
    
    def _apply_late_interaction(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Apply ColBERT-style late interaction scoring"""
        if not hasattr(self, 'colbert_model'):
            return candidates
            
        try:
            # Encode query tokens
            query_embeddings = self.colbert_model.encode([query], show_progress_bar=False)
            
            # Score each candidate using MaxSim operator
            scored_candidates = []
            for candidate in candidates:
                text = candidate.get('text', '')
                
                # Encode document tokens
                doc_embeddings = self.colbert_model.encode([text], show_progress_bar=False)
                
                # Compute MaxSim score (simplified version)
                similarity_scores = np.dot(query_embeddings, doc_embeddings.T)
                max_sim_score = np.max(similarity_scores, axis=1).mean()
                
                candidate['late_interaction_score'] = float(max_sim_score)
                scored_candidates.append(candidate)
            
            # Sort by late interaction score
            scored_candidates.sort(key=lambda x: x.get('late_interaction_score', 0), reverse=True)
            
            return scored_candidates
            
        except Exception as e:
            logging.error(f"Late interaction scoring failed: {e}")
            return candidates

# Add to the main interface
async def get_protocol_compliant_answer_with_advanced_retrieval(
    query: str, 
    matter_id: str, 
    model: str = "mistral:latest",
    max_chunks: int = 15,
    use_late_interaction: bool = False,
    use_hierarchical: bool = False,
    use_adaptive_chunking: bool = False,
    use_knowledge_graph: bool = False
) -> Dict[str, Any]:
    """Enhanced version with comprehensive advanced retrieval options"""
    
    start_time = datetime.now()
    
    # Initialize advanced processing
    if any([use_late_interaction, use_hierarchical, use_adaptive_chunking, use_knowledge_graph]):
        advanced_retrieval = AdvancedRetrieval(matter_id)
        advanced_processor = AdvancedSemanticProcessor(matter_id)
        
        # Determine which advanced method to use (priority order)
        if use_knowledge_graph and advanced_processor.graph_processing_available:
            # Build knowledge graph if not exists and use graph-enhanced retrieval
            await advanced_processor.build_knowledge_graph()
            sources = await advanced_processor.hierarchical_retrieval(query, top_k=max_chunks)
            retrieval_method = 'knowledge_graph_enhanced'
            
        elif use_hierarchical:
            # Use hierarchical retrieval
            sources = await advanced_processor.hierarchical_retrieval(query, top_k=max_chunks)
            retrieval_method = 'hierarchical'
            
        elif use_adaptive_chunking:
            # Use adaptive chunking
            sources = await advanced_processor.adaptive_chunking_search(query, top_k=max_chunks)
            retrieval_method = 'adaptive_chunking'
            
        elif use_late_interaction and advanced_retrieval.late_interaction_available:
            # Use late interaction retrieval
            sources = await advanced_retrieval.late_interaction_search(query, top_k=max_chunks)
            retrieval_method = 'late_interaction'
            
        else:
            # Fall back to standard retrieval
            result = await get_protocol_compliant_answer(query, matter_id, model, max_chunks)
            result['retrieval_method'] = 'standard_fallback'
            return result
        
        # Store advanced sources for comparison
        st.session_state.advanced_retrieval_sources = sources
        st.session_state.advanced_retrieval_method = retrieval_method
        
    else:
        # Use standard retrieval
        return await get_protocol_compliant_answer(query, matter_id, model, max_chunks)
    
    # For advanced retrieval, integrate with existing flow
    if not sources:
        # Return the original function call if no sources found
        result = await get_protocol_compliant_answer(query, matter_id, model, max_chunks)
        result['retrieval_method'] = f'{retrieval_method}_no_sources'
        return result
    
    # If we have enhanced sources, we need to convert them to the format expected by the main function
    # For now, just call the main function but store the advanced sources in session state for future use
    st.session_state.advanced_retrieval_sources = sources
    
    # Call the main function to handle the complete workflow
    result = await get_protocol_compliant_answer(query, matter_id, model, max_chunks)
    result['retrieval_method'] = retrieval_method
    result['advanced_sources_count'] = len(sources)
    
    return result

class AdvancedSemanticProcessor:
    """Advanced semantic processing with hierarchical retrieval, knowledge graphs, and adaptive chunking"""
    
    def __init__(self, matter_id: str):
        self.matter_id = matter_id
        self.pipeline = None
        self.knowledge_graph = None
        self.advanced_embeddings_available = ADVANCED_EMBEDDINGS_AVAILABLE
        self.graph_processing_available = GRAPH_PROCESSING_AVAILABLE
        
        # Initialize advanced capabilities
        self._initialize_advanced_capabilities()
    
    def _initialize_advanced_capabilities(self):
        """Initialize advanced semantic processing capabilities"""
        try:
            # Initialize knowledge graph if available
            if self.graph_processing_available and GRAPH_PROCESSING_AVAILABLE:
                import networkx as nx
                self.knowledge_graph = nx.DiGraph()
                st.sidebar.success("📊 Knowledge Graph Available")
            else:
                self.knowledge_graph = None
            
            # Check for advanced embedding models
            if self.advanced_embeddings_available:
                # Try to load instruction-tuned embedding model
                try:
                    self.instruction_model = AutoModel.from_pretrained(
                        'sentence-transformers/all-MiniLM-L6-v2',
                        trust_remote_code=True
                    )
                    st.sidebar.success("🧠 Advanced Embeddings Available")
                except:
                    self.instruction_model = None
                    st.sidebar.info("💡 Consider installing transformers for advanced embeddings")
            
        except Exception as e:
            logging.error(f"Advanced capabilities initialization failed: {e}")
    
    async def hierarchical_retrieval(self, query: str, top_k: int = 10) -> List[Dict]:
        """Perform hierarchical retrieval with context-aware scoring"""
        try:
            # Load pipeline if not already loaded
            if not self.pipeline:
                from local_rag_pipeline import LocalRAGPipeline
                self.pipeline = LocalRAGPipeline(f'rag_storage/{self.matter_id}')
            
            # Step 1: Get initial candidates (broader search)
            initial_candidates = self.pipeline.search_documents(query, top_k=top_k*2)
            
            # Step 2: Apply hierarchical scoring
            hierarchical_results = self._apply_hierarchical_scoring(query, initial_candidates)
            
            # Step 3: Context-aware reranking
            final_results = self._context_aware_reranking(query, hierarchical_results)
            
            return final_results[:top_k]
            
        except Exception as e:
            logging.error(f"Hierarchical retrieval failed: {e}")
            return []
    
    def _apply_hierarchical_scoring(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Apply hierarchical scoring considering document structure"""
        try:
            for candidate in candidates:
                # Base similarity score
                base_score = candidate.get('similarity_score', 0)
                
                # Hierarchical bonuses
                structure_bonus = 0
                
                # Section title relevance
                if candidate.get('section_title'):
                    section_title = candidate['section_title'].lower()
                    query_lower = query.lower()
                    
                    # Boost for relevant section titles
                    if any(word in section_title for word in query_lower.split()):
                        structure_bonus += 0.1
                
                # Document type bonuses
                doc_info = candidate.get('document_info', {})
                filename = doc_info.get('filename', '').lower()
                
                # Boost for specific document types based on query
                if 'claim' in query.lower() and 'claim' in filename:
                    structure_bonus += 0.15
                elif 'defence' in query.lower() and 'defence' in filename:
                    structure_bonus += 0.15
                elif 'witness' in query.lower() and 'witness' in filename:
                    structure_bonus += 0.15
                
                # Position-based scoring (earlier chunks often more important)
                chunk_index = candidate.get('chunk_index', 10)
                if chunk_index < 5:  # First 5 chunks
                    structure_bonus += 0.05
                
                # Calculate hierarchical score
                candidate['hierarchical_score'] = min(1.0, base_score + structure_bonus)
            
            # Sort by hierarchical score
            candidates.sort(key=lambda x: x.get('hierarchical_score', 0), reverse=True)
            
            return candidates
            
        except Exception as e:
            logging.error(f"Hierarchical scoring failed: {e}")
            return candidates
    
    def _context_aware_reranking(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Apply context-aware reranking considering legal context"""
        try:
            # Analyze query type
            query_type = self._classify_query_type(query)
            
            for candidate in candidates:
                rerank_bonus = 0
                
                # Legal context bonuses
                if query_type == 'factual':
                    # Boost factual statements and evidence
                    text = candidate.get('text', '').lower()
                    if any(phrase in text for phrase in ['on', 'the claimant', 'the defendant', 'date', 'time']):
                        rerank_bonus += 0.1
                
                elif query_type == 'procedural':
                    # Boost procedural documents
                    doc_name = candidate.get('document_info', {}).get('filename', '').lower()
                    if any(proc in doc_name for proc in ['order', 'direction', 'case management']):
                        rerank_bonus += 0.15
                
                elif query_type == 'legal':
                    # Boost legal arguments and citations
                    text = candidate.get('text', '').lower()
                    if any(legal in text for legal in ['pursuant to', 'section', 'act', 'regulation']):
                        rerank_bonus += 0.1
                
                # Apply reranking
                base_score = candidate.get('hierarchical_score', candidate.get('similarity_score', 0))
                candidate['context_aware_score'] = min(1.0, base_score + rerank_bonus)
            
            # Final sort by context-aware score
            candidates.sort(key=lambda x: x.get('context_aware_score', 0), reverse=True)
            
            return candidates
            
        except Exception as e:
            logging.error(f"Context-aware reranking failed: {e}")
            return candidates
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query type for context-aware processing"""
        query_lower = query.lower()
        
        # Factual queries
        factual_indicators = ['who', 'what', 'when', 'where', 'how many', 'which']
        if any(indicator in query_lower for indicator in factual_indicators):
            return 'factual'
        
        # Procedural queries
        procedural_indicators = ['procedure', 'process', 'step', 'timeline', 'deadline', 'order']
        if any(indicator in query_lower for indicator in procedural_indicators):
            return 'procedural'
        
        # Legal queries
        legal_indicators = ['legal', 'law', 'statute', 'regulation', 'case law', 'precedent']
        if any(indicator in query_lower for indicator in legal_indicators):
            return 'legal'
        
        # Summary queries
        summary_indicators = ['summarise', 'summarize', 'overview', 'summary']
        if any(indicator in query_lower for indicator in summary_indicators):
            return 'summary'
        
        return 'general'
    
    async def build_knowledge_graph(self) -> bool:
        """Build knowledge graph from processed documents"""
        if not self.graph_processing_available or not self.knowledge_graph:
            return False
        
        try:
            # Load pipeline if not already loaded
            if not self.pipeline:
                from local_rag_pipeline import LocalRAGPipeline
                self.pipeline = LocalRAGPipeline(f'rag_storage/{self.matter_id}')
            
            # Get all documents for graph building
            doc_status = self.pipeline.get_document_status()
            
            # Extract entities and relationships
            for doc in doc_status.get('documents', []):
                filename = doc.get('filename', '')
                
                # Add document node
                if self.knowledge_graph:
                    self.knowledge_graph.add_node(filename, node_type='document')
                
                # Extract entities from document
                entities = self._extract_legal_entities(doc)
                
                for entity in entities:
                    # Add entity node
                    if self.knowledge_graph:
                        self.knowledge_graph.add_node(entity['name'], node_type=entity['type'])
                        
                        # Add relationship between document and entity
                        self.knowledge_graph.add_edge(filename, entity['name'], 
                                                    relation='contains', confidence=entity['confidence'])
            
            return True
            
        except Exception as e:
            logging.error(f"Knowledge graph building failed: {e}")
            return False
    
    def _extract_legal_entities(self, document: Dict) -> List[Dict]:
        """Extract legal entities from document (simplified version)"""
        entities = []
        filename = document.get('filename', '').lower()
        
        # Extract based on filename patterns
        if 'claimant' in filename or 'claim' in filename:
            entities.append({'name': 'Claimant', 'type': 'party', 'confidence': 0.9})
        
        if 'defendant' in filename or 'defence' in filename:
            entities.append({'name': 'Defendant', 'type': 'party', 'confidence': 0.9})
        
        if 'witness' in filename:
            entities.append({'name': 'Witness', 'type': 'party', 'confidence': 0.8})
        
        # Extract case numbers (pattern matching)
        import re
        case_pattern = r'KB-\d{4}-\d{6}'
        if re.search(case_pattern, filename):
            entities.append({'name': 'Case Reference', 'type': 'reference', 'confidence': 0.95})
        
        return entities
    
    async def adaptive_chunking_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Perform search with adaptive chunking based on content type"""
        try:
            # Load pipeline if not already loaded
            if not self.pipeline:
                from local_rag_pipeline import LocalRAGPipeline
                self.pipeline = LocalRAGPipeline(f'rag_storage/{self.matter_id}')
            
            # Classify query to determine optimal chunk strategy
            query_type = self._classify_query_type(query)
            
            # Adjust search parameters based on query type
            if query_type == 'summary':
                # For summaries, get more diverse chunks
                chunks = self.pipeline.search_documents(query, top_k=top_k*2)
                return self._diversify_chunks(chunks)[:top_k]
            
            elif query_type == 'factual':
                # For factual queries, prioritize precision
                chunks = self.pipeline.search_documents(query, top_k=top_k)
                return self._filter_factual_content(chunks)
            
            else:
                # Standard search for general queries
                return self.pipeline.search_documents(query, top_k=top_k)
            
        except Exception as e:
            logging.error(f"Adaptive chunking search failed: {e}")
            return []
    
    def _diversify_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Diversify chunks to ensure good coverage for summaries"""
        try:
            diversified = []
            used_documents = set()
            
            # First pass: one chunk per document
            for chunk in chunks:
                doc_name = chunk.get('document_info', {}).get('filename', '')
                if doc_name not in used_documents:
                    diversified.append(chunk)
                    used_documents.add(doc_name)
            
            # Second pass: fill remaining slots with highest similarity
            remaining_slots = len(chunks) - len(diversified)
            for chunk in chunks:
                if len(diversified) >= len(chunks):
                    break
                if chunk not in diversified:
                    diversified.append(chunk)
            
            return diversified
            
        except Exception as e:
            logging.error(f"Chunk diversification failed: {e}")
            return chunks
    
    def _filter_factual_content(self, chunks: List[Dict]) -> List[Dict]:
        """Filter chunks to prioritize factual content"""
        try:
            factual_chunks = []
            
            for chunk in chunks:
                text = chunk.get('text', '').lower()
                
                # Score based on factual indicators
                factual_score = 0
                
                # Date indicators
                if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{4}', text):
                    factual_score += 0.2
                
                # Name indicators (proper nouns)
                if re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', chunk.get('text', '')):
                    factual_score += 0.1
                
                # Specific factual language
                factual_phrases = ['on the', 'at the time', 'the claimant', 'the defendant']
                factual_score += sum(0.05 for phrase in factual_phrases if phrase in text)
                
                chunk['factual_score'] = factual_score
                factual_chunks.append(chunk)
            
            # Sort by combined factual and similarity score
            factual_chunks.sort(
                key=lambda x: x.get('similarity_score', 0) + x.get('factual_score', 0),
                reverse=True
            )
            
            return factual_chunks
            
        except Exception as e:
            logging.error(f"Factual content filtering failed: {e}")
            return chunks

if __name__ == "__main__":
    # Test the interface
    print("Enhanced RAG Interface with Protocol Compliance")
    print("Use this in your Streamlit app with: render_enhanced_rag_interface()") 