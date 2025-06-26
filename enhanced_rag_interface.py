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

# Import anonymisation module
try:
    from pseudoanonymisation_module import anonymise_rag_result, global_anonymiser
    ANONYMISATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Pseudoanonymisation module not available: {e}")
    ANONYMISATION_AVAILABLE = False

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
        st.caption("💡 Use the **Document Management** tab to select specific documents for targeted analysis")
    else:
        st.success(f"🎯 **Using Selected Documents** - Query will search only {len(selected_doc_ids)} selected document(s)")
        with st.expander("📋 View Selected Documents", expanded=False):
            try:
                pipeline = rag_session_manager.get_or_create_pipeline(selected_matter)
                doc_status = pipeline.get_document_status()
                
                selected_docs = []
                for doc_id in selected_doc_ids:
                    for doc in doc_status.get('documents', []):
                        if doc['id'] == doc_id:
                            selected_docs.append(doc)
                            break
                
                if selected_docs:
                    for doc in selected_docs:
                        st.write(f"📄 **{doc['filename']}** - {doc.get('chunk_count', 0)} chunks")
                else:
                    st.warning("⚠️ Selected documents not found - may have been deleted")
                    
            except Exception as e:
                st.error(f"Error loading document details: {e}")
        
        if st.button("🔄 Reset to Use All Documents", key="reset_doc_selection"):
            st.session_state.selected_documents_for_query = []
            st.session_state.document_selection_mode = "Use All Documents"
            st.rerun()
    
    # Query input with intelligent suggestions
    query = st.text_area(
        "Ask about your documents:",
        height=120,
        placeholder="What is the case number? Who are the parties involved? What are the main legal claims?",
        help="Enter your question about the documents. Be specific for better results."
    )
    
    # Improve prompt functionality (from main AI analysis page)
    if 'improved_prompt' not in st.session_state:
        st.session_state.improved_prompt = ""
    
    col_prompt1, col_prompt2 = st.columns([3, 1])
    
    with col_prompt1:
        if query.strip() and st.button("💡 Suggest Improved Prompt", key="suggest_improved_rag_prompt"):
            with st.spinner("Improving your prompt for better legal analysis..."):
                try:
                    improved = get_improved_prompt(query, "UK litigation document analysis", selected_model)
                    st.session_state.improved_prompt = improved
                    st.success("Improved prompt generated below. You can edit it or use it as your main question.")
                except Exception as e:
                    st.error(f"Error improving prompt: {e}")
    
    with col_prompt2:
        if st.session_state.improved_prompt and st.button("Use Improved Prompt", key="use_improved_rag_prompt"):
            st.session_state.user_query_improved = st.session_state.improved_prompt
            st.session_state.improved_prompt = ""
            st.rerun()
    
    # Show improved prompt if available
    if st.session_state.improved_prompt:
        st.markdown("**💡 Improved Prompt:**")
        improved_query = st.text_area(
            "Edit improved prompt if needed:",
            value=st.session_state.improved_prompt,
            height=100,
            key="improved_prompt_text_area_rag"
        )
        
        # Use improved query for analysis
        if improved_query.strip():
            query = improved_query
    
    # Check if we should use an improved query from session state
    if hasattr(st.session_state, 'user_query_improved') and st.session_state.user_query_improved:
        query = st.session_state.user_query_improved
        delattr(st.session_state, 'user_query_improved')
    
    # Adaptive chunk recommendation
    if query.strip():
        # Get document status for recommendation
        try:
            pipeline = rag_session_manager.get_or_create_pipeline(selected_matter)
            doc_status = pipeline.get_document_status()
            total_docs = doc_status['total_documents']
            total_chunks = doc_status['total_chunks']
        except:
            total_docs = "multiple"
            total_chunks = 123
        
        summary_keywords = ['summarise', 'summarize', 'overview', 'key points', 'main findings', 'comprehensive analysis', 'full analysis']
        is_summary_query = any(keyword in query.lower() for keyword in summary_keywords)
        
        if is_summary_query and max_chunks < 25:
            recommended_chunks = min(30, int(total_chunks * 0.25))  # 25% of total chunks
            st.info(f"💡 **Tip**: For comprehensive summarization, consider using **{recommended_chunks}+ chunks** instead of {max_chunks} to capture more content from your {total_docs} documents ({total_chunks} total chunks).")
            st.caption(f"Current setting covers {(max_chunks/total_chunks)*100:.1f}% of content. Recommended: {(recommended_chunks/total_chunks)*100:.1f}%")
    
    # Analysis button
    analysis_button_text = "🧠 Generate Protocol-Compliant Analysis"
    if 'enable_anonymisation' in locals() and enable_anonymisation:
        analysis_button_text = "🛡️ Generate Anonymised Analysis (mistral → phi3)"
    
    if st.button(analysis_button_text, type="primary", disabled=not query.strip()):
        
        spinner_text = f"Analyzing documents with {selected_model}..."
        if 'enable_anonymisation' in locals() and enable_anonymisation:
            spinner_text = f"Step 1: Analyzing with {selected_model} → Step 2: Anonymising with phi3..."
        
        with st.spinner(spinner_text):
            
            # Run the analysis
            result = asyncio.run(get_protocol_compliant_answer(
                query, selected_matter, selected_model, max_chunks, 
                anonymise='enable_anonymisation' in locals() and enable_anonymisation
            ))
            
            # Display results
            st.markdown("### 📋 Analysis Results")
            
            # Anonymisation status
            if result.get('anonymisation_applied', False):
                st.success("🔒 **Pseudoanonymisation Applied** - Names and sensitive data replaced")
                if 'anonymisation_info' in result:
                    anon_info = result['anonymisation_info']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Entities Anonymised", anon_info.get('entities_anonymised', 0))
                    with col2:
                        st.metric("Processing Model", result.get('anonymisation_model', 'phi3'))
                    with col3:
                        st.metric("Status", "🟢 Complete")
            elif result.get('anonymisation_error'):
                st.error(f"❌ Anonymisation failed: {result['anonymisation_error']}")
            
            # Protocol compliance report
            st.markdown("#### 🛡️ Protocol Compliance Report")
            render_protocol_compliance_report(result.get('protocol_compliance', {}))
            
            # Answer section
            answer_title = "#### 💬 Answer"
            if result.get('anonymisation_applied', False):
                answer_title = "#### 🔒 Anonymised Answer"
            st.markdown(answer_title)
            answer = result.get('answer', 'No answer generated')
            st.markdown(answer)
            
            # Sources section with enhanced location information
            sources = result.get('sources', [])
            if sources:
                st.markdown("#### 📚 Sources")
                for i, source in enumerate(sources):
                    # Create detailed source location info
                    location_info = []
                    
                    # Check for specific location information
                    if 'section_title' in source and source['section_title']:
                        location_info.append(f"Section: {source['section_title']}")
                    elif 'chunk_index' in source:
                        location_info.append(f"Section: {source['chunk_index'] + 1}")
                    
                    if 'paragraph_index' in source and source['paragraph_index'] is not None:
                        location_info.append(f"Para: {source['paragraph_index'] + 1}")
                    
                    # Build enhanced source title
                    location_str = " | ".join(location_info) if location_info else f"Chunk {i+1}"
                    source_title = f"[Source {i+1}] {source['document']} - {location_str}"
                    
                    with st.expander(f"{source_title} (Similarity: {source['similarity_score']:.3f})"):
                        st.write("**Document:** " + source['document'])
                        if location_info:
                            st.write("**Location:** " + " | ".join(location_info))
                        else:
                            st.write("**Location:** " + f"Chunk {source.get('chunk_index', i)}")
                        st.write("**Similarity Score:** " + str(source['similarity_score']))
                        st.write("**Content Preview:**")
                        st.write(source['text_preview'])
            
            # Metadata section
            with st.expander("🔍 Technical Details"):
                st.write(f"**Analysis Model:** {result.get('model_used', 'Unknown')}")
                st.write(f"**Context Chunks:** {result.get('context_chunks_used', 0)}")
                st.write(f"**Generation Time:** {result.get('generation_time', 0):.2f} seconds")
                st.write(f"**Sources Found:** {len(sources)}")
                
                # Document selection details
                if 'document_selection_info' in result:
                    doc_sel_info = result['document_selection_info']
                    st.write("---")
                    st.write("**📄 Document Selection Details:**")
                    st.write(f"**Selection Mode:** {doc_sel_info.get('mode', 'Unknown')}")
                    st.write(f"**Search Scope:** {doc_sel_info.get('search_scope', 'Unknown')}")
                    if doc_sel_info.get('selected_count', 0) > 0:
                        st.write(f"**Selected Documents:** {doc_sel_info.get('selected_count', 0)}")
                        st.write(f"**Chunks from Selection:** {doc_sel_info.get('chunks_from_selection', 0)}")
                
                # Anonymisation details
                if result.get('anonymisation_applied', False):
                    st.write("---")
                    st.write("**🔒 Anonymisation Details:**")
                    st.write(f"**Anonymisation Model:** {result.get('anonymisation_model', 'phi3:latest')}")
                    if 'anonymisation_info' in result:
                        anon_info = result['anonymisation_info']
                        st.write(f"**Entities Processed:** {anon_info.get('entities_anonymised', 0)}")
                        st.write(f"**Processing Time:** {anon_info.get('processing_time', 'Unknown')}")
                    st.write("**Privacy Status:** ✅ Document-level anonymisation complete")
                
                st.write(f"**Debug Info:** {result.get('debug_info', 'None')}")
    
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
            st.metric("Pseudoanonymisation", "🟢 Available", help="phi3-powered creative anonymisation for privacy protection")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Multi-Agent Status", "🔴 Disabled", help="Multi-agent system disabled to prevent hallucinations")
        
        with col2:
            st.metric("Anti-Hallucination", "🟢 Active", help="Strict prompting and validation active")
        
        with col3:
            st.metric("Protocol Compliance", "🟢 Monitoring", help="Real-time protocol compliance checking")

if __name__ == "__main__":
    # Test the interface
    print("Enhanced RAG Interface with Protocol Compliance")
    print("Use this in your Streamlit app with: render_enhanced_rag_interface()") 