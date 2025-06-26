# enhanced_rag_interface_v2.py
"""
Enhanced RAG Interface v2 - Implementing SOTA Hierarchical RAG
Features:
1. Intelligent pipeline selection (Hierarchical vs Legacy)
2. Document-level summarization during upload
3. Query-adaptive chunk selection (5-50 chunks intelligently chosen)
4. Coarse-to-fine retrieval for comprehensive analysis
5. Coverage optimization and feedback
"""

import asyncio
import streamlit as st
import time
from typing import Dict, Any, List, Optional
import aiohttp
from pathlib import Path
import json

# Import the adaptive RAG system
try:
    from hierarchical_rag_adapter import (
        get_adaptive_rag_pipeline, 
        get_rag_capabilities, 
        AdaptiveRAGPipeline,
        HIERARCHICAL_AVAILABLE
    )
    ADAPTER_AVAILABLE = True
except ImportError:
    ADAPTER_AVAILABLE = False
    print("Warning: Adaptive RAG adapter not available")

# Fallback imports
from local_rag_pipeline import rag_session_manager
from enhanced_rag_interface import (
    create_protocol_compliant_prompt,
    check_protocol_compliance,
    render_protocol_compliance_report
)

# Try to import anonymisation
try:
    from pseudoanonymisation_module import anonymise_rag_result
    ANONYMISATION_AVAILABLE = True
except ImportError:
    ANONYMISATION_AVAILABLE = False

def classify_query_for_chunking(query: str) -> Dict[str, Any]:
    """
    Classify query to determine optimal chunking strategy
    Based on latest RAG research on query-adaptive retrieval
    """
    query_lower = query.lower()
    
    # Simple fact extraction queries
    simple_indicators = ['what is', 'who is', 'when did', 'where is', 'how much', 'what are']
    is_simple = any(indicator in query_lower for indicator in simple_indicators)
    
    # Comprehensive analysis queries  
    comprehensive_indicators = [
        'summarize', 'summarise', 'overview', 'comprehensive', 'all', 'entire', 
        'complete analysis', 'main points', 'key findings', 'full picture'
    ]
    is_comprehensive = any(indicator in query_lower for indicator in comprehensive_indicators)
    
    # Cross-document comparison queries
    cross_doc_indicators = ['compare', 'contrast', 'between', 'versus', 'relationship', 'differences', 'similarities']
    is_cross_document = any(indicator in query_lower for indicator in cross_doc_indicators)
    
    # Legal analysis queries
    legal_indicators = ['legal', 'liability', 'breach', 'contract', 'clause', 'obligation', 'rights', 'damages']
    is_legal = any(indicator in query_lower for indicator in legal_indicators)
    
    # Determine query complexity and recommended chunks
    if is_simple:
        complexity = "simple_fact"
        recommended_chunks = 5
        strategy = "Focused retrieval for specific facts"
    elif is_comprehensive:
        complexity = "comprehensive"
        recommended_chunks = 30
        strategy = "Broad context for summarization"
    elif is_cross_document:
        complexity = "cross_document" 
        recommended_chunks = 25
        strategy = "Multi-document comparison"
    elif is_legal:
        complexity = "legal_analysis"
        recommended_chunks = 20
        strategy = "Legal reasoning with precedent"
    else:
        complexity = "detailed_analysis"
        recommended_chunks = 15
        strategy = "Balanced analysis"
    
    return {
        'complexity': complexity,
        'recommended_chunks': recommended_chunks,
        'strategy': strategy,
        'is_simple': is_simple,
        'is_comprehensive': is_comprehensive,
        'is_cross_document': is_cross_document,
        'is_legal': is_legal
    }

async def get_enhanced_rag_answer(query: str, matter_id: str, model: str, 
                                 max_chunks: int = 15, 
                                 use_adaptive: bool = True,
                                 anonymise: bool = False) -> Dict[str, Any]:
    """
    Get RAG answer using enhanced hierarchical pipeline with intelligent chunking
    """
    
    try:
        # Get the appropriate pipeline
        if use_adaptive and ADAPTER_AVAILABLE:
            pipeline = get_adaptive_rag_pipeline(matter_id)
            search_chunks = await pipeline.intelligent_search(query, max_chunks)
            pipeline_type = "adaptive_hierarchical"
        else:
            # Fallback to legacy pipeline
            pipeline = rag_session_manager.get_or_create_pipeline(matter_id)
            search_chunks = pipeline.search_documents(query, max_chunks)
            pipeline_type = "legacy"
        
        if not search_chunks:
            return {
                'answer': "No relevant documents found for your query.",
                'sources': [],
                'model_used': model,
                'pipeline_type': pipeline_type,
                'protocol_compliance': {
                    'overall_score': 0,
                    'status': 'NO_SOURCES',
                    'message': 'No documents available for analysis'
                },
                'coverage_info': {
                    'chunks_retrieved': 0,
                    'total_available': 0,
                    'coverage_percentage': 0,
                    'quality': 'none'
                }
            }
        
        # Get coverage information
        total_available = 0
        if hasattr(pipeline, 'get_document_status'):
            doc_status = pipeline.get_document_status()
            total_available = doc_status.get('total_chunks', 0)
        
        coverage_percentage = (len(search_chunks) / max(total_available, 1)) * 100
        coverage_quality = get_coverage_quality(coverage_percentage)
        
        # Build enhanced context
        context_parts = []
        valid_sources = []
        
        for i, chunk in enumerate(search_chunks):
            chunk_text = chunk.get('text', '').strip()
            if len(chunk_text) > 20:
                context_parts.append(f"[Source {i+1}] {chunk_text}")
                valid_sources.append({
                    'chunk_id': chunk['id'],
                    'document': chunk.get('document_info', {}).get('filename', 'Unknown'),
                    'similarity_score': chunk.get('similarity_score', 0.0),
                    'text_preview': chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
                })
        
        if not context_parts:
            return {
                'answer': "Document chunks found but contain insufficient content for analysis.",
                'sources': [],
                'model_used': model,
                'pipeline_type': pipeline_type,
                'coverage_info': {
                    'chunks_retrieved': len(search_chunks),
                    'total_available': total_available,
                    'coverage_percentage': coverage_percentage,
                    'quality': 'insufficient'
                }
            }
        
        # Analyze query for optimal prompting
        query_analysis = classify_query_for_chunking(query)
        
        # Create enhanced context
        context = "\n\n".join(context_parts)
        
        # Use enhanced prompt with query-specific instructions
        enhanced_instructions = f"""
Query Analysis: {query_analysis['strategy']} (Complexity: {query_analysis['complexity']})
Context Coverage: {coverage_percentage:.1f}% ({len(context_parts)} of {total_available} chunks)
Pipeline: {pipeline_type.replace('_', ' ').title()}

CRITICAL REQUIREMENTS:
- Every factual statement MUST include [Source X] citation
- Start response with "Based on the provided documents:"
- Adapt analysis depth to query complexity: {query_analysis['strategy']}
"""
        
        prompt = create_protocol_compliant_prompt(query, context, model, "General Litigation", None) + enhanced_instructions
        
        # Generate answer with enhanced parameters
        start_time = time.time()
        timeout = aiohttp.ClientTimeout(total=120)
        
        # Enhanced model parameters for better compliance
        model_params = get_enhanced_model_params(model, query_analysis['complexity'])
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "num_predict": 4000,
                **model_params
            }
            
            async with session.post("http://localhost:11434/api/generate", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    answer = result.get('response', 'No response').strip()
                    generation_time = time.time() - start_time
                    
                    # Enhanced compliance checking
                    compliance_report = check_protocol_compliance(answer, valid_sources)
                    
                    # Build result
                    result_data = {
                        'answer': answer,
                        'sources': valid_sources,
                        'model_used': model,
                        'pipeline_type': pipeline_type,
                        'context_chunks_used': len(context_parts),
                        'generation_time': generation_time,
                        'protocol_compliance': compliance_report,
                        'query_analysis': query_analysis,
                        'coverage_info': {
                            'chunks_retrieved': len(search_chunks),
                            'total_available': total_available,
                            'coverage_percentage': coverage_percentage,
                            'quality': coverage_quality,
                            'recommended_chunks': query_analysis['recommended_chunks']
                        },
                        'enhanced_features': {
                            'hierarchical_search': use_adaptive and ADAPTER_AVAILABLE,
                            'query_adaptive_chunking': True,
                            'intelligent_coverage': True
                        }
                    }
                    
                    # Apply anonymisation if requested
                    if anonymise and ANONYMISATION_AVAILABLE:
                        try:
                            anonymised_result = await anonymise_rag_result(result_data)
                            anonymised_result['anonymisation_applied'] = True
                            return anonymised_result
                        except Exception as anon_error:
                            result_data['anonymisation_error'] = f"Anonymisation failed: {str(anon_error)}"
                            result_data['anonymisation_applied'] = False
                    else:
                        result_data['anonymisation_applied'] = False
                    
                    return result_data
                else:
                    error_text = await response.text()
                    return {
                        'answer': f"Error: HTTP {response.status} - {error_text}",
                        'sources': [],
                        'model_used': model,
                        'pipeline_type': pipeline_type,
                        'coverage_info': {'quality': 'error'}
                    }
    
    except Exception as e:
        return {
            'answer': f"System error: {str(e)}",
            'sources': [],
            'model_used': model,
            'pipeline_type': 'error',
            'coverage_info': {'quality': 'error'}
        }

def get_enhanced_model_params(model: str, complexity: str) -> Dict[str, Any]:
    """Get enhanced model parameters based on query complexity"""
    
    base_params = {
        "temperature": 0.0,
        "top_p": 0.05,
        "top_k": 3,
        "repeat_penalty": 1.5
    }
    
    # Adjust parameters based on complexity
    if complexity == "comprehensive":
        # Allow slightly more creativity for summarization
        base_params.update({
            "temperature": 0.1,
            "top_p": 0.1,
            "top_k": 5
        })
    elif complexity == "simple_fact":
        # Maximum precision for facts
        base_params.update({
            "temperature": 0.0,
            "top_p": 0.01,
            "top_k": 1
        })
    
    # Model-specific adjustments
    if "phi3" in model.lower():
        base_params.update({
            "system": "FACT EXTRACTION ONLY. Format: 'Based on the provided documents: [fact] [Source X].'"
        })
    elif "mistral" in model.lower():
        base_params.update({
            "system": "🚨 CITATION REQUIRED: Every statement needs [Source X]. NO exceptions."
        })
    
    return base_params

def get_coverage_quality(percentage: float) -> str:
    """Determine coverage quality based on percentage"""
    if percentage >= 50:
        return "excellent"
    elif percentage >= 25:
        return "good"
    elif percentage >= 10:
        return "limited"
    else:
        return "very_limited"

def get_available_matters_v2() -> List[str]:
    """Get enhanced list of available matters"""
    
    matters = []
    
    # Check adaptive pipeline matters
    if ADAPTER_AVAILABLE:
        rag_storage_paths = [
            Path("rag_storage"),
            Path("hierarchical_rag")
        ]
        
        for storage_path in rag_storage_paths:
            if storage_path.exists():
                for matter_dir in storage_path.iterdir():
                    if matter_dir.is_dir():
                        metadata_files = [
                            matter_dir / "metadata.json",
                            matter_dir / "hierarchical_metadata.json"
                        ]
                        
                        for metadata_file in metadata_files:
                            if metadata_file.exists():
                                try:
                                    with open(metadata_file, 'r') as f:
                                        metadata = json.load(f)
                                        doc_count = len(metadata.get('documents', {}))
                                        doc_summaries = len(metadata.get('document_summaries', {}))
                                        
                                        if doc_count > 0 or doc_summaries > 0:
                                            total_docs = doc_count + doc_summaries
                                            pipeline_type = "📊 Hierarchical" if doc_summaries > 0 else "📁 Legacy"
                                            matters.append(f"{matter_dir.name} - {pipeline_type} ({total_docs} docs)")
                                            break
                                except:
                                    pass
    
    # Remove duplicates and sort
    matters = list(set(matters))
    matters.sort()
    
    if not matters:
        matters = ["Default Matter - 📁 Legacy", "Document Analysis - 📊 Available", "Legal Review - 📊 Available"]
    
    return matters

def render_enhanced_rag_interface_v2():
    """Render the enhanced RAG interface with SOTA features"""
    
    st.markdown("### 🚀 Enhanced RAG System v2 - Hierarchical Intelligence")
    
    # Show system capabilities
    if ADAPTER_AVAILABLE:
        capabilities = get_rag_capabilities()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status = "🟢 Available" if capabilities['hierarchical_rag_available'] else "🟡 Legacy Only"
            st.metric("Hierarchical RAG", status)
        
        with col2:
            st.metric("Pipeline", "🤖 Adaptive" if capabilities['adaptive_routing'] else "📁 Standard")
        
        with col3:
            feature_count = len(capabilities['features']['hierarchical']) + len(capabilities['features']['adaptive'])
            st.metric("Enhanced Features", f"✨ {feature_count}")
    
    st.info("📈 **New Features**: Document summarization, intelligent chunking, query-adaptive retrieval, coverage optimization")
    
    # Configuration section
    with st.expander("⚙️ Enhanced Configuration", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Model selection with hierarchical recommendations
            available_models = ["mistral:latest", "deepseek-llm:7b", "phi3:latest", "mixtral:latest"]
            model_descriptions = {
                "mistral:latest": "🟢 **Best for hierarchical** - Excellent protocol compliance",
                "deepseek-llm:7b": "🟡 Good balance - Works well with summaries", 
                "phi3:latest": "🟡 Fast - Better with structured chunks",
                "mixtral:latest": "🔴 Slow but capable - Good for complex analysis"
            }
            
            selected_model = st.selectbox(
                "Analysis Model:",
                available_models,
                index=0,
                help="Mistral recommended for best results with hierarchical features"
            )
            
            if selected_model in model_descriptions:
                if "🟢" in model_descriptions[selected_model]:
                    st.success(model_descriptions[selected_model])
                elif "🟡" in model_descriptions[selected_model]:
                    st.warning(model_descriptions[selected_model])
                else:
                    st.error(model_descriptions[selected_model])
        
        with col2:
            # Enhanced matter selection
            available_matters = get_available_matters_v2()
            selected_matter_display = st.selectbox(
                "Document Collection:",
                available_matters,
                index=0,
                help="📊 = Hierarchical features available, 📁 = Legacy mode"
            )
            selected_matter = selected_matter_display.split(' - ')[0]
        
        with col3:
            # Intelligent chunking with query analysis
            max_chunks = st.slider(
                "Max Context Chunks:",
                min_value=1,
                max_value=50,
                value=15,
                help="System will recommend optimal chunk count based on your query"
            )
        
        with col4:
            # Pipeline selection
            use_adaptive = st.checkbox(
                "🚀 Enhanced Pipeline",
                value=ADAPTER_AVAILABLE,
                disabled=not ADAPTER_AVAILABLE,
                help="Use hierarchical RAG with document summarization and intelligent chunking"
            )
    
    # Query input with intelligent analysis
    st.markdown("#### 🔍 Query with Intelligent Analysis")
    query = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="Try: 'Summarize the key findings...' or 'What are the main legal obligations...' or 'Compare the approaches between...'",
        help="The system will analyze your query complexity and adapt chunking strategy automatically"
    )
    
    # Real-time query analysis
    if query.strip():
        query_analysis = classify_query_for_chunking(query)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            complexity_color = {
                "simple_fact": "🟢",
                "detailed_analysis": "🟡", 
                "comprehensive": "🔵",
                "cross_document": "🟣",
                "legal_analysis": "⚖️"
            }
            st.metric(
                "Query Complexity", 
                f"{complexity_color.get(query_analysis['complexity'], '🔍')} {query_analysis['complexity'].replace('_', ' ').title()}"
            )
        
        with col2:
            st.metric("Recommended Chunks", f"📊 {query_analysis['recommended_chunks']}")
            if query_analysis['recommended_chunks'] != max_chunks:
                diff = query_analysis['recommended_chunks'] - max_chunks
                if diff > 0:
                    st.caption(f"💡 Consider increasing by {diff} for better coverage")
                else:
                    st.caption(f"✅ Current setting is good")
        
        with col3:
            st.metric("Strategy", f"🎯 {query_analysis['strategy']}")
        
        # Show adaptive chunking recommendation
        if query_analysis['complexity'] == "comprehensive" and max_chunks < 25:
            st.info(f"💡 **Comprehensive Analysis Detected**: Consider using **{query_analysis['recommended_chunks']}+ chunks** for full document summarization")
        elif query_analysis['complexity'] == "simple_fact" and max_chunks > 10:
            st.info(f"🎯 **Fact Query Detected**: You could use fewer chunks (**~{query_analysis['recommended_chunks']}**) for focused retrieval")
    
    # Anonymisation option
    st.markdown("#### 🔒 Privacy Protection")
    enable_anonymisation = st.checkbox(
        "🛡️ Enable Pseudoanonymisation",
        value=False,
        disabled=not ANONYMISATION_AVAILABLE,
        help="Apply phi3-powered name anonymisation after analysis (mistral → phi3 workflow)"
    )
    
    # Analysis button
    analysis_button_text = "🧠 Generate Enhanced Analysis"
    if enable_anonymisation:
        analysis_button_text = "🛡️ Generate Anonymised Analysis"
    
    if st.button(analysis_button_text, type="primary", disabled=not query.strip()):
        
        spinner_text = f"🚀 Enhanced analysis with {selected_model}..."
        if enable_anonymisation:
            spinner_text = f"Step 1: Analyzing with {selected_model} → Step 2: Anonymising with phi3..."
        
        with st.spinner(spinner_text):
            # Run the enhanced analysis
            result = asyncio.run(get_enhanced_rag_answer(
                query, selected_matter, selected_model, max_chunks, 
                use_adaptive, enable_anonymisation
            ))
            
            # Display enhanced results
            st.markdown("### 📋 Enhanced Analysis Results")
            
            # Pipeline and coverage information
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                pipeline_type = result.get('pipeline_type', 'unknown')
                pipeline_emoji = "🚀" if "hierarchical" in pipeline_type else "📁"
                st.metric("Pipeline", f"{pipeline_emoji} {pipeline_type.replace('_', ' ').title()}")
            
            with col2:
                coverage_info = result.get('coverage_info', {})
                coverage_pct = coverage_info.get('coverage_percentage', 0)
                quality_emoji = {"excellent": "🟢", "good": "🟡", "limited": "🟠", "very_limited": "🔴"}.get(coverage_info.get('quality', 'none'), "⚪")
                st.metric("Coverage", f"{quality_emoji} {coverage_pct:.1f}%")
            
            with col3:
                chunks_used = result.get('context_chunks_used', 0)
                total_available = coverage_info.get('total_available', 0)
                st.metric("Chunks Used", f"📊 {chunks_used}/{total_available}")
            
            with col4:
                gen_time = result.get('generation_time', 0)
                st.metric("Generation Time", f"⏱️ {gen_time:.1f}s")
            
            # Coverage analysis
            if coverage_info:
                coverage_quality = coverage_info.get('quality', 'unknown')
                if coverage_quality == "excellent":
                    st.success(f"🟢 **Excellent Coverage** ({coverage_pct:.1f}%) - Comprehensive analysis with broad context")
                elif coverage_quality == "good":
                    st.info(f"🟡 **Good Coverage** ({coverage_pct:.1f}%) - Solid analysis with adequate context")
                elif coverage_quality == "limited":
                    st.warning(f"🟠 **Limited Coverage** ({coverage_pct:.1f}%) - Consider increasing chunk count for better analysis")
                else:
                    st.error(f"🔴 **Very Limited Coverage** ({coverage_pct:.1f}%) - Increase chunks significantly for comprehensive analysis")
            
            # Anonymisation status
            if result.get('anonymisation_applied', False):
                st.success("🔒 **Pseudoanonymisation Applied** - Names and sensitive identifiers replaced with reversible pseudonyms")
            
            # Protocol compliance
            st.markdown("#### 🛡️ Protocol Compliance Report")
            render_protocol_compliance_report(result.get('protocol_compliance', {}))
            
            # Enhanced answer display
            answer_title = "#### 💬 Enhanced Analysis"
            if result.get('anonymisation_applied', False):
                answer_title = "#### 🔒 Anonymised Analysis"
            
            st.markdown(answer_title)
            answer = result.get('answer', 'No answer generated')
            st.markdown(answer)
            
            # Sources with enhanced metadata
            sources = result.get('sources', [])
            if sources:
                st.markdown("#### 📚 Source Analysis")
                for i, source in enumerate(sources):
                    similarity = source.get('similarity_score', 0)
                    similarity_color = "🟢" if similarity > 0.8 else "🟡" if similarity > 0.6 else "🟠" if similarity > 0.4 else "🔴"
                    
                    with st.expander(f"Source {i+1}: {source['document']} {similarity_color} ({similarity:.3f})"):
                        st.write("**Document:** " + source['document'])
                        st.write(f"**Relevance Score:** {similarity:.3f} {similarity_color}")
                        st.write("**Content Preview:**")
                        st.write(source['text_preview'])
            
            # Technical details with enhanced info
            with st.expander("🔍 Enhanced Technical Details"):
                st.write(f"**Analysis Model:** {result.get('model_used', 'Unknown')}")
                st.write(f"**Pipeline Type:** {result.get('pipeline_type', 'Unknown')}")
                st.write(f"**Context Chunks:** {result.get('context_chunks_used', 0)}")
                st.write(f"**Generation Time:** {result.get('generation_time', 0):.2f} seconds")
                st.write(f"**Sources Found:** {len(sources)}")
                
                # Query analysis details
                if 'query_analysis' in result:
                    qa = result['query_analysis']
                    st.write("---")
                    st.write("**🧠 Query Analysis:**")
                    st.write(f"**Complexity:** {qa['complexity']}")
                    st.write(f"**Strategy:** {qa['strategy']}")
                    st.write(f"**Recommended Chunks:** {qa['recommended_chunks']}")
                
                # Enhanced features
                if 'enhanced_features' in result:
                    ef = result['enhanced_features']
                    st.write("---")
                    st.write("**🚀 Enhanced Features:**")
                    st.write(f"**Hierarchical Search:** {'✅' if ef.get('hierarchical_search') else '❌'}")
                    st.write(f"**Query Adaptive Chunking:** {'✅' if ef.get('query_adaptive_chunking') else '❌'}")
                    st.write(f"**Intelligent Coverage:** {'✅' if ef.get('intelligent_coverage') else '❌'}")
                
                # Anonymisation details
                if result.get('anonymisation_applied', False):
                    st.write("---")
                    st.write("**🔒 Anonymisation Details:**")
                    st.write("**Status:** ✅ Privacy protection applied")
                    st.write("**Method:** Reversible pseudoanonymisation with phi3")
    
    # System status
    st.markdown("---")
    st.markdown("#### 📊 System Status")
    
    if ADAPTER_AVAILABLE:
        try:
            pipeline = get_adaptive_rag_pipeline(selected_matter)
            status = pipeline.get_document_status()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Documents", status.get('total_documents', 0))
            with col2:
                st.metric("Total Chunks", status.get('total_chunks', 0))
            with col3:
                hierarchical_docs = status.get('hierarchical_documents', 0)
                st.metric("Hierarchical Docs", f"📊 {hierarchical_docs}")
            with col4:
                legacy_docs = status.get('legacy_documents', 0)
                st.metric("Legacy Docs", f"📁 {legacy_docs}")
                
            if status.get('total_documents', 0) == 0:
                st.info("📄 Upload documents to begin using the enhanced RAG system")
            else:
                hierarchical_pct = (hierarchical_docs / max(status.get('total_documents', 1), 1)) * 100
                st.success(f"✅ System ready! {hierarchical_pct:.0f}% of documents use hierarchical features")
        
        except Exception as e:
            st.error(f"Error getting system status: {e}")
    else:
        st.warning("🟡 Enhanced RAG features not available - using legacy mode")

# Main function for compatibility
def render_enhanced_rag_interface():
    """Compatibility wrapper for existing imports"""
    render_enhanced_rag_interface_v2() 