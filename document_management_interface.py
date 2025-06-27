# document_management_interface.py
"""
Comprehensive Document Management Interface for Strategic Counsel
Features: Upload, remove, organize, and per-query document selection
Supports both legacy and hierarchical RAG pipelines
"""

import streamlit as st
import pandas as pd
import json
import io
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Set, Optional, Tuple
import asyncio

# Import RAG pipelines
try:
    from local_rag_pipeline import rag_session_manager
    from hierarchical_rag_adapter import get_adaptive_rag_pipeline, HIERARCHICAL_AVAILABLE
    from hierarchical_rag_pipeline import HierarchicalRAGPipeline
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"RAG pipeline import error: {e}")
    RAG_AVAILABLE = False

class DocumentManager:
    """Comprehensive document management for RAG systems"""
    
    def __init__(self, matter_id: str):
        self.matter_id = matter_id
        self.legacy_pipeline = None
        self.hierarchical_pipeline = None
        self.adaptive_pipeline = None
        
        if RAG_AVAILABLE:
            try:
                self.legacy_pipeline = rag_session_manager.get_or_create_pipeline(matter_id)
                if HIERARCHICAL_AVAILABLE:
                    self.adaptive_pipeline = get_adaptive_rag_pipeline(matter_id)
                    self.hierarchical_pipeline = HierarchicalRAGPipeline(matter_id)
            except Exception as e:
                st.warning(f"Could not initialize RAG pipelines: {e}")
    
    def get_all_documents(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all documents from both legacy and hierarchical pipelines"""
        documents = {
            'legacy': [],
            'hierarchical': [],
            'total_count': 0,
            'total_chunks': 0
        }
        
        try:
            # Get legacy documents
            if self.legacy_pipeline:
                legacy_status = self.legacy_pipeline.get_document_status()
                documents['legacy'] = legacy_status.get('documents', [])
                
                # Add pipeline type to each document
                for doc in documents['legacy']:
                    doc['pipeline_type'] = 'legacy'
                    doc['features'] = ['basic_chunking', 'vector_search']
            
            # Get hierarchical documents
            if self.hierarchical_pipeline:
                hierarchical_status = self.hierarchical_pipeline.get_status()
                hierarchical_docs = []
                
                # Convert hierarchical document summaries to standard format
                for doc_id, summary in self.hierarchical_pipeline.document_summaries.items():
                    doc_metadata = self.hierarchical_pipeline.document_metadata.get(doc_id, {})
                    hierarchical_docs.append({
                        'id': doc_id,
                        'filename': summary.filename,
                        'created_at': summary.created_at,
                        'text_length': doc_metadata.get('text_length', 0),
                        'chunk_count': doc_metadata.get('total_chunks', 0),
                        'pipeline_type': 'hierarchical',
                        'summary': summary.full_summary[:200] + "..." if len(summary.full_summary) > 200 else summary.full_summary,
                        'key_topics': summary.key_topics,
                        'content_type': summary.content_type,
                        'features': ['document_summarization', 'hierarchical_chunking', 'multi_level_search']
                    })
                
                documents['hierarchical'] = hierarchical_docs
            
            # Calculate totals
            documents['total_count'] = len(documents['legacy']) + len(documents['hierarchical'])
            documents['total_chunks'] = (
                sum(doc.get('chunk_count', 0) for doc in documents['legacy']) +
                sum(doc.get('chunk_count', 0) for doc in documents['hierarchical'])
            )
            
        except Exception as e:
            st.error(f"Error retrieving documents: {e}")
        
        return documents
    
    async def add_document_smart(self, file_obj, filename: str, 
                               use_hierarchical: bool = True, 
                               ocr_preference: str = "aws") -> Tuple[bool, str, Dict[str, Any]]:
        """Add document with intelligent pipeline selection"""
        
        if use_hierarchical and self.hierarchical_pipeline:
            # Use hierarchical pipeline for advanced processing
            return await self.hierarchical_pipeline.add_document(file_obj, filename, ocr_preference)
        elif self.legacy_pipeline:
            # Use legacy pipeline for basic processing
            return self.legacy_pipeline.add_document(file_obj, filename, ocr_preference)
        else:
            return False, "No RAG pipeline available", {}
    
    def delete_document(self, doc_id: str, pipeline_type: str) -> Tuple[bool, str]:
        """Delete document from specified pipeline"""
        
        try:
            if pipeline_type == 'legacy' and self.legacy_pipeline:
                return self.legacy_pipeline.delete_document(doc_id)
            elif pipeline_type == 'hierarchical' and self.hierarchical_pipeline:
                # Remove from hierarchical pipeline
                if doc_id in self.hierarchical_pipeline.document_summaries:
                    del self.hierarchical_pipeline.document_summaries[doc_id]
                
                if doc_id in self.hierarchical_pipeline.document_metadata:
                    del self.hierarchical_pipeline.document_metadata[doc_id]
                
                # Remove associated chunks
                chunks_to_remove = [chunk_id for chunk_id, chunk in self.hierarchical_pipeline.hierarchical_chunks.items() 
                                  if chunk.doc_id == doc_id]
                for chunk_id in chunks_to_remove:
                    del self.hierarchical_pipeline.hierarchical_chunks[chunk_id]
                
                # Save updated metadata
                self.hierarchical_pipeline._save_metadata()
                self.hierarchical_pipeline._save_vector_indices()
                
                return True, f"Document deleted from hierarchical pipeline"
            else:
                return False, "Invalid pipeline type or pipeline not available"
                
        except Exception as e:
            return False, f"Error deleting document: {str(e)}"
    
    def get_document_preview(self, doc_id: str, pipeline_type: str) -> Optional[str]:
        """Get document content preview"""
        
        try:
            if pipeline_type == 'legacy' and self.legacy_pipeline:
                doc_path = self.legacy_pipeline.documents_path / f"{doc_id}.txt"
                if doc_path.exists():
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        return content[:1000] + "..." if len(content) > 1000 else content
            
            elif pipeline_type == 'hierarchical' and self.hierarchical_pipeline:
                doc_path = self.hierarchical_pipeline.documents_path / f"{doc_id}.txt"
                if doc_path.exists():
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        return content[:1000] + "..." if len(content) > 1000 else content
            
            return None
            
        except Exception as e:
            st.error(f"Error getting document preview: {e}")
            return None

def render_document_management():
    """Main document management interface"""
    
    st.markdown("### 📚 Document Management System")
    st.markdown("Manage your document database with full control over uploads, organization, and per-query selection")
    
    # Matter selection
    col1, col2 = st.columns([3, 1])
    with col1:
        matter_id = st.selectbox(
            "Select Document Collection:",
            ["Corporate Governance", "Legal Analysis", "Default Matter", "Document Review"],
            key="doc_mgmt_matter"
        )
    
    with col2:
        if st.button("🔄 Refresh", help="Refresh document list"):
            st.rerun()
    
    try:
        from local_rag_pipeline import rag_session_manager
        
        # Get pipeline and document status
        pipeline = rag_session_manager.get_or_create_pipeline(matter_id)
        doc_status = pipeline.get_document_status()
        
        # Document statistics
        st.markdown("#### 📊 Document Database Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Documents", doc_status.get('total_documents', 0))
        with col2:
            st.metric("Total Chunks", doc_status.get('total_chunks', 0))
        with col3:
            embedding_model = doc_status.get('embedding_model', 'N/A')
            st.metric("Embedding Model", "all-mpnet-base-v2" if embedding_model != 'N/A' else 'N/A')
        with col4:
            storage_status = "🟢 Active" if doc_status.get('storage_path') else "🔴 Error"
            st.metric("Storage", storage_status)
        
        # Check for hierarchical RAG availability
        hierarchical_available = False
        try:
            from hierarchical_rag_adapter import HIERARCHICAL_AVAILABLE
            hierarchical_available = HIERARCHICAL_AVAILABLE
        except ImportError:
            pass
        
        if hierarchical_available:
            st.success("🚀 **Hierarchical RAG Available** - Upload new documents with advanced processing")
        else:
            st.info("📁 **Legacy Mode** - Basic RAG functionality available")
        
        # Document upload section
        st.markdown("---")
        st.markdown("#### 📤 Upload New Documents")
        
        with st.expander("Upload Documents", expanded=True):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                uploaded_files = st.file_uploader(
                    "Choose documents to upload:",
                    accept_multiple_files=True,
                    type=['pdf', 'txt', 'docx', 'doc', 'rtf'],
                    help="Supported formats: PDF, TXT, DOCX, DOC, RTF",
                    key="doc_upload_files"
                )
            
            with col2:
                ocr_method = st.selectbox(
                    "OCR Method:",
                    ["aws", "local", "none"],
                    index=0,
                    help="AWS Textract (best), Local OCR, or No OCR",
                    key="doc_upload_ocr"
                )
                
                if hierarchical_available:
                    use_hierarchical = st.checkbox(
                        "🚀 Use Hierarchical Processing",
                        value=True,
                        help="Enable document summarization and multi-level chunking",
                        key="doc_upload_hierarchical"
                    )
                else:
                    use_hierarchical = False
                    st.caption("📁 Legacy processing only")
            
            if uploaded_files:
                if st.button("🔄 Process Documents", type="primary", key="process_docs_btn"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    successful_uploads = []
                    failed_uploads = []
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        progress = (i + 1) / len(uploaded_files)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {uploaded_file.name}...")
                        
                        # Process document
                        file_obj = io.BytesIO(uploaded_file.getvalue())
                        
                        # For now, use legacy pipeline (hierarchical integration comes next)
                        success, message, doc_info = pipeline.add_document(
                            file_obj, uploaded_file.name, ocr_method
                        )
                        
                        if success:
                            successful_uploads.append({
                                'filename': uploaded_file.name,
                                'message': message,
                                'info': doc_info
                            })
                        else:
                            failed_uploads.append({
                                'filename': uploaded_file.name,
                                'error': message
                            })
                    
                    progress_bar.progress(1.0)
                    status_text.text("Processing complete!")
                    
                    # Show results
                    if successful_uploads:
                        st.success(f"✅ Successfully processed {len(successful_uploads)} document(s)")
                        for upload in successful_uploads:
                            st.success(f"📄 {upload['filename']}: {upload['message']}")
                    
                    if failed_uploads:
                        st.error(f"❌ Failed to process {len(failed_uploads)} document(s)")
                        for failure in failed_uploads:
                            st.error(f"📄 {failure['filename']}: {failure['error']}")
                    
                    if successful_uploads:
                        st.rerun()
        
        # Document list and management
        if doc_status.get('total_documents', 0) > 0:
            st.markdown("---")
            st.markdown("#### 📋 Document Database")
            
            documents = doc_status.get('documents', [])
            
            # Filter and view options
            col1, col2, col3 = st.columns(3)
            with col1:
                sort_by = st.selectbox(
                    "Sort by:",
                    ["Upload Date", "Filename", "Size", "Chunks"],
                    key="doc_sort"
                )
            
            with col2:
                view_mode = st.selectbox(
                    "View Mode:",
                    ["Detailed", "Compact", "Table"],
                    key="doc_view_mode"
                )
            
            with col3:
                show_previews = st.checkbox(
                    "Show Previews",
                    value=False,
                    key="show_doc_previews"
                )
            
            # Sort documents
            if sort_by == "Upload Date":
                documents.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            elif sort_by == "Filename":
                documents.sort(key=lambda x: x.get('filename', ''))
            elif sort_by == "Size":
                documents.sort(key=lambda x: x.get('text_length', 0), reverse=True)
            elif sort_by == "Chunks":
                documents.sort(key=lambda x: x.get('chunk_count', 0), reverse=True)
            
            # Render documents
            if view_mode == "Table":
                render_documents_table(documents, pipeline)
            else:
                render_documents_detailed(documents, pipeline, view_mode == "Compact", show_previews)
        
        else:
            st.info("📄 No documents found. Upload documents above to get started.")
        
        # Per-query document selection section
        if doc_status.get('total_documents', 0) > 0:
            st.markdown("---")
            render_per_query_document_selection(documents, matter_id)
        
    except Exception as e:
        st.error(f"❌ Error accessing document system: {e}")
        st.info("💡 Please ensure RAG pipeline is properly configured")

def render_documents_table(documents: List[Dict[str, Any]], pipeline):
    """Render documents in table format"""
    
    if not documents:
        return
    
    # Create DataFrame
    table_data = []
    for doc in documents:
        table_data.append({
            'Filename': doc['filename'][:40] + "..." if len(doc['filename']) > 40 else doc['filename'],
            'Upload Date': doc['created_at'][:19] if doc.get('created_at') else 'N/A',
            'Size (chars)': f"{doc.get('text_length', 0):,}",
            'Chunks': doc.get('chunk_count', 0),
            'ID': doc['id'][:12] + "..." if len(doc['id']) > 12 else doc['id']
        })
    
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True)
    
    # Action buttons below table
    st.markdown("**Document Actions:**")
    for i, doc in enumerate(documents):
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        
        with col1:
            st.caption(f"📄 {doc['filename']}")
        
        with col2:
            if st.button("👁️ Preview", key=f"preview_table_{doc['id']}"):
                show_document_preview(doc, pipeline)
        
        with col3:
            if st.button("📊 Details", key=f"details_table_{doc['id']}"):
                show_document_details(doc)
        
        with col4:
            if st.button("📦 Archive", key=f"archive_table_{doc['id']}", type="secondary"):
                confirm_archive_document(doc, pipeline)

def render_documents_detailed(documents: List[Dict[str, Any]], pipeline, compact: bool = False, show_previews: bool = False):
    """Render documents in detailed card format"""
    
    for doc in documents:
        with st.container():
            st.markdown(f"**📄 {doc['filename']}**")
            
            if not compact:
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.caption(f"📅 Uploaded: {doc['created_at'][:19] if doc.get('created_at') else 'N/A'}")
                    st.caption(f"📏 Size: {doc.get('text_length', 0):,} characters")
                    
                with col2:
                    st.caption(f"🧩 Chunks: {doc.get('chunk_count', 0)}")
                    st.caption(f"🔧 Model: {doc.get('embedding_model', 'N/A')}")
                
                with col3:
                    action_col1, action_col2, action_col3 = st.columns(3)
                    
                    with action_col1:
                        if st.button("👁️", key=f"preview_{doc['id']}", help="Preview"):
                            show_document_preview(doc, pipeline)
                    
                    with action_col2:
                        if st.button("📊", key=f"details_{doc['id']}", help="Details"):
                            show_document_details(doc)
                    
                    with action_col3:
                        if st.button("📦", key=f"archive_{doc['id']}", help="Archive", type="secondary"):
                            confirm_archive_document(doc, pipeline)
                
                # Show preview if enabled
                if show_previews:
                    doc_path = pipeline.documents_path / f"{doc['id']}.txt"
                    if doc_path.exists():
                        try:
                            with open(doc_path, 'r', encoding='utf-8') as f:
                                preview = f.read()[:300]
                            st.caption(f"📝 Preview: {preview}..." if len(preview) == 300 else preview)
                        except Exception as e:
                            st.caption(f"❌ Preview error: {e}")
            
            else:
                # Compact view
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.caption(f"📅 {doc['created_at'][:10]} | 🧩 {doc.get('chunk_count', 0)} chunks | 📏 {doc.get('text_length', 0):,} chars")
                with col2:
                    if st.button("🔧", key=f"actions_{doc['id']}", help="Actions"):
                        show_document_details(doc)
            
            st.markdown("---")

def show_document_preview(doc: Dict[str, Any], pipeline):
    """Show document content preview"""
    
    doc_path = pipeline.documents_path / f"{doc['id']}.txt"
    if doc_path.exists():
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
                preview = content[:1000] + "..." if len(content) > 1000 else content
            
            st.text_area(
                f"📄 Preview: {doc['filename']}", 
                value=preview,
                height=300,
                disabled=True,
                key=f"preview_content_{doc['id']}"
            )
        except Exception as e:
            st.error(f"Error reading document: {e}")
    else:
        st.error("Document file not found")

def show_document_details(doc: Dict[str, Any]):
    """Show detailed document information"""
    
    st.markdown(f"#### 📄 Document Details: {doc['filename']}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Basic Information:**")
        st.write(f"- **ID:** {doc['id']}")
        st.write(f"- **Filename:** {doc['filename']}")
        st.write(f"- **Upload Date:** {doc.get('created_at', 'N/A')}")
        st.write(f"- **Text Length:** {doc.get('text_length', 0):,} characters")
        st.write(f"- **Chunk Count:** {doc.get('chunk_count', 0)}")
    
    with col2:
        st.write("**Technical Details:**")
        st.write(f"- **File Hash:** {doc.get('file_hash', 'N/A')}")
        st.write(f"- **Embedding Model:** {doc.get('embedding_model', 'N/A')}")
        st.write(f"- **Chunk Size:** {doc.get('chunk_size', 'N/A')}")
        st.write(f"- **Pipeline:** Legacy RAG")

def confirm_archive_document(doc: Dict[str, Any], pipeline):
    """Confirm and execute document archiving"""
    
    # Use a unique key for the confirmation to avoid conflicts
    confirm_key = f"confirm_archive_{doc['id']}"
    
    if st.button(f"⚠️ Confirm Archive '{doc['filename'][:20]}...'", 
                key=confirm_key, 
                type="secondary"):
        
        success, message = pipeline.delete_document(doc['id'])  # This now archives instead of deletes
        if success:
            st.success(f"✅ {message}")
            st.rerun()
        else:
            st.error(f"❌ {message}")

def render_per_query_document_selection(documents: List[Dict[str, Any]], matter_id: str):
    """Render per-query document selection interface"""
    
    st.markdown("#### 🎯 Per-Query Document Selection")
    st.markdown("Choose specific documents for your next query instead of using all documents")
    
    with st.expander("Configure Document Selection for Query", expanded=False):
        
        # Selection modes
        selection_mode = st.radio(
            "Selection Mode:",
            ["Use All Documents", "Select Specific Documents", "Quick Selection"],
            key="doc_selection_mode",
            help="Choose how to select documents for your next query"
        )
        
        selected_docs = []
        
        if selection_mode == "Select Specific Documents":
            st.markdown("**Choose documents to include in your query:**")
            
            for doc in documents:
                selected = st.checkbox(
                    f"📄 {doc['filename']} ({doc.get('chunk_count', 0)} chunks)",
                    key=f"select_doc_{doc['id']}",
                    help=f"Created: {doc.get('created_at', 'N/A')[:19]} | Size: {doc.get('text_length', 0):,} chars",
                    value=True  # Default to selected
                )
                if selected:
                    selected_docs.append({
                        'id': doc['id'],
                        'filename': doc['filename'],
                        'chunks': doc.get('chunk_count', 0)
                    })
        
        elif selection_mode == "Quick Selection":
            st.markdown("**Quick selection options:**")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📄 Select Largest Documents", key="select_largest"):
                    # Select top 3 largest documents
                    sorted_docs = sorted(documents, key=lambda x: x.get('text_length', 0), reverse=True)
                    selected_docs = [{'id': doc['id'], 'filename': doc['filename'], 'chunks': doc.get('chunk_count', 0)} 
                                   for doc in sorted_docs[:3]]
                    st.session_state.quick_selected_docs = selected_docs
            
            with col2:
                if st.button("🕒 Select Most Recent", key="select_recent"):
                    # Select most recent 3 documents
                    sorted_docs = sorted(documents, key=lambda x: x.get('created_at', ''), reverse=True)
                    selected_docs = [{'id': doc['id'], 'filename': doc['filename'], 'chunks': doc.get('chunk_count', 0)} 
                                   for doc in sorted_docs[:3]]
                    st.session_state.quick_selected_docs = selected_docs
            
            # Show quick selection results
            if hasattr(st.session_state, 'quick_selected_docs'):
                st.markdown("**Quick Selection Results:**")
                for doc in st.session_state.quick_selected_docs:
                    st.write(f"✅ {doc['filename']} - {doc['chunks']} chunks")
                selected_docs = st.session_state.quick_selected_docs
        
        # Show selection summary
        if selection_mode != "Use All Documents" and selected_docs:
            st.markdown("**Selection Summary:**")
            total_chunks = sum(doc['chunks'] for doc in selected_docs)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Selected Documents", len(selected_docs))
            with col2:
                st.metric("Total Chunks", total_chunks)
            with col3:
                coverage = (len(selected_docs) / len(documents)) * 100 if documents else 0
                st.metric("Coverage", f"{coverage:.0f}%")
            
            # Store selection in session state for use in query
            st.session_state.selected_documents_for_query = [doc['id'] for doc in selected_docs]
            st.session_state.document_selection_mode = selection_mode
            
            st.success(f"✅ Document selection saved for next query")
            
            # Show selected documents
            with st.expander("Selected Documents Details", expanded=False):
                for doc in selected_docs:
                    st.write(f"📄 **{doc['filename']}** - {doc['chunks']} chunks")
        
        elif selection_mode == "Use All Documents":
            # Clear selection if using all documents
            st.session_state.selected_documents_for_query = []
            st.session_state.document_selection_mode = "Use All Documents"
            
            total_chunks = sum(doc.get('chunk_count', 0) for doc in documents)
            st.info(f"📚 Using all {len(documents)} documents with {total_chunks} total chunks")
        
        else:
            st.info("👆 Select documents above to configure your query scope")

if __name__ == "__main__":
    render_document_management() 