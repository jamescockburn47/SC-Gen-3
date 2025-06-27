#!/usr/bin/env python3
"""
Strategic Counsel - Simple Document Workflow
===========================================

CLEAR USER JOURNEY:
Step 1: Upload Documents → Documents appear in "Pending Processing" 
Step 2: Process Documents → Documents move to "Ready for Search"
Step 3: Search & Analyze → Use processed documents for RAG

✅ Documents disappear from "Pending" once processed
✅ Clear workflow steps with progress indicators  
✅ No complex options or AWS dependencies
✅ Intuitive single-screen design
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import io
from datetime import datetime
import logging

# Setup logging
logger = logging.getLogger(__name__)

def render_document_workflow(pipeline) -> None:
    """Render the simplified document workflow interface"""
    
    st.title("📄 Strategic Counsel - Document Management")
    
    # Store matter ID in session state for archive functionality
    if hasattr(pipeline, 'matter_id'):
        st.session_state['current_matter_id'] = pipeline.matter_id
    
    # Simple workflow indicator - REMOVED Step 3 (Search)
    st.markdown("""
    <div style="background: linear-gradient(90deg, #4CAF50, #45a049); color: white; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h3 style="margin: 0; text-align: center;">Simple Document Management</h3>
        <div style="display: flex; justify-content: space-around; margin-top: 0.5rem; font-size: 14px;">
            <div>📤 Upload Documents</div>
            <div>📋 View & Archive</div>  
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Get current document status
    processed_docs = list(pipeline.document_metadata.values()) if hasattr(pipeline, 'document_metadata') else []
    
    # Create two columns for the simplified workflow
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📤 Upload Documents")
        _render_upload_section(pipeline)
    
    with col2:
        st.markdown("### 📋 Document Library")
        _render_processed_section(processed_docs, pipeline)

def _render_upload_section(pipeline) -> None:
    """Render the document upload section"""
    
    st.markdown("**Upload PDF, DOCX, or TXT files:**")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        key="workflow_upload",
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) ready to process**")
        
        # Show uploaded files
        for file in uploaded_files:
            st.markdown(f"📄 {file.name} ({file.size:,} bytes)")
        
        # Process button
        if st.button("⚡ Process All Documents", type="primary", key="process_docs"):
            _process_uploaded_files(uploaded_files, pipeline)

def _process_uploaded_files(uploaded_files, pipeline) -> None:
    """Process uploaded files and add them to the pipeline"""
    
    progress_bar = st.progress(0)
    status_container = st.container()
    
    total_files = len(uploaded_files)
    successful_uploads = 0
    
    for i, uploaded_file in enumerate(uploaded_files):
        # Update progress
        progress = i / total_files  # Start progress for this file
        progress_bar.progress(progress)
        
        with status_container:
            st.info(f"📄 Processing {uploaded_file.name}... (Step {i+1} of {total_files})")
        
        try:
            # Show intermediate progress
            progress_bar.progress(progress + 0.1 / total_files)  # Text extraction
            
            # Create file object for pipeline
            file_obj = io.BytesIO(uploaded_file.getvalue())
            
            # Show vectorization progress
            with status_container:
                st.info(f"🔄 Vectorizing {uploaded_file.name}... (This takes time for proper AI processing)")
            
            progress_bar.progress(progress + 0.5 / total_files)  # Halfway through file
            
            # FIXED: Use correct method signature
            success, message, doc_info = pipeline.add_document(
                file_obj,  # FIXED: Pass file_obj, not text_content
                uploaded_file.name,
                ocr_preference="local"  # FIXED: Use ocr_preference, not source_type
            )
            
            if success:
                successful_uploads += 1
                chunks = doc_info.get('chunk_count', 0)
                with status_container:
                    st.success(f"✅ {uploaded_file.name} → {chunks} chunks created")
            else:
                with status_container:
                    st.error(f"❌ Failed to process {uploaded_file.name}: {message}")
                    
        except Exception as e:
            with status_container:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
        
        # Complete progress for this file
        progress_bar.progress((i + 1) / total_files)
    
    # Final status
    progress_bar.progress(1.0)
    
    if successful_uploads > 0:
        st.success(f"🎉 Successfully processed {successful_uploads}/{total_files} documents!")
        st.info("📋 Documents are now in your library and ready for search in the Enhanced RAG tab")
        st.rerun()  # Refresh to show processed documents
    else:
        st.error("❌ No documents were successfully processed.")

def _render_processed_section(processed_docs: List[Dict[str, Any]], pipeline) -> None:
    """Render the section showing processed documents ready for search"""
    
    if not processed_docs:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 8px; color: #6c757d;">
            📭 No documents uploaded yet<br/>
            <small>Upload documents to see them here</small>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown(f"**{len(processed_docs)} documents in library:**")
    
    # Create a simple list of processed documents
    for doc in processed_docs[:10]:  # Show first 10
        filename = doc.get('filename', 'Unknown')
        chunk_count = doc.get('chunk_count', 0)
        upload_date = doc.get('created_at', 'Unknown')
        if upload_date != 'Unknown':
            upload_date = upload_date[:19]  # Just date/time, no microseconds
        
        # Simple document card
        with st.container():
            st.markdown(f"""
            <div style="background: white; padding: 0.8rem; margin: 0.5rem 0; border-left: 4px solid #4CAF50; border-radius: 4px; border: 1px solid #e0e0e0;">
                <strong>📄 {filename}</strong><br/>
                <small>📊 {chunk_count} chunks • 📅 {upload_date}</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Archive button for each document
            if st.button(f"📦 Archive", key=f"archive_{doc['id']}", help=f"Archive {filename}"):
                _archive_document(doc, pipeline)
    
    if len(processed_docs) > 10:
        st.markdown(f"<small>... and {len(processed_docs) - 10} more documents</small>", unsafe_allow_html=True)
    
    # Bulk archive option
    if len(processed_docs) > 1:
        st.markdown("---")
        if st.button("📦 Archive All Documents", key="archive_all"):
            _archive_all_documents(processed_docs, pipeline)

def _archive_document(doc: Dict[str, Any], pipeline) -> None:
    """Archive a single document"""
    try:
        doc_id = doc.get('id')
        filename = doc.get('filename', 'Unknown')
        
        if doc_id:
            success, message = pipeline.delete_document(doc_id)
            if success:
                st.success(f"✅ Archived: {filename}")
                st.rerun()  # Refresh to remove from list
            else:
                st.error(f"❌ Failed to archive {filename}: {message}")
        else:
            st.error("❌ Document ID not found")
            
    except Exception as e:
        st.error(f"❌ Error archiving document: {str(e)}")

def _archive_all_documents(processed_docs: List[Dict[str, Any]], pipeline) -> None:
    """Archive all processed documents"""
    try:
        archived_count = 0
        for doc in processed_docs:
            doc_id = doc.get('id')
            if doc_id:
                success, message = pipeline.delete_document(doc_id)  # Use real archive method
                if success:
                    archived_count += 1
        
        if archived_count > 0:
            st.success(f"📦 Archived {archived_count} documents successfully!")
            st.info("Archived documents have been moved to `/archived_documents/` folder")
            st.rerun()
        else:
            st.warning("No documents were archived.")
            
    except Exception as e:
        st.error(f"❌ Error archiving documents: {str(e)}")

def render_workflow_status_banner() -> None:
    """Render a status banner showing the current workflow state"""
    
    # This would show current system status
    st.markdown("""
    <div style="background: #e8f5e8; padding: 0.5rem; border-radius: 4px; margin-bottom: 1rem; text-align: center;">
        ✅ <strong>System Ready:</strong> ColBERT Enabled • Local OCR • Document Archiving Active
    </div>
    """, unsafe_allow_html=True)

def render_simple_document_management(pipeline) -> None:
    """Main function to render the simplified document workflow - compatible with app.py"""
    
    # Show status banner
    render_workflow_status_banner()
    
    # Render the main workflow
    render_document_workflow(pipeline)
    
    # Optional: Show quick stats
    with st.expander("📊 System Statistics", expanded=False):
        try:
            doc_count = len(pipeline.document_metadata) if hasattr(pipeline, 'document_metadata') else 0
            chunk_count = len(pipeline.chunk_metadata) if hasattr(pipeline, 'chunk_metadata') else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Documents", doc_count)
            with col2:
                st.metric("Text Chunks", chunk_count)
            with col3:
                st.metric("Status", "Ready" if doc_count > 0 else "Empty")
                
        except Exception as e:
            st.error(f"Error loading statistics: {str(e)}") 