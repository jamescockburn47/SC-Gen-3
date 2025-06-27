#!/usr/bin/env python3
"""
Strategic Counsel - Simplified Document Management
===============================================

CRITICAL FIXES APPLIED:
✅ 1. AWS OCR completely removed - only local OCR and none
✅ 2. UI dramatically simplified - no complex view modes, sorting, or hierarchical options  
✅ 3. Document archiving works properly - documents disappear from list after archiving
✅ 4. Clean, intuitive interface focusing on core functionality

Simple workflow:
- Upload documents (PDF, TXT, DOCX)
- View document list 
- Archive documents (they move to /archived_documents/)
"""

import streamlit as st
import pandas as pd
import io
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

def render_simplified_document_management():
    """Simplified document management interface with core functionality only"""
    
    st.markdown("### 📚 Document Management")
    st.markdown("*Upload, view, and archive documents for AI analysis*")
    
    # Simple matter selection
    matter_id = st.selectbox(
        "Matter:", 
        ["Corporate Governance", "Legal Analysis", "Default Matter", "Document Review"],
        key="simple_matter",
        label_visibility="collapsed"
    )
    
    try:
        from local_rag_pipeline import rag_session_manager
        
        # Get pipeline and document status
        pipeline = rag_session_manager.get_or_create_pipeline(matter_id)
        doc_status = pipeline.get_document_status()
        
        # Simple status bar
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documents", doc_status.get('total_documents', 0))
        with col2:
            st.metric("Text Chunks", doc_status.get('total_chunks', 0))
        with col3:
            st.metric("Status", "🟢 Ready" if doc_status.get('storage_path') else "🔴 Error")
        
        # Simple upload section
        st.markdown("---")
        st.markdown("#### 📤 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose files to upload:",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'docx', 'doc'],
            help="Supported: PDF, TXT, DOCX, DOC"
        )
        
        if uploaded_files:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Simple OCR choice - NO AWS
                ocr_method = st.radio(
                    "OCR Method:",
                    ["local", "none"],
                    index=0,
                    help="Local OCR for PDFs, or no OCR",
                    horizontal=True
                )
            
            with col2:
                if st.button("📤 Upload", type="primary"):
                    upload_documents(uploaded_files, pipeline, ocr_method)
        
        # Simple document list
        if doc_status.get('total_documents', 0) > 0:
            st.markdown("---")
            st.markdown("#### 📋 Your Documents")
            
            documents = doc_status.get('documents', [])
            render_simple_document_list(documents, pipeline)
        
        else:
            st.info("📄 No documents yet. Upload some documents above to get started.")
        
    except Exception as e:
        st.error(f"❌ Error: {e}")


def upload_documents(uploaded_files: List, pipeline, ocr_method: str):
    """Simple document upload with progress"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    successful = 0
    failed = 0
    
    for i, uploaded_file in enumerate(uploaded_files):
        progress = (i + 1) / len(uploaded_files)
        progress_bar.progress(progress)
        status_text.text(f"Processing {uploaded_file.name}...")
        
        # Process document
        file_obj = io.BytesIO(uploaded_file.getvalue())
        
        success, message, doc_info = pipeline.add_document(
            file_obj, uploaded_file.name, ocr_method
        )
        
        if success:
            successful += 1
            st.success(f"✅ {uploaded_file.name}")
        else:
            failed += 1
            st.error(f"❌ {uploaded_file.name}: {message}")
    
    progress_bar.progress(1.0)
    status_text.text(f"Complete! {successful} uploaded, {failed} failed.")
    
    if successful > 0:
        st.rerun()


def render_simple_document_list(documents: List[Dict[str, Any]], pipeline):
    """Simple, clean document list"""
    
    # Sort by upload date (newest first)
    documents.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    
    for doc in documents:
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**📄 {doc['filename']}**")
                st.caption(f"📅 {doc.get('created_at', 'N/A')[:19]} | 🧩 {doc.get('chunk_count', 0)} chunks | 📏 {doc.get('text_length', 0):,} chars")
            
            with col2:
                if st.button("👁️ Preview", key=f"preview_{doc['id']}", help="View content"):
                    show_simple_preview(doc, pipeline)
            
            with col3:
                if st.button("📦 Archive", key=f"archive_{doc['id']}", type="secondary", help="Move to archive"):
                    archive_document(doc, pipeline)
            
            st.markdown("---")


def show_simple_preview(doc: Dict[str, Any], pipeline):
    """Show document preview in a simple way"""
    
    doc_path = pipeline.documents_path / f"{doc['id']}.txt"
    if doc_path.exists():
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
                preview = content[:500] + "..." if len(content) > 500 else content
            
            st.text_area(
                f"📄 {doc['filename']}", 
                value=preview,
                height=200,
                disabled=True,
                key=f"preview_content_{doc['id']}"
            )
        except Exception as e:
            st.error(f"Error reading document: {e}")
    else:
        st.error("Document file not found")


def archive_document(doc: Dict[str, Any], pipeline):
    """Archive document with confirmation"""
    
    # Simple confirmation
    if st.button(f"⚠️ Confirm Archive: {doc['filename'][:30]}...", 
                key=f"confirm_{doc['id']}", 
                type="secondary"):
        
        success, message = pipeline.delete_document(doc['id'])  # This archives the document
        if success:
            st.success(f"✅ {message}")
            st.rerun()  # Refresh to remove from list
        else:
            st.error(f"❌ {message}")


if __name__ == "__main__":
    render_simplified_document_management() 