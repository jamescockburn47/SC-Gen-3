#!/usr/bin/env python3
"""Strategic Counsel v3.3 - CH Year Order, Export/Memory, Protocol/Red Flag UI

Key Changes:
- CH Summaries now automatically use a Gemini model (via ch_pipeline.py logic).
- Sidebar AI model selector is now only for 'Consult Counsel & Digest Updates'.
- CH Pipeline returns AI summarization costs, displayed in UI.
- Implemented more accurate token counting for Gemini in 'Consult Counsel'.
- Corrected attribute access for Gemini SDK check.
- CH Results display uses st.expander per company.
- Added UI for keyword-based filtering in CH analysis (backend logic placeholder).
- Added "Copy Summary" to CH expanders (via st.code).
- CH Summaries can now be selected for injection into Counsel chat.
- Added Protocol status display in sidebar.
- Attempt to highlight "Red Flags" section from CH summaries.
- Sidebar sections collapsed with expanders for easier navigation.
"""

from __future__ import annotations

import streamlit as st
import datetime as _dt # Added for date operations
import pathlib as _pl # Added for Path operations
import hashlib as _hashlib # Added for hashing
import json # Added for json operations
import io # Added for io operations
import logging # Standard library logging
logger = logging.getLogger(__name__)   # ← ADD (before the try/except)
logger.setLevel(logging.INFO)          # optional: default level here

import sys
from datetime import datetime

# Ensure the root directory is in sys.path for module imports
APP_ROOT_DIR = _pl.Path(__file__).resolve().parent
if str(APP_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(APP_ROOT_DIR))

# App-specific modules
import config
import ch_pipeline
import app_utils
from about_page import render_about_page # Changed from show_about_page
from instructions_page import render_instructions_page
try:
    import group_structure_utils
    GROUP_STRUCTURE_AVAILABLE = True
except ImportError as e:
    logger.error(
        "Failed to import group_structure_utils: %s. "
        "Group-structure tab may not function correctly.", e
    )
    GROUP_STRUCTURE_AVAILABLE = False
except Exception as e:
    logger.error(
        "Error importing group_structure_utils module: %s",
        e,
        exc_info=True,
    )
    GROUP_STRUCTURE_AVAILABLE = False

if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# Define APP_BASE_PATH - this should ideally be defined once, consistently
APP_BASE_PATH = APP_ROOT_DIR

# --- Logger Setup ---
# Basic logger configuration (customize as needed)
LOG_DIR = APP_BASE_PATH / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE_PATH = LOG_DIR / f"app_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Create handlers
# c_handler = logging.StreamHandler() # Console handler (optional, Streamlit handles stdout)
f_handler = logging.FileHandler(LOG_FILE_PATH)
# c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# c_handler.setFormatter(log_format)
f_handler.setFormatter(log_format)

# Add handlers to the logger
# logger.addHandler(c_handler)
if not logger.handlers: # Avoid adding multiple file handlers on Streamlit reruns
    logger.addHandler(f_handler)

# --- Initialize Session State ---

import streamlit as st
st.set_page_config(
    page_title="Strategic Counsel", page_icon="⚖️", layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': "# Strategic Counsel v3.3\nModular AI Legal Assistant Workspace."}
)

# Load Harcus Parker inspired CSS
def load_local_css(path: str | _pl.Path) -> None:
    """Utility to inject local CSS into the Streamlit app."""
    try:
        css = _pl.Path(path).read_text()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except Exception as e:
        logger.error("Could not load CSS %s: %s", path, e)

load_local_css(APP_BASE_PATH / "static" / "harcus_parker_style.css")

try:
    import config
    logger = config.logger
    from ch_pipeline import TEXTRACT_AVAILABLE as CH_PIPELINE_TEXTRACT_FLAG_FROM_MODULE
    config.CH_PIPELINE_TEXTRACT_FLAG = CH_PIPELINE_TEXTRACT_FLAG_FROM_MODULE
except ImportError as e_initial_imports:
    st.error(f"Fatal Error: Could not import core modules (config, ch_pipeline): {e_initial_imports}")
    st.stop()
except Exception as e_conf:
    st.error(f"Fatal Error during config.py import or setup: {e_conf}")
    st.stop()

import datetime as _dt
import hashlib as _hashlib
import io
import json
import os
import pathlib as _pl
import re # For Red Flag parsing
import tempfile
import csv
from typing import List, Tuple, Dict, Optional, Any

import pandas as pd

try:
    from app_utils import (
        summarise_with_title,
        fetch_url_content,
        find_company_number,
        extract_text_from_uploaded_file,
        build_consult_docx,
        extract_legal_citations,
        verify_citations,
    )
    from about_page import render_about_page
    from instructions_page import render_instructions_page
    from ch_pipeline import run_batch_company_analysis
    from ai_utils import get_improved_prompt, check_protocol_compliance
    # from ai_utils import _gemini_generate_content_with_retry_and_tokens # Not directly used in app.py typically
except ImportError as e_app_utils_more:
    st.error(f"Fatal Error: Could not import app utilities or CH pipeline: {e_app_utils_more}")
    logger.error(f"ImportError from app_utils/about_page/ch_pipeline/ai_utils: {e_app_utils_more}", exc_info=True)
    st.stop()

APP_BASE_PATH: _pl.Path = config.APP_BASE_PATH
OPENAI_API_KEY_PRESENT = bool(config.OPENAI_API_KEY and config.OPENAI_API_KEY.startswith("sk-"))
CH_API_KEY_PRESENT = bool(config.CH_API_KEY)
GEMINI_API_KEY_PRESENT = bool(config.GEMINI_API_KEY and config.genai) # Corrected to check config.genai

REQUIRED_DIRS_REL = ("memory", "memory/digests", "summaries", "exports", "logs", "static")
for rel_p in REQUIRED_DIRS_REL:
    abs_p = APP_BASE_PATH / rel_p
    try: abs_p.mkdir(parents=True, exist_ok=True)
    except OSError as e_mkdir: st.error(f"Fatal Error creating directory {abs_p.name}: {e_mkdir}"); st.stop()

MODEL_PRICES_PER_1K_TOKENS_GBP: Dict[str, float] = {
    "gpt-4.1": 0.0080,
    "gemini-3.5": 0.0028,
}
MODEL_ENERGY_WH_PER_1K_TOKENS: Dict[str, float] = {
    "gpt-4.1": 0.4,
    "gemini-3.5": 0.2,
}
KETTLE_WH: int = 360

PROTO_PATH = APP_BASE_PATH / "strategic_protocols.txt"
PROTO_TEXT: str
PROTO_HASH = ""
PROTO_LOAD_SUCCESS = False # Flag for successful load

if not PROTO_PATH.exists():
    PROTO_TEXT = config.PROTO_TEXT_FALLBACK
    logger.warning(f"Protocol file {PROTO_PATH.name} not found. Using fallback.")
    config.LOADED_PROTO_PATH_NAME = PROTO_PATH.name # For about_page.py
    config.LOADED_PROTO_TEXT = PROTO_TEXT # For about_page.py
    PROTO_LOAD_SUCCESS = False
else:
    try:
        PROTO_TEXT = PROTO_PATH.read_text(encoding="utf-8")
        PROTO_HASH = _hashlib.sha256(PROTO_TEXT.encode()).hexdigest()[:8]
        config.PROTO_TEXT_FALLBACK = PROTO_TEXT # Update fallback if successfully loaded
        config.LOADED_PROTO_PATH_NAME = PROTO_PATH.name # For about_page.py
        config.LOADED_PROTO_TEXT = PROTO_TEXT # For about_page.py
        logger.info(f"Successfully loaded protocol from {PROTO_PATH.name}")
        PROTO_LOAD_SUCCESS = True
    except Exception as e_proto:
        PROTO_TEXT = config.PROTO_TEXT_FALLBACK
        logger.error(f"Error loading protocol file {PROTO_PATH.name}: {e_proto}. Using fallback.", exc_info=True)
        config.LOADED_PROTO_PATH_NAME = PROTO_PATH.name # Still set for about_page.py
        config.LOADED_PROTO_TEXT = PROTO_TEXT # Still set for about_page.py
        PROTO_LOAD_SUCCESS = False


CH_CATEGORIES: Dict[str, str] = {
    "Accounts": "accounts", "Confirmation Stmt": "confirmation-statement", "Officers": "officers",
    "Capital": "capital", "Charges": "mortgage", "Insolvency": "insolvency",
    "PSC": "persons-with-significant-control", "Name Change": "change-of-name",
    "Reg. Office": "registered-office-address",
}

def init_session_state():
    defaults = {
        "current_topic": "general_default_topic", "session_history": [], "loaded_memories": [],
        "processed_summaries": [], # (src_id, title, summary_text) for uploaded docs/URLs
        "selected_summary_texts": [], # Texts of selected uploaded doc/URL summaries for PRIMARY context
        "latest_digest_content": "",
        "document_processing_complete": True, "ch_last_digest_path": None, "ch_last_df": None,
        "ch_last_narrative": None, "ch_last_batch_metrics": {},
        "consult_digest_model": config.OPENAI_MODEL_DEFAULT if 'config' in globals() and hasattr(config, 'OPENAI_MODEL_DEFAULT') else "gpt-4.1", # Fallback if config not loaded
        "ch_analysis_summaries_for_injection": [], # List of (company_id, title_for_list, summary_text)
        
        # For "Improve Prompt" in Consult Counsel
        "user_instruction_main_text_area_value": "", 
        "original_user_instruction_main": "", 
        "user_instruction_main_is_improved": False,

        # For "Improve Prompt" in CH Analysis
        "additional_ai_instructions_ch_text_area_value": "", 
        "original_additional_ai_instructions_ch": "", 
        "additional_ai_instructions_ch_is_improved": False,

        # New session state for CH document selection flow
        "ch_available_documents": [], 
        "ch_document_selection": {}, 
        "ch_start_year_input_main": _dt.date.today().year - 4, # Default start year (e.g., 4 years ago)
        "ch_end_year_input_main": _dt.date.today().year, # Default end year (current year)
        # Session states for Company Group Structure Analysis - ALIGNED WITH UI
        "group_structure_cn_for_analysis": "", # Matches UI
        "group_structure_report": [], # Matches UI
        "group_structure_viz_data": None, # Matches UI
        "suggested_parent_cn_for_rerun": None, # Matches UI
        "group_structure_parent_timeline": [] # Matches UI
        ,"last_ai_response_text": ""
        ,"last_protocol_compliance_report": ""
        ,"last_protocol_compliance_tokens": (0,0)
        ,"auto_protocol_compliance": True
        # Note: "company_group_analysis_results" and "company_group_ultimate_parent_cn"
        # from the original init did not have direct equivalents in the UI's init block.
        # If they are needed elsewhere, ensure they are handled consistently or add them here
        # with the 'group_structure_' prefix if they belong to this feature.
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value
init_session_state()

with st.sidebar:
    st.markdown("## Configuration")
    current_topic_input = st.text_input("Matter / Topic ID", st.session_state.current_topic, key="topic_input_sidebar")
    if current_topic_input != st.session_state.current_topic:
        st.session_state.current_topic = current_topic_input
        # Reset topic-specific states
        st.session_state.session_history = []
        st.session_state.processed_summaries = []
        st.session_state.selected_summary_texts = []
        st.session_state.loaded_memories = []
        st.session_state.latest_digest_content = ""
        st.session_state.ch_last_digest_path = None
        st.session_state.ch_last_df = None
        st.session_state.ch_last_narrative = None
        st.session_state.ch_last_batch_metrics = {}
        st.session_state.ch_analysis_summaries_for_injection = [] # Crucially reset this
        # Reset group structure states
        st.session_state.group_structure_cn_for_analysis = ""
        st.session_state.group_structure_report = []
        st.session_state.group_structure_viz_data = None
        st.session_state.suggested_parent_cn_for_rerun = None
        st.session_state.group_structure_parent_timeline = []
        st.rerun()

    def _topic_color_style(topic_str: str) -> str:
        color_hue = int(_hashlib.sha1(topic_str.encode()).hexdigest(), 16) % 360
        # Return CSS variables for dynamic styling in the custom CSS block
        return f"--topic-hue:{color_hue};"

    st.markdown(f'''<div class="topic-display-box" style="{_topic_color_style(st.session_state.current_topic)}">
                    Topic: <strong>{st.session_state.current_topic}</strong>
                 </div>''', unsafe_allow_html=True)

    # --- Protocol Status Display ---
    with st.expander("System Status", expanded=False):
        protocol_status_message = ""
        protocol_status_type = "info"  # Can be "success", "warning", "error"

        if PROTO_LOAD_SUCCESS:
            protocol_status_message = f"Protocol '{PROTO_PATH.name}' loaded (Hash: {PROTO_HASH})."
            protocol_status_type = "success"
        elif not PROTO_PATH.exists():
            protocol_status_message = f"Protocol file '{PROTO_PATH.name}' not found. Using default protocol."
            protocol_status_type = "warning"
        else:
            protocol_status_message = f"Error loading protocol '{PROTO_PATH.name}'. Using default protocol."
            protocol_status_type = "error"

        if protocol_status_type == "success":
            st.success(protocol_status_message)
        elif protocol_status_type == "warning":
            st.warning(protocol_status_message)
        else:
            st.error(protocol_status_message)
    # --- End Protocol Status Display ---


    with st.expander("AI Model Selection", expanded=False):
        if not OPENAI_API_KEY_PRESENT:
            st.error("‼️ OpenAI API Key missing. OpenAI models will fail.")
        if not GEMINI_API_KEY_PRESENT:
            st.warning("⚠️ Gemini API Key missing. Gemini models unavailable for consultation.")

        all_available_models = list(MODEL_PRICES_PER_1K_TOKENS_GBP.keys())
        gpt_models = [m for m in all_available_models if m.startswith("gpt-") and OPENAI_API_KEY_PRESENT]
        gemini_models_consult = [m for m in all_available_models if m.startswith("gemini-") and GEMINI_API_KEY_PRESENT]

        selectable_models_consult = gpt_models + gemini_models_consult
        if not selectable_models_consult:
            st.error("No AI models available for Consultation/Digests!")

        default_consult_model_index = 0
        if "consult_digest_model" in st.session_state and st.session_state.consult_digest_model in selectable_models_consult:
            try:
                default_consult_model_index = selectable_models_consult.index(st.session_state.consult_digest_model)
            except ValueError:
                default_consult_model_index = 0
        elif selectable_models_consult:
            st.session_state.consult_digest_model = selectable_models_consult[0]
        else:
            st.session_state.consult_digest_model = None

        st.session_state.consult_digest_model = st.selectbox(
            "Model for Consultation & Digests:",
            selectable_models_consult,
            index=default_consult_model_index,
            key="consult_digest_model_selector_main",
            disabled=not selectable_models_consult,
        )
        if st.session_state.consult_digest_model:
            price_consult = MODEL_PRICES_PER_1K_TOKENS_GBP.get(st.session_state.consult_digest_model, 0.0)
            st.caption(f"Est. Cost/1k Tokens: £{price_consult:.5f}")
        else:
            st.caption("Est. Cost/1k Tokens: N/A")

        st.caption("CH Summaries will use Gemini by default for speed (if configured), or fallback to OpenAI.")

        ai_temp = st.slider(
            "AI Creativity (Temperature)",
            0.0,
            1.0,
            0.2,
            0.05,
            key="ai_temp_slider_sidebar",
        )

    with st.expander("Context Injection", expanded=False):
        memory_file_path = APP_BASE_PATH / "memory" / f"{st.session_state.current_topic}.json"
        loaded_memories_from_file: List[str] = []
        if memory_file_path.exists():
            try:
                mem_data = json.loads(memory_file_path.read_text(encoding="utf-8"))
                if isinstance(mem_data, list):
                    loaded_memories_from_file = [str(item) for item in mem_data if isinstance(item, str)]
            except Exception as e_mem_load:
                st.warning(f"Could not load memory file {memory_file_path.name}: {e_mem_load}")
        selected_mem_snippets = st.multiselect(
            "Inject Memories",
            loaded_memories_from_file,
            default=[mem for mem in st.session_state.loaded_memories if mem in loaded_memories_from_file],
            key="mem_multiselect_sidebar",
        )
        st.session_state.loaded_memories = selected_mem_snippets

        digest_file_path = APP_BASE_PATH / "memory" / "digests" / f"{st.session_state.current_topic}.md"
        if digest_file_path.exists():
            try:
                st.session_state.latest_digest_content = digest_file_path.read_text(encoding="utf-8")
            except Exception as e_digest_load:
                st.warning(f"Could not load digest {digest_file_path.name}: {e_digest_load}")
                st.session_state.latest_digest_content = ""
        else:
            st.session_state.latest_digest_content = ""
        inject_digest_checkbox = st.checkbox(
            "Inject Digest",
            value=bool(st.session_state.latest_digest_content),
            key="inject_digest_checkbox_sidebar",
            disabled=not bool(st.session_state.latest_digest_content),
        )

        st.markdown("---"); st.markdown("### Document Intake (for Context)")
        uploaded_docs_list = st.file_uploader(
            "Upload Docs (PDF, DOCX, TXT)",
            ["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="doc_uploader_sidebar",
        )
        urls_input_str = st.text_area("Paste URLs (one per line)", key="url_textarea_sidebar", height=80)
        urls_to_process = [u.strip() for u in urls_input_str.splitlines() if u.strip().startswith("http")]

        current_source_identifiers = {f.name for f in uploaded_docs_list} | set(urls_to_process)
        processed_summary_ids_in_session = {s_tuple[0] for s_tuple in st.session_state.processed_summaries}
        sources_needing_processing = current_source_identifiers - processed_summary_ids_in_session
    
        newly_processed_summaries_for_this_run_sidebar: List[Tuple[str, str, str]] = [] # Define here for wider scope
        if sources_needing_processing and st.session_state.document_processing_complete:
            st.session_state.document_processing_complete = False # Prevent re-processing during rerun
            summaries_cache_dir_for_topic = APP_BASE_PATH / "summaries" / st.session_state.current_topic
            summaries_cache_dir_for_topic.mkdir(parents=True, exist_ok=True)
    
            with st.spinner(f"Processing {len(sources_needing_processing)} new document(s)/URL(s)..."):
                progress_bar_docs = st.progress(0.0)
                for idx, src_id in enumerate(list(sources_needing_processing)): # Convert set to list for indexing
                    title, summary = "Error", "Processing Failed"
                    # Simple cache key based on source identifier hash
                    cache_file_name = f"summary_{_hashlib.sha256(src_id.encode()).hexdigest()[:16]}.json"
                    summary_cache_file = summaries_cache_dir_for_topic / cache_file_name
    
                    if summary_cache_file.exists():
                        try:
                            cached_data = json.loads(summary_cache_file.read_text(encoding="utf-8"))
                            title, summary = cached_data.get("t", "Cache Title Error"), cached_data.get("s", "Cache Summary Error")
                        except Exception: title, summary = "Error", "Processing Failed (Cache Read)" # More specific cache error
    
                    if title == "Error" or "Cache" in title : # If cache load failed or it was an error state
                        raw_content, error_msg = None, None
                        # Check if it's an uploaded file or a URL
                        if src_id in {f.name for f in uploaded_docs_list}: # Is it an uploaded file?
                            file_obj = next((f for f in uploaded_docs_list if f.name == src_id), None)
                            if file_obj: raw_content, error_msg = extract_text_from_uploaded_file(io.BytesIO(file_obj.getvalue()), src_id)
                        elif src_id in urls_to_process: # Is it a URL?
                            raw_content, error_msg = fetch_url_content(src_id)
    
                        if error_msg: title, summary = f"Error: {src_id[:40]}...", error_msg
                        elif not raw_content or not raw_content.strip(): title, summary = f"Empty: {src_id[:40]}...", "No text content found or extracted."
                        else: # Successfully got raw content
                            # Use a cost-effective model for these quick summaries
                            title, summary = summarise_with_title(raw_content, config.OPENAI_MODEL_DEFAULT, st.session_state.current_topic, PROTO_TEXT)
    
                        if "Error" not in title and "Empty" not in title: # Cache if successfully processed
                            try: summary_cache_file.write_text(json.dumps({"t":title,"s":summary,"src":src_id}),encoding="utf-8")
                            except Exception as e_c: logger.warning(f"Cache write failed for {src_id}: {e_c}")
    
                    newly_processed_summaries_for_this_run_sidebar.append((src_id, title, summary))
                    progress_bar_docs.progress((idx + 1) / len(sources_needing_processing))
    
                # Update session state: keep existing ones that are still valid, add new ones
                existing_to_keep = [s for s in st.session_state.processed_summaries if s[0] in current_source_identifiers and s[0] not in sources_needing_processing]
                st.session_state.processed_summaries = existing_to_keep + newly_processed_summaries_for_this_run_sidebar
                progress_bar_docs.empty()
            st.session_state.document_processing_complete = True; st.rerun() # Rerun to update UI with new summaries
    
        # Selection for Uploaded/URL Summaries
        st.session_state.selected_summary_texts = [] # Reset before populating based on checkbox state
        if st.session_state.processed_summaries:
            st.markdown("---"); st.markdown("### Available Doc/URL Summaries (Select to Inject)")
            for idx, (s_id, title, summary_text) in enumerate(st.session_state.processed_summaries):
                checkbox_key = f"sum_sel_{_hashlib.md5(s_id.encode()).hexdigest()}"
                is_newly_processed = any(s_id == item[0] for item in newly_processed_summaries_for_this_run_sidebar)
                # Default to checked if newly processed, or if previously checked (and still exists)
                default_checked = is_newly_processed or st.session_state.get(checkbox_key, False)
                is_injected = st.checkbox(f"{idx+1}. {title[:40]}...", value=default_checked, key=checkbox_key, help=f"Source: {s_id}\nSummary: {summary_text[:200]}...")
                if is_injected: st.session_state.selected_summary_texts.append(f"UPLOADED DOCUMENT/URL SUMMARY ('{title}'):\n{summary_text}")
    
    
        # Selection for CH Analysis Summaries
        selected_ch_summary_texts_for_injection_temp = [] # Temp list for this run
        if st.session_state.ch_analysis_summaries_for_injection:
            st.markdown("---"); st.markdown("### CH Analysis Summaries (Select to Inject)")
            for idx, (company_id, title_for_list, summary_text) in enumerate(st.session_state.ch_analysis_summaries_for_injection):
                ch_checkbox_key = f"ch_sum_sel_{_hashlib.md5(company_id.encode() + title_for_list.encode()).hexdigest()}"
                # Default to False unless explicitly checked by the user.
                is_ch_summary_injected = st.checkbox(f"{idx+1}. CH: {title_for_list[:40]}...", value=st.session_state.get(ch_checkbox_key, False), key=ch_checkbox_key, help=f"Company: {company_id}\nSummary: {summary_text[:200]}...")
                if is_ch_summary_injected:
                    selected_ch_summary_texts_for_injection_temp.append(f"COMPANIES HOUSE ANALYSIS SUMMARY FOR {company_id} ({title_for_list}):\n{summary_text}")
        # This state is now dynamically built when creating context for AI rather than storing selection list in session_state permanently


    st.markdown("---")
    if st.button("End Session & Update Digest", key="end_session_button_sidebar"):
        if not st.session_state.session_history: st.warning("No new interactions to add to digest.")
        elif not st.session_state.consult_digest_model: st.error("No AI model selected for Digest Update.")
        else:
            with st.spinner("Updating Digest..."):
                new_interactions_block = "\n\n---\n\n".join(st.session_state.session_history)
                existing_digest_text = st.session_state.latest_digest_content
                update_digest_prompt = (f"Consolidate the following notes. Integrate the NEW interactions into the EXISTING digest, "
                                    f"maintaining a coherent and concise summary. Aim for a maximum of around 2000 words for the entire updated digest. "
                                    f"Preserve key facts and decisions.\n\n"
                                    f"EXISTING DIGEST (for topic: {st.session_state.current_topic}):\n{existing_digest_text}\n\n"
                                    f"NEW INTERACTIONS (to integrate for topic: {st.session_state.current_topic}):\n{new_interactions_block}")
                try:
                    current_ai_model_for_digest = st.session_state.consult_digest_model
                    updated_digest_text = "Error updating digest."
                    if current_ai_model_for_digest.startswith("gpt-"):
                        client = config.get_openai_client(); assert client
                        resp = client.chat.completions.create(model=current_ai_model_for_digest, temperature=0.1, max_tokens=3000, messages=[{"role": "system", "content": PROTO_TEXT}, {"role": "user", "content": update_digest_prompt}])
                        updated_digest_text = resp.choices[0].message.content.strip()
                    elif current_ai_model_for_digest.startswith("gemini-"):
                        client = config.get_gemini_model(current_ai_model_for_digest); assert client and config.genai # Check config.genai
                        full_prompt_gemini = f"{PROTO_TEXT}\n\n{update_digest_prompt}" # Combine for Gemini
                        resp = client.generate_content(full_prompt_gemini, generation_config=config.genai.types.GenerationConfig(temperature=0.1, max_output_tokens=3000)) # Use config.genai
                        updated_digest_text = resp.text.strip() # Check for block reason

                    digest_file_path.write_text(updated_digest_text, encoding="utf-8")
                    historical_digest_path = APP_BASE_PATH / "memory" / "digests" / f"history_{st.session_state.current_topic}.md"
                    with historical_digest_path.open("a", encoding="utf-8") as fp_hist:
                        fp_hist.write(f"\n\n### Update: {_dt.datetime.now():%Y-%m-%d %H:%M} (Model: {current_ai_model_for_digest})\n{updated_digest_text}\n---\n")
                    st.success(f"Digest for '{st.session_state.current_topic}' updated."); st.session_state.session_history = []; st.session_state.latest_digest_content = updated_digest_text; st.rerun()
                except Exception as e_digest_update:
                    st.error(f"Digest update failed: {e_digest_update}"); logger.error(f"Digest update error: {e_digest_update}", exc_info=True)

    with st.expander("Protocol Compliance", expanded=False):
        st.session_state.auto_protocol_compliance = st.checkbox(
            "Auto-check after each response",
            value=st.session_state.get("auto_protocol_compliance", True),
            key="auto_protocol_compliance_checkbox",
        )
        if st.button("Run Protocol Compliance Report", key="protocol_compliance_button"):
            latest_output = st.session_state.get("last_ai_response_text", "")
            if not latest_output:
                st.warning("No AI output available for compliance check.")
            else:
                with st.spinner("Checking compliance..."):
                    report_text, rpt_p, rpt_c = check_protocol_compliance(latest_output, PROTO_TEXT)
                st.session_state.last_protocol_compliance_report = report_text
                st.session_state.last_protocol_compliance_tokens = (rpt_p, rpt_c)
                if "Error:" in report_text:
                    st.error(report_text)
                else:
                    st.success("Compliance check complete. See report below.")

        if st.session_state.get("last_protocol_compliance_report"):
            with st.expander("Latest Compliance Report", expanded=False):
                st.text_area(
                    "Protocol Compliance Report",
                    st.session_state.last_protocol_compliance_report,
                    height=250,
                )
                p_tok, c_tok = st.session_state.last_protocol_compliance_tokens
                st.caption(f"Prompt tokens: {p_tok}, Completion tokens: {c_tok}")

# ── Main Application Area UI (Using Tabs) ─────────────────────────
st.markdown(f"## 🏛️ Strategic Counsel: {st.session_state.current_topic}")

# Define tabs based on what functionality is available
if 'GROUP_STRUCTURE_AVAILABLE' in globals() and GROUP_STRUCTURE_AVAILABLE:
    tab_consult, tab_ch_analysis, tab_group_structure, tab_about_rendered, tab_instructions = st.tabs([
        "💬 Consult Counsel",
        "🇬🇧 Companies House Analysis",
        "🕸️ Company Group Structure",  # Include group structure tab
        "ℹ️ About",
        "📖 Instructions"
    ])
else:
    # Fall back to three tabs if group structure not available
    tab_consult, tab_ch_analysis, tab_about_rendered, tab_instructions = st.tabs([
        "💬 Consult Counsel",
        "🇬🇧 Companies House Analysis",
        "ℹ️ About",
        "📖 Instructions"
    ])
    # Create a placeholder for tab_group_structure to avoid errors later in the code
    class PlaceholderTab:
        def __enter__(self): pass
        def __exit__(self, *args): pass
    tab_group_structure = PlaceholderTab()

with tab_consult:
    st.markdown("Provide instructions and context (using sidebar options) for drafting, analysis, or advice.")
    
    # Text area for user's main instruction. Its value is stored in st.session_state.main_instruction_area_consult_tab (by Streamlit)
    # and mirrored to st.session_state.user_instruction_main_text_area_value by the on_change callback.
    st.text_area(
        "Your Instruction:", 
        value=st.session_state.user_instruction_main_text_area_value, # Display value from our dedicated session state
        height=200, 
        key="main_instruction_area_consult_tab", # Key for this specific widget
        on_change=lambda: st.session_state.update(user_instruction_main_text_area_value=st.session_state.main_instruction_area_consult_tab) # Update our dedicated state from widget's state
    )

    col_improve_main, col_cancel_main, col_spacer_main = st.columns([2,2,3]) # Adjusted column ratios
    with col_improve_main:
        if st.button("💡 Suggest Improved Prompt", key="improve_prompt_main_button", help="Let AI refine your instruction for better results.", use_container_width=True):
            current_text_in_area = st.session_state.user_instruction_main_text_area_value 
            if current_text_in_area and current_text_in_area.strip():
                if not st.session_state.user_instruction_main_is_improved: 
                    st.session_state.original_user_instruction_main = current_text_in_area
                
                with st.spinner("Improving prompt..."):
                    improved_prompt = get_improved_prompt(current_text_in_area, "Strategic Counsel general query")
                    if "Error:" not in improved_prompt and improved_prompt.strip():
                        st.session_state.user_instruction_main_text_area_value = improved_prompt 
                        st.session_state.user_instruction_main_is_improved = True
                        st.rerun() 
                    elif "Error:" in improved_prompt:
                        st.warning(f"Could not improve prompt: {improved_prompt}")
                    # If prompt is empty or only whitespace after improvement, no change is made to the text area.
            else:
                st.info("Please enter an instruction first to improve it.")

    with col_cancel_main:
        if st.session_state.user_instruction_main_is_improved:
            if st.button("↩️ Revert to Original", key="cancel_improve_prompt_main_button", use_container_width=True):
                st.session_state.user_instruction_main_text_area_value = st.session_state.original_user_instruction_main
                st.session_state.user_instruction_main_is_improved = False
                st.rerun()

    consult_model_name = st.session_state.get("consult_digest_model")

    if st.button("✨ Consult Counsel", type="primary", key="run_ai_button_consult_tab"):
        current_instruction_to_use = st.session_state.user_instruction_main_text_area_value

        if not current_instruction_to_use.strip(): st.warning("Please enter your instructions.")
        elif not consult_model_name: st.error("No AI model selected for Consultation.")
        else:
            with st.spinner(f"Consulting {consult_model_name}..."):
                messages_for_ai = [{"role": "system", "content": PROTO_TEXT + f"\n[Protocol Hash:{PROTO_HASH}]"}]
                context_parts_for_ai = []
                if inject_digest_checkbox and st.session_state.latest_digest_content: context_parts_for_ai.append(f"CURRENT DIGEST:\n{st.session_state.latest_digest_content}")
                if st.session_state.loaded_memories: context_parts_for_ai.append("INJECTED MEMORIES:\n" + "\n---\n".join(st.session_state.loaded_memories))

                combined_selected_summaries = []
                if st.session_state.selected_summary_texts: 
                    combined_selected_summaries.extend(st.session_state.selected_summary_texts)
                
                if "ch_analysis_summaries_for_injection" in st.session_state and st.session_state.ch_analysis_summaries_for_injection:
                    for idx, (company_id, title_for_list, summary_text) in enumerate(st.session_state.ch_analysis_summaries_for_injection):
                        ch_checkbox_key = f"ch_sum_sel_{_hashlib.md5(company_id.encode() + title_for_list.encode()).hexdigest()}"
                        if st.session_state.get(ch_checkbox_key, False): 
                            combined_selected_summaries.append(f"COMPANIES HOUSE ANALYSIS SUMMARY FOR {company_id} ({title_for_list}):\n{summary_text}")
                
                if combined_selected_summaries:
                    context_parts_for_ai.append("SELECTED DOCUMENT SUMMARIES & ANALYSIS:\n" + "\n===\n".join(combined_selected_summaries))

                if context_parts_for_ai: messages_for_ai.append({"role": "system", "content": "ADDITIONAL CONTEXT:\n\n" + "\n\n".join(context_parts_for_ai)})
                messages_for_ai.append({"role": "user", "content": current_instruction_to_use}) # Use the potentially improved instruction

                try:
                    ai_response_text = "Error: AI response could not be generated."
                    prompt_tokens, completion_tokens = 0, 0

                    if consult_model_name.startswith("gpt-"):
                        openai_client = config.get_openai_client(); assert openai_client
                        ai_api_response = openai_client.chat.completions.create(
                            model=consult_model_name, temperature=ai_temp, messages=messages_for_ai, max_tokens=3500
                        )
                        ai_response_text = ai_api_response.choices[0].message.content.strip()
                        if ai_api_response.usage:
                            prompt_tokens = ai_api_response.usage.prompt_tokens
                            completion_tokens = ai_api_response.usage.completion_tokens
                    elif consult_model_name.startswith("gemini-"):
                        gemini_model_client = config.get_gemini_model(consult_model_name); assert gemini_model_client and config.genai
                        try: 
                            full_prompt_str_gemini = "\n\n".join([f"{m['role']}: {m['content']}" for m in messages_for_ai])
                            count_resp_prompt = gemini_model_client.count_tokens(full_prompt_str_gemini)
                            prompt_tokens = count_resp_prompt.total_tokens
                        except Exception as e_gem_count_p: logger.warning(f"Gemini prompt token count failed: {e_gem_count_p}"); prompt_tokens = 0

                        gemini_api_response = gemini_model_client.generate_content(
                            contents=messages_for_ai,
                            generation_config=config.genai.types.GenerationConfig(temperature=ai_temp, max_output_tokens=3500)
                        )
                        if hasattr(gemini_api_response, 'text') and gemini_api_response.text:
                             ai_response_text = gemini_api_response.text.strip()
                             try: 
                                 count_resp_completion = gemini_model_client.count_tokens(ai_response_text)
                                 completion_tokens = count_resp_completion.total_tokens
                             except Exception as e_gem_count_c: logger.warning(f"Gemini completion token count failed: {e_gem_count_c}"); completion_tokens = 0
                        elif hasattr(gemini_api_response, 'prompt_feedback') and gemini_api_response.prompt_feedback.block_reason:
                            block_reason_str = config.genai.types.BlockedReason(gemini_api_response.prompt_feedback.block_reason).name
                            ai_response_text = f"Error: Gemini content generation blocked. Reason: {block_reason_str}."
                            logger.error(f"Gemini content blocked. Reason: {block_reason_str}. Feedback: {gemini_api_response.prompt_feedback}")
                        else:
                             ai_response_text = "Error: Gemini response was empty or malformed."
                             logger.error(f"Gemini empty/malformed response: {gemini_api_response}")
                    else:
                        raise ValueError(f"Unsupported model type for consultation: {consult_model_name}")

                    st.session_state.session_history.append(f"Instruction:\n{current_instruction_to_use}\n\nResponse ({consult_model_name}):\n{ai_response_text}") # Log the used instruction
                    citations_found = extract_legal_citations(ai_response_text)
                    verification = verify_citations(
                        citations_found,
                        uploaded_docs_list,
                        APP_BASE_PATH / "verified_sources.json",
                    )
                    unverified = [c for c, ok in verification.items() if not ok]
                    for c in unverified:
                        ai_response_text = ai_response_text.replace(c, f"{c} [UNVERIFIED]")
                    st.session_state.last_ai_response_text = ai_response_text
                    if unverified:
                        st.warning(
                            "The following citations could not be verified:\n" +
                            "\n".join(f"- {c}" for c in unverified) +
                            "\nPlease upload the source or provide a direct link."
                        )
                        with st.form(key=f"citation_links_{len(st.session_state.session_history)}"):
                            link_inputs = {}
                            for cit in unverified:
                                input_key = f"link_{_hashlib.md5(cit.encode()).hexdigest()}"
                                link_inputs[cit] = st.text_input(f"Source for {cit}", key=input_key)
                            if st.form_submit_button("Submit Links"):
                                links_path = APP_BASE_PATH / "verified_sources.json"
                                try:
                                    existing = json.loads(links_path.read_text()) if links_path.exists() else {}
                                except Exception:
                                    existing = {}
                                for cit, link in link_inputs.items():
                                    if not link:
                                        continue
                                    entry = existing.get(cit)
                                    if isinstance(entry, dict):
                                        entry["url"] = link
                                    elif isinstance(entry, bool):
                                        entry = {"verified": entry, "url": link}
                                    else:
                                        entry = {"verified": False, "url": link}
                                    existing[cit] = entry
                                try:
                                    links_path.write_text(json.dumps(existing, indent=2))
                                    st.success("Citation links saved.")
                                except Exception as e_save:
                                    st.error(f"Failed to save links: {e_save}")
                    with st.chat_message("assistant", avatar="⚖️"): st.markdown(ai_response_text)

                    if st.session_state.get("auto_protocol_compliance", True):
                        with st.spinner("Checking protocol compliance..."):
                            rpt_text, rpt_p, rpt_c = check_protocol_compliance(ai_response_text, PROTO_TEXT)
                        st.session_state.last_protocol_compliance_report = rpt_text
                        st.session_state.last_protocol_compliance_tokens = (rpt_p, rpt_c)
                        if "error:" in rpt_text.lower():
                            st.warning(rpt_text)
                        elif "non-compliant" in rpt_text.lower():
                            st.error("Protocol issues detected. See details below.")
                        else:
                            st.success("Protocol compliance OK.")
                        with st.expander("Protocol Compliance Report", expanded=False):
                            st.text_area("Protocol Compliance Report", rpt_text, height=250)
                            st.caption(f"Prompt tokens: {rpt_p}, Completion tokens: {rpt_c}")

                    with st.expander("📊 Run Details & Export"):
                        total_tokens = prompt_tokens + completion_tokens
                        cost = (total_tokens / 1000) * MODEL_PRICES_PER_1K_TOKENS_GBP.get(consult_model_name,0.0) if total_tokens > 0 else 0.0
                        energy_model_wh = MODEL_ENERGY_WH_PER_1K_TOKENS.get(consult_model_name, 0.0)
                        energy_wh = (total_tokens / 1000) * energy_model_wh if total_tokens > 0 else 0.0

                        st.metric("Total Tokens", f"{total_tokens:,}", f"{prompt_tokens:,} prompt + {completion_tokens:,} completion")
                        st.metric("Est. Cost", f"£{cost:.5f}")
                        if energy_model_wh > 0 and energy_wh > 0:
                            st.metric("Est. Energy", f"{energy_wh:.3f}Wh", f"~{(energy_wh / KETTLE_WH * 100):.1f}% Kettle" if KETTLE_WH > 0 else "")

                        ts_now_str = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                        docx_filename = f"{st.session_state.current_topic}_{ts_now_str}_response.docx"
                        docx_export_path = APP_BASE_PATH / "exports" / docx_filename
                        try:
                            convo_entry = (
                                f"Instruction:\n{current_instruction_to_use}\n\n"
                                f"Response ({consult_model_name} @ {ts_now_str}):\n{ai_response_text}"
                            )
                            build_consult_docx([convo_entry], docx_export_path)
                            with open(docx_export_path, "rb") as fp_docx:
                                st.download_button(
                                    "Download .docx",
                                    fp_docx,
                                    docx_filename,
                                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                )
                        except Exception as e_docx:
                            st.error(f"DOCX export error: {e_docx}")

                        log_filename = f"{st.session_state.current_topic}_{ts_now_str}_log.json"
                        log_export_path = APP_BASE_PATH / "logs" / log_filename
                        log_data = {"topic":st.session_state.current_topic, "timestamp":ts_now_str, "model":consult_model_name, "temp":ai_temp, "tokens":{"p":prompt_tokens,"c":completion_tokens,"t":total_tokens}, "cost_gbp":cost, "energy_wh":energy_wh, "user_instr":current_instruction_to_use[:200]+("..." if len(current_instruction_to_use) > 200 else ""), "resp_preview":ai_response_text[:200]+("..." if len(ai_response_text) > 200 else "")} # Use current_instruction_to_use
                        try: log_export_path.write_text(json.dumps(log_data, indent=2), encoding="utf-8")
                        except Exception as e_log: st.error(f"Log save error: {e_log}")

                except Exception as e_ai_consult:
                    st.error(f"AI Consultation Error with {consult_model_name}: {e_ai_consult}", icon="🚨")
                    logger.error(f"AI Consultation Error ({consult_model_name}): {e_ai_consult}", exc_info=True)

    if st.session_state.session_history:
        st.markdown("---"); st.subheader("📜 Current Session History (Newest First)")
        history_display_container = st.container(height=400) # Ensure fixed height for scroll
        for i, entry_text in enumerate(reversed(st.session_state.session_history)):
            history_display_container.markdown(f"**Interaction {len(st.session_state.session_history)-i}:**\n---\n{entry_text}\n\n")

with tab_ch_analysis:
    st.markdown("### Companies House Document Analysis")
    st.markdown(
        "Enter company numbers, select document categories and date ranges, "
        "then search for available documents. Review the list and select documents for detailed AI analysis."
    )

    ch_company_numbers_input = st.text_area(
        "Company Number(s) (one per line or comma-separated)",
        value=st.session_state.get("ch_company_numbers_input_main", ""),
        key="ch_company_numbers_input_main",
        height=80,
        help="e.g., 12345678 or SC012345, 09876543"
    )

    col_dates_ch1, col_dates_ch2 = st.columns(2)
    with col_dates_ch1:
        st.session_state.ch_start_year_input_main = st.number_input("Start Year (YYYY)", min_value=1900, max_value=2099, value=st.session_state.ch_start_year_input_main, format="%d", key="ch_start_year_widget")
    with col_dates_ch2:
        st.session_state.ch_end_year_input_main = st.number_input("End Year (YYYY)", min_value=1900, max_value=2099, value=st.session_state.ch_end_year_input_main, format="%d", key="ch_end_year_widget")

    ch_category_options_friendly = list(CH_CATEGORIES.keys())
    ch_selected_categories_friendly = st.multiselect(
        "Select Document Categories (leave blank for all major types if unsure)",
        ch_category_options_friendly,
        default=st.session_state.get("ch_categories_multiselect_main", []),
        key="ch_categories_multiselect_main"
    )
    ch_selected_categories_api = [CH_CATEGORIES[f_name] for f_name in ch_selected_categories_friendly if f_name in CH_CATEGORIES]

    st.markdown("---")
    st.markdown("#### Step 1: Find Available Company Documents")

    ch_company_numbers_list = []
    if ch_company_numbers_input: # Use the current value from the widget
        raw_list = [num.strip() for num in ch_company_numbers_input.replace(',', '\\n').splitlines() if num.strip()]
        ch_company_numbers_list = list(dict.fromkeys(raw_list))

    if st.button("🔍 Search for Available Documents", key="ch_search_documents_button"):
        if not ch_company_numbers_list:
            st.warning("Please enter at least one company number.")
        elif st.session_state.ch_start_year_input_main > st.session_state.ch_end_year_input_main:
            st.warning("Start Year cannot be after End Year.")
        elif 'ch_pipeline' not in globals() or not hasattr(ch_pipeline, 'get_relevant_filings_metadata'):
            st.error("CH Search function is not available. Please check ch_pipeline.py.")
        else:
            with st.spinner("Searching for available documents... This may take a moment."):
                try:
                    # Call the correct function signature
                    all_docs, profiles_map, meta_error = ch_pipeline.get_relevant_filings_metadata(
                        company_numbers_list=ch_company_numbers_list,
                        api_key=config.CH_API_KEY,
                        selected_categories_api=ch_selected_categories_api,
                        start_year=st.session_state.ch_start_year_input_main,
                        end_year=st.session_state.ch_end_year_input_main
                    )
                    # Assign a unique 'id' for UI selection (use transaction_id or fallback)
                    for doc in all_docs:
                        doc['id'] = doc.get('transaction_id') or doc.get('document_metadata_link') or f"{doc.get('company_number','')}_{doc.get('date','')}_{doc.get('type','')}_{doc.get('description','')[:10]}"
                    st.session_state.ch_available_documents = all_docs
                    st.session_state.ch_document_selection = {
                        doc['id']: True for doc in st.session_state.ch_available_documents
                    }
                    st.session_state.ch_company_profiles_map = profiles_map
                    if meta_error:
                        st.warning(f"Some errors occurred during metadata retrieval. See logs for details.\n{meta_error}")
                    if not st.session_state.ch_available_documents and ch_company_numbers_list:
                        st.info("No documents found matching your criteria. Try adjusting categories or date range.")
                    elif st.session_state.ch_available_documents:
                        st.success(f"Found {len(st.session_state.ch_available_documents)} potentially relevant document(s). Please review and select below.")
                except Exception as e_fetch_meta:
                    st.error(f"Error fetching CH document metadata: {e_fetch_meta}")
                st.rerun()

    if st.session_state.ch_available_documents:
        st.markdown("---")
        st.markdown("#### Step 2: Select Documents for Analysis")
        st.caption("Review the list of documents found. Uncheck any you don't want to include in the analysis.")

        docs_by_company_display = {}
        for doc_detail in st.session_state.ch_available_documents:
            docs_by_company_display.setdefault(doc_detail['company_number'], []).append(doc_detail)

        for company_num, docs_list in docs_by_company_display.items():
            with st.expander(f"Documents for {company_num} ({len(docs_list)} found)", expanded=True):
                for doc_detail in docs_list:
                    # Compose a display name for the document
                    display_name = (
                        f"{doc_detail.get('date', 'N/A')} | {doc_detail.get('type', 'N/A')} | "
                        f"{doc_detail.get('description', 'N/A')}"
                    )
                    doc_id = doc_detail['id']
                    checked = st.session_state.ch_document_selection.get(doc_id, True)
                    st.session_state.ch_document_selection[doc_id] = st.checkbox(
                        display_name,
                        value=checked,
                        key=f"ch_doc_select_{doc_id}",
                        help=f"Company: {company_num}\nType: {doc_detail.get('type', 'N/A')}\nDate: {doc_detail.get('date', 'N/A')}\nDescription: {doc_detail.get('description', 'N/A')}"
                    )

        st.markdown("---")

    st.markdown("#### Step 3: Configure and Run Analysis")
    st.text_area(
        "Additional AI Instructions for Summaries (Optional):",
        value=st.session_state.additional_ai_instructions_ch_text_area_value,
        height=100,
        key="additional_ai_instructions_ch_text_area_main",
        on_change=lambda: st.session_state.update(additional_ai_instructions_ch_text_area_value=st.session_state.additional_ai_instructions_ch_text_area_main),
        help="e.g., 'Focus on financial risks and director changes.' This will be applied to each selected document's summary."
    )
    
    ch_keywords_for_filter_input = st.text_input(
        "Keywords to Highlight/Filter in Analysis (comma-separated, optional)",
        key="ch_keywords_input_main",
        help="These keywords can be used to guide the AI or highlight sections in the final report."
    )
    ch_keywords_for_filter = [kw.strip() for kw in ch_keywords_for_filter_input.split(',') if kw.strip()]

    # Update disabled state logic for the run button
    is_any_doc_selected = any(st.session_state.ch_document_selection.values()) if st.session_state.ch_document_selection else False

    if st.button("📊 Run Analysis on Selected Documents", type="primary", key="ch_run_analysis_selected_button", disabled=not is_any_doc_selected):
        if not st.session_state.ch_document_selection:
            st.warning("Please select at least one document for analysis.")
        elif not st.session_state.ch_available_documents:
            st.warning("No available documents to analyze. Please search for documents first.")
        else:
            with st.spinner("Running Companies House analysis..."):
                try:
                    docs_to_process_by_company = {
                        company_num: [
                            doc for doc in st.session_state.ch_available_documents
                            if doc['company_number'] == company_num and st.session_state.ch_document_selection.get(doc['id'], False)
                        ]
                        for company_num in ch_company_numbers_list
                    }
                    docs_to_process_by_company = {k: v for k, v in docs_to_process_by_company.items() if v}  # Remove empty entries

                    if not docs_to_process_by_company:
                        st.warning("No documents selected for analysis. Please select documents and try again.")
                    else:
                        # Call the function and unpack its two return values
                        # output_csv_path here is the path to the "digest" CSV.
                        output_csv_path, batch_metrics = ch_pipeline.run_batch_company_analysis(
                            company_numbers_list=list(docs_to_process_by_company.keys()),
                            selected_filings_metadata_by_company=docs_to_process_by_company,
                            company_profiles_map=st.session_state.ch_company_profiles_map,
                            ch_api_key_batch=config.CH_API_KEY,
                            model_prices_gbp=MODEL_PRICES_PER_1K_TOKENS_GBP,
                            specific_ai_instructions=st.session_state.additional_ai_instructions_ch_text_area_value,
                            filter_keywords_str=ch_keywords_for_filter,
                            base_scratch_dir=APP_BASE_PATH / "temp_ch_runs",
                            keep_days=7,
                            use_textract_ocr=(config.CH_PIPELINE_TEXTRACT_FLAG if hasattr(config, 'CH_PIPELINE_TEXTRACT_FLAG') else False),
                            textract_workers=config.MAX_TEXTRACT_WORKERS,
                        )

                        # Path for the DOCX report download button
                        st.session_state.ch_last_digest_path = batch_metrics.get("output_docx_path")
                        
                        st.session_state.ch_last_batch_metrics = batch_metrics
                        st.session_state.ch_last_df = None  # For individual document details table
                        st.session_state.ch_analysis_summaries_for_injection = []  # For individual summaries
                        st.session_state.ch_last_narrative = "Analysis initiated..."

                        main_json_data_list = None
                        main_json_path_str = batch_metrics.get("output_json_path")
                        json_processed_successfully = False
                        
                        if main_json_path_str:
                            main_json_path = _pl.Path(main_json_path_str)
                            if (main_json_path.exists()):
                                try:
                                    with open(main_json_path, 'r', encoding='utf-8') as f_json:
                                        loaded_json_content = json.load(f_json)
                                    
                                    if isinstance(loaded_json_content, list):
                                        main_json_data_list = loaded_json_content
                                    elif isinstance(loaded_json_content, dict):
                                        if "processed_documents" in loaded_json_content and isinstance(loaded_json_content["processed_documents"], list):
                                            main_json_data_list = loaded_json_content["processed_documents"]
                                        elif isinstance(loaded_json_content, dict):
                                            if "processed_documents" in loaded_json_content and isinstance(loaded_json_content["processed_documents"], list):
                                                main_json_data_list = loaded_json_content["processed_documents"]
                                            elif "output_data_rows" in loaded_json_content and isinstance(loaded_json_content["output_data_rows"], list):
                                                main_json_data_list = loaded_json_content["output_data_rows"]
                                    
                                    if main_json_data_list:
                                        json_processed_successfully = True
                                        if 'logger' in globals() and hasattr(logger, 'info'):
                                            logger.info(f"Successfully loaded and parsed main JSON ({main_json_path.name}) for individual document details.")
                                        
                                        st.session_state.ch_last_df = pd.DataFrame(main_json_data_list)
                                        
                                        temp_summaries_for_injection = []
                                        for item_idx, item in enumerate(main_json_data_list):
                                            if isinstance(item, dict):
                                                company_id = str(item.get('company_number', f'Comp_N/A_{item_idx}'))
                                                doc_date = item.get('date', 'N/A')
                                                doc_type = item.get('type', 'N/A')
                                                doc_desc_raw = item.get('description', 'Document')
                                                doc_description = str(doc_desc_raw if pd.notna(doc_desc_raw) and str(doc_desc_raw).strip() else item.get('type', 'Document'))
                                                title = f"{doc_date} - {doc_type} - {doc_description[:75]}"
                                                if len(doc_description) > 75: title += "..."
                                                
                                                individual_summary_text = str(item.get('summary', 'Individual summary not available in JSON.'))
                                                
                                                status = item.get('processing_status', item.get('status'))
                                                if status and str(status).lower() not in ["success", "completed", "ok", "processed", "summarized"]:
                                                    title += f" (Status: {status})"
                                                temp_summaries_for_injection.append((company_id, title, individual_summary_text))
                                        st.session_state.ch_analysis_summaries_for_injection = temp_summaries_for_injection
                                    else: 
                                        msg = f"Main JSON ({main_json_path.name}) loaded but document list not found in expected structure."
                                        if 'logger' in globals() and hasattr(logger, 'warning'): logger.warning(msg)
                                        st.warning(msg)
                                except Exception as e_json:
                                    msg = f"Error processing main JSON output from {main_json_path_str}: {e_json}"
                                    if 'logger' in globals() and hasattr(logger, 'error'): logger.error(msg, exc_info=True)
                                    st.error(msg)
                            else: 
                                msg = f"Main JSON output file specified by pipeline but not found: {main_json_path_str}"
                                if 'logger' in globals() and hasattr(logger, 'warning'): logger.warning(msg)
                                st.warning(msg)
                        else: 
                            msg = "Pipeline did not provide a path for the main JSON output (for individual document details)."
                            if 'logger' in globals() and hasattr(logger, 'warning'): logger.warning(msg)
                            st.warning(msg)

                        # --- Narrative Generation ---
                        narrative_parts = []
                        num_companies_processed = batch_metrics.get("total_companies_processed", 0)
                        num_docs_analyzed = batch_metrics.get("total_documents_analyzed", 0)

                        # Report pipeline metrics first
                        # num_companies_processed and num_docs_analyzed are assumed to be populated from batch_metrics (e.g., lines 950-952).
                        if num_companies_processed > 0 or num_docs_analyzed > 0:
                            narrative_parts.append(f"Pipeline metrics indicate processing for **{num_companies_processed}** company/ies and **{num_docs_analyzed}** document(s).")
                        else:
                            narrative_parts.append("Pipeline metrics did not report specific company or document counts from the overall analysis.")

                        # Then, report on what was found in the JSON DataFrame
                        if json_processed_successfully and st.session_state.ch_last_df is not None and not st.session_state.ch_last_df.empty:
                            df_company_count = st.session_state.ch_last_df['company_number'].nunique() if 'company_number' in st.session_state.ch_last_df.columns else 0
                            df_doc_count = len(st.session_state.ch_last_df)

                            if df_company_count > 0 or df_doc_count > 0:
                                narrative_parts.append(f"The loaded JSON data provides details for **{df_company_count}** company/ies and **{df_doc_count}** document(s).")
                            else:
                                narrative_parts.append("The loaded JSON data was empty or did not contain countable company/document details.")
                            
                            # Existing individual summary logic (originally lines 960-964)
                            actual_individual_summaries_count = sum(1 for s_inj in st.session_state.ch_analysis_summaries_for_injection if s_inj[2] != 'Individual summary not available in JSON.')
                            if actual_individual_summaries_count > 0:
                                narrative_parts.append(f"Found **{actual_individual_summaries_count}** individual document summaries in JSON for review/injection.")
                            else:
                                narrative_parts.append("Individual document summaries were not found or were placeholders in the JSON data. This needs to be fixed in ch_pipeline.py if individual summaries are expected.")
                        
                        else: # This 'else' corresponds to 'if json_processed_successfully and ...' (i.e., JSON failed or df empty)
                            # This replaces the original else block's narrative appends (lines 966-968 in the provided snippet)
                            if not json_processed_successfully: 
                                narrative_parts.append("Detailed information from the JSON output was not available or failed to load. Any counts mentioned above are based on overall pipeline metrics only.")
                            elif st.session_state.ch_last_df is None or st.session_state.ch_last_df.empty: 
                                narrative_parts.append("The JSON data, even if loaded, appeared empty or did not yield document details. Any counts mentioned above are based on overall pipeline metrics only.")

                        # Attempt to read CONSOLIDATED summary from the digest CSV
                        consolidated_summary_from_csv = None
                        if output_csv_path and _pl.Path(output_csv_path).exists():
                            try:
                                df_csv_digest = pd.read_csv(output_csv_path)
                                if not df_csv_digest.empty and 'Summary of Findings' in df_csv_digest.columns:
                                    consolidated_summary_from_csv = df_csv_digest['Summary of Findings'].iloc[0]
                                    if pd.notna(consolidated_summary_from_csv) and consolidated_summary_from_csv.strip():
                                        narrative_parts.append("A consolidated 'Summary of Findings' was successfully extracted from the CSV digest.")
                                    else:
                                        consolidated_summary_from_csv = None 
                                        narrative_parts.append("CSV digest was read, but 'Summary of Findings' was empty or not found.")
                                elif not df_csv_digest.empty:
                                    narrative_parts.append("CSV digest was read, but the 'Summary of Findings' column was missing.")
                                else:
                                    narrative_parts.append("CSV digest file was empty.")
                            except KeyError as e_csv_key:
                                narrative_parts.append(f"Could not read 'Summary of Findings' from CSV digest due to missing column: {e_csv_key}. The CSV ({_pl.Path(output_csv_path).name}) might be malformed or not the expected digest format.")
                                if 'logger' in globals() and hasattr(logger, 'warning'): logger.warning(f"KeyError reading CSV digest {output_csv_path}: {e_csv_key}")
                            except Exception as e_csv:
                                narrative_parts.append(f"Failed to read or process the CSV digest at {output_csv_path}: {e_csv}")
                                if 'logger' in globals() and hasattr(logger, 'error'): logger.error(f"Error reading CSV digest {output_csv_path}: {e_csv}", exc_info=True)
                        elif output_csv_path:
                            narrative_parts.append(f"CSV digest file specified by pipeline but not found at {output_csv_path}.")
                        
                        cost_str = ""
                        if "total_cost_gbp" in batch_metrics and isinstance(batch_metrics["total_cost_gbp"], (float, int)):
                            cost_str = f"£{batch_metrics['total_cost_gbp']:.4f}"
                        elif "total_cost_usd" in batch_metrics:
                            cost_usd_val = batch_metrics['total_cost_usd']
                            if isinstance(cost_usd_val, str):
                                try: cost_usd_val = float(cost_usd_val.replace('$', ''))
                                except ValueError: cost_usd_val = None
                            if isinstance(cost_usd_val, (float, int)): cost_str = f"${cost_usd_val:.2f} (USD)"
                        if cost_str: narrative_parts.append(f"Estimated processing cost: {cost_str}.")
                        
                        st.session_state.ch_last_narrative = " ".join(narrative_parts)
                        
                        if consolidated_summary_from_csv:
                            # This will be displayed in the "Analysis Results" section by existing code
                            # if st.session_state.get('ch_last_narrative') includes it,
                            # or we can add a specific st.markdown for it here if preferred.
                            # For now, it's part of the narrative string.
                            # To display it separately and more prominently:
                            st.session_state.ch_consolidated_summary = consolidated_summary_from_csv
                        else:
                            st.session_state.ch_consolidated_summary = None

                        st.success("Companies House analysis processing complete.")
                except Exception as e_batch: # Corrected indentation to 16 spaces
                    st.error(f"An error occurred during the Companies House analysis: {e_batch}") # Corrected indentation to 20 spaces
                    if 'logger' in globals() and hasattr(logger, 'error'): # Corrected indentation to 20 spaces
                        logger.error(f"Error during CH analysis batch processing: {e_batch}", exc_info=True) # Corrected indentation to 24 spaces
                    # Update session state to reflect failure
                    st.session_state.ch_last_narrative = "Analysis failed. Please check error messages and logs." # Corrected indentation to 20 spaces
                    st.session_state.ch_last_df = None # Corrected indentation to 20 spaces
                    st.session_state.ch_last_digest_path = None # Corrected indentation to 20 spaces
                    st.session_state.ch_last_batch_metrics = None # Corrected indentation to 20 spaces
    # <<< END OF THE 'if st.button("📊 Run Analysis on Selected Documents", ...)' block's logic >>>

    st.markdown("---")
    st.markdown("#### Analysis Results")
    # Check if DataFrame is None or empty
    df_is_empty = st.session_state.get('ch_last_df') is None or st.session_state.get('ch_last_df', pd.DataFrame()).empty # Added default to empty df
    
    # Display Narrative
    if st.session_state.get('ch_last_narrative'):
        st.markdown("##### Narrative Summary & Key Findings")
        st.markdown(st.session_state.ch_last_narrative, unsafe_allow_html=True)
    elif df_is_empty : # Only show "Run analysis" if no narrative AND no df
        st.info("Run an analysis to see results here.")

    # Display Consolidated Summary if available from CSV
    if st.session_state.get('ch_consolidated_summary'):
        st.markdown("---")
        st.markdown("##### Consolidated Summary of Findings (from CSV Digest)")
        st.markdown(st.session_state.ch_consolidated_summary, unsafe_allow_html=True) # Assuming it might contain markdown

    # Display DOCX Download Button
    if st.session_state.get('ch_last_digest_path'):
        try:
            digest_path_obj = _pl.Path(st.session_state.ch_last_digest_path)
            if digest_path_obj.exists():
                with open(digest_path_obj, "rb") as fp:
                    st.download_button(
                        label="📥 Download Full CH Report (DOCX)", data=fp,
                        file_name=digest_path_obj.name,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
            else:
                st.warning(f"Report file (DOCX) not found at: {digest_path_obj}")
        except Exception as e_dl:
            st.warning(f"Could not prepare CH report (DOCX) for download: {e_dl}")
            if 'logger' in globals() and hasattr(logger, 'error'):
                logger.error(f"Error preparing CH report (DOCX) for download {st.session_state.get('ch_last_digest_path')}: {e_dl}", exc_info=True)
    
    # Display Detailed Document Information Table (from JSON)
    if st.session_state.get('ch_last_df') is not None and not st.session_state.ch_last_df.empty:
        st.markdown("---")
        st.markdown("##### Detailed Document Information (from JSON)")
        st.dataframe(st.session_state.ch_last_df)
    elif not df_is_empty : # If df is not empty but previous check failed, means it's None
        st.info("Detailed document information from JSON is not available or failed to load.")


    if st.session_state.get('ch_last_batch_metrics'):
        st.markdown("##### Processing Metrics")
        metrics_data = st.session_state.ch_last_batch_metrics
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Companies Processed", metrics_data.get("total_companies_processed", 0))
        m_col2.metric("Documents Analyzed", metrics_data.get("total_documents_analyzed", 0))
        
        total_cost_raw = metrics_data.get('total_cost_usd', 0.0)
        if isinstance(total_cost_raw, str):
            try: total_cost_raw = float(total_cost_raw.replace('$', ''))
            except ValueError: total_cost_raw = 0.0
        
        cost_display = f"£{total_cost_raw * 0.80:.4f}" 
        if "total_cost_gbp" in metrics_data:
             cost_display = f"£{metrics_data.get('total_cost_gbp', 0.0)::.4f}"
        m_col3.metric("Est. Cost", cost_display)

# --- START: REVISED TAB FOR GROUP STRUCTURE VISUALIZATION ---
with tab_group_structure:
    if 'GROUP_STRUCTURE_AVAILABLE' in globals() and GROUP_STRUCTURE_AVAILABLE:
        # --- OCR Handler Configuration for Group Structure Tab ---
        ocr_handler_for_group_tab = None  # Default to no OCR handler

        can_use_textract_generally = ch_pipeline.TEXTRACT_AVAILABLE and \
                                     hasattr(ch_pipeline, 'perform_textract_ocr') and \
                                     hasattr(ch_pipeline, 'initialize_textract_aws_clients')

        if can_use_textract_generally:
            if "group_structure_use_textract_checkbox" not in st.session_state:
                st.session_state.group_structure_use_textract_checkbox = False

            use_textract_gs = st.checkbox(
                "🔬 Use AWS Textract for PDF OCR in Group Structure Analysis",
                value=st.session_state.group_structure_use_textract_checkbox,
                key="group_structure_use_textract_checkbox_widget",
                on_change=lambda: st.session_state.update(group_structure_use_textract_checkbox=st.session_state.group_structure_use_textract_checkbox_widget),
                help="If checked, Textract will be used for OCR on PDF documents encountered during group structure analysis. This may incur AWS costs."
            )

            if use_textract_gs:
                if ch_pipeline.initialize_textract_aws_clients():
                    ocr_handler_for_group_tab = ch_pipeline.perform_textract_ocr
                else:
                    st.warning("AWS Textract was selected, but failed to initialize AWS clients. OCR will not be used for group structure analysis.")
        else:
            st.caption("AWS Textract for PDF OCR is not available or not fully configured in the system (ch_pipeline).")
        
        st.markdown("---")

        try:
            from group_structure_utils import render_group_structure_ui
            
            render_group_structure_ui(
                api_key=config.CH_API_KEY,
                base_scratch_dir=APP_BASE_PATH / "temp_group_structure_runs",
                logger=logger,
                ocr_handler=ocr_handler_for_group_tab
            )
        except ImportError as e_import_gs:
            st.error(f"Group Structure functionality could not be loaded due to an import error: {e_import_gs}. Please check the installation of 'group_structure_utils'.")
            if 'logger' in globals() and hasattr(logger, 'error'): # Check logger existence
                 logger.error(f"ImportError for group_structure_utils in tab: {e_import_gs}", exc_info=True)
        except Exception as e_render_gs:
            st.error(f"An error occurred while rendering or operating the Group Structure tab: {str(e_render_gs)}")
            if 'logger' in globals() and hasattr(logger, 'error'): # Check logger existence
                logger.error(f"Error in Group Structure tab (app.py level): {e_render_gs}", exc_info=True)
    else:
        st.error("Group Structure functionality is not available. The 'group_structure_utils' module might have failed to load.")
# --- END: REVISED TAB FOR GROUP STRUCTURE VISUALIZATION ---

with tab_about_rendered:
    render_about_page()

with tab_instructions:
    render_instructions_page()

# --- End of Main App Area UI (Using Tabs) ---