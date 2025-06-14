#!/usr/bin/env python3
"""Strategic Counsel v3.6 - Corrected logger argument for group structure UI

Key Changes:
- Fixed TypeError by passing 'logger_param' instead of 'logger' to render_group_structure_ui.
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
import datetime as _dt 
import pathlib as _pl 
import hashlib as _hashlib 
import json 
import io 
import logging 
logger = logging.getLogger(__name__)   
logger.setLevel(logging.INFO)          

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
import ai_utils 
from about_page import render_about_page
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

# Import pandas and docx at a higher level as they are fundamental if used
try:
    import pandas as pd
except ImportError:
    logger.error("Pandas library not found. Parts of the application may fail.")
    pd = None 

try:
    from docx import Document
except ImportError:
    logger.error("python-docx library not found. DOCX export will fail.")
    Document = None 


if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

APP_BASE_PATH = APP_ROOT_DIR
LOGO_PATH = APP_BASE_PATH / "static" / "logo" / "logo1.png"

LOG_DIR = APP_BASE_PATH / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE_PATH = LOG_DIR / f"app_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

f_handler = logging.FileHandler(LOG_FILE_PATH)
f_handler.setLevel(logging.INFO)
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
f_handler.setFormatter(log_format)

if not logger.handlers: 
    logger.addHandler(f_handler)

st.set_page_config(
    page_title="Strategic Counsel",
    page_icon=str(LOGO_PATH),
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': "# Strategic Counsel v3.6\nModular AI Legal Assistant Workspace."}
)

# Custom CSS (remains the same)
st.markdown("""
    <style>
        /* Base & Body */
        .stApp {
            background-color: #f0f2f5; 
        }
        body {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            color: #333333; 
        }
        .main .block-container {
            padding-top: 2rem; padding-bottom: 2rem;
            padding-left: 2rem; padding-right: 2rem;
        }
        .st-emotion-cache-16txtl3 { 
            background-color: #001f3f; 
            color: #ffffff;
        }
        .st-emotion-cache-16txtl3 h1, .st-emotion-cache-16txtl3 h2, .st-emotion-cache-16txtl3 h3, .st-emotion-cache-16txtl3 h4, .st-emotion-cache-16txtl3 h5, .st-emotion-cache-16txtl3 p, .st-emotion-cache-16txtl3 label {
            color: #ffffff !important;
        }
        .st-emotion-cache-16txtl3 .stSlider label, .st-emotion-cache-16txtl3 .stSelectbox label, .st-emotion-cache-16txtl3 .stMultiSelect label, .st-emotion-cache-16txtl3 .stCheckbox label {
             color: #ffffff !important;
        }
         .st-emotion-cache-16txtl3 .stButton>button {
            border: 1px solid #d4a017; background-color: transparent; color: #d4a017;
        }
        .st-emotion-cache-16txtl3 .stButton>button:hover {
            background-color: #d4a017; color: #001f3f;
        }
        .stTabs [data-baseweb="tab-list"] {
            background-color: #e1e8ed; border-radius: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 44px; background-color: transparent; color: #001f3f; font-weight: 600;
        }
        .stTabs [data-baseweb="tab--selected"] {
            background-color: #001f3f; color: #ffffff; border-radius: 8px 8px 0 0;
        }
        .stTabs [data-baseweb="tab-highlight"] {
            background-color: #d4a017; 
        }
        .stButton>button {
            border: 2px solid #001f3f; background-color: #001f3f; color: #ffffff; 
            padding: 0.5rem 1rem; border-radius: 5px; font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #003366; border-color: #003366; color: #ffffff;
        }
        .stButton>button:active {
            background-color: #002244 !important; border-color: #002244 !important;
        }
        .stButton[kind="primary"]>button, .stButton>button[kind="primary"] {
             background-color: #d4a017 !important; border-color: #d4a017 !important;
             color: #001f3f !important; 
        }
        .stButton[kind="primary"]>button:hover, .stButton>button[kind="primary"]:hover {
            background-color: #b8860b !important; border-color: #b8860b !important;
        }
        h1, h2, h3 { color: #001f3f; }
        .stExpander { border: 1px solid #e1e8ed; border-radius: 8px; }
        .stExpander header {
            background-color: #e1e8ed; color: #001f3f; font-weight: 600;
            border-radius: 8px 8px 0 0;
        }
        .stTextInput input, .stTextArea textarea {
            border: 1px solid #ced4da; border-radius: 4px; padding: 0.5rem;
        }
        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: #001f3f; box-shadow: 0 0 0 0.2rem rgba(0, 31, 63, 0.25);
        }
        .stMetric {
            background-color: #ffffff; border: 1px solid #e1e8ed;
            border-radius: 8px; padding: 1rem;
        }
        .stMetric label { color: #001f3f; }
        .stMetric value { color: #333333; }
        .stDataFrame { border: 1px solid #e1e8ed; border-radius: 8px; }
        .topic-display-box {
            background-color: hsl(var(--topic-hue, 210), 70%, 90%); 
            padding: 8px 12px; border-radius: 8px; margin: 8px 0; 
            text-align: center; color: #333;
        }
    </style>
""", unsafe_allow_html=True)

try:
    from app_utils import (
        summarise_with_title, fetch_url_content, find_company_number,
        extract_text_from_uploaded_file
    )
    from about_page import render_about_page
    from ch_pipeline import run_batch_company_analysis
    from ai_utils import get_improved_prompt, check_protocol_compliance 
except ImportError as e_app_utils_more:
    st.error(f"Fatal Error: Could not import app utilities or CH pipeline: {e_app_utils_more}")
    logger.error(f"ImportError from app_utils/about_page/ch_pipeline/ai_utils: {e_app_utils_more}", exc_info=True)
    st.stop() 

try:
    logger = config.logger 
    from ch_pipeline import TEXTRACT_AVAILABLE as CH_PIPELINE_TEXTRACT_FLAG_FROM_MODULE
    config.CH_PIPELINE_TEXTRACT_FLAG = CH_PIPELINE_TEXTRACT_FLAG_FROM_MODULE
except ImportError as e_initial_imports:
    st.error(f"Fatal Error: Could not import core modules (config, ch_pipeline): {e_initial_imports}")
    st.stop()
except Exception as e_conf:
    st.error(f"Fatal Error during config.py import or setup: {e_conf}")
    st.stop()

APP_BASE_PATH: _pl.Path = config.APP_BASE_PATH
OPENAI_API_KEY_PRESENT = bool(config.OPENAI_API_KEY and config.OPENAI_API_KEY.startswith("sk-"))
CH_API_KEY_PRESENT = bool(config.CH_API_KEY)
GEMINI_API_KEY_PRESENT = bool(config.GEMINI_API_KEY and config.genai) 

REQUIRED_DIRS_REL = ("memory", "memory/digests", "summaries", "exports", "logs", "static")
for rel_p in REQUIRED_DIRS_REL:
    abs_p = APP_BASE_PATH / rel_p
    try: abs_p.mkdir(parents=True, exist_ok=True)
    except OSError as e_mkdir: st.error(f"Fatal Error creating directory {abs_p.name}: {e_mkdir}"); st.stop()

MODEL_PRICES_PER_1K_TOKENS_GBP: Dict[str, float] = {
    "gpt-4o": 0.0040, "gpt-4-turbo": 0.0080, "gpt-3.5-turbo": 0.0004, "gpt-4o-mini": 0.00012,
    config.GEMINI_MODEL_DEFAULT: 0.0028, 
    "gemini-1.5-pro-latest": 0.0028, 
    "gemini-1.5-flash-latest": 0.00028,
    config.GEMINI_2_5_PRO_MODEL: 0.0050, 
}
MODEL_ENERGY_WH_PER_1K_TOKENS: Dict[str, float] = {
    "gpt-4o": 0.15, "gpt-4-turbo": 0.4, "gpt-3.5-turbo": 0.04, "gpt-4o-mini": 0.02,
    config.GEMINI_MODEL_DEFAULT: 0.2, 
    "gemini-1.5-pro-latest": 0.2, 
    "gemini-1.5-flash-latest": 0.05,
    config.GEMINI_2_5_PRO_MODEL: 0.25, 
}
KETTLE_WH: int = 360

PROTO_PATH = APP_BASE_PATH / "strategic_protocols.txt"
PROTO_TEXT: str
PROTO_HASH = ""
PROTO_LOAD_SUCCESS = False 

if not PROTO_PATH.exists():
    PROTO_TEXT = config.PROTO_TEXT_FALLBACK
    logger.warning(f"Protocol file {PROTO_PATH.name} not found. Using fallback.")
    config.LOADED_PROTO_PATH_NAME = PROTO_PATH.name 
    config.LOADED_PROTO_TEXT = PROTO_TEXT 
    PROTO_LOAD_SUCCESS = False
else:
    try:
        PROTO_TEXT = PROTO_PATH.read_text(encoding="utf-8")
        PROTO_HASH = _hashlib.sha256(PROTO_TEXT.encode()).hexdigest()[:8]
        config.PROTO_TEXT_FALLBACK = PROTO_TEXT 
        config.LOADED_PROTO_PATH_NAME = PROTO_PATH.name 
        config.LOADED_PROTO_TEXT = PROTO_TEXT 
        logger.info(f"Successfully loaded protocol from {PROTO_PATH.name}")
        PROTO_LOAD_SUCCESS = True
    except Exception as e_proto:
        PROTO_TEXT = config.PROTO_TEXT_FALLBACK
        logger.error(f"Error loading protocol file {PROTO_PATH.name}: {e_proto}. Using fallback.", exc_info=True)
        config.LOADED_PROTO_PATH_NAME = PROTO_PATH.name 
        config.LOADED_PROTO_TEXT = PROTO_TEXT 
        PROTO_LOAD_SUCCESS = False


CH_CATEGORIES: Dict[str, str] = {
    "Accounts": "accounts", "Confirmation Stmt": "confirmation-statement", "Officers": "officers",
    "Capital": "capital", "Charges": "mortgage", "Insolvency": "insolvency",
    "PSC": "persons-with-significant-control", "Name Change": "change-of-name",
    "Reg. Office": "registered-office-address",
}

def init_session_state():
    """Initialize Streamlit session state with default values"""
    defaults = {
        "current_topic": "general_default_topic",
        "session_history": [],
        "loaded_memories": [],
        "processed_summaries": [],
        "selected_summary_texts": [],
        "latest_digest_content": "",
        "document_processing_complete": True,
        "ch_last_digest_path": None,
        "ch_last_df": None,
        "ch_last_narrative": None,
        "ch_last_batch_metrics": {},
        "consult_digest_model": config.OPENAI_MODEL_DEFAULT if 'config' in globals() and hasattr(config, 'OPENAI_MODEL_DEFAULT') else "gpt-4",
        "ch_analysis_summaries_for_injection": [],
        "ocr_method": "aws",
        "ocr_method_radio": 0,
        "user_instruction_main_text_area_value": "",
        "original_user_instruction_main": "",
        "user_instruction_main_is_improved": False,
        "additional_ai_instructions_ch_text_area_value": "",
        "original_additional_ai_instructions_ch": "",
        "additional_ai_instructions_ch_is_improved": False,
        "ch_available_documents": [],
        "ch_document_selection": {},
        "ch_start_year_input_main": _dt.date.today().year - 4,
        "ch_end_year_input_main": _dt.date.today().year,
        "group_structure_cn_for_analysis": "",
        "group_structure_report": [],
        "group_structure_viz_data": None,
        "suggested_parent_cn_for_rerun": None,
        "group_structure_parent_timeline": [],
        "latest_ai_response_for_protocol_check": None,
        "ch_company_profiles_map": {},
    }

    # Initialize session state with defaults if not present
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Ensure OCR method and radio index are in sync
    ocr_method_map = {"aws": 0, "google": 1, "none": 2}
    current_method = st.session_state.get("ocr_method", "aws")
    
    # Validate and correct OCR method if needed
    if current_method not in ocr_method_map:
        current_method = "aws"
        st.session_state.ocr_method = current_method
    
    # Ensure radio index matches the method
    st.session_state.ocr_method_radio = ocr_method_map[current_method]

def main():
    init_session_state()
    with st.sidebar:
        st.image(str(LOGO_PATH), use_column_width=False)
        st.markdown("## Configuration")
        current_topic_input = st.text_input("Matter / Topic ID", st.session_state.current_topic, key="topic_input_sidebar")
        if current_topic_input != st.session_state.current_topic:
            st.session_state.current_topic = current_topic_input
            st.session_state.session_history = []
            st.session_state.processed_summaries = []
            st.session_state.selected_summary_texts = []
            st.session_state.loaded_memories = []
            st.session_state.latest_digest_content = ""
            st.session_state.ch_last_digest_path = None
            st.session_state.ch_last_df = None
            st.session_state.ch_last_narrative = None
            st.session_state.ch_last_batch_metrics = {}
            st.session_state.ch_analysis_summaries_for_injection = [] 
            st.session_state.group_structure_cn_for_analysis = ""
            st.session_state.group_structure_report = []
            st.session_state.group_structure_viz_data = None
            st.session_state.suggested_parent_cn_for_rerun = None
            st.session_state.group_structure_parent_timeline = []
            st.session_state.latest_ai_response_for_protocol_check = None
            st.session_state.ch_company_profiles_map = {}
            st.rerun()

        def _topic_color_style(topic_str: str) -> str:
            color_hue = int(_hashlib.sha1(topic_str.encode()).hexdigest(), 16) % 360
            return f"--topic-hue:{color_hue};"

        st.markdown(f'''<div class="topic-display-box" style="{_topic_color_style(st.session_state.current_topic)}">
                        Topic: <strong>{st.session_state.current_topic}</strong>
                     </div>''', unsafe_allow_html=True)

        st.markdown("---"); st.markdown("### System Status")
        protocol_status_message = ""
        protocol_status_type = "info" 

        if PROTO_LOAD_SUCCESS :
            protocol_status_message = f"Protocol '{PROTO_PATH.name}' loaded (Hash: {PROTO_HASH})."
            protocol_status_type = "success"
        elif not PROTO_PATH.exists(): 
            protocol_status_message = f"Protocol file '{PROTO_PATH.name}' not found. Using default protocol."
            protocol_status_type = "warning"
        else: 
            protocol_status_message = f"Error loading protocol '{PROTO_PATH.name}'. Using default protocol."
            protocol_status_type = "error"
        
        if protocol_status_type == "success": st.success(protocol_status_message)
        elif protocol_status_type == "warning": st.warning(protocol_status_message)
        else: st.error(protocol_status_message)

        with st.expander("AI Model Selection", expanded=False):
            if not OPENAI_API_KEY_PRESENT:
                st.error("‼️ OpenAI API Key missing. OpenAI models will fail.")
            if not GEMINI_API_KEY_PRESENT:
                st.warning("⚠️ Gemini API Key missing. Gemini models unavailable for consultation.")

        all_available_models_from_config = list(MODEL_PRICES_PER_1K_TOKENS_GBP.keys())
        gpt_models_selectable = [m for m in all_available_models_from_config if m.startswith("gpt-") and OPENAI_API_KEY_PRESENT]
        
        gemini_models_selectable = [
            m for m in all_available_models_from_config 
            if m.startswith("gemini-") and GEMINI_API_KEY_PRESENT
        ]
        if config.GEMINI_2_5_PRO_MODEL not in gemini_models_selectable and \
           config.GEMINI_2_5_PRO_MODEL in all_available_models_from_config and \
           GEMINI_API_KEY_PRESENT:
            gemini_models_selectable.append(config.GEMINI_2_5_PRO_MODEL)
        
        gemini_models_selectable = sorted(list(set(gemini_models_selectable)))

        selectable_models_consult = gpt_models_selectable + gemini_models_selectable
        if not selectable_models_consult: st.error("No AI models available for Consultation/Digests!");

        default_consult_model_index = 0
        if "consult_digest_model" in st.session_state and \
           st.session_state.consult_digest_model in selectable_models_consult:
            try: default_consult_model_index = selectable_models_consult.index(st.session_state.consult_digest_model)
            except ValueError: default_consult_model_index = 0 
        elif selectable_models_consult: 
            st.session_state.consult_digest_model = selectable_models_consult[0]
        else: 
            st.session_state.consult_digest_model = None

        st.session_state.consult_digest_model = st.selectbox(
            "Model for Consultation & Digests:", selectable_models_consult,
            index=default_consult_model_index,
            key="consult_digest_model_selector_main",
            disabled=not selectable_models_consult
        )
        if st.session_state.consult_digest_model:
            price_consult = MODEL_PRICES_PER_1K_TOKENS_GBP.get(st.session_state.consult_digest_model, 0.0)
            st.caption(f"Est. Cost/1k Tokens: £{price_consult:.5f}")
        else:
            st.caption("Est. Cost/1k Tokens: N/A")

        ai_temp = st.slider(
            "AI Creativity (Temperature)",
            0.0,
            1.0,
            0.2,
            0.05,
            key="ai_temp_slider_sidebar",
        )

        with st.expander("Context Injection", expanded=False):
            # Memory file handling
            memory_file_path = APP_BASE_PATH / "memory" / f"{st.session_state.current_topic}.json"
            loaded_memories_from_file: List[str] = []
            if memory_file_path.exists():
                try:
                    mem_data = json.loads(memory_file_path.read_text(encoding="utf-8"))
                    if isinstance(mem_data, list):
                        loaded_memories_from_file = [str(item) for item in mem_data if isinstance(item, str)]
                except Exception as e_mem_load:
                    st.warning(f"Could not load memory file {memory_file_path.name}: {e_mem_load}")
            
            # Memory selection widget with unique key
            selected_mem_snippets = st.multiselect(
                "Inject Memories",
                loaded_memories_from_file,
                default=[mem for mem in st.session_state.loaded_memories if mem in loaded_memories_from_file],
                key="mem_multiselect_sidebar_context"
            )
            st.session_state.loaded_memories = selected_mem_snippets

            # Digest file handling
            digest_file_path = APP_BASE_PATH / "memory" / "digests" / f"{st.session_state.current_topic}.md"
            if digest_file_path.exists():
                try: 
                    st.session_state.latest_digest_content = digest_file_path.read_text(encoding="utf-8")
                except Exception as e_digest_load: 
                    st.warning(f"Could not load digest {digest_file_path.name}: {e_digest_load}")
                    st.session_state.latest_digest_content = ""
            else: 
                st.session_state.latest_digest_content = ""
            
            # Digest injection checkbox with unique key
            inject_digest_checkbox = st.checkbox(
                "Inject Digest", 
                value=bool(st.session_state.latest_digest_content), 
                key="inject_digest_checkbox_context", 
                disabled=not bool(st.session_state.latest_digest_content)
            )

        st.checkbox(
            "Summarise uploads",
            value=st.session_state.get("summarise_uploads", True),
            key="summarise_uploads",
        )
        summary_level_option = st.slider(
            "Summary detail level",
            1,
            3,
            value=st.session_state.get("summary_detail_level", 2),
            format="%d",
            key="summary_detail_level",
        )
        # Avoid setting session state after widget creation to prevent
        # StreamlitAPIException. The slider already updates the session
        # state value when interacted with.
        if "summary_detail_level" not in st.session_state:
            st.session_state.summary_detail_level = summary_level_option

        # Update the OCR method radio button section
        ocr_options = ["AWS Textract", "Google Drive", "None"]
        ocr_method_map = {"aws": 0, "google": 1, "none": 2}

        # Get current OCR method from session state
        current_ocr_method = st.session_state.get("ocr_method", "aws")
        if current_ocr_method not in ocr_method_map:
            current_ocr_method = "aws"
            st.session_state.ocr_method = current_ocr_method

        # Get the index for the current method
        ocr_index = ocr_method_map[current_ocr_method]

        # Update the radio button
        selected_index = st.radio(
            "OCR method",
            ocr_options,
            index=ocr_index,
            key="ocr_method_radio",
            on_change=lambda: st.session_state.update(
                ocr_method={0: "aws", 1: "google", 2: "none"}[st.session_state.ocr_method_radio]
            )
        )

        # Ensure OCR method and radio index stay in sync
        if st.session_state.ocr_method_radio != ocr_index:
            st.session_state.ocr_method = {0: "aws", 1: "google", 2: "none"}[st.session_state.ocr_method_radio]

        newly_processed_summaries_for_this_run_sidebar: List[Tuple[str, str, str]] = [] 
        if sources_needing_processing and st.session_state.document_processing_complete:
            st.session_state.document_processing_complete = False 
            summaries_cache_dir_for_topic = APP_BASE_PATH / "summaries" / st.session_state.current_topic
            summaries_cache_dir_for_topic.mkdir(parents=True, exist_ok=True)

            with st.spinner(f"Processing {len(sources_needing_processing)} new document(s)/URL(s)..."):
                progress_bar_docs = st.progress(0.0)
                for idx, src_id in enumerate(list(sources_needing_processing)): 
                    title, summary = "Error", "Processing Failed"
                    cache_file_name = f"summary_{_hashlib.sha256(src_id.encode()).hexdigest()[:16]}.json"
                    summary_cache_file = summaries_cache_dir_for_topic / cache_file_name

                    if summary_cache_file.exists():
                        try:
                            cached_data = json.loads(summary_cache_file.read_text(encoding="utf-8"))
                            title, summary = cached_data.get("t", "Cache Title Error"), cached_data.get("s", "Cache Summary Error")
                        except Exception: title, summary = "Error", "Processing Failed (Cache Read)" 

                    if title == "Error" or "Cache" in title : 
                        raw_content, error_msg = None, None
                        if src_id in {f.name for f in uploaded_docs_list}: 
                            file_obj = next((f for f in uploaded_docs_list if f.name == src_id), None)
                            if file_obj: 
                                if callable(extract_text_from_uploaded_file):
                                    raw_content, error_msg = extract_text_from_uploaded_file(io.BytesIO(file_obj.getvalue()), src_id)
                                else:
                                    error_msg = "Text extraction utility not available."
                                    logger.error("extract_text_from_uploaded_file is not callable.")
                        elif src_id in urls_to_process: 
                            if callable(fetch_url_content):
                                raw_content, error_msg = fetch_url_content(src_id)
                            else:
                                error_msg = "URL fetching utility not available."
                                logger.error("fetch_url_content is not callable.")


                        if error_msg: title, summary = f"Error: {src_id[:40]}...", error_msg
                        elif not raw_content or not raw_content.strip(): title, summary = f"Empty: {src_id[:40]}...", "No text content found or extracted."
                        else: 
                            if callable(summarise_with_title):
                                title, summary = summarise_with_title(raw_content, "gpt-4o-mini", st.session_state.current_topic) 
                            else:
                                title, summary = "Error", "Summarization utility not available."
                                logger.error("summarise_with_title is not callable.")


                        if "Error" not in title and "Empty" not in title: 
                            try: summary_cache_file.write_text(json.dumps({"t":title,"s":summary,"src":src_id}),encoding="utf-8")
                            except Exception as e_c: logger.warning(f"Cache write failed for {src_id}: {e_c}")

                    newly_processed_summaries_for_this_run_sidebar.append((src_id, title, summary))
                    progress_bar_docs.progress((idx + 1) / len(sources_needing_processing))

                existing_to_keep = [s for s in st.session_state.processed_summaries if s[0] in current_source_identifiers and s[0] not in sources_needing_processing]
                st.session_state.processed_summaries = existing_to_keep + newly_processed_summaries_for_this_run_sidebar
                progress_bar_docs.empty()
            st.session_state.document_processing_complete = True; st.rerun() 

        st.session_state.selected_summary_texts = [] 
        if st.session_state.processed_summaries:
            st.markdown("---"); st.markdown("### Available Doc/URL Summaries (Select to Inject)")
            for idx, (s_id, title, summary_text) in enumerate(st.session_state.processed_summaries):
                checkbox_key = f"sum_sel_{_hashlib.md5(s_id.encode()).hexdigest()}"
                is_newly_processed = any(s_id == item[0] for item in newly_processed_summaries_for_this_run_sidebar)
                default_checked = is_newly_processed or st.session_state.get(checkbox_key, False)
                is_injected = st.checkbox(f"{idx+1}. {title[:40]}...", value=default_checked, key=checkbox_key, help=f"Source: {s_id}\nSummary: {summary_text[:200]}...")
                if is_injected: st.session_state.selected_summary_texts.append(f"UPLOADED DOCUMENT/URL SUMMARY ('{title}'):\n{summary_text}")

        selected_ch_summary_texts_for_injection_temp = [] 
        if st.session_state.ch_analysis_summaries_for_injection:
            st.markdown("---"); st.markdown("### CH Analysis Summaries (Select to Inject)")
            for idx, (company_id, title_for_list, summary_text) in enumerate(st.session_state.ch_analysis_summaries_for_injection):
                ch_checkbox_key = f"ch_sum_sel_{_hashlib.md5(company_id.encode() + title_for_list.encode()).hexdigest()}"
                is_ch_summary_injected = st.checkbox(f"{idx+1}. CH: {title_for_list[:40]}...", value=st.session_state.get(ch_checkbox_key, False), key=ch_checkbox_key, help=f"Company: {company_id}\nSummary: {summary_text[:200]}...")
                if is_ch_summary_injected:
                    selected_ch_summary_texts_for_injection_temp.append(f"COMPANIES HOUSE ANALYSIS SUMMARY FOR {company_id} ({title_for_list}):\n{summary_text}")

        st.markdown("---")
        st.markdown("#### Step 1: Find Available Company Documents")

        ch_company_numbers_list = []
        if ch_company_numbers_input: 
            raw_list = [num.strip() for num in ch_company_numbers_input.replace(',', '\\n').splitlines() if num.strip()]
            ch_company_numbers_list = list(dict.fromkeys(raw_list))

        if st.button("🔍 Search for Available Documents", key="ch_search_documents_button"):
            if not ch_company_numbers_list:
                st.warning("Please enter at least one company number.")
            elif st.session_state.ch_start_year_input_main > st.session_state.ch_end_year_input_main:
                st.warning("Start Year cannot be after End Year.")
            elif not CH_API_KEY_PRESENT:
                 st.error("Companies House API Key is not configured. Cannot search for documents.")
            elif 'ch_pipeline' not in globals() or not hasattr(ch_pipeline, 'get_relevant_filings_metadata'): 
                st.error("CH Search function is not available. Please check ch_pipeline.py.")
            else:
                with st.spinner("Searching for available documents... This may take a moment."):
                    try:
                        all_docs_for_all_companies = []
                        all_profiles_map = {} 
                        any_meta_error = None

                        for company_no_iter in ch_company_numbers_list:
                            if not config.CH_API_KEY: 
                                st.error(f"CH API Key missing for processing {company_no_iter}.")
                                any_meta_error = f"CH API Key missing for {company_no_iter}"
                                continue

                            years_back_calc = st.session_state.ch_end_year_input_main - st.session_state.ch_start_year_input_main
                            if years_back_calc < 0: years_back_calc = 0 

                            docs_for_this_co, profile_for_this_co, meta_err_this_co = ch_pipeline.get_relevant_filings_metadata(
                                company_number=company_no_iter,
                                api_key=config.CH_API_KEY, 
                                years_back=years_back_calc +1, 
                                categories_to_fetch=ch_selected_categories_api if ch_selected_categories_api else list(CH_CATEGORIES.values()),
                            )

                            if meta_err_this_co:
                                any_meta_error = meta_err_this_co 
                                st.warning(f"Error fetching metadata for {company_no_iter}: {meta_err_this_co}")
                            if docs_for_this_co:
                                for doc in docs_for_this_co: 
                                    doc['company_number'] = company_no_iter 
                                all_docs_for_all_companies.extend(docs_for_this_co)
                            if profile_for_this_co:
                                all_profiles_map[company_no_iter] = profile_for_this_co
                        
                        for doc in all_docs_for_all_companies:
                            doc['id'] = doc.get('transaction_id') or doc.get('links', {}).get('document_metadata') or \
                                        f"{doc.get('company_number','')}_{doc.get('date','')}_{doc.get('type','')}_{doc.get('description','N/A')[:10]}"
                        
                        st.session_state.ch_available_documents = all_docs_for_all_companies
                        st.session_state.ch_document_selection = {
                            doc['id']: True for doc in st.session_state.ch_available_documents
                        }
                        st.session_state.ch_company_profiles_map = all_profiles_map 

                        if any_meta_error:
                            st.warning(f"Some errors occurred during metadata retrieval. Check logs. First error: {any_meta_error}")
                        if not st.session_state.ch_available_documents and ch_company_numbers_list:
                            st.info("No documents found matching your criteria. Try adjusting categories or date range.")
                        elif st.session_state.ch_available_documents:
                            st.success(f"Found {len(st.session_state.ch_available_documents)} potentially relevant document(s). Please review and select below.")
                    except Exception as e_fetch_meta:
                        st.error(f"Error fetching CH document metadata: {e_fetch_meta}")
                        logger.error(f"Error in CH metadata fetch (app.py): {e_fetch_meta}", exc_info=True)
                    st.rerun()

        if st.session_state.ch_available_documents:
            st.markdown("---")
            st.markdown("#### Step 2: Select Documents for Analysis")
            st.caption("Review the list of documents found. Uncheck any you don't want to include in the analysis.")

            docs_by_company_display = {}
            for doc_detail in st.session_state.ch_available_documents:
                docs_by_company_display.setdefault(doc_detail['company_number'], []).append(doc_detail)

            for company_num_iter_display, docs_list_display in docs_by_company_display.items():
                company_name_display = company_num_iter_display 
                profile_data = st.session_state.ch_company_profiles_map.get(company_num_iter_display)
                if profile_data and profile_data.get("company_name"):
                    company_name_display = f"{profile_data.get('company_name')} ({company_num_iter_display})"

                with st.expander(f"Documents for {company_name_display} ({len(docs_list_display)} found)", expanded=True):
                    for doc_detail_display in docs_list_display:
                        display_name = (
                            f"{doc_detail_display.get('date', 'N/A')} | {doc_detail_display.get('type', 'N/A')} | "
                            f"{doc_detail_display.get('description', 'N/A')}"
                        )
                        doc_id_display = doc_detail_display['id']
                        checked_display = st.session_state.ch_document_selection.get(doc_id_display, True)
                        st.session_state.ch_document_selection[doc_id_display] = st.checkbox(
                            display_name, value=checked_display,
                            key=f"ch_doc_select_{doc_id_display.replace('/','_')}", 
                            help=(f"Company: {doc_detail_display.get('company_number', 'N/A')}\n"
                                  f"Type: {doc_detail_display.get('type', 'N/A')}\n"
                                  f"Date: {doc_detail_display.get('date', 'N/A')}\n"
                                  f"Description: {doc_detail_display.get('description', 'N/A')}\n"
                                  f"Transaction ID: {doc_detail_display.get('transaction_id', 'N/A')}")
                        )
            st.markdown("---")

        st.markdown("#### Step 3: Configure and Run Analysis")
        st.text_area(
            "Additional AI Instructions for Summaries (Optional):",
            value=st.session_state.additional_ai_instructions_ch_text_area_value,
            height=100, key="additional_ai_instructions_ch_text_area_main",
            on_change=lambda: st.session_state.update(additional_ai_instructions_ch_text_area_value=st.session_state.additional_ai_instructions_ch_text_area_main),
            help="e.g., 'Focus on financial risks and director changes.' This will be applied to each selected document's summary."
        )
        
        ch_keywords_for_filter_input = st.text_input(
            "Keywords to Highlight/Filter in Analysis (comma-separated, optional)",
            key="ch_keywords_input_main",
            help="These keywords can be used to guide the AI or highlight sections in the final report."
        )
        ch_keywords_for_filter = [kw.strip() for kw in ch_keywords_for_filter_input.split(',') if kw.strip()]

        is_any_doc_selected = any(st.session_state.ch_document_selection.values()) if st.session_state.ch_document_selection else False

        if st.button("📊 Run Analysis on Selected Documents", type="primary", key="ch_run_analysis_selected_button", disabled=not is_any_doc_selected):
            if not CH_API_KEY_PRESENT:
                st.error("Companies House API Key is not configured. Cannot run analysis.")
            elif not st.session_state.ch_document_selection: 
                st.warning("Please select at least one document for analysis.")
            elif not st.session_state.ch_available_documents: 
                st.warning("No available documents to analyze. Please search for documents first.")
            else:
                with st.spinner("Running Companies House analysis..."):
                    try:
                        company_numbers_with_selected_docs = list(set(
                            doc['company_number'] for doc_id, is_selected in st.session_state.ch_document_selection.items() if is_selected
                            for doc in st.session_state.ch_available_documents if doc['id'] == doc_id
                        ))

                        docs_to_process_by_company_map = {cn: [] for cn in company_numbers_with_selected_docs}
                        for doc_id, is_selected in st.session_state.ch_document_selection.items():
                            if is_selected:
                                selected_doc_detail = next((doc for doc in st.session_state.ch_available_documents if doc['id'] == doc_id), None)
                                if selected_doc_detail:
                                    company_num_of_doc = selected_doc_detail['company_number']
                                    if company_num_of_doc in docs_to_process_by_company_map:
                                         docs_to_process_by_company_map[company_num_of_doc].append(selected_doc_detail)
                        
                        docs_to_process_by_company_map = {k: v for k, v in docs_to_process_by_company_map.items() if v}

                        if not docs_to_process_by_company_map:
                            st.warning("No documents selected for analysis. Please select documents and try again.")
                        elif not config.CH_API_KEY: 
                            st.error("CH API Key missing. Cannot run batch analysis.")
                        else:
                            output_csv_path, batch_metrics = ch_pipeline.run_batch_company_analysis(
                                company_numbers_list=list(docs_to_process_by_company_map.keys()), 
                                selected_filings_metadata_by_company=docs_to_process_by_company_map,
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
                            
                            st.session_state.ch_last_digest_path = batch_metrics.get("output_docx_path")
                            st.session_state.ch_last_batch_metrics = batch_metrics
                            st.session_state.ch_last_df = None  
                            st.session_state.ch_analysis_summaries_for_injection = []  
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
                                        
                                        if isinstance(loaded_json_content, list): main_json_data_list = loaded_json_content
                                        elif isinstance(loaded_json_content, dict):
                                            if "processed_documents" in loaded_json_content and isinstance(loaded_json_content["processed_documents"], list): main_json_data_list = loaded_json_content["processed_documents"]
                                            elif "output_data_rows" in loaded_json_content and isinstance(loaded_json_content["output_data_rows"], list): main_json_data_list = loaded_json_content["output_data_rows"]
                                        
                                        if main_json_data_list and pd is not None: 
                                            json_processed_successfully = True
                                            logger.info(f"Successfully loaded and parsed main JSON ({main_json_path.name}) for individual document details.")
                                            st.session_state.ch_last_df = pd.DataFrame(main_json_data_list)
                                            
                                            temp_summaries_for_injection = []
                                            for item_idx, item in enumerate(main_json_data_list):
                                                if isinstance(item, dict):
                                                    company_id_item = str(item.get('company_number', f'Comp_N/A_{item_idx}'))
                                                    doc_date_item = item.get('document_date', item.get('date', 'N/A')) 
                                                    doc_type_item = item.get('document_type', item.get('type', 'N/A')) 
                                                    doc_desc_raw_item = item.get('document_description', item.get('description', 'Document')) 
                                                    doc_description_item = str(doc_desc_raw_item if pd.notna(doc_desc_raw_item) and str(doc_desc_raw_item).strip() else doc_type_item) if pd is not None else str(doc_desc_raw_item or doc_type_item)
                                                    title_item = f"{doc_date_item} - {doc_type_item} - {doc_description_item[:75]}"
                                                    if len(doc_description_item) > 75: title_item += "..."
                                                    
                                                    individual_summary_text_item = str(item.get('summary', 'Individual summary not available in JSON.'))
                                                    
                                                    status_item = item.get('processing_status', item.get('ai_summary_status', item.get('text_extraction_status'))) 
                                                    if status_item and str(status_item).lower() not in ["success", "completed", "ok", "processed", "summarized"] and "error" not in str(status_item).lower() and "fail" not in str(status_item).lower():
                                                        title_item += f" (Status: {status_item})"
                                                    elif "error" in str(status_item).lower() or "fail" in str(status_item).lower():
                                                        title_item += f" (Processing Issue)"

                                                    temp_summaries_for_injection.append((company_id_item, title_item, individual_summary_text_item))
                                            st.session_state.ch_analysis_summaries_for_injection = temp_summaries_for_injection
                                        elif pd is None:
                                            msg = "Pandas library is not available. Cannot process JSON data into DataFrame."
                                            logger.error(msg); st.error(msg)
                                        else: 
                                            msg = f"Main JSON ({main_json_path.name}) loaded but document list not found in expected structure."
                                            logger.warning(msg); st.warning(msg)
                                    except Exception as e_json_load:
                                        msg = f"Error processing main JSON output from {main_json_path_str}: {e_json_load}"
                                        logger.error(msg, exc_info=True); st.error(msg)
                                else: 
                                    msg = f"Main JSON output file specified by pipeline but not found: {main_json_path_str}"
                                    logger.warning(msg); st.warning(msg)
                            else: 
                                msg = "Pipeline did not provide a path for the main JSON output (for individual document details)."
                                logger.warning(msg); st.warning(msg)

                            narrative_parts = []
                            num_companies_processed = batch_metrics.get("total_companies_processed", 0)
                            num_docs_analyzed = batch_metrics.get("total_documents_analyzed", 0)

                            if num_companies_processed > 0 or num_docs_analyzed > 0: narrative_parts.append(f"Pipeline metrics indicate processing for **{num_companies_processed}** company/ies and **{num_docs_analyzed}** document(s).")
                            else: narrative_parts.append("Pipeline metrics did not report specific company or document counts from the overall analysis.")

                            if json_processed_successfully and st.session_state.ch_last_df is not None and not st.session_state.ch_last_df.empty:
                                df_company_count = st.session_state.ch_last_df['company_number'].nunique() if 'company_number' in st.session_state.ch_last_df.columns else 0
                                df_doc_count = len(st.session_state.ch_last_df)
                                if df_company_count > 0 or df_doc_count > 0: narrative_parts.append(f"The loaded JSON data provides details for **{df_company_count}** company/ies and **{df_doc_count}** document(s).")
                                else: narrative_parts.append("The loaded JSON data was empty or did not contain countable company/document details.")
                                
                                actual_individual_summaries_count = sum(1 for s_inj in st.session_state.ch_analysis_summaries_for_injection if s_inj[2] != 'Individual summary not available in JSON.')
                                if actual_individual_summaries_count > 0: narrative_parts.append(f"Found **{actual_individual_summaries_count}** individual document summaries in JSON for review/injection.")
                                else: narrative_parts.append("Individual document summaries were not found or were placeholders in the JSON data.")
                            else: 
                                if not json_processed_successfully: narrative_parts.append("Detailed information from the JSON output was not available or failed to load.")
                                elif st.session_state.ch_last_df is None or (pd is not None and st.session_state.ch_last_df.empty): narrative_parts.append("The JSON data, even if loaded, appeared empty or did not yield document details.")
                                elif pd is None: narrative_parts.append("Pandas not available, cannot determine if JSON data is empty.")


                            consolidated_summary_from_csv = None
                            if output_csv_path and _pl.Path(output_csv_path).exists() and pd is not None:
                                try:
                                    df_csv_digest = pd.read_csv(output_csv_path)
                                    if not df_csv_digest.empty and 'Summary of Findings' in df_csv_digest.columns:
                                        consolidated_summary_from_csv = df_csv_digest['Summary of Findings'].iloc[0]
                                        if pd.notna(consolidated_summary_from_csv) and consolidated_summary_from_csv.strip(): narrative_parts.append("A consolidated 'Summary of Findings' was successfully extracted from the CSV digest.")
                                        else: consolidated_summary_from_csv = None; narrative_parts.append("CSV digest was read, but 'Summary of Findings' was empty or not found.")
                                    elif not df_csv_digest.empty: narrative_parts.append("CSV digest was read, but the 'Summary of Findings' column was missing.")
                                    else: narrative_parts.append("CSV digest file was empty.")
                                except KeyError as e_csv_key:
                                    narrative_parts.append(f"Could not read 'Summary of Findings' from CSV digest due to missing column: {e_csv_key}."); logger.warning(f"KeyError reading CSV digest {output_csv_path}: {e_csv_key}")
                                except Exception as e_csv_read:
                                    narrative_parts.append(f"Failed to read or process the CSV digest at {output_csv_path}: {e_csv_read}"); logger.error(f"Error reading CSV digest {output_csv_path}: {e_csv_read}", exc_info=True)
                            elif output_csv_path and pd is None:
                                narrative_parts.append(f"Pandas not available. Cannot read CSV digest at {output_csv_path}.")
                            elif output_csv_path: narrative_parts.append(f"CSV digest file specified by pipeline but not found at {output_csv_path}.")
                            
                            cost_str_display = ""
                            total_ai_cost_gbp = batch_metrics.get("total_ai_summarization_cost_gbp")
                            if isinstance(total_ai_cost_gbp, (float, int)): cost_str_display = f"£{total_ai_cost_gbp:.4f} (AI Summaries)"
                            
                            total_textract_cost_gbp = batch_metrics.get("aws_textract_costs", {})
                            if isinstance(total_textract_cost_gbp, dict): 
                                total_textract_cost_gbp_metric_final = total_textract_cost_gbp.get("total_estimated_aws_cost_gbp_for_ocr")
                                if isinstance(total_textract_cost_gbp_metric_final, (float, int)):
                                    cost_str_display += f"{' + ' if cost_str_display else ''}£{total_textract_cost_gbp_metric_final:.4f} (Textract OCR)"
                            
                            if cost_str_display: narrative_parts.append(f"Estimated processing cost: {cost_str_display}.")
                            
                            st.session_state.ch_last_narrative = " ".join(narrative_parts)
                            st.session_state.ch_consolidated_summary = consolidated_summary_from_csv if consolidated_summary_from_csv else None
                            st.success("Companies House analysis processing complete.")
                    except Exception as e_batch_run: 
                        st.error(f"An error occurred during the Companies House analysis: {e_batch_run}") 
                        logger.error(f"Error during CH analysis batch processing: {e_batch_run}", exc_info=True) 
                        st.session_state.ch_last_narrative = "Analysis failed. Please check error messages and logs." 
                        st.session_state.ch_last_df = None 
                        st.session_state.ch_last_digest_path = None 
                        st.session_state.ch_last_batch_metrics = {}
                        st.session_state.ch_analysis_summaries_for_injection = []
                        st.session_state.ch_consolidated_summary = None


        st.markdown("---")
        st.markdown("#### Analysis Results")
        df_results_available_ch = st.session_state.get('ch_last_df') is not None and \
                               (pd is not None and not st.session_state.ch_last_df.empty)
        
        if st.session_state.get('ch_last_narrative'):
            st.markdown("##### Narrative Summary & Key Findings")
            st.markdown(st.session_state.ch_last_narrative, unsafe_allow_html=True)
        elif not df_results_available_ch: 
            st.info("Run an analysis to see results here.")

        if st.session_state.get('ch_consolidated_summary'):
            st.markdown("---")
            st.markdown("##### Consolidated Summary of Findings (from CSV Digest)")
            st.markdown(st.session_state.ch_consolidated_summary, unsafe_allow_html=True) 

        ch_last_digest_path_val_results = st.session_state.get('ch_last_digest_path')
        if ch_last_digest_path_val_results and isinstance(ch_last_digest_path_val_results, (str, _pl.Path)):
            try:
                digest_path_obj_results_display = _pl.Path(ch_last_digest_path_val_results)
                if digest_path_obj_results_display.exists():
                    with open(digest_path_obj_results_display, "rb") as fp_results_display:
                        st.download_button(
                            label="📥 Download Full CH Report (DOCX)", data=fp_results_display,
                            file_name=digest_path_obj_results_display.name,
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                else: st.warning(f"Report file (DOCX) not found at: {digest_path_obj_results_display}")
            except Exception as e_dl_results_display:
                st.warning(f"Could not prepare CH report (DOCX) for download: {e_dl_results_display}")
                logger.error(f"Error preparing CH report (DOCX) for download {ch_last_digest_path_val_results}: {e_dl_results_display}", exc_info=True)
        
        if df_results_available_ch:
            st.markdown("---")
            st.markdown("##### Detailed Document Information (from JSON)")
            cols_to_show = ['company_number', 'document_date', 'document_type', 'document_description', 'text_extraction_status', 'extracted_text_length', 'ai_summary_status', 'summary', 'ocr_pages_processed', 'processing_error']
            if st.session_state.ch_last_df is not None: # Check again before column access
                display_df_ch = st.session_state.ch_last_df[[col for col in cols_to_show if col in st.session_state.ch_last_df.columns]]
                st.dataframe(display_df_ch)
        elif st.session_state.get('ch_last_narrative'): 
            st.info("Detailed document information from JSON is not available or failed to load.")


        if st.session_state.get('ch_last_batch_metrics'):
            st.markdown("##### Processing Metrics")
            metrics_data_display_final = st.session_state.ch_last_batch_metrics
            m_col1_disp_final, m_col2_disp_final, m_col3_disp_final = st.columns(3)
            m_col1_disp_final.metric("Companies Processed", metrics_data_display_final.get("total_companies_processed", 0))
            m_col2_disp_final.metric("Documents Analyzed", metrics_data_display_final.get("total_documents_analyzed", 0))
            
            cost_display_final_metric = "N/A"
            if "total_ai_summarization_cost_gbp" in metrics_data_display_final:
                cost_display_final_metric = f"£{metrics_data_display_final.get('total_ai_summarization_cost_gbp', 0.0):.4f} (AI)"
            
            aws_costs_data_final = metrics_data_display_final.get("aws_textract_costs", {})
            if isinstance(aws_costs_data_final, dict): 
                total_textract_cost_gbp_metric_final = aws_costs_data_final.get("total_estimated_aws_cost_gbp_for_ocr")
                if isinstance(total_textract_cost_gbp_metric_final, (float, int)):
                    cost_display_final_metric += f"{' + ' if cost_display_final_metric != 'N/A' else ''}£{total_textract_cost_gbp_metric_final:.4f} (OCR)"
            
            m_col3_disp_final.metric("Est. Cost", cost_display_final_metric if cost_display_final_metric != "N/A" else "£0.0000")


    with tab_group_structure:
        if 'GROUP_STRUCTURE_AVAILABLE' in globals() and GROUP_STRUCTURE_AVAILABLE:
            ocr_handler_for_group_tab = None  
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
                    on_change=lambda: st.session_state.update(group_structure_use_textract_checkbox=st.session_state.group_structure_use_textract_checkbox),
                    help="If checked, Textract will be used for OCR on PDF documents. This may incur AWS costs."
                )

                if use_textract_gs:
                    if callable(getattr(ch_pipeline, 'initialize_textract_aws_clients', None)) and ch_pipeline.initialize_textract_aws_clients():
                        ocr_handler_for_group_tab = ch_pipeline.perform_textract_ocr
                    else:
                        st.warning("AWS Textract selected, but failed to initialize AWS clients. OCR will not be used.")
            else:
                st.caption("AWS Textract for PDF OCR is not available or not fully configured (ch_pipeline).")
            
            st.markdown("---")

            try:
                if callable(getattr(group_structure_utils, 'render_group_structure_ui', None)):
                    if not config.CH_API_KEY:
                         st.error("Companies House API Key is not configured. Group Structure Analysis cannot proceed.")
                    else:
                        # Pass logger_param as expected by the function definition
                        group_structure_utils.render_group_structure_ui(
                            api_key=config.CH_API_KEY, 
                            base_scratch_dir=APP_BASE_PATH / "temp_group_structure_runs",
                            logger_param=logger, # Corrected argument name
                            ocr_handler=ocr_handler_for_group_tab
                        )
                else:
                    st.error("Group Structure UI utility not available.")
                    logger.error("group_structure_utils.render_group_structure_ui is not callable.")
            except ImportError as e_import_gs_tab: 
                st.error(f"Group Structure UI could not be loaded: {e_import_gs_tab}.")
                logger.error(f"ImportError for group_structure_utils in tab: {e_import_gs_tab}", exc_info=True)
            except Exception as e_render_gs_tab:
                st.error(f"An error occurred in the Group Structure tab: {str(e_render_gs_tab)}")
                logger.error(f"Error in Group Structure tab (app.py level): {e_render_gs_tab}", exc_info=True)
        else:
            st.error("Group Structure functionality is not available ('group_structure_utils' module failed to load).")

    # --- END: REVISED TAB FOR GROUP STRUCTURE VISUALIZATION ---

    with tab_timeline:
        st.markdown("### Case Timeline")
        uploaded = st.file_uploader(
            "Upload docket file (CSV, JSON or PDF)",
            type=["csv", "json", "pdf"],
            key="case_timeline_uploader",
        )
        if uploaded is not None:
            tmp_path = APP_BASE_PATH / "_tmp_timeline_upload"
            tmp_path.write_bytes(uploaded.getvalue())
            try:
                events = timeline_utils.parse_docket_file(tmp_path)
                events_sorted = sorted(events, key=lambda e: e.get("date", ""))
                st.session_state.case_timeline_events = events_sorted
            except Exception as e_parse:
                st.error(f"Failed to parse docket file: {e_parse}")
                st.session_state.case_timeline_events = []
            tmp_path.unlink(missing_ok=True)

        events = st.session_state.get("case_timeline_events", [])
        if events:
            if STREAMLIT_TIMELINE_AVAILABLE:
                tl_events = []
                for ev in events:
                    date_parts = ev.get("date", "").split("-")
                    start_date = {}
                    if len(date_parts) == 3:
                        start_date = {
                            "year": int(date_parts[0]),
                            "month": int(date_parts[1]),
                            "day": int(date_parts[2]),
                        }
                    tl_events.append({
                        "start_date": start_date,
                        "text": {"text": ev.get("description", "")},
                    })
                st_timeline({"events": tl_events}, height=500)
            else:
                st.table(events)
        else:
            st.info("Upload a docket file to display events.")

    with tab_about_rendered:
        render_about_page()

    with tab_instructions:
        render_instructions_page()

    # --- End of Main App Area UI (Using Tabs) ---

if __name__ == "__main__":
    main()
