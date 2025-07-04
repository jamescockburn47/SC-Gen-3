#!/usr/bin/env python3
"""Strategic Counsel v4.0 - Enhanced LawMA Legal Specialist Integration

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
import os
from typing import List, Dict, Tuple, Optional, Any
import warnings

# Ensure the root directory is in sys.path for module imports
APP_ROOT_DIR = _pl.Path(__file__).resolve().parent
if str(APP_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(APP_ROOT_DIR))

# App-specific modules
import config 
import ch_pipeline
import app_utils 
import ai_utils 
# Legacy imports removed - now using consolidated help page

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
LOGO_PATH = APP_ROOT_DIR / "static" / "logo" / "logo1.png"

LOG_DIR = APP_BASE_PATH / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE_PATH = LOG_DIR / f"app_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

f_handler = logging.FileHandler(LOG_FILE_PATH)
f_handler.setLevel(logging.INFO)
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
f_handler.setFormatter(log_format)

if not logger.handlers: 
    logger.addHandler(f_handler)

# === RADICAL UI CUSTOMIZATION FOR TEXT VISIBILITY ===
st.set_page_config(
    page_title="Strategic Counsel Gen 4 - AI Legal Platform",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/jamescockburn47/SC-Gen-4',
        'Report a bug': "https://github.com/jamescockburn47/SC-Gen-4/issues",
        'About': "# Strategic Counsel Gen 4\nAI-powered legal analysis platform with LawMA integration"
    }
)

# Load Harcus Parker branded theme
try:
    css_file_path = APP_ROOT_DIR / "static" / "harcus_parker_style.css"
    with open(css_file_path, 'r') as f:
        css_content = f.read()
    st.markdown(f"""
    <style>
    {css_content}
    </style>
    """, unsafe_allow_html=True)
except FileNotFoundError:
    st.error(f"CSS file not found: {css_file_path}")
    # Fallback to basic dark theme
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a2332 100%);
        color: #f8f6f0;
    }
    .main {
        background: #1a2332;
        color: #f8f6f0;
    }
    </style>
    """, unsafe_allow_html=True)


# Harcus Parker branding enhancement with JavaScript
st.markdown("""
<script>
// Enhance Harcus Parker professional theme
document.addEventListener('DOMContentLoaded', function() {
    // Add professional law firm animations
    const elements = document.querySelectorAll('.stApp > div');
    elements.forEach((el, index) => {
        el.style.animation = `fadeIn 0.5s ease-out ${index * 0.1}s both`;
    });
    
    // Professional hover effects for buttons
    const buttons = document.querySelectorAll('button');
    buttons.forEach(button => {
        button.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
            this.style.transition = 'all 0.3s ease';
        });
        button.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
    
    // Add gold accent to focus elements
    const focusElements = document.querySelectorAll('input, textarea, select');
    focusElements.forEach(el => {
        el.addEventListener('focus', function() {
            this.style.boxShadow = '0 0 0 3px rgba(212, 175, 55, 0.3)';
        });
        el.addEventListener('blur', function() {
            this.style.boxShadow = '';
        });
    });
});

// Animation keyframes
const style = document.createElement('style');
style.textContent = `
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
`;
document.head.appendChild(style);
</script>
""", unsafe_allow_html=True)

try:
    from app_utils import (
        summarise_with_title, fetch_url_content, find_company_number,
        extract_text_from_uploaded_file
    )
    # Legacy about_page import removed
    from ch_pipeline import run_batch_company_analysis
    from ai_utils import get_improved_prompt, check_protocol_compliance, comprehensive_legal_consultation_with_protocols 
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

# Model pricing per 1k tokens in GBP (approximate)
MODEL_PRICES_PER_1K_TOKENS_GBP: Dict[str, float] = {
    "gpt-4.1": 0.0020, "gpt-4.1-mini": 0.0004, "gpt-4.1-nano": 0.0001,
    "gpt-4o": 0.0025, "gpt-4o-mini": 0.00015,
    "o3": 0.0020, "o3-pro": 0.0200, "o4-mini": 0.0011
}

# Model pricing for output per 1k tokens in GBP
MODEL_OUTPUT_PRICES_PER_1K_TOKENS_GBP: Dict[str, float] = {
    "gpt-4.1": 0.0080, "gpt-4.1-mini": 0.0016, "gpt-4.1-nano": 0.0014,
    "gpt-4o": 0.0100, "gpt-4o-mini": 0.0006,
    "o3": 0.0080, "o3-pro": 0.0800, "o4-mini": 0.0044
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

# Predefined topics for the AI consultation
PREDEFINED_TOPICS = {
    "Corporate Governance": "Corporate governance, director duties, and compliance",
    "Contract Law": "Contract interpretation, disputes, and negotiations", 
    "Employment Law": "Employment rights, dismissals, and workplace issues",
    "Commercial Law": "Commercial transactions, business disputes, and regulatory compliance",
    "Property Law": "Real estate transactions, leases, and property disputes",
    "Intellectual Property": "Patents, trademarks, copyrights, and IP disputes",
    "Data Protection": "GDPR compliance, data breaches, and privacy issues",
    "Regulatory Compliance": "Industry regulations, licensing, and compliance issues",
    "Dispute Resolution": "Litigation strategy, mediation, and arbitration",
    "General Legal Advice": "General legal questions and guidance"
}

def init_session_state():
    """Initialize Streamlit session state with default values"""
    
    # Clean up any corrupted widget states first
    corrupted_keys = []
    for key, value in st.session_state.items():
        if isinstance(key, str) and (key.endswith("_radio") or key.endswith("_selectbox")):
            if not isinstance(value, (int, str, type(None))):
                corrupted_keys.append(key)
    
    # Remove corrupted keys
    for key in corrupted_keys:
        del st.session_state[key]
        logger.warning(f"Removed corrupted session state key: {key}")
    
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
        "consult_digest_model": config.OPENAI_MODEL_DEFAULT if 'config' in globals() and hasattr(config, 'OPENAI_MODEL_DEFAULT') else "gpt-4.1",
        "ch_analysis_summaries_for_injection": [],
        "ocr_method": "local",
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
    ocr_method_map = {"local": 0, "none": 1}
    current_method = st.session_state.get("ocr_method", "local")
    
    # Validate and correct OCR method if needed
    if current_method not in ocr_method_map:
        current_method = "local"
        st.session_state.ocr_method = current_method
    
    # Ensure radio index matches the method and is valid
    expected_radio_index = ocr_method_map[current_method]
    current_radio_index = st.session_state.get("ocr_method_radio", expected_radio_index)
    
    # Validate radio index is within bounds (0-1 for 2 options)
    if not isinstance(current_radio_index, int) or current_radio_index < 0 or current_radio_index > 1:
        current_radio_index = expected_radio_index
    
    st.session_state.ocr_method_radio = current_radio_index

def main():
    init_session_state()
    with st.sidebar:
        st.image(str(LOGO_PATH), use_container_width=False)
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

        # Update the OCR method selection - use selectbox instead of radio for better stability
        ocr_options = ["Local OCR", "None"]
        ocr_method_map = {"local": 0, "none": 1}
        reverse_ocr_map = {0: "local", 1: "none"}

        # Get current OCR method from session state with extensive validation
        current_ocr_method = st.session_state.get("ocr_method", "local")
        if current_ocr_method not in ocr_method_map:
            current_ocr_method = "local"
            st.session_state.ocr_method = current_ocr_method

        # Get the index for the current method and ensure it's valid
        ocr_index = ocr_method_map[current_ocr_method]
        
        # Validate that the index is within bounds
        if ocr_index < 0 or ocr_index >= len(ocr_options):
            ocr_index = 0  # Default to first option if invalid
            st.session_state.ocr_method = "local"
            st.session_state.ocr_method_radio = 0

        # Clean up any corrupted state before creating widgets
        if "ocr_method_radio" in st.session_state:
            if not isinstance(st.session_state.ocr_method_radio, int) or \
               st.session_state.ocr_method_radio < 0 or \
               st.session_state.ocr_method_radio >= len(ocr_options):
                # Force reset corrupted state
                st.session_state.ocr_method_radio = ocr_index
                st.session_state.ocr_method = current_ocr_method

        # Use selectbox instead of radio widget for better stability
        try:
            selected_option = st.selectbox(
                "OCR method",
                ocr_options,
                index=ocr_index,
                key="ocr_method_selectbox"
            )
            
            # Update session state based on selection
            selected_index = ocr_options.index(selected_option) if selected_option in ocr_options else 0
            if selected_index in reverse_ocr_map:
                st.session_state.ocr_method = reverse_ocr_map[selected_index]
                st.session_state.ocr_method_radio = selected_index
            else:
                # Fallback if something goes wrong
                st.session_state.ocr_method = "local"
                st.session_state.ocr_method_radio = 0
                
        except Exception as e:
            # Ultimate fallback if widget creation fails
            logger.error(f"Error creating OCR method widget: {e}")
            st.error("Error with OCR method selector. Using default: Local OCR")
            st.session_state.ocr_method = "local"
            st.session_state.ocr_method_radio = 0

    # Clear workflow guidance banner
    st.markdown("""
    <div style="background: linear-gradient(90deg, #d4af37 0%, #b8941f 100%); padding: 20px; border-radius: 10px; margin-bottom: 30px;">
        <h2 style="color: #1a2332; margin: 0; text-align: center;">📚 Strategic Counsel Document Analysis</h2>
        <div style="text-align: center; margin-top: 15px;">
            <span style="background: rgba(26,35,50,0.1); padding: 8px 15px; border-radius: 20px; margin: 0 5px; color: #1a2332; font-weight: bold;">
                📁 1. Upload Documents
            </span>
            <span style="color: #1a2332; font-size: 18px; margin: 0 10px;">→</span>
            <span style="background: rgba(26,35,50,0.1); padding: 8px 15px; border-radius: 20px; margin: 0 5px; color: #1a2332; font-weight: bold;">
                🔍 2. Search & Analyze  
            </span>
            <span style="color: #1a2332; font-size: 18px; margin: 0 10px;">→</span>
            <span style="background: rgba(26,35,50,0.1); padding: 8px 15px; border-radius: 20px; margin: 0 5px; color: #1a2332; font-weight: bold;">
                📊 3. Results
            </span>
        </div>
        <p style="text-align: center; margin: 10px 0 0 0; color: #1a2332; opacity: 0.8;">
            <strong>Simple workflow:</strong> Upload your documents first, then ask questions to get AI-powered analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Create the main tab structure for the application
    tab_upload, tab_search, tab_advanced = st.tabs([
        "📁 Upload Documents", "🔍 Search & Analyze", "⚙️ Advanced Tools"
    ])

    with tab_upload:
        st.markdown("### 📁 Step 1: Upload Your Documents")
        st.markdown("Start here! Upload documents to begin analysis.")
        
        # Import and render simplified document workflow interface
        try:
            from simple_document_workflow import render_simple_document_management as render_document_management
            
            # Get the current RAG pipeline for the matter
            from local_rag_pipeline import rag_session_manager
            current_matter = st.session_state.current_topic
            pipeline = rag_session_manager.get_or_create_pipeline(current_matter)
            
            # Show current status prominently
            status = pipeline.get_document_status()
            if status['total_documents'] > 0:
                st.success(f"✅ Ready! You have {status['total_documents']} documents ({status['total_chunks']} text chunks) loaded and ready for analysis.")
                st.info("👉 **Next step**: Go to the 'Search & Analyze' tab to ask questions about your documents.")
            else:
                st.warning("⚠️ No documents uploaded yet. Upload documents below to get started.")
            
            # Render the simplified workflow with the pipeline
            render_document_management(pipeline)
            
        except ImportError as e:
            st.error(f"❌ Document Management interface not available: {e}")
            st.info("💡 Please ensure simple_document_workflow.py is available")
        except Exception as e:
            st.error(f"❌ Error in Document Management: {e}")
            logger.error(f"Error in Document Management tab: {e}", exc_info=True)

    with tab_search:
        st.markdown("### 🔍 Step 2: Search & Analyze Your Documents")
        
        # Check if documents are available first
        try:
            from local_rag_pipeline import rag_session_manager
            current_matter = st.session_state.current_topic
            pipeline = rag_session_manager.get_or_create_pipeline(current_matter)
            status = pipeline.get_document_status()
            
            if status['total_documents'] == 0:
                st.warning("⚠️ **No documents loaded yet!**")
                st.info("👈 Please go to the '📁 Upload Documents' tab first to upload documents.")
                st.markdown("---")
                st.markdown("#### What you can do once documents are uploaded:")
                st.markdown("- Ask questions about your documents")
                st.markdown("- Get AI-powered analysis and summaries")
                st.markdown("- Extract key facts and insights")
                st.markdown("- Generate timeline analyses")
            else:
                st.success(f"✅ Ready to analyze {status['total_documents']} documents ({status['total_chunks']} text chunks)")
                
                # Render the Simple RAG interface
                try:
                    from simple_rag_interface import render_simple_rag_interface
                    render_simple_rag_interface()
                except ImportError as e:
                    st.error(f"❌ Search interface not available: {e}")
                    st.info("💡 Please check that simple_rag_interface.py is available")
                except Exception as e:
                    st.error(f"❌ Error in search interface: {e}")
                    st.info("Please check the system logs for details")
                
        except Exception as e:
            st.error(f"❌ Error accessing document system: {e}")

    with tab_advanced:
        st.markdown("### ⚙️ Advanced Tools")
        st.markdown("Additional analysis and configuration options.")
        
        # Create sub-tabs for advanced features
        adv_tab_ai, adv_tab_ch, adv_tab_group, adv_tab_help = st.tabs([
            "🤖 AI Consultation", "🏢 Companies House", "📊 Group Analysis", "❓ Help"
        ])

        with adv_tab_ai:
            st.markdown("### AI Legal Consultation")
            
            # AI consultation should work independently without requiring document uploads
            st.markdown("**Ask your legal question or describe your situation:**")
            
            # Topic selection
            selected_topic = st.selectbox(
                "Select consultation topic:",
                options=list(PREDEFINED_TOPICS.keys()),
                index=0,
                key="ai_consultation_topic"
            )
            st.session_state.current_topic = selected_topic
            
            # In the AI Consultation section, add this before the main consultation button:
            if 'improved_prompt' not in st.session_state:
                st.session_state.improved_prompt = ""
            if 'prompt_value' not in st.session_state:
                st.session_state.prompt_value = ""
            
            # User question input - use the prompt_value if available, otherwise empty
            user_question = st.text_area(
                "Your question or situation description:",
                height=150,
                placeholder="e.g., I need advice on corporate governance issues...",
                key="ai_consultation_question",
                value=st.session_state.prompt_value
            )
            
            # Optional document upload for additional context
            with st.expander("📁 Optional: Upload Supporting Documents", expanded=False):
                st.caption("You can optionally upload documents to provide additional context to your consultation.")
                uploaded_docs_ai = st.file_uploader(
                    "Upload documents (optional):",
                    accept_multiple_files=True,
                    type=['pdf', 'txt', 'docx'],
                    key="ai_consultation_docs"
                )
                
                # Process uploaded documents if any
                if uploaded_docs_ai:
                    if st.button("Process Uploaded Documents", key="process_ai_docs"):
                        # Process documents similar to the existing logic but simplified
                        st.session_state.uploaded_docs = uploaded_docs_ai
                        st.success(f"Uploaded {len(uploaded_docs_ai)} document(s) for additional context.")

            # AI model selection
            selected_model = st.selectbox(
                "Select AI Model:",
                options=list(MODEL_PRICES_PER_1K_TOKENS_GBP.keys()),
                index=list(MODEL_PRICES_PER_1K_TOKENS_GBP.keys()).index("gpt-4.1"),
                key="ai_consultation_model"
            )
            
            if st.button("💡 Suggest Improved Prompt", key="suggest_improved_prompt"):
                with st.spinner("Improving your prompt..."):
                    improved = get_improved_prompt(user_question, selected_topic, selected_model)
                    st.session_state.improved_prompt = improved
                    st.success("Improved prompt generated below. You can edit it or use it as your main question.")

            if st.session_state.improved_prompt:
                st.markdown("**Improved Prompt:**")
                improved = st.text_area("Edit improved prompt if needed:", value=st.session_state.improved_prompt, key="improved_prompt_text_area")
                if st.button("Use Improved Prompt", key="use_improved_prompt"):
                    st.session_state.prompt_value = improved
                    st.session_state.improved_prompt = ""
                    st.rerun()

            # Generate consultation
            if st.button("🤖 Get AI Consultation", type="primary", key="generate_ai_consultation"):
                if not user_question.strip():
                    st.warning("Please enter your question or describe your situation.")
                else:
                    with st.spinner("Generating AI consultation..."):
                        try:
                            # Load protocol text
                            protocol_text = ""
                            try:
                                with open("strategic_protocols.txt", "r") as f:
                                    protocol_text = f.read()
                            except Exception:
                                protocol_text = ""
                            # Call the comprehensive consultation function
                            result = comprehensive_legal_consultation_with_protocols(
                                user_question=user_question,
                                legal_topic=selected_topic,
                                topic_description=PREDEFINED_TOPICS.get(selected_topic, "General legal advice"),
                                document_context=st.session_state.latest_digest_content,
                                protocol_text=protocol_text,
                                model_name=selected_model
                            )
                            st.markdown("### 🤖 AI Legal Consultation")
                            st.markdown(f"**Legal Topic:** {selected_topic}")
                            st.markdown(f"**AI Model:** {selected_model}")
                            st.markdown("---")
                            st.markdown(result["response"])
                            # Protocol compliance report window
                            with st.expander("🛡️ Protocol Compliance Report", expanded=False):
                                st.markdown(result["protocol_report"] or "No protocol compliance report available.")
                            # Citation verification display
                            if result["citations"]:
                                st.markdown("---")
                                st.markdown("#### ⚖️ Citations & Legal References")
                                st.markdown("The following case law and legislation references were detected and automatically verified:")
                                
                                verified_citations = result.get("verified_citations", {})
                                citations_found = []
                                citations_verified = []
                                citations_unverified = []
                                
                                for citation in result["citations"]:
                                    citations_found.append(citation)
                                    if verified_citations.get(citation, False):
                                        citations_verified.append(citation)
                                    else:
                                        citations_unverified.append(citation)
                                
                                # Display verification results
                                if citations_verified:
                                    st.markdown("##### ✅ Automatically Verified Citations")
                                    for citation in citations_verified:
                                        bailii_url = f"https://www.bailii.org/search?q={citation.replace(' ', '+')}"
                                        st.markdown(f"• [{citation}]({bailii_url}) - Found and verified")
                                
                                if citations_unverified:
                                    st.markdown("##### ⚠️ Unverified Citations")
                                    for citation in citations_unverified:
                                        st.markdown(f"• {citation} - Could not automatically verify")
                                
                                # Manual citation enhancement section
                                if citations_unverified:
                                    st.markdown("---")
                                    st.markdown("##### 🔗 Manual Citation Links (Optional)")
                                    st.markdown("If you have specific Bailii or legislation.gov.uk links for the unverified citations, you can provide them below:")
                                    
                                    citation_links = {}
                                    for citation in citations_unverified:
                                        link = st.text_input(
                                            f"Link for: {citation}", 
                                            key=f"citation_link_{citation}",
                                            placeholder="https://www.bailii.org/... or https://www.legislation.gov.uk/...",
                                            help="Paste the direct link to this case or legislation"
                                        )
                                        citation_links[citation] = link
                                    
                                    if st.button("🔍 Validate and Update Citations", key="update_with_manual_citations"):
                                        # Validate and process manual citations
                                        verified_response = result["response"]
                                        manual_updates = 0
                                        validation_results = []
                                        
                                        with st.spinner("Validating provided citation links..."):
                                            for citation, link in citation_links.items():
                                                if link.strip():
                                                    # Validate the provided link
                                                    from app_utils import fetch_url_content
                                                    content, error = fetch_url_content(link.strip())
                                                    
                                                    if error:
                                                        validation_results.append({
                                                            "citation": citation,
                                                            "link": link.strip(),
                                                            "status": "error",
                                                            "message": f"Could not access link: {error}"
                                                        })
                                                        continue
                                                    
                                                    if not content:
                                                        validation_results.append({
                                                            "citation": citation,
                                                            "link": link.strip(),
                                                            "status": "error",
                                                            "message": "No content found at the provided link"
                                                        })
                                                        continue
                                                    
                                                    # AI-powered validation: Check if the case actually supports the legal proposition
                                                    from ai_utils import get_openai_client
                                                    
                                                    # Extract the context around the citation in the consultation
                                                    consultation_text = result["response"]
                                                    citation_context = ""
                                                    
                                                    # Find sentences containing the citation
                                                    sentences = consultation_text.split('.')
                                                    for i, sentence in enumerate(sentences):
                                                        if citation in sentence:
                                                            # Get surrounding context (previous and next sentences)
                                                            start_idx = max(0, i-1)
                                                            end_idx = min(len(sentences), i+2)
                                                            citation_context = '. '.join(sentences[start_idx:end_idx]).strip()
                                                            break
                                                    
                                                    if not citation_context:
                                                        citation_context = "Citation appears in consultation but context unclear."
                                                    
                                                    # Prepare AI validation prompt
                                                    validation_prompt = f"""
You are a legal expert reviewing whether a case citation properly supports a legal proposition.

CONSULTATION CONTEXT WHERE CITATION APPEARS:
{citation_context}

CITATION BEING CHECKED:
{citation}

CASE/LEGISLATION CONTENT FROM PROVIDED LINK:
{content[:8000]}  

ANALYSIS REQUIRED:
1. Does the provided content actually contain the cited case/legislation?
2. What are the key legal principles/holdings from this case/legislation?
3. Does this case/legislation actually support the proposition made in the consultation context?
4. Is the citation being used appropriately for the legal point being made?

Respond with a JSON object containing:
- "is_correct_document": true/false (is this the right case/legislation?)
- "supports_proposition": true/false (does it support the legal point?)
- "confidence": "high"/"medium"/"low"
- "legal_principle": "brief description of what the case/legislation establishes"
- "analysis": "detailed explanation of whether and how it supports the consultation's point"
- "issues": "any problems with how the citation is being used"
"""
                                                    
                                                    openai_client = get_openai_client()
                                                    if openai_client:
                                                        try:
                                                            ai_response = openai_client.chat.completions.create(
                                                                model="gpt-4o",
                                                                temperature=0.1,
                                                                max_tokens=1000,
                                                                messages=[
                                                                    {"role": "system", "content": "You are a legal expert analyzing case citations for accuracy and relevance."},
                                                                    {"role": "user", "content": validation_prompt}
                                                                ],
                                                                response_format={"type": "json_object"}
                                                            )
                                                            
                                                            import json as json_module
                                                            response_content = ai_response.choices[0].message.content
                                                            if response_content:
                                                                ai_analysis = json_module.loads(response_content)
                                                            else:
                                                                raise Exception("Empty response from AI")
                                                            
                                                            is_correct_doc = ai_analysis.get("is_correct_document", False)
                                                            supports_prop = ai_analysis.get("supports_proposition", False)
                                                            confidence = ai_analysis.get("confidence", "low")
                                                            legal_principle = ai_analysis.get("legal_principle", "Unknown")
                                                            analysis = ai_analysis.get("analysis", "Analysis unavailable")
                                                            issues = ai_analysis.get("issues", "")
                                                            
                                                            if is_correct_doc and supports_prop and confidence in ["high", "medium"]:
                                                                # Citation is valid and supports the proposition
                                                                verified_response = verified_response.replace(
                                                                    f"{citation} [UNVERIFIED]", 
                                                                    f"[{citation}]({link.strip()})"
                                                                )
                                                                if "[UNVERIFIED]" not in result["response"]:
                                                                    verified_response = verified_response.replace(
                                                                        citation, 
                                                                        f"[{citation}]({link.strip()})",
                                                                        1
                                                                    )
                                                                manual_updates += 1
                                                                validation_results.append({
                                                                    "citation": citation,
                                                                    "link": link.strip(),
                                                                    "status": "verified",
                                                                    "message": f"✅ **Validated ({confidence} confidence)**: {legal_principle}",
                                                                    "analysis": analysis
                                                                })
                                                            else:
                                                                # Citation doesn't properly support the proposition
                                                                rejection_reason = []
                                                                if not is_correct_doc:
                                                                    rejection_reason.append("wrong document")
                                                                if not supports_prop:
                                                                    rejection_reason.append("doesn't support proposition")
                                                                if confidence == "low":
                                                                    rejection_reason.append("low confidence in analysis")
                                                                
                                                                validation_results.append({
                                                                    "citation": citation,
                                                                    "link": link.strip(),
                                                                    "status": "rejected",
                                                                    "message": f"❌ **Rejected**: {', '.join(rejection_reason)}",
                                                                    "analysis": analysis,
                                                                    "issues": issues
                                                                })
                                                        
                                                        except Exception as e:
                                                            validation_results.append({
                                                                "citation": citation,
                                                                "link": link.strip(),
                                                                "status": "error",
                                                                "message": f"AI validation failed: {str(e)}"
                                                            })
                                                    else:
                                                        validation_results.append({
                                                            "citation": citation,
                                                            "link": link.strip(),
                                                            "status": "error",
                                                            "message": "AI validation unavailable - OpenAI client not configured"
                                                        })
                                        
                                        # Display validation results
                                        st.markdown("#### 🔍 Citation Link Validation Results")
                                        for result_item in validation_results:
                                            if result_item["status"] == "verified":
                                                st.success(f"✅ **{result_item['citation']}**")
                                                st.success(result_item['message'])
                                                if 'analysis' in result_item:
                                                    with st.expander("📋 Detailed Legal Analysis", expanded=False):
                                                        st.markdown(result_item['analysis'])
                                            elif result_item["status"] == "rejected":
                                                st.warning(f"⚠️ **{result_item['citation']}**")
                                                st.warning(result_item['message'])
                                                if 'analysis' in result_item:
                                                    with st.expander("📋 Why This Citation Was Rejected", expanded=False):
                                                        st.markdown(f"**Analysis:** {result_item['analysis']}")
                                                        if 'issues' in result_item and result_item['issues']:
                                                            st.markdown(f"**Issues:** {result_item['issues']}")
                                            else:  # error
                                                st.error(f"❌ **{result_item['citation']}**: {result_item['message']}")
                                        
                                        if manual_updates > 0:
                                            # Re-run protocol compliance check
                                            from ai_utils import check_protocol_compliance
                                            protocol_report, ptok, ctok = check_protocol_compliance(verified_response, protocol_text)
                                            
                                            st.markdown("### 📄 Updated Consultation with Validated Citations")
                                            st.success(f"Successfully validated and updated {manual_updates} citation(s).")
                                            st.markdown(verified_response)
                                            with st.expander("🛡️ Updated Protocol Compliance Report", expanded=False):
                                                st.markdown(protocol_report or "No protocol compliance report available.")
                                        else:
                                            if validation_results:
                                                st.warning("No citations were validated successfully. Please check that your links contain relevant legal content.")
                                            else:
                                                st.warning("Please provide at least one citation link to validate.")
                                            
                                if not citations_found:
                                    st.markdown("No legal citations detected in this consultation.")
                                else:
                                    st.markdown(f"**Summary:** {len(citations_verified)} automatically verified, {len(citations_unverified)} unverified out of {len(citations_found)} total citations.")
                            # Cost estimate
                            if selected_model in MODEL_PRICES_PER_1K_TOKENS_GBP:
                                total_tokens = result["prompt_tokens"] + result["completion_tokens"]
                                estimated_cost = (total_tokens / 1000) * MODEL_PRICES_PER_1K_TOKENS_GBP[selected_model]
                                st.caption(f"Estimated consultation cost: £{estimated_cost:.4f} (Tokens: {total_tokens})")
                            # Export option
                            if st.button("📄 Export Consultation to DOCX", key="export_consultation"):
                                try:
                                    from app_utils import build_consult_docx
                                    import tempfile
                                    consultation_content = [
                                        f"AI Legal Consultation Report",
                                        f"Topic: {selected_topic}",
                                        f"Model: {selected_model}",
                                        f"Date: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                                        "",
                                        f"Client Question: {user_question}",
                                        "",
                                        result["response"]
                                    ]
                                    with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp_file:
                                        build_consult_docx(consultation_content, _pl.Path(tmp_file.name))
                                        with open(tmp_file.name, 'rb') as f:
                                            docx_data = f.read()
                                        st.download_button(
                                            label="📥 Download Consultation Report",
                                            data=docx_data,
                                            file_name=f"legal_consultation_{selected_topic.replace(' ', '_')}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                                        )
                                        _pl.Path(tmp_file.name).unlink()
                                except Exception as e:
                                    st.error(f"Error exporting consultation: {e}")
                                    logger.error(f"Error exporting consultation: {e}", exc_info=True)
                        except Exception as e:
                            st.error(f"Error generating consultation: {e}")
                            logger.error(f"Error in AI consultation: {e}", exc_info=True)

        with adv_tab_ch:
            st.markdown("### Companies House Analysis")
            
            # Companies House specific inputs
            ch_company_numbers_input = st.text_area(
                "Company Numbers (one per line or comma-separated):",
                height=100,
                placeholder="00000001\n12345678\nor: 00000001, 12345678",
                key="ch_company_numbers_input_main"
            )
            
            col_ch_1, col_ch_2 = st.columns(2)
            with col_ch_1:
                st.session_state.ch_start_year_input_main = st.number_input(
                    "Start Year:", min_value=1990, max_value=2030, value=2020, key="ch_start_year_main"
                )
            with col_ch_2:
                st.session_state.ch_end_year_input_main = st.number_input(
                    "End Year:", min_value=1990, max_value=2030, value=2024, key="ch_end_year_main"
                )
            
            ch_selected_categories_multiselect = st.multiselect(
                "Document Categories:",
                options=list(CH_CATEGORIES.keys()),
                default=["Accounts", "Confirmation Stmt"],
                key="ch_categories_multiselect_main"
            )
            ch_selected_categories_api = [CH_CATEGORIES[cat] for cat in ch_selected_categories_multiselect]

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
                                    use_textract_ocr=False,
                                    textract_workers=1,
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
                                if isinstance(total_ai_cost_gbp, (float, int)): 
                                    cost_str_display = f"£{total_ai_cost_gbp:.4f} (AI Processing)"
                                
                                if cost_str_display: 
                                    narrative_parts.append(f"Estimated processing cost: {cost_str_display}.")
                                
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
            ch_last_df = st.session_state.get('ch_last_df')
            df_results_available_ch = (ch_last_df is not None and 
                                   pd is not None and 
                                   hasattr(ch_last_df, 'empty') and
                                   not ch_last_df.empty)
            
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
                
                m_col3_disp_final.metric("Est. Cost", cost_display_final_metric if cost_display_final_metric != "N/A" else "£0.0000")

        with adv_tab_group:
            if 'GROUP_STRUCTURE_AVAILABLE' in globals() and GROUP_STRUCTURE_AVAILABLE:
                # Simplified OCR - no AWS Textract options
                ocr_handler_for_group_tab = None  
                st.caption("📄 Local OCR available for PDF processing in Group Structure Analysis")
                
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
                                logger=logger, # Fixed parameter name
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

        with adv_tab_help:
            try:
                from help_page import render_help_page
                render_help_page()
            except Exception as e:
                st.error(f"Error rendering Help page: {e}")
                logger.error(f"Error rendering Help page: {e}", exc_info=True)
                st.info("Help page functionality will be available when help_page module is properly configured.")

    # --- End of Main App Area UI (Using Tabs) ---

if __name__ == "__main__":
    main()
