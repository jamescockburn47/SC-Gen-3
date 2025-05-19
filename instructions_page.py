import streamlit as st

"""Instructions page for Strategic Counsel"""


def render_instructions_page():
    """Render step-by-step usage instructions."""
    st.markdown("## 📖 Instructions: Using Strategic Counsel")
    st.markdown(
        """
        Follow these guidelines to make the most of each feature:
        1. **Set your Matter / Topic ID** in the sidebar. This keeps workspaces separate.
        2. **Consult Counsel** – Draft or analyse documents:
           - Enter your instruction in the text area.
           - Optionally click **Suggest Improved Prompt** to refine it.
           - Use the sidebar to inject previous digests, memories, uploads or web links.
           - Submit to receive AI-generated output and optionally export to DOCX or update your digest.
           - The system runs a protocol compliance check after each response. Disable this via **Auto-check** in the sidebar if needed.
        3. **Companies House Analysis** – Summarise UK filings:
           - Enter company numbers and select year range and document categories.
           - Run the analysis; review and download summaries or CSV results.
           - Tick summaries you want to inject back into **Consult Counsel**.
        4. **Company Group Structure** (if enabled):
           - Provide a company number and follow the step buttons to fetch profiles, parent data and subsidiary details.
        5. **About** – Overview of the system and environment details.
        
        For best results keep instructions concise and load only the most relevant context.
        """
    )

