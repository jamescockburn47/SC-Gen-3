# ai_utils.py

import logging
import json
import time
import re
import random
from typing import Optional, List, Tuple, Dict, Any, Union
from datetime import datetime

# Attempt to import Google Generative AI
try:
    import google.generativeai as genai_sdk
    from google.generativeai.types import HarmCategory, HarmBlockThreshold # For safety settings
    from google.api_core import exceptions as GoogleAPICoreExceptions
except ImportError:
    genai_sdk = None
    GoogleAPICoreExceptions = None
    HarmCategory = None # Define for type hinting if import fails
    HarmBlockThreshold = None # Define for type hinting if import fails


from config import (
    get_openai_client,
    get_gemini_model,
    OPENAI_MODEL_DEFAULT,
    GEMINI_MODEL_DEFAULT,
    GEMINI_MODEL_FOR_PROTOCOL_CHECK, # New model for protocol checks
    logger, # Use central logger
    LOADED_PROTO_TEXT # Use loaded protocol text from config
)

OBJECTIVE_CH_SUMMARIZATION_BASE_PROMPT = """
You are an expert financial and legal analyst AI. Your task is to perform a rigorous and objective analysis of excerpts from UK Companies House filings.
The provided text is derived from official company documents such as Annual Accounts (AA), Confirmation Statements (CS01), director change forms (AP01, TM01), charge registrations (MR01), etc.
Extract and synthesize key factual information pertaining *only* to the content of the provided text. Adhere strictly to the data presented.

MANDATORY FOCUS AREAS & REQUESTED OUTPUT STRUCTURE (ensure these headings are present if relevant information exists):

1.  **FINANCIAL PERFORMANCE & POSITION (Primarily from Annual Accounts):**
    * Identify and state key financial figures: Revenue/Turnover, Profit/Loss Before Tax (PBT), Operating Profit, Total Assets, Net Assets (Total Assets minus Total Liabilities or Shareholders' Funds), Total Liabilities, Current/Non-Current Assets & Liabilities breakdown if detailed, and key Cash Flow figures (Operating, Investing, Financing) if available.
    * For each figure, state the value and the corresponding financial year or period end date clearly. Use 'N/A' if a specific figure is not found.
    * **If data for multiple financial periods is present in the text, PRIORITIZE A COMPARATIVE PRESENTATION.** For key metrics (e.g., Revenue, PBT, Net Assets), clearly state the values for each available year. Explicitly calculate and state significant year-on-year percentage or absolute changes for these core metrics where possible.
    * **When presenting financial data for multiple periods (e.g., in tables or lists of figures), ensure years are in ASCENDING order (e.g., 2021, 2022, 2023 from left to right or top to bottom for tables, or in chronological sequence for lists).**
    * Mention any stated accounting policies crucial for understanding the figures, if briefly detailed (e.g., basis of consolidation, revenue recognition).

2.  **COMPANY STRUCTURE & CAPITAL:**
    * Describe the company's legal structure if stated (e.g., private limited, PLC). If identified as a subsidiary, note the Parent Company Name and Registration Number if provided in the text.
    * Report any changes in share capital (e.g., statement of capital in CS01), share classes, or significant share allotments/transfers if detailed.

3.  **GOVERNANCE & OFFICERS:**
    * List any director appointments or resignations/terminations, including full names and effective dates if provided.
    * Report changes in Persons with Significant Control (PSCs), including names and dates of change if mentioned.
    * Note the name of the appointed auditor and any change in auditor if specified.

4.  **MATERIAL OBLIGATIONS & COMMITMENTS:** (Excluding items covered under "RED FLAGS / KEY RISKS" unless they are distinct contractual obligations not qualifying as an immediate red flag)
    * List any new or existing registered charges (mortgages), including the amount secured, date registered, charge code, and persons entitled, if detailed.
    * Report significant contractual obligations or commitments if disclosed (e.g., operating leases, capital commitments) not covered elsewhere.

5.  **SIGNIFICANT CORPORATE EVENTS:**
    * Detail any mentions of mergers, acquisitions, disposals of significant assets or business segments, or major restructuring activities, including dates and parties involved if available.
    * Note any changes to the company's registered office address or name, including dates.

6.  **RED FLAGS / KEY RISKS:**
    * **Explicitly list any identified material risks to the business (e.g., from Strategic Report or Directors' Report), 'going concern' statements (especially if adverse or with material uncertainties highlighted by directors or auditors), adverse audit opinions/qualifications from the Auditor's Report (e.g., disclaimer of opinion, qualified opinion, emphasis of matter related to going concern), significant actual or pending legal proceedings, or other explicitly stated warnings found in the text. If none, state "No specific red flags identified in the provided text." Be precise and quote or closely paraphrase where appropriate.**

STYLE AND TONE REQUIREMENTS:
* Output must be objective, factual, precise, and technical. Use formal business language.
* Financial figures, names, and dates must be extracted accurately.
* Avoid speculation, inference beyond the explicit text, or subjective commentary.
* The summary should be well-organized, using bullet points for lists (e.g., directors, charges, key financial figures per year) and concise paragraphs for narrative elements. Structure with clear headings for each focus area as outlined above.
* **When presenting financial data for multiple periods, ensure clarity, facilitate easy year-on-year comparison, and list years in ASCENDING order.**

EXPLICIT EXCLUSIONS (CRITICAL - DO NOT INCLUDE THE FOLLOWING):
* DO NOT mention generic corporate social responsibility (CSR) statements, environmental initiatives ('green credentials'), community engagement, employee welfare programs, corporate culture, or general market commentary UNLESS these are explicitly linked to a material financial impact, a stated significant risk by the directors/auditors, or a major operational change clearly detailed in the filing.
* DO NOT include any marketing language, PR-style statements, or aspirational goals found in the documents.
* DO NOT infer positive or negative sentiment unless it's a direct quote or a formally stated opinion from an auditor or director regarding a material issue (e.g., an adverse audit opinion).
* Focus strictly on the verifiable, substantive information as per a formal regulatory filing. This is a technical analysis, not a public relations summary.

If additional specific instructions are provided by the user (appended below), address those *within the same rigorous and objective framework* outlined above, integrating them logically into the relevant sections, including any requests for comparative analysis.
"""

# Chunking threshold for GPT models (or Gemini models if they exceed their large context)
MAX_CHARS_PER_AI_CHUNK_STANDARD = 100_000
# Effective single-pass threshold for large-context Gemini models (e.g., 1.5M chars ~ 375k tokens, well within 1M token limit)
MAX_CHARS_FOR_GEMINI_LARGE_CONTEXT_SINGLE_PASS = 1_500_000
MIN_CHARS_FOR_AI_SUMMARY = 150 # Minimum characters for AI to attempt summarization

MAX_TOKENS_FOR_SUMMARY_RESPONSE = 4000 # For OpenAI models
MAX_TOKENS_FOR_GEMINI_SUMMARY_RESPONSE = 8192 # Max output for Gemini (can be up to 8192 for Pro 1.5)

# Define default safety settings for Gemini (can be adjusted)
DEFAULT_GEMINI_SAFETY_SETTINGS = None
if HarmCategory and HarmBlockThreshold: # Check if imported
    DEFAULT_GEMINI_SAFETY_SETTINGS = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }


def _prepare_full_prompt(base_prompt: str, specific_instructions: str, company_no: str, text_chunk: str, chunk_info: str = "") -> Tuple[str, str]:
    final_system_prompt = base_prompt
    if specific_instructions and specific_instructions.strip():
        final_system_prompt += f"\n\nADDITIONAL USER-SPECIFIC INSTRUCTIONS (Apply within the objective framework above):\n{specific_instructions.strip()}"
    user_content = (
        f"Company Reference (for context only, data is below): {company_no} {chunk_info}\n\n"
        f"Document Text (derived from JSON, XHTML, or OCR'd PDF):\n---\n{text_chunk}\n---"
    )
    return final_system_prompt, user_content


def gpt_summarise_ch_docs(
    text_to_summarize: str,
    company_no: str,
    specific_instructions: str = "",
    model_to_use: Optional[str] = None
) -> Tuple[str, int, int]:
    openai_client = get_openai_client()
    total_prompt_tokens = 0
    total_completion_tokens = 0

    if not openai_client:
        logger.error(f"OpenAI client not configured. Cannot perform summarization for {company_no}.")
        return "Error: OpenAI client not configured for summarization.", 0, 0
    
    stripped_text = text_to_summarize.strip()
    if not stripped_text:
        logger.warning(f"No text provided to summarise (OpenAI) for {company_no}.")
        return "No content provided to AI for summarization.", 0, 0
    
    if len(stripped_text) < MIN_CHARS_FOR_AI_SUMMARY:
        logger.warning(f"{company_no}: Input text for OpenAI summarization is too short ({len(stripped_text)} chars, threshold {MIN_CHARS_FOR_AI_SUMMARY}). Skipping AI summarization.")
        return f"Input text too short ({len(stripped_text)} chars) for meaningful summarization by AI.", 0, 0

    current_model_for_summary = model_to_use or OPENAI_MODEL_DEFAULT
    logger.info(f"{company_no}: Starting OpenAI summarization with model {current_model_for_summary}. Total text length: {len(stripped_text)} chars.")

    if len(stripped_text) > MAX_CHARS_PER_AI_CHUNK_STANDARD:
        sub_summaries = []
        num_chunks = (len(stripped_text) + MAX_CHARS_PER_AI_CHUNK_STANDARD - 1) // MAX_CHARS_PER_AI_CHUNK_STANDARD
        logger.info(f"{company_no}: Text exceeds {MAX_CHARS_PER_AI_CHUNK_STANDARD} chars, splitting into {num_chunks} chunks for OpenAI.")

        for i in range(0, len(stripped_text), MAX_CHARS_PER_AI_CHUNK_STANDARD):
            chunk = stripped_text[i:i + MAX_CHARS_PER_AI_CHUNK_STANDARD]
            chunk_label = f"(Chunk {i // MAX_CHARS_PER_AI_CHUNK_STANDARD + 1}/{num_chunks})"
            system_prompt, user_content = _prepare_full_prompt(
                OBJECTIVE_CH_SUMMARIZATION_BASE_PROMPT, specific_instructions, company_no, chunk, chunk_label
            )
            try:
                logger.debug(f"{company_no}: Summarizing chunk {chunk_label} ({len(chunk)} chars) with {current_model_for_summary}.")
                response = openai_client.chat.completions.create(
                    model=current_model_for_summary, temperature=0.1,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
                    max_tokens=MAX_TOKENS_FOR_SUMMARY_RESPONSE
                )
                sub_summary = response.choices[0].message.content.strip()
                if response.usage:
                    total_prompt_tokens += response.usage.prompt_tokens
                    total_completion_tokens += response.usage.completion_tokens
                sub_summaries.append(sub_summary)
                logger.info(f"{company_no}: OpenAI sub-summary {chunk_label} complete. Tokens: P{response.usage.prompt_tokens if response.usage else 'N/A'}/C{response.usage.completion_tokens if response.usage else 'N/A'}")
            except Exception as e_chunk:
                sub_summaries.append(f"Error summarizing {chunk_label}: {e_chunk}")
                logger.error(f"OpenAI chunk summarization failed for {company_no} at {chunk_label}: {e_chunk}", exc_info=True)


        if not any("Error summarizing" not in s for s in sub_summaries):
            return "Error: All text chunks failed during OpenAI summarization.", total_prompt_tokens, total_completion_tokens

        aggregation_input_text = "\n\n---\n\n".join(
            f"Section from Chunk {idx+1}:\n{s_text}" if "Error summarizing" not in s_text
            else f"Section from Chunk {idx+1} (error):\n{s_text}"
            for idx, s_text in enumerate(sub_summaries)
        )
        
        aggregation_system_prompt = (
            f"{OBJECTIVE_CH_SUMMARIZATION_BASE_PROMPT}\n\n"
            "You are now in an AGGREGATION step. The following sections are pre-summaries from different parts of a large document or multiple related documents for the SAME COMPANY. "
            "Your task is to synthesize these sections into a single, coherent, and comprehensive summary that adheres to all the structural and content requirements of the base prompt. "
            "Pay particular attention to consolidating financial data across years (maintaining ascending year order) and identifying overall themes or red flags that emerge from the combined information. "
            "Ensure all mandatory focus areas from the base prompt are covered if the information is present across the chunks."
        )
        aggregation_user_content = (
             f"Company Reference: {company_no} (AGGREGATED VIEW FROM MULTIPLE CHUNKS)\n\n"
             f"Pre-summarized Sections (Chunks) to Aggregate:\n---\n{aggregation_input_text}\n---"
        )
        try:
            agg_response = openai_client.chat.completions.create(
                model=current_model_for_summary, temperature=0.1,
                messages=[{"role": "system", "content": aggregation_system_prompt}, {"role": "user", "content": aggregation_user_content}],
                max_tokens=MAX_TOKENS_FOR_SUMMARY_RESPONSE 
            )
            final_summary = agg_response.choices[0].message.content.strip()
            if agg_response.usage:
                total_prompt_tokens += agg_response.usage.prompt_tokens
                total_completion_tokens += agg_response.usage.completion_tokens
            logger.info(f"{company_no}: OpenAI aggregated summary created. Tokens: P{agg_response.usage.prompt_tokens if agg_response.usage else 'N/A'}/C{agg_response.usage.completion_tokens if agg_response.usage else 'N/A'}")
            return final_summary, total_prompt_tokens, total_completion_tokens
        except Exception as e_agg:
            logger.error(f"{company_no}: OpenAI aggregation step failed: {e_agg}.", exc_info=True)
            return f"Error during aggregation: {e_agg}.\n\nConcatenated (partial data may be useful):\n{aggregation_input_text}", total_prompt_tokens, total_completion_tokens
    else: 
        system_prompt, user_content = _prepare_full_prompt(
            OBJECTIVE_CH_SUMMARIZATION_BASE_PROMPT, specific_instructions, company_no, stripped_text
        )
        try:
            logger.debug(f"{company_no}: Summarizing text ({len(stripped_text)} chars) single pass with {current_model_for_summary}.")
            response = openai_client.chat.completions.create(
                model=current_model_for_summary, temperature=0.1,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
                max_tokens=MAX_TOKENS_FOR_SUMMARY_RESPONSE
            )
            summary_content = response.choices[0].message.content.strip()
            if response.usage:
                total_prompt_tokens = response.usage.prompt_tokens
                total_completion_tokens = response.usage.completion_tokens
            logger.info(f"OpenAI summary received for {company_no}. Tokens: P{total_prompt_tokens}/C{total_completion_tokens}")
            return summary_content, total_prompt_tokens, total_completion_tokens
        except Exception as e_openai_api:
            logger.error(f"{company_no}: OpenAI API error: {e_openai_api}", exc_info=True)
            return f"Error: GPT summarization failed: {str(e_openai_api)}", 0, 0


def _gemini_generate_content_with_retry_and_tokens(
    gemini_model_client: Any, # genai.GenerativeModel
    prompt_parts: List[Union[str, Any]], # List of strings or Content parts
    generation_config: Any, # genai.types.GenerationConfig
    safety_settings: Optional[Dict[Any, Any]], # Optional safety settings
    company_no: str,
    context_label: str,
    max_retries: int = 2,
    initial_delay: float = 5.0
) -> Tuple[str, int, int]:
    if not genai_sdk or not GoogleAPICoreExceptions or not gemini_model_client:
        err_msg = "Error: Gemini SDK, Google API Core Exceptions, or model client not available."
        logger.error(f"{company_no} ({context_label}): {err_msg}")
        return err_msg, 0, 0

    prompt_tokens = 0
    completion_tokens = 0

    try:
        prompt_tokens_response = gemini_model_client.count_tokens(prompt_parts)
        prompt_tokens = prompt_tokens_response.total_tokens
    except Exception as e_count_prompt:
        logger.warning(f"{company_no} ({context_label}): Failed to count Gemini prompt tokens: {e_count_prompt}. Proceeding with 0.")
        prompt_tokens = 0

    generated_text = "Error: Generation failed before API call."
    for attempt in range(max_retries + 1):
        try:
            response = gemini_model_client.generate_content(
                contents=prompt_parts,
                generation_config=generation_config,
                safety_settings=safety_settings # Pass safety settings
            )

            if response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason_str = genai_sdk.types.BlockedReason(response.prompt_feedback.block_reason).name if genai_sdk else f"UNKNOWN_REASON_CODE_{response.prompt_feedback.block_reason}"
                logger.error(f"{company_no} ({context_label}): Gemini content generation blocked by API. Reason: {block_reason_str}. Details: {response.prompt_feedback.safety_ratings}")
                return f"Error: Gemini content generation blocked by API (Reason: {block_reason_str}). Please check safety ratings in logs.", prompt_tokens, 0

            text_parts_from_response = []
            if hasattr(response, 'parts') and response.parts:
                text_parts_from_response = [part.text for part in response.parts if hasattr(part, 'text')]
            elif hasattr(response, 'text'):
                 text_parts_from_response = [response.text]

            if text_parts_from_response:
                generated_text = "".join(text_parts_from_response).strip()
                try:
                    completion_tokens_response = gemini_model_client.count_tokens(generated_text)
                    completion_tokens = completion_tokens_response.total_tokens
                except Exception as e_count_completion:
                    logger.warning(f"{company_no} ({context_label}): Failed to count Gemini completion tokens: {e_count_completion}. Defaulting to 0.")
                    completion_tokens = 0
                return generated_text, prompt_tokens, completion_tokens
            else:
                if response.candidates and response.candidates[0].finish_reason:
                    finish_reason_str = genai_sdk.types.FinishReason(response.candidates[0].finish_reason).name if genai_sdk else f"UNKNOWN_FINISH_REASON_{response.candidates[0].finish_reason}"
                    if finish_reason_str not in ["STOP", "MAX_TOKENS"]:
                        logger.error(f"{company_no} ({context_label}): Gemini generation finished with reason: {finish_reason_str}. Safety ratings: {response.candidates[0].safety_ratings if response.candidates else 'N/A'}")
                        return f"Error: Gemini generation stopped (Reason: {finish_reason_str}). Content may be blocked or incomplete.", prompt_tokens, 0
                
                generated_text = "Error: Gemini response did not contain usable text output or was empty."
                logger.error(f"{company_no} ({context_label}): Gemini returned empty or malformed response. Full response object: {response}")
                return generated_text, prompt_tokens, 0

        except (GoogleAPICoreExceptions.ResourceExhausted, GoogleAPICoreExceptions.DeadlineExceeded, GoogleAPICoreExceptions.ServiceUnavailable) as e_retryable:
            error_type = e_retryable.__class__.__name__
            if attempt < max_retries:
                delay = (initial_delay * (2 ** attempt)) + (random.uniform(0, 1))
                logger.warning(f"{company_no} ({context_label}): Gemini API {error_type} (Attempt {attempt + 1}/{max_retries + 1}). Retrying in {delay:.1f}s.")
                time.sleep(delay)
            else:
                logger.error(f"{company_no} ({context_label}): Gemini API {error_type} after {max_retries + 1} attempts: {e_retryable}")
                return f"Error: Gemini API {error_type} after retries: {e_retryable}", prompt_tokens, 0
        except Exception as e:
            logger.error(f"{company_no} ({context_label}): Gemini generate_content() failed (Attempt {attempt + 1}): {e}", exc_info=True)
            if attempt == max_retries:
                 return f"Error: Gemini generate_content() failed with unretryable error: {e}", prompt_tokens, 0
            time.sleep(initial_delay * (1.5 ** attempt)) # Delay for other errors too
            generated_text = f"Error: Gemini generate_content() failed on attempt {attempt+1}: {e}"

    return generated_text, prompt_tokens, completion_tokens # Fallback


def gemini_summarise_ch_docs(
    text_to_summarize: str,
    company_no: str,
    specific_instructions: str = "",
    model_name: Optional[str] = None
) -> Tuple[str, int, int]:
    if not genai_sdk:
        logger.error("Google Generative AI SDK not installed. Cannot perform Gemini summarization.")
        return "Error: Google Generative AI SDK not installed.", 0, 0

    current_model_name = model_name or GEMINI_MODEL_DEFAULT
    gemini_model_client, init_error = get_gemini_model(current_model_name)
    total_prompt_tokens = 0
    total_completion_tokens = 0

    if not gemini_model_client:
        err_msg = init_error or f"Could not initialize Gemini model '{current_model_name}'."
        logger.error(err_msg)
        return f"Error: {err_msg}", 0, 0
    
    stripped_text = text_to_summarize.strip()
    if not stripped_text:
        logger.warning(f"No text provided to summarise (Gemini) for {company_no}.")
        return "No content provided to AI for summarization.", 0, 0
        
    if len(stripped_text) < MIN_CHARS_FOR_AI_SUMMARY:
        logger.warning(f"{company_no}: Input text for Gemini summarization is too short ({len(stripped_text)} chars, threshold {MIN_CHARS_FOR_AI_SUMMARY}). Skipping AI summarization.")
        return f"Input text too short ({len(stripped_text)} chars) for meaningful summarization by AI.", 0, 0

    logger.info(f"{company_no}: Starting Gemini summarization with model {current_model_name}. Length: {len(stripped_text)} chars.")
    
    generation_config_params = {
        "temperature": 0.1,
        "max_output_tokens": MAX_TOKENS_FOR_GEMINI_SUMMARY_RESPONSE
    }
    generation_config = genai_sdk.types.GenerationConfig(**generation_config_params) if genai_sdk else None


    if len(stripped_text) <= MAX_CHARS_FOR_GEMINI_LARGE_CONTEXT_SINGLE_PASS:
        logger.info(f"{company_no}: Text length ({len(stripped_text)}) within single-pass threshold for Gemini. Attempting single summarization.")
        system_prompt_part, user_content_part = _prepare_full_prompt(
            OBJECTIVE_CH_SUMMARIZATION_BASE_PROMPT, specific_instructions, company_no, stripped_text, "Single Pass"
        )
        full_prompt_single_pass = f"{system_prompt_part}\n\n{user_content_part}"

        summary_content, p_tokens, c_tokens = _gemini_generate_content_with_retry_and_tokens(
            gemini_model_client, [full_prompt_single_pass], generation_config, DEFAULT_GEMINI_SAFETY_SETTINGS, company_no, "Single Pass Large Context"
        )
        total_prompt_tokens += p_tokens
        total_completion_tokens += c_tokens
        if "Error:" not in summary_content and "blocked" not in summary_content.lower():
            logger.info(f"Gemini summary (single pass) received for {company_no}. Tokens P:{p_tokens}/C:{c_tokens}")
        else:
             logger.error(f"{company_no}: Gemini single-pass summarization FAILED or was blocked. Error: {summary_content}")
        return summary_content, total_prompt_tokens, total_completion_tokens
    else:
        logger.info(f"{company_no}: Text length ({len(stripped_text)}) exceeds single-pass threshold. Using standard chunking for Gemini (chunk size: {MAX_CHARS_PER_AI_CHUNK_STANDARD} chars).")
        sub_summaries_text = []
        num_chunks = (len(stripped_text) + MAX_CHARS_PER_AI_CHUNK_STANDARD - 1) // MAX_CHARS_PER_AI_CHUNK_STANDARD
        logger.info(f"Splitting into {num_chunks} chunks.")

        for i in range(0, len(stripped_text), MAX_CHARS_PER_AI_CHUNK_STANDARD):
            chunk = stripped_text[i:i + MAX_CHARS_PER_AI_CHUNK_STANDARD]
            chunk_label = f"Chunk {i // MAX_CHARS_PER_AI_CHUNK_STANDARD + 1}/{num_chunks}"
            system_prompt_part, user_content_part = _prepare_full_prompt(
                OBJECTIVE_CH_SUMMARIZATION_BASE_PROMPT, specific_instructions, company_no, chunk, chunk_label
            )
            full_prompt_for_chunk = f"{system_prompt_part}\n\n{user_content_part}"

            logger.debug(f"{company_no}: Summarizing {chunk_label} ({len(chunk)} chars) with {current_model_name}.")
            sub_summary, p_tokens, c_tokens = _gemini_generate_content_with_retry_and_tokens(
                gemini_model_client, [full_prompt_for_chunk], generation_config, DEFAULT_GEMINI_SAFETY_SETTINGS, company_no, chunk_label
            )
            total_prompt_tokens += p_tokens
            total_completion_tokens += c_tokens
            sub_summaries_text.append(sub_summary)
            if "Error:" not in sub_summary and "blocked" not in sub_summary.lower():
                logger.info(f"{company_no}: Gemini sub-summary {chunk_label} complete. Tokens P:{p_tokens}/C:{c_tokens}")
            else:
                logger.error(f"{company_no}: Gemini sub-summary {chunk_label} FAILED or was blocked. Error: {sub_summary}")

        if not any(("Error:" not in s and "blocked" not in s.lower()) for s in sub_summaries_text):
            return "Error: All text chunks failed during Gemini summarization or were blocked.", total_prompt_tokens, total_completion_tokens

        aggregation_input_text = "\n\n---\n\n".join(
            f"Section from Chunk {idx+1}:\n{s_text}" if ("Error:" not in s_text and "blocked" not in s_text.lower())
            else f"Section from Chunk {idx+1} (error/blocked):\n{s_text}"
            for idx, s_text in enumerate(sub_summaries_text)
        )
        
        aggregation_system_prompt_part = (
            f"{OBJECTIVE_CH_SUMMARIZATION_BASE_PROMPT}\n\n"
            "You are now in an AGGREGATION step. The following sections are pre-summaries from different parts of a large document or multiple related documents for the SAME COMPANY. "
            "Your task is to synthesize these sections into a single, coherent, and comprehensive summary that adheres to all the structural and content requirements of the base prompt. "
            "Pay particular attention to consolidating financial data across years (maintaining ascending year order) and identifying overall themes or red flags that emerge from the combined information. "
            "Ensure all mandatory focus areas from the base prompt are covered if the information is present across the chunks. If some chunks reported errors, note that information was missing for those parts but proceed with summarizing available content."
        )
        aggregation_user_content_part = (
             f"Company Reference: {company_no} (AGGREGATED VIEW FROM MULTIPLE CHUNKS)\n\n"
             f"Pre-summarized Sections (Chunks) to Aggregate:\n---\n{aggregation_input_text}\n---"
        )
        full_prompt_for_aggregation = f"{aggregation_system_prompt_part}\n\n{aggregation_user_content_part}"

        final_summary, p_tokens_agg, c_tokens_agg = _gemini_generate_content_with_retry_and_tokens(
            gemini_model_client, [full_prompt_for_aggregation], generation_config, DEFAULT_GEMINI_SAFETY_SETTINGS, company_no, "Aggregation"
        )
        total_prompt_tokens += p_tokens_agg
        total_completion_tokens += c_tokens_agg

        if "Error:" not in final_summary and "blocked" not in final_summary.lower():
            logger.info(f"{company_no}: Gemini aggregated summary created. Tokens P:{p_tokens_agg}/C:{c_tokens_agg}")
        else:
            logger.error(f"{company_no}: Gemini aggregation failed or was blocked. Error: {final_summary}.")
            return f"Error during Gemini aggregation: {final_summary}.\n\nConcatenated sub-summaries (some may contain errors):\n{aggregation_input_text}", total_prompt_tokens, total_completion_tokens
        return final_summary, total_prompt_tokens, total_completion_tokens


def get_improved_prompt(
    original_prompt: str,
    prompt_context: str,
    model_name: Optional[str] = None
) -> str:
    """
    Uses a Gemini model to refine and improve a user's input prompt.
    """
    if not genai_sdk:
        return "Error: Google Generative AI SDK not installed. Cannot improve prompt."

    # Use the general purpose Gemini model for this, or a specific one if desired.
    current_model_name = model_name or GEMINI_MODEL_DEFAULT
    gemini_model_client, init_error = get_gemini_model(current_model_name)

    if not gemini_model_client:
        err_msg = init_error or f"Could not initialize Gemini model '{current_model_name}' for prompt improvement."
        return f"Error: {err_msg}"
    if not original_prompt or not original_prompt.strip():
        return original_prompt 

    logger.info(f"Attempting to improve prompt for context: '{prompt_context}' using {current_model_name}. Original: '{original_prompt[:100]}...'")

    system_instruction = (
        "You are an expert AI assistant specializing in crafting effective prompts for other AI models. "
        "The primary user is a UK Litigator, and all outputs should be tailored for this persona, focusing on UK-specific legal and business contexts unless the prompt explicitly states otherwise. "
        "Your task is to refine the user's input (provided below) to make it a better, more structured, and more effective prompt for an AI that will perform tasks such as legal analysis, document summarization, or research. "
        "Do NOT answer or execute the user's original prompt. Your sole output should be the improved prompt text itself. "
        "The improved prompt should:"
        "\n- Be clear, concise, and unambiguous."
        "\n- Explicitly state or strongly imply that the context is UK law and UK-based entities if relevant to the original prompt's intent."
        "\n- If the original prompt is vague, try to make it more specific by adding reasonable assumptions or by structuring it to elicit more detailed information, always from a UK litigator's perspective."
        "\n- Maintain the core intent of the original prompt."
        "\n- Be suitable for direct use as input to another advanced AI model."
        "\n- If the original prompt is already excellent, you can return it with minimal or no changes, perhaps with a brief affirmation like 'This prompt is already well-structured. Using as is:' followed by the prompt."
        "\n- Do not add conversational fluff or explanations about why you changed the prompt. Only output the refined prompt text."
    )
    
    full_prompt_for_improvement = f"{system_instruction}\n\nUser's Original Prompt (for context: {prompt_context}):\n---\n{original_prompt}\n---\nImproved Prompt:"

    generation_config = genai_sdk.types.GenerationConfig(
        temperature=0.3, 
        max_output_tokens=1024 
    ) if genai_sdk else None

    improved_prompt_text, p_tokens, c_tokens = _gemini_generate_content_with_retry_and_tokens(
        gemini_model_client,
        [full_prompt_for_improvement], 
        generation_config,
        DEFAULT_GEMINI_SAFETY_SETTINGS, # Use default safety settings
        company_no="N/A_PromptImprovement", 
        context_label="PromptImprovement"
    )

    if "Error:" in improved_prompt_text or "blocked" in improved_prompt_text.lower():
        logger.error(f"Failed to improve prompt. AI returned: {improved_prompt_text}")
        return f"Error: Could not improve prompt. AI service issue: {improved_prompt_text}"
    
    if improved_prompt_text.startswith("This prompt is already well-structured. Using as is:"):
        improved_prompt_text = improved_prompt_text.replace("This prompt is already well-structured. Using as is:", "").strip()
    
    logger.info(f"Prompt improvement successful. Tokens P:{p_tokens}/C:{c_tokens}. Improved: '{improved_prompt_text[:100]}...'")
    return improved_prompt_text.strip()

def check_protocol_compliance(
    ai_text_output: str,
    protocol_text: str,
    model_name: Optional[str] = None
) -> Tuple[str, int, int]:
    """
    Uses a Gemini model to check AI text output against strategic protocols.

    Args:
        ai_text_output: The AI-generated text to check.
        protocol_text: The strategic protocols.
        model_name: The specific Gemini model to use.

    Returns:
        A tuple containing the compliance report string, prompt tokens, and completion tokens.
    """
    if not genai_sdk:
        return "Error: Google Generative AI SDK not installed. Cannot check protocol compliance.", 0, 0

    current_model_name = model_name or GEMINI_MODEL_FOR_PROTOCOL_CHECK
    gemini_model_client, init_error = get_gemini_model(current_model_name)

    if not gemini_model_client:
        err_msg = init_error or f"Could not initialize Gemini model '{current_model_name}' for protocol compliance check."
        return f"Error: {err_msg}", 0, 0

    if not ai_text_output or not ai_text_output.strip():
        return "Error: No AI text provided for compliance check.", 0, 0
    if not protocol_text or not protocol_text.strip():
        return "Error: No protocol text provided for compliance check.", 0, 0

    logger.info(f"Starting protocol compliance check using {current_model_name}.")

    # Instruction for the AI performing the compliance check
    # Based on M.E.4: "A sidebar button must run a “Protocol Compliance Report” listing section-by-section adherence or red flags."
    compliance_check_prompt = (
        "You are an AI Compliance Officer. Your task is to meticulously review an AI-generated text against a given set of 'Strategic Protocols'. "
        "Produce a 'Protocol Compliance Report' that assesses adherence to each protocol section (e.g., M.A, M.B, etc.) and its sub-points (e.g., M.A.1, M.A.2). "
        "For each protocol point, state whether the AI text appears compliant, non-compliant, or if the protocol point is not applicable or not assessable from the given text. "
        "If non-compliant or potentially problematic, briefly explain the issue and cite the specific part of the AI text if possible. "
        "If compliant, a brief affirmation is sufficient. "
        "The report should be structured, clear, and directly address each protocol point mentioned in the 'Strategic Protocols' document.\n\n"
        "STRATEGIC PROTOCOLS:\n"
        "----------------------\n"
        f"{protocol_text}\n"
        "----------------------\n\n"
        "AI-GENERATED TEXT TO REVIEW:\n"
        "----------------------------\n"
        f"{ai_text_output}\n"
        "----------------------------\n\n"
        "PROTOCOL COMPLIANCE REPORT (Section-by-Section Analysis):"
    )

    generation_config = genai_sdk.types.GenerationConfig(
        temperature=0.1,  # Low temperature for factual, objective assessment
        max_output_tokens=MAX_TOKENS_FOR_GEMINI_SUMMARY_RESPONSE # Allow ample space for detailed report
    ) if genai_sdk else None
    
    # Using the same retry and token counting wrapper
    report_text, p_tokens, c_tokens = _gemini_generate_content_with_retry_and_tokens(
        gemini_model_client,
        [compliance_check_prompt], # Pass as a list
        generation_config,
        DEFAULT_GEMINI_SAFETY_SETTINGS, # Use default safety settings
        company_no="N/A_ProtocolCheck", # Contextual, not a real company_no here
        context_label="ProtocolComplianceCheck"
    )

    if "Error:" in report_text or "blocked" in report_text.lower():
        logger.error(f"Protocol compliance check failed. AI returned: {report_text}")
        # Return the error message from the AI
    else:
        logger.info(f"Protocol compliance check successful. Tokens P:{p_tokens}/C:{c_tokens}. Report length: {len(report_text)} chars.")
    
    return report_text, p_tokens, c_tokens
