import pytest
from ai_utils import (
    gpt_summarise_ch_docs,
    gemini_summarise_ch_docs,
    get_improved_prompt,
    check_protocol_compliance
)

def test_gpt_summarise_ch_docs(test_env):
    """Test GPT document summarization"""
    sample_text = """
    This is a sample document about a company's financial performance.
    The company reported revenue of £1 million in 2023.
    Operating costs were £500,000.
    Net profit was £200,000.
    The company plans to expand into new markets in 2024.
    """
    
    summary, prompt_tokens, completion_tokens = gpt_summarise_ch_docs(sample_text)
    assert summary is not None
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert prompt_tokens > 0
    assert completion_tokens > 0

def test_gemini_summarise_ch_docs(test_env):
    """Test Gemini document summarization"""
    sample_text = """
    This is a sample document about a company's financial performance.
    The company reported revenue of £1 million in 2023.
    Operating costs were £500,000.
    Net profit was £200,000.
    The company plans to expand into new markets in 2024.
    """
    
    summary, prompt_tokens, completion_tokens = gemini_summarise_ch_docs(sample_text)
    assert summary is not None
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert prompt_tokens > 0
    assert completion_tokens > 0

def test_get_improved_prompt(test_env):
    """Test prompt improvement"""
    original_prompt = "Summarize this document"
    improved_prompt = get_improved_prompt(original_prompt)
    assert improved_prompt is not None
    assert isinstance(improved_prompt, str)
    assert len(improved_prompt) > len(original_prompt)

def test_check_protocol_compliance(test_env):
    """Test protocol compliance check"""
    sample_text = """
    The company has implemented robust financial controls.
    Regular audits are conducted quarterly.
    All transactions are properly documented.
    Risk management procedures are in place.
    """
    
    result = check_protocol_compliance(sample_text)
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0

def test_ai_utils_error_handling(test_env):
    """Test error handling in AI utilities"""
    # Test with empty text
    summary, prompt_tokens, completion_tokens = gpt_summarise_ch_docs("")
    assert summary is None
    assert prompt_tokens == 0
    assert completion_tokens == 0
    
    # Test with very long text
    long_text = "x" * 100000  # Exceeds typical token limits
    summary, prompt_tokens, completion_tokens = gpt_summarise_ch_docs(long_text)
    assert summary is None
    assert prompt_tokens == 0
    assert completion_tokens == 0 