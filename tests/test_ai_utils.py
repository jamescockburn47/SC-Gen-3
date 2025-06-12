import pytest
from unittest.mock import Mock, patch
from ai_utils import (
    gpt_summarise_ch_docs,
    gemini_summarise_ch_docs,
    get_improved_prompt,
    check_protocol_compliance
)

@pytest.fixture
def mock_openai_client():
    with patch('ai_utils.get_openai_client') as mock:
        mock_client = Mock()
        mock.return_value = mock_client
        yield mock_client

@pytest.fixture
def mock_gemini_model():
    with patch('ai_utils.get_gemini_model') as mock:
        mock_model = Mock()
        mock.return_value = mock_model
        yield mock_model

def test_gpt_summarise_ch_docs(mock_openai_client):
    # Test successful summarization
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Test summary"))]
    mock_response.usage = Mock(prompt_tokens=100, completion_tokens=50)
    mock_openai_client.chat.completions.create.return_value = mock_response

    result, prompt_tokens, completion_tokens = gpt_summarise_ch_docs(
        "Test text", "12345678"
    )
    assert result == "Test summary"
    assert prompt_tokens == 100
    assert completion_tokens == 50

def test_gpt_summarise_ch_docs_error(mock_openai_client):
    # Test error handling
    mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
    
    result, prompt_tokens, completion_tokens = gpt_summarise_ch_docs(
        "Test text", "12345678"
    )
    assert "Error: GPT summarization failed" in result
    assert prompt_tokens == 0
    assert completion_tokens == 0

def test_gemini_summarise_ch_docs(mock_gemini_model):
    # Test successful summarization
    mock_response = Mock()
    mock_response.text = "Test summary"
    mock_gemini_model.generate_content.return_value = mock_response
    
    result, prompt_tokens, completion_tokens = gemini_summarise_ch_docs(
        "Test text", "12345678"
    )
    assert result == "Test summary"
    assert prompt_tokens >= 0
    assert completion_tokens >= 0

def test_gemini_summarise_ch_docs_error(mock_gemini_model):
    # Test error handling
    mock_gemini_model.generate_content.side_effect = Exception("API Error")
    
    result, prompt_tokens, completion_tokens = gemini_summarise_ch_docs(
        "Test text", "12345678"
    )
    assert "Error: Gemini summarization failed" in result
    assert prompt_tokens == 0
    assert completion_tokens == 0

def test_get_improved_prompt(mock_openai_client):
    # Test successful prompt improvement
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Improved prompt"))]
    mock_openai_client.chat.completions.create.return_value = mock_response

    result = get_improved_prompt("Original prompt", "Context")
    assert result == "Improved prompt"

def test_get_improved_prompt_error(mock_openai_client):
    # Test error handling
    mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
    
    result = get_improved_prompt("Original prompt", "Context")
    assert result == "Original prompt"

def test_check_protocol_compliance(mock_openai_client):
    # Test successful protocol check
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Compliant"))]
    mock_openai_client.chat.completions.create.return_value = mock_response

    result, prompt_tokens, completion_tokens = check_protocol_compliance(
        "AI output", "Protocol text"
    )
    assert result == "Compliant"
    assert prompt_tokens >= 0
    assert completion_tokens >= 0

def test_check_protocol_compliance_error(mock_openai_client):
    # Test error handling
    mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
    
    result, prompt_tokens, completion_tokens = check_protocol_compliance(
        "AI output", "Protocol text"
    )
    assert "Error: Protocol compliance check failed" in result
    assert prompt_tokens == 0
    assert completion_tokens == 0 