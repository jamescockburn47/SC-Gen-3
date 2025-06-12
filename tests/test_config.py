import pytest
from config import (
    get_openai_client,
    get_ch_session,
    get_gemini_model,
    OPENAI_API_KEY,
    CH_API_KEY,
    GEMINI_API_KEY,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION_DEFAULT,
    S3_TEXTRACT_BUCKET,
    LOG_LEVEL
)

def test_environment_variables(mock_env_vars):
    """Test that environment variables are properly loaded"""
    assert OPENAI_API_KEY == "test-openai-key"
    assert CH_API_KEY == "test-ch-key"
    assert GEMINI_API_KEY == "test-gemini-key"
    assert AWS_ACCESS_KEY_ID == "test-aws-key"
    assert AWS_SECRET_ACCESS_KEY == "test-aws-secret"
    assert AWS_REGION_DEFAULT == "eu-west-2"
    assert S3_TEXTRACT_BUCKET == "test-bucket"
    assert LOG_LEVEL == "DEBUG"

def test_get_openai_client(mock_openai_client, mock_env_vars):
    """Test OpenAI client initialization"""
    client = get_openai_client()
    assert client is not None
    mock_openai_client.assert_called_once()

def test_get_ch_session(mock_ch_session, mock_env_vars):
    """Test Companies House session initialization"""
    session = get_ch_session()
    assert session is not None
    mock_ch_session.assert_called_once()

def test_get_ch_session_with_key_override(mock_ch_session, mock_env_vars):
    """Test Companies House session initialization with key override"""
    custom_key = "custom-ch-key"
    session = get_ch_session(api_key_override=custom_key)
    assert session is not None
    mock_ch_session.assert_called_once()

def test_get_gemini_model(mock_gemini_model, mock_env_vars):
    """Test Gemini model initialization"""
    model = get_gemini_model("gemini-1.5-pro-latest")
    assert model is not None
    mock_gemini_model.assert_called_once_with("gemini-1.5-pro-latest")

def test_get_gemini_model_with_invalid_model(mock_gemini_model, mock_env_vars):
    """Test Gemini model initialization with invalid model name"""
    with pytest.raises(Exception):
        get_gemini_model("invalid-model-name")

def test_missing_api_keys():
    """Test behavior when API keys are missing"""
    with pytest.raises(ValueError):
        get_openai_client()
    
    with pytest.raises(ValueError):
        get_ch_session()
    
    with pytest.raises(ValueError):
        get_gemini_model("gemini-1.5-pro-latest") 