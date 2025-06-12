import pytest
from unittest.mock import Mock, patch
from ch_pipeline import (
    CompanyHouseDocumentPipeline,
    run_batch_company_analysis,
    get_relevant_filings_metadata
)

@pytest.fixture
def mock_ch_api():
    with patch('ch_pipeline.get_ch_documents_metadata') as mock:
        yield mock

@pytest.fixture
def mock_ai_utils():
    with patch('ch_pipeline.gpt_summarise_ch_docs') as mock:
        yield mock

def test_get_relevant_filings_metadata(mock_ch_api):
    # Test successful metadata fetch
    mock_ch_api.return_value = (
        [{"type": "accounts", "date": "2023-01-01"}],
        {"company_name": "Test Company"},
        None
    )
    
    result, profile, error = get_relevant_filings_metadata(
        "12345678",
        "test-api-key",
        years_back=2,
        categories_to_fetch=["accounts"]
    )
    assert len(result) == 1
    assert profile["company_name"] == "Test Company"
    assert error is None

def test_get_relevant_filings_metadata_error(mock_ch_api):
    # Test error handling
    mock_ch_api.return_value = ([], None, "API Error")
    
    result, profile, error = get_relevant_filings_metadata(
        "12345678",
        "test-api-key"
    )
    assert len(result) == 0
    assert profile is None
    assert error == "API Error"

def test_company_house_document_pipeline(mock_ch_api, mock_ai_utils):
    # Test successful pipeline run
    mock_ch_api.return_value = (
        [{"type": "accounts", "date": "2023-01-01"}],
        {"company_name": "Test Company"},
        None
    )
    mock_ai_utils.return_value = ("Summary", 100, 50)
    
    pipeline = CompanyHouseDocumentPipeline("12345678")
    result = pipeline.run(years_back=2)
    assert isinstance(result, dict)
    assert "company_name" in result
    assert "summary" in result

def test_company_house_document_pipeline_error(mock_ch_api):
    # Test error handling
    mock_ch_api.return_value = ([], None, "API Error")
    
    pipeline = CompanyHouseDocumentPipeline("12345678")
    result = pipeline.run(years_back=2)
    assert isinstance(result, dict)
    assert "error" in result

def test_run_batch_company_analysis(mock_ch_api, mock_ai_utils):
    # Test successful batch analysis
    mock_ch_api.return_value = (
        [{"type": "accounts", "date": "2023-01-01"}],
        {"company_name": "Test Company"},
        None
    )
    mock_ai_utils.return_value = ("Summary", 100, 50)
    
    results = run_batch_company_analysis(
        ["12345678"],
        {"12345678": [{"type": "accounts"}]},
        {"12345678": {"company_name": "Test Company"}},
        "test-api-key",
        {"gpt-4": 0.01}
    )
    assert isinstance(results, tuple)
    assert len(results) == 2 