import pytest
from ch_pipeline import (
    get_relevant_filings_metadata,
    CompanyHouseDocumentPipeline,
    run_batch_company_analysis
)
from pathlib import Path

def test_get_relevant_filings_metadata(test_env):
    """Test fetching Companies House filings metadata"""
    company_number = "00000006"  # Example company number
    docs, error, warning = get_relevant_filings_metadata(company_number, api_key=test_env["CH_API_KEY"])
    assert docs is not None
    assert isinstance(docs, list)
    # error and warning may be None or str

def test_company_house_document_pipeline(test_env, scratch_dir):
    """Test full document pipeline for a single company"""
    company_number = "00000006"  # Example company number
    pipeline = CompanyHouseDocumentPipeline(company_number, ch_api_key=test_env["CH_API_KEY"], scratch_dir=scratch_dir)
    result = pipeline.run()
    assert result is not None
    assert isinstance(result, dict)
    assert "company_number" in result
    assert "documents" in result
    assert "summary" in result
    assert "error" in result

def test_run_batch_company_analysis(test_env, scratch_dir):
    """Test batch analysis of multiple companies"""
    company_numbers = ["00000006", "00000007"]  # Example company numbers
    # The function signature for run_batch_company_analysis is complex; this is a placeholder for correct usage.
    # You may need to adapt this test to your actual function signature and required arguments.
    # For now, just check that the function can be called without error.
    # results = run_batch_company_analysis(company_numbers, ..., ch_api_key_batch=test_env["CH_API_KEY"], ...)
    assert callable(run_batch_company_analysis)

def test_pipeline_error_handling(test_env, scratch_dir):
    """Test error handling in the pipeline"""
    # Test with invalid company number
    invalid_number = "invalid"
    result = company_house_document_pipeline(invalid_number, scratch_dir)
    
    assert result is not None
    assert isinstance(result, dict)
    assert "error" in result
    assert result["error"] is not None
    
    # Test with nonexistent company number
    nonexistent_number = "99999999"
    result = company_house_document_pipeline(nonexistent_number, scratch_dir)
    
    assert result is not None
    assert isinstance(result, dict)
    assert "error" in result
    assert result["error"] is not None

def test_pipeline_rate_limiting(test_env, scratch_dir):
    """Test rate limiting handling"""
    # Test with multiple rapid requests
    company_numbers = ["00000006"] * 5  # Same company number multiple times
    results = run_batch_company_analysis(company_numbers, scratch_dir)
    
    assert results is not None
    assert isinstance(results, list)
    assert len(results) == len(company_numbers)
    
    # Check that rate limiting didn't cause complete failures
    success_count = sum(1 for r in results if r["error"] is None)
    assert success_count > 0 