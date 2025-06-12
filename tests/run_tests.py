#!/usr/bin/env python3
import os
import sys
import pytest
import coverage

def run_tests():
    """Run all tests with coverage reporting"""
    # Start coverage
    cov = coverage.Coverage(
        branch=True,
        source=['.'],
        omit=[
            'tests/*',
            'venv/*',
            '*/__pycache__/*',
            '*/site-packages/*'
        ]
    )
    cov.start()

    # Run tests
    test_args = [
        '--verbose',
        '--cov=.',
        '--cov-report=term-missing',
        '--cov-report=html',
        '--cov-report=xml',
        '--junitxml=test-results.xml'
    ]
    
    # Add test files
    test_files = [
        'test_config.py',
        'test_ch_pipeline.py',
        'test_ai_utils.py',
        'test_text_extraction_utils.py',
        'test_app_utils.py',
        'test_aws_textract_utils.py',
        'test_group_structure_utils.py',
        'test_app.py'
    ]
    
    # Run pytest
    result = pytest.main(test_args + test_files)
    
    # Stop coverage and save report
    cov.stop()
    cov.save()
    
    # Generate coverage reports
    cov.report()
    cov.html_report(directory='coverage_html')
    cov.xml_report(outfile='coverage.xml')
    
    return result

if __name__ == '__main__':
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # Run tests
    exit_code = run_tests()
    sys.exit(exit_code) 