#!/usr/bin/env python3
import os
import sys
import pytest
import coverage
import argparse
from datetime import datetime
from pathlib import Path

def setup_test_environment():
    """Setup the test environment"""
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # Create test output directory
    test_output_dir = Path('test_output')
    test_output_dir.mkdir(exist_ok=True)
    
    return test_output_dir

def discover_test_files():
    """Discover all test files in the tests directory."""
    tests_dir = Path(__file__).parent
    test_files = set()
    for pattern in ["test_*.py", "*_test.py"]:
        for f in tests_dir.glob(pattern):
            if f.name != "conftest.py" and f.name != "run_tests.py":
                test_files.add(str(f))
    return sorted(test_files)

def run_tests(verbose=False):
    """Run tests with coverage reporting"""
    test_output_dir = setup_test_environment()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
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

    # Discover test files
    test_files = discover_test_files()
    if not test_files:
        print("No test files found.")
        return 1

    # Prepare test arguments
    test_args = [
        '--verbose' if verbose else '-v',
        '--cov=.',
        '--cov-report=term-missing',
        '--cov-report=html',
        '--cov-report=xml',
        f'--junitxml={test_output_dir}/test-results_{timestamp}.xml',
        '--capture=tee-sys',  # Capture stdout/stderr
        '-p', 'no:warnings'  # Disable warning capture
    ]
    
    # Run pytest
    result = pytest.main(test_args + test_files)
    
    # Stop coverage and save report
    cov.stop()
    cov.save()
    
    # Generate coverage reports
    cov.report()
    cov.html_report(directory=f'{test_output_dir}/coverage_html_{timestamp}')
    cov.xml_report(outfile=f'{test_output_dir}/coverage_{timestamp}.xml')
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Run all discovered tests with coverage reporting')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    exit_code = run_tests(args.verbose)
    sys.exit(exit_code)

if __name__ == '__main__':
    main() 