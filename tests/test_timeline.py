import pytest
from unittest.mock import MagicMock, patch
import os
import sys
from pathlib import Path
import json
import datetime as dt

# Add the parent directory to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from timeline_utils import (
    parse_docket_file,
    create_timeline_events,
    format_timeline_event,
    validate_timeline_event,
    sort_timeline_events,
    filter_timeline_events,
    export_timeline_events,
    import_timeline_events,
    merge_timeline_events,
    calculate_event_duration,
    get_event_statistics,
    validate_timeline_data,
    format_timeline_date,
    parse_timeline_date,
    group_timeline_events,
    calculate_timeline_metrics,
    generate_timeline_report,
    validate_timeline_format,
    process_timeline_data,
    handle_timeline_error
)

@pytest.fixture
def sample_timeline_data():
    """Fixture to provide sample timeline data"""
    return {
        'events': [
            {
                'date': '2023-01-01',
                'description': 'Event 1',
                'category': 'meeting',
                'duration': 60,
                'participants': ['John', 'Jane']
            },
            {
                'date': '2023-01-02',
                'description': 'Event 2',
                'category': 'call',
                'duration': 30,
                'participants': ['John']
            }
        ]
    }

@pytest.fixture
def sample_docket_file():
    """Fixture to provide sample docket file content"""
    return """Date,Description,Category,Duration,Participants
2023-01-01,Event 1,meeting,60,"John, Jane"
2023-01-02,Event 2,call,30,John"""

def test_parse_docket_file(sample_docket_file):
    """Test docket file parsing"""
    # Test CSV parsing
    with patch('builtins.open', MagicMock(return_value=MagicMock(read=lambda: sample_docket_file))):
        result = parse_docket_file("test.csv")
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]['date'] == '2023-01-01'
        assert result[0]['description'] == 'Event 1'

    # Test JSON parsing
    json_data = json.dumps({'events': [
        {'date': '2023-01-01', 'description': 'Event 1'},
        {'date': '2023-01-02', 'description': 'Event 2'}
    ]})
    with patch('builtins.open', MagicMock(return_value=MagicMock(read=lambda: json_data))):
        result = parse_docket_file("test.json")
        assert isinstance(result, list)
        assert len(result) == 2

def test_timeline_event_creation(sample_timeline_data):
    """Test timeline event creation and formatting"""
    # Test event creation
    events = create_timeline_events(sample_timeline_data['events'])
    assert isinstance(events, list)
    assert len(events) == 2
    
    # Test event formatting
    formatted = format_timeline_event(events[0])
    assert isinstance(formatted, str)
    assert 'Event 1' in formatted
    
    # Test event validation
    assert validate_timeline_event(events[0]) is True
    assert validate_timeline_event({'date': 'invalid'}) is False

def test_timeline_event_manipulation(sample_timeline_data):
    """Test timeline event manipulation functions"""
    events = create_timeline_events(sample_timeline_data['events'])
    
    # Test sorting
    sorted_events = sort_timeline_events(events)
    assert sorted_events[0]['date'] <= sorted_events[1]['date']
    
    # Test filtering
    filtered_events = filter_timeline_events(events, category='meeting')
    assert len(filtered_events) == 1
    assert filtered_events[0]['category'] == 'meeting'
    
    # Test merging
    merged_events = merge_timeline_events(events, events)
    assert len(merged_events) == 4

def test_timeline_export_import(sample_timeline_data):
    """Test timeline export and import functionality"""
    events = create_timeline_events(sample_timeline_data['events'])
    
    # Test export
    export_path = "test_output/timeline_export.json"
    export_timeline_events(events, export_path)
    assert os.path.exists(export_path)
    
    # Test import
    imported_events = import_timeline_events(export_path)
    assert isinstance(imported_events, list)
    assert len(imported_events) == len(events)

def test_timeline_calculations(sample_timeline_data):
    """Test timeline calculations and metrics"""
    events = create_timeline_events(sample_timeline_data['events'])
    
    # Test duration calculation
    duration = calculate_event_duration(events[0])
    assert duration == 60
    
    # Test statistics
    stats = get_event_statistics(events)
    assert isinstance(stats, dict)
    assert 'total_events' in stats
    assert 'total_duration' in stats
    
    # Test metrics
    metrics = calculate_timeline_metrics(events)
    assert isinstance(metrics, dict)
    assert 'average_duration' in metrics
    assert 'events_per_category' in metrics

def test_timeline_validation(sample_timeline_data):
    """Test timeline validation functions"""
    # Test data validation
    assert validate_timeline_data(sample_timeline_data) is True
    assert validate_timeline_data({'invalid': 'data'}) is False
    
    # Test format validation
    assert validate_timeline_format(sample_timeline_data['events'][0]) is True
    assert validate_timeline_format({'invalid': 'format'}) is False

def test_timeline_date_handling():
    """Test timeline date handling functions"""
    # Test date formatting
    formatted_date = format_timeline_date('2023-01-01')
    assert isinstance(formatted_date, str)
    
    # Test date parsing
    parsed_date = parse_timeline_date('2023-01-01')
    assert isinstance(parsed_date, dt.date)
    
    # Test invalid date handling
    assert parse_timeline_date('invalid') is None

def test_timeline_grouping(sample_timeline_data):
    """Test timeline event grouping"""
    events = create_timeline_events(sample_timeline_data['events'])
    
    # Test grouping by category
    grouped = group_timeline_events(events, by='category')
    assert isinstance(grouped, dict)
    assert 'meeting' in grouped
    assert 'call' in grouped
    
    # Test grouping by date
    grouped = group_timeline_events(events, by='date')
    assert isinstance(grouped, dict)
    assert '2023-01-01' in grouped
    assert '2023-01-02' in grouped

def test_timeline_report_generation(sample_timeline_data):
    """Test timeline report generation"""
    events = create_timeline_events(sample_timeline_data['events'])
    
    # Test report generation
    report = generate_timeline_report(events)
    assert isinstance(report, str)
    assert 'Event 1' in report
    assert 'Event 2' in report
    
    # Test report with statistics
    report = generate_timeline_report(events, include_stats=True)
    assert isinstance(report, str)
    assert 'Statistics' in report

def test_timeline_error_handling():
    """Test timeline error handling"""
    # Test with invalid data
    result = process_timeline_data({'invalid': 'data'})
    assert result is None
    
    # Test with invalid file
    result = parse_docket_file("nonexistent.csv")
    assert result is None
    
    # Test error handling
    error = handle_timeline_error(Exception("Test error"))
    assert isinstance(error, dict)
    assert 'error' in error

def test_timeline_performance():
    """Test timeline performance with large datasets"""
    # Create large dataset
    large_events = [
        {
            'date': f'2023-01-{i:02d}',
            'description': f'Event {i}',
            'category': 'meeting' if i % 2 == 0 else 'call',
            'duration': 60,
            'participants': ['John', 'Jane']
        }
        for i in range(1, 101)  # 100 events
    ]
    
    # Test processing large dataset
    events = create_timeline_events(large_events)
    assert len(events) == 100
    
    # Test sorting large dataset
    sorted_events = sort_timeline_events(events)
    assert len(sorted_events) == 100
    
    # Test grouping large dataset
    grouped = group_timeline_events(events, by='category')
    assert len(grouped) == 2  # meeting and call categories 