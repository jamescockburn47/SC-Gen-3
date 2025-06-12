import pytest
from unittest.mock import patch, MagicMock
from group_structure_utils import (
    analyze_company_group_structure,
    find_subsidiaries,
    build_company_tree,
    validate_company_relationship,
    format_group_structure
)

@pytest.fixture
def sample_company_hierarchy():
    return {
        "parent": {
            "company_number": "12345678",
            "company_name": "Parent Company Ltd",
            "subsidiaries": [
                {
                    "company_number": "87654321",
                    "company_name": "Subsidiary 1 Ltd",
                    "subsidiaries": []
                },
                {
                    "company_number": "23456789",
                    "company_name": "Subsidiary 2 Ltd",
                    "subsidiaries": [
                        {
                            "company_number": "34567890",
                            "company_name": "Sub-Subsidiary Ltd",
                            "subsidiaries": []
                        }
                    ]
                }
            ]
        }
    }

def test_analyze_company_group_structure(sample_company_hierarchy):
    """Test group structure analysis"""
    result = analyze_company_group_structure(sample_company_hierarchy)
    assert isinstance(result, dict)
    assert "total_companies" in result
    assert "max_depth" in result
    assert result["total_companies"] == 4
    assert result["max_depth"] == 2

def test_find_subsidiaries(sample_company_hierarchy):
    """Test subsidiary finding"""
    result = find_subsidiaries(sample_company_hierarchy["parent"])
    assert len(result) == 3  # Including sub-subsidiary
    assert any(s["company_number"] == "87654321" for s in result)
    assert any(s["company_number"] == "23456789" for s in result)
    assert any(s["company_number"] == "34567890" for s in result)

def test_build_company_tree(sample_company_hierarchy):
    """Test company tree building"""
    result = build_company_tree(sample_company_hierarchy["parent"])
    assert isinstance(result, dict)
    assert "name" in result
    assert "children" in result
    assert len(result["children"]) == 2

def test_validate_company_relationship():
    """Test company relationship validation"""
    # Valid relationships
    assert validate_company_relationship("parent", "subsidiary") is True
    assert validate_company_relationship("subsidiary", "sub-subsidiary") is True
    
    # Invalid relationships
    assert validate_company_relationship("subsidiary", "parent") is False
    assert validate_company_relationship("unrelated", "unrelated") is False

def test_format_group_structure(sample_company_hierarchy):
    """Test group structure formatting"""
    result = format_group_structure(sample_company_hierarchy)
    assert isinstance(result, str)
    assert "Parent Company Ltd" in result
    assert "Subsidiary 1 Ltd" in result
    assert "Subsidiary 2 Ltd" in result
    assert "Sub-Subsidiary Ltd" in result

def test_analyze_company_group_structure_empty():
    """Test group structure analysis with empty input"""
    result = analyze_company_group_structure({})
    assert result["total_companies"] == 0
    assert result["max_depth"] == 0

def test_find_subsidiaries_empty():
    """Test subsidiary finding with empty input"""
    result = find_subsidiaries({})
    assert len(result) == 0

def test_build_company_tree_empty():
    """Test company tree building with empty input"""
    result = build_company_tree({})
    assert result == {}

def test_format_group_structure_empty():
    """Test group structure formatting with empty input"""
    result = format_group_structure({})
    assert result == ""

def test_analyze_company_group_structure_invalid():
    """Test group structure analysis with invalid input"""
    with pytest.raises(ValueError):
        analyze_company_group_structure(None)

def test_find_subsidiaries_invalid():
    """Test subsidiary finding with invalid input"""
    with pytest.raises(ValueError):
        find_subsidiaries(None)

def test_build_company_tree_invalid():
    """Test company tree building with invalid input"""
    with pytest.raises(ValueError):
        build_company_tree(None)

def test_format_group_structure_invalid():
    """Test group structure formatting with invalid input"""
    with pytest.raises(ValueError):
        format_group_structure(None) 