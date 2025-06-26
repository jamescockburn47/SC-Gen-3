#!/usr/bin/env python3
# test_intelligent_chunking.py
"""
Test Script: Intelligent Chunking and Hierarchical RAG
Demonstrates SOTA query-adaptive retrieval strategies

Tests:
1. Query complexity classification
2. Adaptive chunk recommendation
3. Hierarchical vs legacy pipeline selection
4. Coverage optimization analysis
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from typing import Dict, Any, List
import json

# Import the enhanced systems
try:
    from enhanced_rag_interface_v2 import classify_query_for_chunking, get_coverage_quality
    from hierarchical_rag_adapter import get_rag_capabilities, HIERARCHICAL_AVAILABLE
    ENHANCED_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced RAG not available: {e}")
    ENHANCED_AVAILABLE = False

def test_query_classification():
    """Test the query complexity classification system"""
    
    print("\n🧠 Query Complexity Classification Test")
    print("=" * 50)
    
    test_queries = [
        # Simple fact queries
        "What is the contract start date?",
        "Who is the primary defendant?",
        "When did the breach occur?",
        
        # Detailed analysis queries
        "Analyze the liability provisions in the contract",
        "What are the key terms and conditions?",
        "Explain the dispute resolution mechanism",
        
        # Comprehensive queries
        "Summarize the entire legal case",
        "Provide a comprehensive overview of all documents",
        "Give me the complete analysis of the agreement",
        
        # Cross-document queries
        "Compare the witness statements",
        "What are the differences between the contracts?",
        "Analyze the relationship between all parties",
        
        # Legal-specific queries
        "What are the legal obligations of each party?",
        "Assess the potential damages and liability",
        "Identify any breaches of contract clauses"
    ]
    
    for query in test_queries:
        if ENHANCED_AVAILABLE:
            analysis = classify_query_for_chunking(query)
            
            complexity = analysis['complexity']
            recommended_chunks = analysis['recommended_chunks']
            strategy = analysis['strategy']
            
            # Color coding for complexity
            complexity_colors = {
                "simple_fact": "🟢",
                "detailed_analysis": "🟡",
                "comprehensive": "🔵", 
                "cross_document": "🟣",
                "legal_analysis": "⚖️"
            }
            
            color = complexity_colors.get(complexity, "🔍")
            
            print(f"\nQuery: '{query}'")
            print(f"  {color} Complexity: {complexity}")
            print(f"  📊 Recommended Chunks: {recommended_chunks}")
            print(f"  🎯 Strategy: {strategy}")
        else:
            print(f"\nQuery: '{query}' -> Analysis not available")

def test_coverage_analysis():
    """Test coverage quality assessment"""
    
    print("\n📊 Coverage Quality Analysis Test")
    print("=" * 50)
    
    coverage_scenarios = [
        (5, 100, "Limited document set"),
        (15, 100, "Standard analysis"),
        (30, 100, "Comprehensive analysis"),
        (50, 100, "Full document coverage"),
        (5, 20, "Small document collection"),
        (15, 25, "Medium document collection"),
        (1, 123, "Single chunk from large collection"),
        (50, 123, "Half coverage of large collection")
    ]
    
    for chunks_used, total_available, scenario in coverage_scenarios:
        percentage = (chunks_used / max(total_available, 1)) * 100
        
        if ENHANCED_AVAILABLE:
            quality = get_coverage_quality(percentage)
            
            quality_emoji = {
                "excellent": "🟢",
                "good": "🟡", 
                "limited": "🟠",
                "very_limited": "🔴"
            }
            
            emoji = quality_emoji.get(quality, "⚪")
            
            print(f"\n{scenario}:")
            print(f"  📊 Chunks: {chunks_used}/{total_available} ({percentage:.1f}%)")
            print(f"  {emoji} Quality: {quality}")
            
            # Recommendations
            if quality == "very_limited":
                print(f"  💡 Recommendation: Increase to {min(total_available, 25)} chunks for better analysis")
            elif quality == "limited" and percentage < 20:
                print(f"  💡 Recommendation: Consider {min(total_available, 30)} chunks for comprehensive coverage")
        else:
            print(f"\n{scenario}: Coverage analysis not available")

def test_adaptive_pipeline_selection():
    """Test when to use hierarchical vs legacy pipeline"""
    
    print("\n🚀 Adaptive Pipeline Selection Test")
    print("=" * 50)
    
    test_scenarios = [
        ("What is the plaintiff's name?", "simple_fact", 5),
        ("Summarize the entire case", "comprehensive", 30),
        ("Compare all witness statements", "cross_document", 25),
        ("Analyze the contract terms", "detailed_analysis", 15),
        ("What are the legal obligations?", "legal_analysis", 20)
    ]
    
    for query, expected_complexity, expected_chunks in test_scenarios:
        print(f"\nQuery: '{query}'")
        
        if ENHANCED_AVAILABLE:
            analysis = classify_query_for_chunking(query)
            
            # Determine pipeline recommendation
            is_comprehensive = "comprehensive" in analysis['complexity']
            is_cross_document = "cross_document" in analysis['complexity']
            
            if HIERARCHICAL_AVAILABLE and (is_comprehensive or is_cross_document):
                pipeline_rec = "🚀 Hierarchical Pipeline (SOTA)"
                reason = "Complex query benefits from document summarization and multi-level chunking"
            elif HIERARCHICAL_AVAILABLE:
                pipeline_rec = "🤖 Adaptive Pipeline"
                reason = "Intelligent routing based on query complexity"
            else:
                pipeline_rec = "📁 Legacy Pipeline"
                reason = "Hierarchical features not available"
            
            print(f"  Recommended: {pipeline_rec}")
            print(f"  Reason: {reason}")
            print(f"  Optimal Chunks: {analysis['recommended_chunks']}")
            
            # Verify expectations
            if analysis['complexity'] == expected_complexity:
                print(f"  ✅ Complexity classification correct")
            else:
                print(f"  ❌ Expected {expected_complexity}, got {analysis['complexity']}")
        else:
            print(f"  📁 Legacy Pipeline (Enhanced features not available)")

def test_system_capabilities():
    """Test system capability detection"""
    
    print("\n🛠️  System Capabilities Test")
    print("=" * 50)
    
    if ENHANCED_AVAILABLE:
        try:
            capabilities = get_rag_capabilities()
            
            print("Available Features:")
            print(f"  📁 Legacy RAG: {'✅' if capabilities['legacy_rag_available'] else '❌'}")
            print(f"  🚀 Hierarchical RAG: {'✅' if capabilities['hierarchical_rag_available'] else '❌'}")
            print(f"  🤖 Adaptive Routing: {'✅' if capabilities['adaptive_routing'] else '❌'}")
            
            print("\nFeature Details:")
            for category, features in capabilities['features'].items():
                print(f"  {category.title()}:")
                for feature in features:
                    print(f"    • {feature}")
            
        except Exception as e:
            print(f"Error getting capabilities: {e}")
    else:
        print("Enhanced capabilities not available")

def analyze_chunking_strategy_comparison():
    """Compare old vs new chunking strategies"""
    
    print("\n📈 Chunking Strategy Comparison")
    print("=" * 50)
    
    print("❌ OLD APPROACH (Random Chunking):")
    print("  • Fixed 5-15 chunks regardless of query complexity")
    print("  • Random selection based purely on vector similarity")
    print("  • No consideration of document structure or hierarchy")
    print("  • Poor coverage: 5/123 chunks = 4% (your reported issue)")
    print("  • No query-specific optimization")
    
    print("\n✅ NEW APPROACH (Intelligent Hierarchical):")
    print("  • Query-adaptive chunk allocation (5-50 based on complexity)")
    print("  • Document-level summarization during upload")
    print("  • Hierarchical retrieval: Document → Section → Paragraph → Sentence")
    print("  • Coverage optimization with quality feedback")
    print("  • Coarse-to-fine strategy for comprehensive queries")
    
    print("\n🎯 QUERY-SPECIFIC EXAMPLES:")
    examples = [
        ("Simple fact", "What is X?", "5 chunks, focused retrieval"),
        ("Comprehensive", "Summarize all...", "30+ chunks, broad context"),
        ("Cross-document", "Compare A vs B", "25 chunks, multi-document"),
        ("Legal analysis", "Assess liability", "20 chunks, precedent context")
    ]
    
    for query_type, example, strategy in examples:
        print(f"  {query_type}: '{example}' → {strategy}")

def main():
    """Run all intelligent chunking tests"""
    
    print("🚀 Intelligent Chunking System Test Suite")
    print("=" * 60)
    print("Testing SOTA query-adaptive retrieval strategies")
    print("Based on latest research: LongRAG, MacRAG, LongRefiner")
    
    # Run tests
    test_query_classification()
    test_coverage_analysis()
    test_adaptive_pipeline_selection()
    test_system_capabilities()
    analyze_chunking_strategy_comparison()
    
    print("\n" + "=" * 60)
    print("✅ Test Suite Complete!")
    
    if ENHANCED_AVAILABLE and HIERARCHICAL_AVAILABLE:
        print("🚀 **Ready for Enhanced RAG**: All systems available")
        print("💡 **Next Steps**: Upload documents and test with complex queries")
    elif ENHANCED_AVAILABLE:
        print("🟡 **Partial Ready**: Enhanced interface available, hierarchical features pending")
        print("💡 **Next Steps**: Install dependencies for full hierarchical support")
    else:
        print("🔴 **Legacy Mode**: Enhanced features not available")
        print("💡 **Next Steps**: Fix import issues to enable intelligent chunking")

if __name__ == "__main__":
    main() 