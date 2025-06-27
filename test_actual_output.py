#!/usr/bin/env python3
"""
Test Actual Output Quality - Strategic Counsel RAG System
========================================================

This script tests the ACTUAL OUTPUT QUALITY that users see,
not just whether functions work without errors.
"""

import sys
import os
from pathlib import Path
import asyncio
from typing import Dict, Any, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_actual_rag_output():
    """Test the actual RAG output quality that users experience"""
    
    print("🔍 TESTING ACTUAL RAG OUTPUT QUALITY")
    print("=" * 50)
    
    try:
        # Import the RAG pipeline
        from local_rag_pipeline import rag_session_manager
        from simple_rag_interface import get_protocol_compliant_answer
        
        # Get the current pipeline
        matter_id = "test_matter" 
        pipeline = rag_session_manager.get_or_create_pipeline(matter_id)
        
        print(f"✅ Pipeline loaded: {pipeline.documents_count} documents, {pipeline.chunks_count} chunks")
        
        # Test different types of queries
        test_queries = [
            "What are the key dates mentioned in the documents?",
            "Summarize the main legal issues",
            "Who are the parties involved?",
            "What timeline of events can you identify?",
            "What are the contractual obligations mentioned?"
        ]
        
        print("\n🧪 TESTING SEARCH RESULTS:")
        print("-" * 30)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: '{query}'")
            
            try:
                # Test raw search results
                search_results = pipeline.search_documents(query, top_k=5)
                
                if not search_results:
                    print("   ❌ NO SEARCH RESULTS RETURNED")
                    continue
                    
                print(f"   ✅ Found {len(search_results)} search results")
                
                # Check result quality
                for j, result in enumerate(search_results[:2], 1):
                    if hasattr(result, 'content') and result.content:
                        content_preview = result.content[:100].replace('\n', ' ')
                        print(f"   📄 Result {j}: {content_preview}...")
                        
                        if hasattr(result, 'score'):
                            print(f"      Score: {result.score:.3f}")
                    else:
                        print(f"   ❌ Result {j}: Empty or invalid content")
                
                # Test full RAG answer
                print(f"   🤖 Testing full RAG answer...")
                try:
                    answer = get_protocol_compliant_answer(
                        query=query,
                        search_results=search_results,
                        matter_context=f"Matter: {matter_id}"
                    )
                    
                    if answer and len(answer.strip()) > 10:
                        print(f"   ✅ Generated answer ({len(answer)} chars)")
                        
                        # Check answer quality indicators
                        if any(word in answer.lower() for word in ['document', 'based on', 'according to']):
                            print("   ✅ Answer references documents")
                        else:
                            print("   ⚠️  Answer may not reference source documents")
                            
                        if len(answer) > 50:
                            print("   ✅ Answer is substantive")
                        else:
                            print("   ⚠️  Answer seems too brief")
                            
                        # Show answer preview
                        answer_preview = answer[:200].replace('\n', ' ')
                        print(f"   📝 Preview: {answer_preview}...")
                        
                    else:
                        print("   ❌ Generated answer is empty or too short")
                        print(f"   Raw answer: '{answer}'")
                        
                except Exception as e:
                    print(f"   ❌ RAG Answer failed: {e}")
                    
            except Exception as e:
                print(f"   ❌ Search failed: {e}")
                
        print("\n" + "="*50)
        
        # Test timeline-specific functionality
        print("\n🕐 TESTING TIMELINE FUNCTIONALITY:")
        print("-" * 30)
        
        timeline_query = "Give me a chronological timeline of events"
        print(f"Timeline Query: '{timeline_query}'")
        
        try:
            # Test if timeline detection works
            timeline_keywords = ['timeline', 'chronological', 'sequence', 'dates', 'history', 'progression', 'order', 'when', 'date']
            is_timeline_query = any(keyword in timeline_query.lower() for keyword in timeline_keywords)
            
            if is_timeline_query:
                print("   ✅ Timeline detection working")
                
                # Test timeline-specific search (should use more chunks)
                timeline_results = pipeline.search_documents(timeline_query, top_k=50)  # Timeline uses 50 chunks
                print(f"   ✅ Retrieved {len(timeline_results)} chunks for timeline (should be ~50)")
                
                # Test timeline answer
                timeline_answer = get_protocol_compliant_answer(
                    query=timeline_query,
                    search_results=timeline_results,
                    matter_context=f"Matter: {matter_id}"
                )
                
                if timeline_answer:
                    print(f"   ✅ Timeline answer generated ({len(timeline_answer)} chars)")
                    
                    # Check for chronological indicators
                    chronological_indicators = ['first', 'then', 'next', 'after', 'before', 'subsequently', 'initially', 'finally']
                    has_chronology = any(indicator in timeline_answer.lower() for indicator in chronological_indicators)
                    
                    if has_chronology:
                        print("   ✅ Answer contains chronological indicators")
                    else:
                        print("   ⚠️  Answer may lack chronological structure")
                        
                    # Show timeline preview
                    timeline_preview = timeline_answer[:300].replace('\n', ' ')
                    print(f"   📝 Timeline Preview: {timeline_preview}...")
                else:
                    print("   ❌ Timeline answer is empty")
            else:
                print("   ❌ Timeline detection failed")
                
        except Exception as e:
            print(f"   ❌ Timeline test failed: {e}")
            
        print("\n" + "="*50)
        print("✅ OUTPUT QUALITY TEST COMPLETE")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test Error: {e}")
        return False

def test_enhanced_rag_output():
    """Test the Enhanced RAG interface output"""
    
    print("\n🚀 TESTING ENHANCED RAG OUTPUT:")
    print("-" * 30)
    
    try:
        from enhanced_rag_interface import enhanced_rag_search
        
        # Test enhanced search with different configurations
        test_query = "What are the main contractual terms and conditions?"
        
        configurations = [
            {"use_colbert": True, "use_hierarchical": True, "use_adaptive": True},
            {"use_colbert": False, "use_hierarchical": True, "use_adaptive": True},
            {"use_colbert": True, "use_hierarchical": False, "use_adaptive": True},
        ]
        
        for i, config in enumerate(configurations, 1):
            print(f"\n{i}. Testing Enhanced RAG Config: {config}")
            
            try:
                result = enhanced_rag_search(
                    query=test_query,
                    matter_id="test_matter",
                    **config
                )
                
                if result and 'answer' in result:
                    answer = result['answer']
                    print(f"   ✅ Enhanced answer generated ({len(answer)} chars)")
                    
                    # Check for enhancement indicators
                    if 'metadata' in result:
                        metadata = result['metadata']
                        print(f"   📊 Metadata: {metadata}")
                        
                    # Show answer preview
                    enhanced_preview = answer[:200].replace('\n', ' ')
                    print(f"   📝 Preview: {enhanced_preview}...")
                    
                else:
                    print("   ❌ Enhanced RAG returned empty result")
                    
            except Exception as e:
                print(f"   ❌ Enhanced RAG config {i} failed: {e}")
                
    except ImportError as e:
        print(f"⚠️  Enhanced RAG not available: {e}")
    except Exception as e:
        print(f"❌ Enhanced RAG test error: {e}")

def test_simple_interface_output():
    """Test the Simple RAG Interface output"""
    
    print("\n📝 TESTING SIMPLE INTERFACE OUTPUT:")
    print("-" * 30)
    
    try:
        from simple_rag_interface import perform_rag_search
        
        # Test simple interface
        test_query = "What are the key findings in the documents?"
        print(f"Simple Query: '{test_query}'")
        
        result = perform_rag_search(test_query, "test_matter")
        
        if result:
            print(f"   ✅ Simple interface result generated ({len(result)} chars)")
            
            # Check result quality
            if len(result) > 100:
                print("   ✅ Result is substantial")
            else:
                print("   ⚠️  Result seems brief")
                
            # Show preview
            simple_preview = result[:200].replace('\n', ' ')
            print(f"   📝 Preview: {simple_preview}...")
            
        else:
            print("   ❌ Simple interface returned empty result")
            
    except ImportError as e:
        print(f"⚠️  Simple interface not available: {e}")
    except Exception as e:
        print(f"❌ Simple interface test error: {e}")

if __name__ == "__main__":
    print("🎯 TESTING ACTUAL OUTPUT QUALITY - WHAT USERS REALLY SEE")
    print("=" * 60)
    
    # Run comprehensive output tests
    success = test_actual_rag_output()
    
    if success:
        test_enhanced_rag_output()
        test_simple_interface_output()
        
        print("\n" + "="*60)
        print("🎯 CONCLUSION: Check the output quality above")
        print("✅ = Working well")
        print("⚠️  = Needs attention") 
        print("❌ = Broken/Poor quality")
        print("="*60)
    else:
        print("\n❌ BASIC RAG OUTPUT TEST FAILED - SYSTEM NOT READY") 