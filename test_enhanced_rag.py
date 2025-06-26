#!/usr/bin/env python3
"""
Test script for the enhanced RAG interface with protocol compliance
"""

import asyncio
import sys
from enhanced_rag_interface import get_protocol_compliant_answer, check_protocol_compliance

async def test_enhanced_rag():
    """Test the enhanced RAG system with protocol compliance"""
    
    print("🛡️ Testing Enhanced RAG Interface")
    print("=" * 50)
    
    # Test queries
    test_queries = [
        "What is this dispute about?",
        "Who are the parties involved?",
        "What are the main legal claims?",
        "What is the case number?",
        "What dates are mentioned in the documents?"
    ]
    
    # Test with different models
    test_models = ["phi3:latest", "deepseek-llm:7b", "mistral:latest"]
    
    print(f"Testing {len(test_queries)} queries with {len(test_models)} models")
    print("-" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}/{len(test_queries)}: {query}")
        print("-" * 30)
        
        for model in test_models:
            print(f"\n🤖 Testing with {model}...")
            
            try:
                result = await get_protocol_compliant_answer(
                    query, 
                    'Corporate Governance',  # Use default matter for testing
                    model, 
                    max_chunks=5
                )
                
                answer = result.get('answer', '')
                sources = result.get('sources', [])
                compliance = result.get('protocol_compliance', {})
                
                # Display basic results
                print(f"   ✅ Generated: {len(answer)} chars")
                print(f"   📚 Sources: {len(sources)}")
                print(f"   ⏱️ Time: {result.get('generation_time', 0):.2f}s")
                
                # Protocol compliance summary
                overall_score = compliance.get('overall_score', 0)
                if overall_score >= 0.8:
                    print(f"   🟢 Protocol Compliance: {overall_score:.1%} (Excellent)")
                elif overall_score >= 0.6:
                    print(f"   🟡 Protocol Compliance: {overall_score:.1%} (Good)")
                else:
                    print(f"   🔴 Protocol Compliance: {overall_score:.1%} (Needs Improvement)")
                
                # Show any violations
                violations = []
                for check_name, check_data in compliance.get('compliance_checks', {}).items():
                    if check_data.get('status') == 'FAIL':
                        violations.append(check_name.replace('_', ' ').title())
                
                if violations:
                    print(f"   ⚠️ Issues: {', '.join(violations)}")
                else:
                    print(f"   ✅ All protocol checks passed")
                
                # Show recommendations
                recommendations = compliance.get('recommendations', [])
                if recommendations:
                    print(f"   💡 Recommendations: {len(recommendations)} suggestions")
                
                # Show answer preview
                answer_preview = answer[:100] + "..." if len(answer) > 100 else answer
                print(f"   📝 Answer: {answer_preview}")
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
        
        print()
    
    print("🎯 Enhanced RAG Test Summary:")
    print("- Enhanced interface includes protocol compliance monitoring")
    print("- Real-time hallucination detection")
    print("- Source citation validation")
    print("- Model-agnostic query processing")
    print("- Comprehensive compliance reporting")

async def test_single_enhanced_query(query: str, model: str = "phi3:latest"):
    """Test a single query with detailed output"""
    
    print(f"🔍 Enhanced RAG Test: {query}")
    print(f"🤖 Model: {model}")
    print("=" * 60)
    
    result = await get_protocol_compliant_answer(query, 'Corporate Governance', model)
    
    # Display full answer
    print("📋 ANSWER:")
    print("-" * 40)
    print(result['answer'])
    print()
    
    # Display protocol compliance report
    compliance = result.get('protocol_compliance', {})
    print("🛡️ PROTOCOL COMPLIANCE REPORT:")
    print("-" * 40)
    
    overall_score = compliance.get('overall_score', 0)
    print(f"Overall Score: {overall_score:.1%}")
    
    checks = compliance.get('compliance_checks', {})
    for check_name, check_data in checks.items():
        status = check_data.get('status', 'UNKNOWN')
        score = check_data.get('score', 0)
        details = check_data.get('details', '')
        
        status_emoji = "✅" if status == "PASS" else "⚠️" if status == "PARTIAL" else "❌"
        print(f"{status_emoji} {check_name.replace('_', ' ').title()}: {score:.1%}")
        print(f"   {details}")
    
    recommendations = compliance.get('recommendations', [])
    if recommendations:
        print("\n💡 RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"   • {rec}")
    
    # Display sources
    sources = result.get('sources', [])
    if sources:
        print(f"\n📚 SOURCES ({len(sources)} found):")
        print("-" * 40)
        for i, source in enumerate(sources, 1):
            print(f"{i}. {source['document']} (Score: {source['similarity_score']:.3f})")
            print(f"   Preview: {source['text_preview'][:100]}...")
            print()
    
    # Display technical details
    print("🔧 TECHNICAL DETAILS:")
    print("-" * 40)
    print(f"Model Used: {result.get('model_used')}")
    print(f"Context Chunks: {result.get('context_chunks_used', 0)}")
    print(f"Generation Time: {result.get('generation_time', 0):.2f} seconds")
    print(f"Debug Info: {result.get('debug_info', 'None')}")

def main():
    """Main function"""
    
    if len(sys.argv) > 1:
        query = sys.argv[1]
        model = sys.argv[2] if len(sys.argv) > 2 else "phi3:latest"
        asyncio.run(test_single_enhanced_query(query, model))
    else:
        asyncio.run(test_enhanced_rag())

if __name__ == "__main__":
    main() 