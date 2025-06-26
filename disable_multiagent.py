#!/usr/bin/env python3
"""
Protocol-Enforced Multi-Agent RAG Test
Tests with MCP Strategic Counsel protocols applied to multi-agent responses
"""

import asyncio
from local_rag_pipeline import rag_session_manager

async def protocol_multi_agent_test():
    """Test multi-agent RAG with protocol enforcement"""
    
    print("🤖 PROTOCOL-ENFORCED MULTI-AGENT RAG TEST")
    print("=" * 70)
    
    try:
        # Import multi-agent orchestrator
        from multi_agent_rag_orchestrator import get_orchestrator
        
        matter_id = 'Corporate Governance'
        orchestrator = get_orchestrator(matter_id)
        pipeline = rag_session_manager.get_or_create_pipeline(matter_id)
        
        test_queries = [
            "What are the main legal claims in this case?",
            "Identify all parties and their roles in this matter",
            "What are the key dates and deadlines mentioned?",
            "Summarize any contract terms or obligations",
            "What compliance or regulatory issues are identified?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 MULTI-AGENT TEST {i}/{len(test_queries)}")
            print(f"Query: {query}")
            print("-" * 50)
            
            try:
                # Process with multi-agent orchestrator
                result = await orchestrator.process_query(query, max_context_chunks=3)
                
                if result.get('answer'):
                    print("✅ Multi-Agent Response Generated")
                    
                    # Now apply MCP protocol enforcement to the result
                    try:
                        from mcp_rag_server import mcp_rag_server
                        
                        # Enforce protocols on the multi-agent result
                        is_compliant, compliance_msg, enforcement_metadata = mcp_rag_server.enforce_protocol_on_response(
                            matter_id, result
                        )
                        
                        compliance_status = "✅ COMPLIANT" if is_compliant else "⚠️ NON-COMPLIANT"
                        print(f"🔒 Protocol Status: {compliance_status}")
                        
                        if compliance_msg:
                            print(f"   Compliance Note: {compliance_msg}")
                        
                        # Show agent breakdown
                        agents_used = result.get('agents_used', [])
                        print(f"🤖 Agents Used: {len(agents_used)}")
                        for agent in agents_used:
                            print(f"   • {agent}")
                        
                        print(f"⏱️ Execution Time: {result.get('execution_time', 'Unknown')}s")
                        
                        # Show answer (truncated)
                        answer = result.get('answer', '')
                        print(f"📄 Answer ({len(answer)} chars):")
                        print(answer[:300] + "..." if len(answer) > 300 else answer)
                        
                        # Show any protocol warnings
                        if enforcement_metadata.get('warnings'):
                            print(f"⚠️ Protocol Warnings:")
                            for warning in enforcement_metadata['warnings']:
                                print(f"   - {warning}")
                        
                        # Audit citations
                        citation_audit = mcp_rag_server.audit_citation_provenance(result)
                        print(f"🔍 Citation Reliability: {citation_audit['overall_reliability']:.2%}")
                        
                    except ImportError:
                        print("⚠️ MCP server not available - no protocol enforcement")
                        print(f"📄 Answer: {result.get('answer', '')[:200]}...")
                        
                else:
                    print("❌ No response generated")
                    print(f"Error: {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                print(f"❌ Multi-agent error: {str(e)}")
                
                # Fallback to single-model with protocol enforcement
                print("🔄 Falling back to single-model with protocols...")
                try:
                    fallback_result = await pipeline.generate_rag_answer(
                        query, "phi3:latest", max_context_chunks=3, temperature=0.1, enforce_protocols=True
                    )
                    
                    compliance_status = "✅ COMPLIANT" if fallback_result.get('protocol_compliance') else "⚠️ NON-COMPLIANT"
                    print(f"🔒 Fallback Protocol Status: {compliance_status}")
                    print(f"📄 Fallback Answer: {fallback_result.get('answer', '')[:200]}...")
                    
                except Exception as fallback_error:
                    print(f"❌ Fallback also failed: {fallback_error}")
            
            print("-" * 70)
            
    except ImportError as e:
        print(f"❌ Multi-agent system not available: {e}")
        print("🔄 Testing single-model with protocol enforcement instead...")
        
        # Test single model with protocols
        await test_single_model_protocols()

async def test_single_model_protocols():
    """Test single model with protocol enforcement as fallback"""
    
    print("\n🔄 SINGLE-MODEL PROTOCOL TEST")
    print("=" * 50)
    
    matter_id = 'Corporate Governance'
    pipeline = rag_session_manager.get_or_create_pipeline(matter_id)
    
    query = "What are the main legal claims and parties in this case?"
    
    try:
        result = await pipeline.generate_rag_answer(
            query, "phi3:latest", max_context_chunks=3, temperature=0.1, enforce_protocols=True
        )
        
        print(f"Query: {query}")
        print(f"Protocol Enforcement: {result.get('protocol_enforcement_requested', 'Unknown')}")
        print(f"Protocol Compliance: {result.get('protocol_compliance', 'Unknown')}")
        
        if result.get('compliance_message'):
            print(f"Compliance Note: {result['compliance_message']}")
        
        print(f"Answer: {result.get('answer', 'No answer')[:300]}...")
        
    except Exception as e:
        print(f"❌ Single-model test failed: {e}")

async def main():
    """Run protocol-enforced multi-agent tests"""
    await protocol_multi_agent_test()

if __name__ == "__main__":
    asyncio.run(main()) 