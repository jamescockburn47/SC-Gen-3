#!/usr/bin/env python3

import asyncio
from quick_rag_fix import get_non_hallucinating_answer

async def main():
    print("Getting answer about the dispute...")
    result = await get_non_hallucinating_answer(
        'What is this dispute about?', 
        'Corporate Governance', 
        'phi3:latest'
    )
    
    print("\n" + "="*60)
    print("📋 ANSWER:")
    print("="*60)
    print(result['answer'])
    
    print("\n" + "="*60)
    print("📚 SOURCES:")
    print("="*60)
    for i, source in enumerate(result.get('sources', [])):
        print(f"{i+1}. {source['document']} (Similarity: {source['similarity_score']:.3f})")
        print(f"   Preview: {source['text_preview'][:100]}...")
        print()
    
    if result.get('hallucination_flags'):
        print("⚠️ Hallucination flags:", result['hallucination_flags'])

if __name__ == "__main__":
    asyncio.run(main()) 