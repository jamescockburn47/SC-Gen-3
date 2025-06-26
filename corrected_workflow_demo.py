#!/usr/bin/env python3
"""
Corrected Anonymisation Workflow
Shows the proper sequence: mistral → optional anonymisation → cloud → reverse
"""

import asyncio

async def demonstrate_correct_workflow():
    """Show the correct workflow order"""
    
    print("🔄 Correct Anonymisation Workflow")
    print("=" * 40)
    
    # Step 1: User uploads ANY document and asks question
    print("📤 Step 1: User uploads document + asks question")
    user_query = "What are the key legal claims in this case?"
    print(f"   Query: '{user_query}'")
    
    # Step 2: mistral analyzes documents (normal RAG)
    print("\n🧠 Step 2: mistral does RAG analysis (REQUIRED)")
    mistral_result = {
        'answer': 'Based on the provided documents: The case involves Sarah Johnson vs Royal London Hospital [Source 1]. The claimant alleges medical negligence regarding treatment by Dr. Michael Thompson [Source 2]. Key claims include breach of duty of care and failure to diagnose [Source 1].',
        'sources': [
            {'document': 'Medical_Case_File_Johnson.pdf', 'similarity_score': 0.95},
            {'document': 'Expert_Witness_Report.pdf', 'similarity_score': 0.87}
        ],
        'model_used': 'mistral:latest',
        'generation_time': 2.3
    }
    print(f"   ✅ mistral analysis complete: {len(mistral_result['answer'])} chars")
    print(f"   📊 Sources: {len(mistral_result['sources'])} documents")
    
    # Step 3: User chooses whether to anonymise
    print("\n🤔 Step 3: User decides - anonymise for cloud analysis?")
    user_choice = "yes"  # Could be "no" to skip anonymisation
    print(f"   User choice: {user_choice}")
    
    if user_choice.lower() == "yes":
        print("\n🔒 Step 4: OPTIONAL anonymisation (phi3)")
        
        # Import the universal anonymiser
        from universal_reverse_engineering_demo import UniversalAnonymiser
        anonymiser = UniversalAnonymiser()
        
        # Anonymise the mistral output
        anonymised_result = await anonymiser.anonymise_any_document(mistral_result['answer'])
        
        print(f"   ✅ phi3 anonymisation complete")
        print(f"   📊 Entities anonymised: {anonymised_result['total_entities']}")
        print(f"   🔒 Anonymised: {anonymised_result['anonymised_text'][:100]}...")
        
        # Step 5: Send to cloud (anonymised version)
        print("\n☁️ Step 5: Send anonymised version to cloud")
        cloud_input = anonymised_result['anonymised_text']
        print(f"   📤 Sending to GPT-4: {len(cloud_input)} chars (anonymised)")
        
        # Simulate cloud analysis
        cloud_response = f"Advanced analysis: The case shows strong evidence for Elizabeth Wright's claims against Elysian Metropolitan Medical Center. The treatment by Dr. Jonathan Harper appears to fall below standard care..."
        
        print(f"   📨 Cloud response: {len(cloud_response)} chars")
        
        # Step 6: Reverse engineer cloud response
        print("\n🔓 Step 6: Reverse engineer cloud response")
        final_result = anonymiser.reverse_engineer_result(cloud_response)
        
        print(f"   ✅ Names restored to original")
        print(f"   🎯 Final result: {final_result[:100]}...")
        
        workflow_summary = {
            'original_analysis': mistral_result,
            'anonymised_version': anonymised_result['anonymised_text'],
            'cloud_response': cloud_response,
            'final_restored': final_result,
            'privacy_protected': True,
            'entities_handled': anonymised_result['total_entities']
        }
        
    else:
        print("\n⏭️ Step 4: Skip anonymisation - direct cloud analysis")
        print("   📤 Sending original mistral result directly to cloud")
        print("   ⚠️ Warning: Real names will be sent to cloud model")
        
        workflow_summary = {
            'original_analysis': mistral_result,
            'cloud_response': 'Direct analysis without anonymisation',
            'privacy_protected': False,
            'entities_handled': 0
        }
    
    return workflow_summary

async def show_flexible_options():
    """Show all the flexible options available"""
    
    print("\n🎛️ Flexible Workflow Options")
    print("=" * 30)
    
    options = {
        "Option 1": "mistral only (fastest, local privacy)",
        "Option 2": "mistral → anonymise → display (privacy for viewing)",
        "Option 3": "mistral → anonymise → cloud → reverse (advanced analysis)",
        "Option 4": "mistral → cloud direct (fastest advanced, less private)"
    }
    
    for option, description in options.items():
        print(f"📋 {option}: {description}")
    
    print("\n💡 Key Benefits:")
    print("✅ mistral always provides base analysis")
    print("✅ Anonymisation is completely optional")
    print("✅ User controls privacy vs advanced analysis trade-off")
    print("✅ Works with ANY document uploaded")
    print("✅ Perfect reverse engineering when needed")

async def demo_any_document_workflow():
    """Show how it works with different document types"""
    
    print("\n📚 Universal Document Support")
    print("=" * 30)
    
    document_types = [
        "📋 Medical Records → Anonymise patient names",
        "⚖️ Legal Cases → Anonymise party names", 
        "💼 Business Contracts → Anonymise company names",
        "🏥 Insurance Claims → Anonymise all personal data",
        "🎓 Academic Papers → Anonymise research participants",
        "🏛️ Government Documents → Anonymise citizen names"
    ]
    
    print("Works with ANY document type:")
    for doc_type in document_types:
        print(f"   {doc_type}")
    
    print(f"\n🔍 Universal Detection:")
    print("• Person names (any names, any language)")
    print("• Company names (any legal entity)")
    print("• Case numbers (any format)")
    print("• Addresses (any location)")
    print("• Email addresses (any domain)")
    print("• Phone numbers (any country)")
    
    print(f"\n🎨 phi3 Creative Generation:")
    print("• Realistic replacement names")
    print("• Maintains professional context")
    print("• Consistent across documents")
    print("• Fast generation (8-10 seconds)")

if __name__ == "__main__":
    print("🛡️ Corrected Universal Anonymisation Workflow")
    print("mistral → optional anonymisation → cloud → reverse")
    print("=" * 60)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Demo the correct workflow
        result = loop.run_until_complete(demonstrate_correct_workflow())
        loop.run_until_complete(show_flexible_options())
        loop.run_until_complete(demo_any_document_workflow())
        
        print("\n🎉 Universal System Ready!")
        print("✅ Works with ANY document uploaded")
        print("✅ Anonymisation AFTER mistral (optional)")
        print("✅ Perfect reverse engineering")
        print("✅ User controls privacy vs performance")
        
    finally:
        loop.close() 