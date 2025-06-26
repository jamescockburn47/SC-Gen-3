#!/usr/bin/env python3
"""
Strategic Counsel Model Selection Guide
Helps users choose the best model for their RAG queries
"""

def print_model_guide():
    """Print a comprehensive model selection guide"""
    
    print("🧠 Strategic Counsel Model Selection Guide")
    print("=" * 50)
    print()
    
    print("📊 Your Available Models & Recommendations:")
    print()
    
    # Model recommendations based on user's system
    models = {
        "phi3:latest": {
            "size": "2.2 GB",
            "speed": "⚡ FASTEST",
            "use_cases": [
                "Quick questions and testing",
                "Simple document summaries", 
                "Fast entity extraction",
                "Basic legal queries"
            ],
            "pros": ["Fastest response time", "Low memory usage", "Good for iterative testing"],
            "cons": ["Less detailed analysis", "May miss nuanced legal points"],
            "best_for": "Quick answers and exploration"
        },
        
        "deepseek-llm:7b": {
            "size": "4.0 GB", 
            "speed": "🚀 FAST",
            "use_cases": [
                "Detailed legal analysis",
                "Complex document interpretation",
                "Multi-step reasoning",
                "Comprehensive case summaries"
            ],
            "pros": ["Excellent legal reasoning", "Detailed responses", "Good balance of speed/quality"],
            "cons": ["Slower than phi3", "Uses more memory"],
            "best_for": "Primary legal analysis work"
        },
        
        "mistral:latest": {
            "size": "4.1 GB",
            "speed": "🚀 FAST", 
            "use_cases": [
                "Professional document analysis",
                "Contract review",
                "Risk assessment",
                "Citation-heavy responses"
            ],
            "pros": ["Strong citation accuracy", "Professional tone", "Good legal terminology"],
            "cons": ["Can be verbose", "Moderate speed"],
            "best_for": "Professional client-ready analysis"
        },
        
        "mixtral:latest": {
            "size": "26 GB",
            "speed": "🐌 SLOWER",
            "use_cases": [
                "Complex multi-document analysis",
                "Advanced legal reasoning",
                "Comparative analysis",
                "High-stakes legal work"
            ],
            "pros": ["Most sophisticated analysis", "Excellent for complex cases", "Superior reasoning"],
            "cons": ["Much slower", "High memory usage", "May timeout on complex queries"],
            "best_for": "Complex cases requiring deep analysis (use sparingly)"
        }
    }
    
    for model, info in models.items():
        print(f"🤖 **{model}** ({info['size']}) - {info['speed']}")
        print(f"   Best for: {info['best_for']}")
        print(f"   Use cases:")
        for use_case in info['use_cases']:
            print(f"     • {use_case}")
        print(f"   ✅ Pros: {', '.join(info['pros'])}")
        print(f"   ⚠️  Cons: {', '.join(info['cons'])}")
        print()
    
    print("💡 RECOMMENDATIONS BY TASK TYPE:")
    print("-" * 35)
    print("📋 Quick exploration: phi3:latest")
    print("⚖️  Primary legal work: deepseek-llm:7b") 
    print("📄 Professional reports: mistral:latest")
    print("🧠 Complex analysis: mixtral:latest (when time allows)")
    print()
    
    print("🎯 WORKFLOW SUGGESTIONS:")
    print("-" * 25)
    print("1. Start with phi3 for quick understanding")
    print("2. Use deepseek-7b for detailed analysis") 
    print("3. Use mistral for client-ready summaries")
    print("4. Use mixtral only for complex cases")
    print()
    
    print("⚡ PERFORMANCE TIPS:")
    print("-" * 20)
    print("• phi3: ~2-5 seconds per query")
    print("• deepseek-7b: ~5-15 seconds per query")
    print("• mistral: ~8-20 seconds per query") 
    print("• mixtral: ~30-60+ seconds per query")
    print()
    
    print("🚨 AVOID:")
    print("-" * 10)
    print("❌ Using mixtral for simple questions (overkill)")
    print("❌ Using phi3 for complex legal analysis (insufficient)")
    print("❌ Running multiple large models simultaneously")
    print()
    
    print("🎛️  MULTI-AGENT MODE:")
    print("-" * 20)
    print("When multi-agent mode is available, the system automatically:")
    print("• Uses phi3 for quick extraction")
    print("• Uses deepseek-7b for detailed analysis")
    print("• Uses mistral for professional formatting")
    print("• Combines results for comprehensive answers")
    print("💡 This gives you the best of all models automatically!")

if __name__ == "__main__":
    print_model_guide() 