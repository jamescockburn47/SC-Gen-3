#!/usr/bin/env python3
"""
Quick LLM Connection Test
Tests Ollama models for responsiveness and basic functionality
"""

import asyncio
import aiohttp
import time
import json
from typing import Dict, Any

class OllamaTestSuite:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        
    async def test_connection(self) -> bool:
        """Test basic Ollama connection"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags", timeout=aiohttp.ClientTimeout(total=10)) as response:
                    return response.status == 200
        except:
            return False
    
    async def get_models(self) -> list:
        """Get available models"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('models', [])
            return []
        except:
            return []
    
    async def test_model_response(self, model_name: str, timeout: int = 60) -> Dict[str, Any]:
        """Test a specific model with a simple prompt"""
        
        test_prompt = "Hello! Please respond with exactly these words: 'Model test successful'"
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": model_name,
                    "prompt": test_prompt,
                    "temperature": 0.1,
                    "stream": False
                }
                
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    
                    end_time = time.time()
                    response_time = end_time - start_time
                    
                    if response.status == 200:
                        result = await response.json()
                        answer = result.get('response', '').strip()
                        
                        return {
                            'success': True,
                            'response_time': response_time,
                            'answer': answer,
                            'answer_length': len(answer),
                            'correct_response': 'Model test successful' in answer
                        }
                    else:
                        error_text = await response.text()
                        return {
                            'success': False,
                            'error': f"HTTP {response.status}: {error_text}",
                            'response_time': response_time
                        }
                        
        except asyncio.TimeoutError:
            return {
                'success': False,
                'error': f'Timeout after {timeout} seconds',
                'response_time': timeout
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response_time': time.time() - start_time
            }
    
    async def run_comprehensive_test(self):
        """Run comprehensive test of all models"""
        
        print("🧪 Ollama LLM Test Suite")
        print("=" * 50)
        
        # Test connection
        print("🔗 Testing Ollama connection...")
        if not await self.test_connection():
            print("❌ Failed to connect to Ollama!")
            print(f"   Make sure Ollama is running on {self.base_url}")
            return
        
        print("✅ Ollama connection successful!")
        
        # Get models
        print("\n📋 Getting available models...")
        models = await self.get_models()
        
        if not models:
            print("❌ No models found!")
            return
        
        print(f"✅ Found {len(models)} models:")
        for model in models:
            size_gb = model.get('size', 0) / (1024**3)
            print(f"   • {model['name']} ({size_gb:.1f} GB)")
        
        # Test each model
        print("\n🚀 Testing model responses...")
        
        results = {}
        
        for model in models:
            model_name = model['name']
            print(f"\n🧠 Testing {model_name}...")
            
            # Determine timeout based on model size
            model_size_gb = model.get('size', 0) / (1024**3)
            timeout = 30 if model_size_gb < 10 else 60 if model_size_gb < 30 else 120
            
            result = await self.test_model_response(model_name, timeout)
            results[model_name] = result
            
            if result['success']:
                print(f"✅ Response in {result['response_time']:.1f}s")
                if result['correct_response']:
                    print("   ✅ Correct response format")
                else:
                    print("   ⚠️  Unexpected response format")
                print(f"   💬 Answer: {result['answer'][:100]}...")
            else:
                print(f"❌ Failed: {result['error']}")
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        successful_models = [name for name, result in results.items() if result['success']]
        failed_models = [name for name, result in results.items() if not result['success']]
        
        print(f"✅ Successful models: {len(successful_models)}/{len(models)}")
        print(f"❌ Failed models: {len(failed_models)}")
        
        if successful_models:
            print("\n🏆 Fastest models:")
            sorted_models = sorted(
                [(name, results[name]['response_time']) for name in successful_models],
                key=lambda x: x[1]
            )
            for name, time_taken in sorted_models[:3]:
                print(f"   • {name}: {time_taken:.1f}s")
        
        if failed_models:
            print("\n⚠️  Failed models:")
            for name in failed_models:
                print(f"   • {name}: {results[name]['error']}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS")
        print("-" * 30)
        
        if successful_models:
            fastest = min(successful_models, key=lambda x: results[x]['response_time'])
            print(f"🚀 Fastest model: {fastest} ({results[fastest]['response_time']:.1f}s)")
            
            # Find best balance of speed and size
            balanced_models = [
                (name, results[name]['response_time']) 
                for name in successful_models 
                if results[name]['response_time'] < 30
            ]
            
            if balanced_models:
                balanced = min(balanced_models, key=lambda x: x[1])
                print(f"⚖️  Recommended for RAG: {balanced[0]}")
        
        print("\n🎯 Next steps:")
        print("   1. Use fastest model for quick testing")
        print("   2. Use larger models for complex legal analysis")
        print("   3. Configure RAG system with working models")

async def main():
    """Run the test suite"""
    tester = OllamaTestSuite()
    await tester.run_comprehensive_test()

if __name__ == "__main__":
    asyncio.run(main()) 