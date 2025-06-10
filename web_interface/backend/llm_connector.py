"""
LLM Connector using OpenRouter's free models
Provides prompt enhancement, summarization, and analysis capabilities
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any
import httpx
from datetime import datetime

# OpenRouter free models (as of 2024)
FREE_MODELS = {
    "mistral-7b": {
        "id": "mistralai/mistral-7b-instruct:free",
        "name": "Mistral 7B Instruct",
        "description": "Fast and capable for most tasks",
        "context_length": 32768,
        "good_for": ["analysis", "summarization", "enhancement"]
    },
    "llama-3.1-8b": {
        "id": "meta-llama/llama-3.1-8b-instruct:free", 
        "name": "Llama 3.1 8B Instruct",
        "description": "Meta's latest instruction-tuned model",
        "context_length": 131072,
        "good_for": ["creative", "analysis", "coding"]
    },
    "qwen-72b": {
        "id": "qwen/qwen-2-72b-instruct:free",
        "name": "Qwen2 72B Instruct", 
        "description": "Large multilingual model",
        "context_length": 32768,
        "good_for": ["complex_analysis", "enhancement", "creative"]
    },
    "gemma-7b": {
        "id": "google/gemma-7b-it:free",
        "name": "Gemma 7B IT",
        "description": "Google's instruction-tuned model",
        "context_length": 8192,
        "good_for": ["analysis", "summarization"]
    }
}

class OpenRouterLLM:
    """OpenRouter LLM connector for prompt enhancement features"""
    
    def __init__(self, api_key: Optional[str] = None, default_model: str = "mistral-7b"):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.default_model = default_model
        self.client = httpx.AsyncClient(timeout=30.0)
        
        if not self.api_key:
            print("⚠️ OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable")
            print("   or get a free key at: https://openrouter.ai/keys")
        
    async def _make_request(self, messages: List[Dict], model: str = None, **kwargs) -> Optional[str]:
        """Make request to OpenRouter API"""
        if not self.api_key:
            return None
            
        model_id = FREE_MODELS.get(model or self.default_model, {}).get("id")
        if not model_id:
            model_id = FREE_MODELS[self.default_model]["id"]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/prompt-library",
            "X-Title": "The Big Everything Prompt Library"
        }
        
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000),
            "top_p": kwargs.get("top_p", 0.9)
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                print(f"OpenRouter API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Error calling OpenRouter API: {e}")
            return None
    
    async def enhance_prompt(self, original_prompt: str, enhancement_type: str = "improve") -> Optional[Dict[str, Any]]:
        """Enhance a prompt using LLM suggestions"""
        
        enhancement_prompts = {
            "improve": """Analyze this AI prompt and suggest improvements. Focus on:
1. Clarity and specificity
2. Better structure and formatting
3. Missing context or instructions
4. More effective phrasing

Original prompt:
{prompt}

Provide your analysis and an improved version.""",
            
            "expand": """Take this AI prompt and expand it with more detailed instructions, examples, and context. Make it more comprehensive while maintaining the original intent.

Original prompt:
{prompt}

Provide an expanded, more detailed version.""",
            
            "variants": """Create 3 different variants of this AI prompt, each with a different approach or style while maintaining the core objective.

Original prompt:
{prompt}

Provide 3 distinct variants with brief explanations.""",
            
            "analyze": """Analyze this AI prompt for effectiveness. Consider:
1. Clarity of instructions
2. Potential ambiguities
3. Missing context
4. Target audience suitability
5. Likely output quality

Original prompt:
{prompt}

Provide detailed analysis and recommendations."""
        }
        
        prompt_template = enhancement_prompts.get(enhancement_type, enhancement_prompts["improve"])
        user_message = prompt_template.format(prompt=original_prompt)
        
        messages = [
            {"role": "system", "content": "You are an expert at analyzing and improving AI prompts. Provide practical, actionable suggestions."},
            {"role": "user", "content": user_message}
        ]
        
        result = await self._make_request(messages, model="llama-3.1-8b")
        
        if result:
            return {
                "enhancement_type": enhancement_type,
                "original_prompt": original_prompt,
                "enhanced_content": result,
                "model_used": FREE_MODELS["llama-3.1-8b"]["name"],
                "timestamp": datetime.now().isoformat()
            }
        
        return None
    
    async def summarize_prompt(self, prompt_content: str, max_length: int = 200) -> Optional[str]:
        """Generate a concise summary of a prompt"""
        
        messages = [
            {"role": "system", "content": f"Summarize AI prompts concisely in {max_length} characters or less. Focus on the main purpose and key features."},
            {"role": "user", "content": f"Summarize this AI prompt:\n\n{prompt_content}"}
        ]
        
        return await self._make_request(messages, model="mistral-7b", max_tokens=100)
    
    async def generate_tags(self, prompt_content: str, title: str = "") -> Optional[List[str]]:
        """Generate relevant tags for a prompt"""
        
        content = f"Title: {title}\n\n{prompt_content}" if title else prompt_content
        
        messages = [
            {"role": "system", "content": "Generate 3-7 relevant tags for AI prompts. Return only comma-separated tags, no explanation."},
            {"role": "user", "content": f"Generate tags for this AI prompt:\n\n{content}"}
        ]
        
        result = await self._make_request(messages, model="mistral-7b", max_tokens=50)
        
        if result:
            # Parse tags from response
            tags = [tag.strip() for tag in result.split(',')]
            # Clean and filter tags
            tags = [tag.lower() for tag in tags if tag and len(tag) > 2 and len(tag) < 20]
            return tags[:7]  # Limit to 7 tags
        
        return None
    
    async def compare_prompts(self, prompt1: str, prompt2: str) -> Optional[str]:
        """Compare two prompts and analyze differences"""
        
        messages = [
            {"role": "system", "content": "Compare two AI prompts and analyze their differences, strengths, and use cases."},
            {"role": "user", "content": f"Compare these two AI prompts:\n\nPrompt 1:\n{prompt1}\n\nPrompt 2:\n{prompt2}"}
        ]
        
        return await self._make_request(messages, model="qwen-72b")
    
    async def suggest_use_cases(self, prompt_content: str) -> Optional[List[str]]:
        """Suggest use cases for a prompt"""
        
        messages = [
            {"role": "system", "content": "Suggest 3-5 specific use cases for AI prompts. Return as numbered list."},
            {"role": "user", "content": f"What are good use cases for this AI prompt?\n\n{prompt_content}"}
        ]
        
        result = await self._make_request(messages, model="mistral-7b")
        
        if result:
            # Parse use cases from numbered list
            use_cases = []
            for line in result.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove numbering and clean up
                    clean_line = line.split('.', 1)[-1].strip() if '.' in line else line[1:].strip()
                    if clean_line:
                        use_cases.append(clean_line)
            return use_cases[:5]
        
        return None
    
    async def detect_prompt_type(self, prompt_content: str) -> Optional[str]:
        """Classify the type/category of a prompt"""
        
        messages = [
            {"role": "system", "content": "Classify AI prompts into categories: coding, writing, analysis, creative, business, education, gaming, roleplay, other. Return only the category name."},
            {"role": "user", "content": f"What category is this AI prompt?\n\n{prompt_content}"}
        ]
        
        result = await self._make_request(messages, model="gemma-7b", max_tokens=10)
        return result.strip().lower() if result else None
    
    async def batch_enhance_prompts(self, prompts: List[Dict[str, str]], enhancement_type: str = "improve") -> List[Dict[str, Any]]:
        """Enhance multiple prompts in batch with rate limiting"""
        results = []
        
        for i, prompt_info in enumerate(prompts):
            prompt_content = prompt_info.get("content", "")
            if not prompt_content:
                continue
                
            print(f"Enhancing prompt {i+1}/{len(prompts)}: {prompt_info.get('title', 'Untitled')[:50]}...")
            
            result = await self.enhance_prompt(prompt_content, enhancement_type)
            if result:
                result.update({
                    "original_id": prompt_info.get("id"),
                    "original_title": prompt_info.get("title", "")
                })
                results.append(result)
            
            # Rate limiting - wait between requests
            if i < len(prompts) - 1:
                await asyncio.sleep(1)
        
        return results
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available free models"""
        return FREE_MODELS
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

# Factory function for optional LLM integration
def create_llm_connector(api_key: Optional[str] = None) -> Optional[OpenRouterLLM]:
    """Create LLM connector if API key is available"""
    try:
        connector = OpenRouterLLM(api_key=api_key)
        if connector.api_key:
            return connector
        else:
            print("LLM connector not initialized - no API key provided")
            return None
    except Exception as e:
        print(f"Could not initialize LLM connector: {e}")
        return None

# Utility functions for prompt enhancement workflows
async def enhance_prompt_workflow(llm: OpenRouterLLM, prompt_content: str, include_all: bool = False) -> Dict[str, Any]:
    """Complete prompt enhancement workflow"""
    if not llm:
        return {}
    
    workflow_results = {}
    
    # Basic enhancement
    enhanced = await llm.enhance_prompt(prompt_content, "improve")
    if enhanced:
        workflow_results["enhancement"] = enhanced
    
    # Generate tags
    tags = await llm.generate_tags(prompt_content)
    if tags:
        workflow_results["suggested_tags"] = tags
    
    # Detect type
    prompt_type = await llm.detect_prompt_type(prompt_content)
    if prompt_type:
        workflow_results["detected_type"] = prompt_type
    
    if include_all:
        # Use cases
        use_cases = await llm.suggest_use_cases(prompt_content)
        if use_cases:
            workflow_results["use_cases"] = use_cases
        
        # Summary
        summary = await llm.summarize_prompt(prompt_content)
        if summary:
            workflow_results["summary"] = summary
        
        # Variants
        variants = await llm.enhance_prompt(prompt_content, "variants")
        if variants:
            workflow_results["variants"] = variants
    
    return workflow_results