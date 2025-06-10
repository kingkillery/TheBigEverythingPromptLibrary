# 🤖 LLM Enhancement Features Setup

The web interface includes powerful AI enhancement features using OpenRouter's free models to analyze, improve, and generate insights about prompts.

## Features

🔍 **Prompt Analysis** - Comprehensive analysis of prompt effectiveness  
✨ **Prompt Improvement** - AI-powered suggestions for better prompts  
🔄 **Variants Generation** - Create different versions of prompts  
🏷️ **Tag Generation** - Automatically generate relevant tags  
📊 **Use Case Suggestions** - Discover new applications for prompts  
🆚 **Prompt Comparison** - Compare effectiveness of different prompts  

## Setup Instructions

### 1. Get a Free OpenRouter API Key

1. Visit [OpenRouter.ai](https://openrouter.ai/keys)
2. Sign up for a free account
3. Generate an API key
4. Copy your API key

### 2. Set Environment Variable

**Option A: Set in your terminal session**
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

**Option B: Create a .env file**
```bash
cd web_interface
echo "OPENROUTER_API_KEY=your-api-key-here" > .env
```

**Option C: Set system-wide (Windows)**
```cmd
setx OPENROUTER_API_KEY "your-api-key-here"
```

### 3. Start the Server

```bash
python start_server.py
```

If the API key is set correctly, you'll see:
```
✅ LLM connector initialized with 4 free models
```

## Available Free Models

The connector uses these free models from OpenRouter:

- **Mistral 7B Instruct** - Fast and capable for most tasks
- **Llama 3.1 8B Instruct** - Meta's latest instruction-tuned model  
- **Qwen2 72B Instruct** - Large multilingual model
- **Gemma 7B IT** - Google's instruction-tuned model

## Usage

### In the Web Interface

1. Search for any prompt
2. Click the **🤖 Enhance** button next to any result
3. Choose from available enhancement options:
   - **🔍 Analyze Prompt** - Get detailed analysis and recommendations
   - **✨ Improve** - Get an improved version of the prompt
   - **🔄 Create Variants** - Generate 3 different versions
   - **🏷️ Generate Tags** - Get relevant tags for the prompt

### API Endpoints

**Enhance a prompt:**
```bash
curl -X POST "http://localhost:8000/api/llm/enhance" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Your prompt here", "type": "improve"}'
```

**Analyze a prompt:**
```bash
curl -X POST "http://localhost:8000/api/llm/analyze" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Your prompt here", "include_all": true}'
```

**Generate tags:**
```bash
curl -X POST "http://localhost:8000/api/llm/generate-tags" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Your prompt here", "title": "Optional title"}'
```

## Enhancement Types

### `improve`
- Analyzes clarity and specificity
- Suggests better structure and formatting
- Identifies missing context or instructions
- Provides more effective phrasing

### `expand`
- Adds detailed instructions and examples
- Provides more comprehensive context
- Maintains original intent while adding depth

### `variants`
- Creates 3 different approaches to the same objective
- Explores different styles and methodologies
- Maintains core purpose with varied execution

### `analyze`
- Evaluates prompt effectiveness
- Identifies potential ambiguities
- Assesses target audience suitability
- Provides actionable recommendations

## Cost Information

**All models used are FREE** with reasonable rate limits:
- No API charges for the free tier
- Rate limited to prevent abuse
- Automatically handles retries and error cases

## Troubleshooting

### "LLM connector not available"
- Ensure `httpx` is installed: `pip install httpx`
- Check that the LLM connector file exists

### "LLM connector not initialized - no API key provided"  
- Set the `OPENROUTER_API_KEY` environment variable
- Restart the server after setting the key

### "OpenRouter API error: 401"
- Check that your API key is correct
- Ensure the key is properly set in the environment

### "OpenRouter API error: 429"
- You've hit the rate limit
- Wait a few minutes before trying again
- Free tier has usage limits per hour

### Enhancement requests timing out
- Free models may be busy during peak times
- Try again in a few minutes
- The system automatically retries failed requests

## Security Notes

- API keys are only stored in environment variables
- No API keys are logged or stored in files
- All requests are made directly to OpenRouter's secure endpoints
- No prompt content is stored permanently

## Rate Limits

The free models have generous but finite limits:
- ~50-100 requests per hour per model
- ~1000 tokens per request maximum
- Automatic switching between models if one is overloaded

## Privacy

- Prompts are sent to OpenRouter for processing
- OpenRouter's privacy policy applies to API usage
- No conversation history is stored locally
- Consider using generic examples for sensitive prompts