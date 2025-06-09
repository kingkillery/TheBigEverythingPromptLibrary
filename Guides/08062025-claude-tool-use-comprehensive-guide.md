# Claude Tool Use - Comprehensive Prompting Guide

## Description
Comprehensive guide for implementing tool use with Claude's API, covering tool definition, tool choice strategies, structured data extraction, and advanced use cases. This guide synthesizes practical techniques from Anthropic's official cookbooks for building powerful tool-enabled AI applications.

## Source
Anthropic Cookbook - Tool Use Collection (https://github.com/anthropics/anthropic-cookbook/tree/main/tool_use)

## Overview

Claude's tool use capability allows the model to interact with external functions and APIs, enabling complex workflows, structured data extraction, and dynamic responses. This guide covers all aspects of tool implementation, from basic calculator tools to sophisticated multi-tool orchestration.

**Key Benefits:**
- **Structured Output**: Force specific JSON schemas for reliable data extraction
- **Dynamic Tool Selection**: Intelligent routing between multiple available tools
- **Function Integration**: Seamless connection to external APIs and services
- **Workflow Orchestration**: Chain multiple tools for complex tasks

---

## Core Concepts

### Tool Definition Structure

All tools follow a standardized schema:

```python
tool_definition = {
    "name": "function_name",
    "description": "Clear description of what the tool does and when to use it",
    "input_schema": {
        "type": "object",
        "properties": {
            "parameter_name": {
                "type": "string",  # string, number, array, object, boolean
                "description": "Clear parameter description"
            }
        },
        "required": ["parameter_name"]
    }
}
```

### Tool Choice Options

Claude supports three tool choice strategies:

1. **`auto`** (default): Claude decides whether to use tools
2. **`tool`**: Force Claude to use a specific tool
3. **`any`**: Force Claude to use one of the available tools

---

## Pattern 1: Basic Tool Implementation

### Simple Calculator Tool

```python
def calculate(expression):
    # Remove any non-digit or non-operator characters
    expression = re.sub(r'[^0-9+\-*/().]', '', expression)
    
    try:
        result = eval(expression)  # Note: eval is unsafe in production
        return str(result)
    except (SyntaxError, ZeroDivisionError, NameError, TypeError, OverflowError):
        return "Error: Invalid expression"

calculator_tool = {
    "name": "calculator",
    "description": "A simple calculator that performs basic arithmetic operations.",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate (e.g., '2 + 3 * 4')."
            }
        },
        "required": ["expression"]
    }
}
```

### Tool Processing Function

```python
def process_tool_call(tool_name, tool_input):
    if tool_name == "calculator":
        return calculate(tool_input["expression"])
    # Add more tool handlers here
    
def chat_with_claude(user_message):
    message = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=4096,
        messages=[{"role": "user", "content": user_message}],
        tools=[calculator_tool],
    )

    if message.stop_reason == "tool_use":
        tool_use = next(block for block in message.content if block.type == "tool_use")
        tool_result = process_tool_call(tool_use.name, tool_use.input)
        
        # Send result back to Claude
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4096,
            messages=[
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": message.content},
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": tool_result,
                    }],
                },
            ],
            tools=[calculator_tool],
        )
        return response.content[0].text
    
    return message.content[0].text
```

---

## Pattern 2: Tool Choice Strategies

### Auto Tool Selection

Best for scenarios where Claude should intelligently decide when to use tools:

```python
def chat_with_web_search(user_query):
    system_prompt = f"""
    Answer as many questions as you can using your existing knowledge.  
    Only search the web for queries that you cannot confidently answer.
    Today's date is {date.today().strftime("%B %d %Y")}
    If you think a user's question involves something recent or in the future, use the search tool.
    """

    response = client.messages.create(
        system=system_prompt,
        model="claude-3-sonnet-20240229",
        messages=[{"role": "user", "content": user_query}],
        max_tokens=1000,
        tool_choice={"type": "auto"},  # Let Claude decide
        tools=[web_search_tool]
    )
```

### Forced Tool Usage

Force Claude to always use a specific tool:

```python
# Force Claude to always use sentiment analysis tool
response = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=4096,
    tools=tools,
    tool_choice={"type": "tool", "name": "print_sentiment_scores"},
    messages=[{"role": "user", "content": query}]
)
```

### Any Tool Requirement

Force Claude to use one of the available tools (useful for chatbots that must always take action):

```python
# SMS chatbot that must always send a message or look up info
system_prompt = """
All your communication with a user is done via text message.
Only call tools when you have enough information to accurately call them.  
Do not call the get_customer_info tool until a user has provided you with their username.
If you do not know a user's username, simply ask a user for their username.
"""

response = client.messages.create(
    system=system_prompt,
    model="claude-3-sonnet-20240229",
    max_tokens=4096,
    tools=[send_text_tool, get_customer_info_tool],
    tool_choice={"type": "any"},  # Must use one of the tools
    messages=[{"role": "user", "content": user_message}]
)
```

---

## Pattern 3: Structured Data Extraction

### Article Summarization Tool

```python
article_summary_tool = {
    "name": "print_summary",
    "description": "Prints a summary of the article.",
    "input_schema": {
        "type": "object",
        "properties": {
            "author": {"type": "string", "description": "Name of the article author"},
            "topics": {
                "type": "array",
                "items": {"type": "string"},
                "description": 'Array of topics, e.g. ["tech", "politics"]. Should be as specific as possible.'
            },
            "summary": {"type": "string", "description": "Summary of the article. One or two paragraphs max."},
            "coherence": {"type": "integer", "description": "Coherence of the article's key points, 0-100 (inclusive)"},
            "persuasion": {"type": "number", "description": "Article's persuasion score, 0.0-1.0 (inclusive)"}
        },
        "required": ['author', 'topics', 'summary', 'coherence', 'persuasion']
    }
}
```

### Named Entity Recognition Tool

```python
entity_extraction_tool = {
    "name": "print_entities",
    "description": "Prints extracted named entities.",
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "The extracted entity name."},
                        "type": {"type": "string", "description": "The entity type (e.g., PERSON, ORGANIZATION, LOCATION)."},
                        "context": {"type": "string", "description": "The context in which the entity appears in the text."}
                    },
                    "required": ["name", "type", "context"]
                }
            }
        },
        "required": ["entities"]
    }
}
```

### Sentiment Analysis Tool

```python
sentiment_tool = {
    "name": "print_sentiment_scores",
    "description": "Prints the sentiment scores of a given text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "positive_score": {"type": "number", "description": "The positive sentiment score, ranging from 0.0 to 1.0."},
            "negative_score": {"type": "number", "description": "The negative sentiment score, ranging from 0.0 to 1.0."},
            "neutral_score": {"type": "number", "description": "The neutral sentiment score, ranging from 0.0 to 1.0."}
        },
        "required": ["positive_score", "negative_score", "neutral_score"]
    }
}
```

---

## Pattern 4: Dynamic Schema Tools

For cases where you don't know the exact JSON structure upfront:

```python
dynamic_extraction_tool = {
    "name": "print_all_characteristics",
    "description": "Prints all characteristics which are provided.",
    "input_schema": {
        "type": "object",
        "additionalProperties": True  # Allow any properties
    }
}

# Usage prompt
query = """Given a description of a character, your task is to extract all the characteristics of the character and print them using the print_all_characteristics tool.

The print_all_characteristics tool takes an arbitrary number of inputs where the key is the characteristic name and the value is the characteristic value (age: 28 or eye_color: green).

<description>
The man is tall, with a beard and a scar on his left cheek. He has a deep voice and wears a black leather jacket.
</description>

Now use the print_all_characteristics tool."""
```

---

## Pattern 5: Customer Service Agent

### Multi-Tool Service Bot

```python
customer_service_tools = [
    {
        "name": "send_text_to_user",
        "description": "Sends a text message to a user",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The piece of text to be sent to the user via text message"},
            },
            "required": ["text"]
        }
    },
    {
        "name": "get_customer_info",
        "description": "Gets information on a customer based on the customer's username. Response includes email, username, and previous purchases. Only call this tool once a user has provided you with their username",
        "input_schema": {
            "type": "object",
            "properties": {
                "username": {"type": "string", "description": "The username of the user in question."},
            },
            "required": ["username"]
        }
    }
]

system_prompt = """
All your communication with a user is done via text message.
Only call tools when you have enough information to accurately call them.  
Do not call the get_customer_info tool until a user has provided you with their username. This is important.
If you do not know a user's username, simply ask a user for their username.
"""
```

---

## Pattern 6: Memory and Context Management

### Conversation Memory Tool

```python
memory_tool = {
    "name": "update_memory",
    "description": "Update the conversation memory with important information",
    "input_schema": {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Memory key (e.g., 'user_preferences', 'project_status')"},
            "value": {"type": "string", "description": "Value to store"},
            "action": {"type": "string", "enum": ["store", "retrieve", "update"], "description": "Action to perform"}
        },
        "required": ["key", "action"]
    }
}

def process_memory_tool(key, value, action, memory_store={}):
    if action == "store":
        memory_store[key] = value
        return f"Stored {key}: {value}"
    elif action == "retrieve":
        return memory_store.get(key, "No information found")
    elif action == "update":
        if key in memory_store:
            memory_store[key] = value
            return f"Updated {key}: {value}"
        else:
            memory_store[key] = value
            return f"Created new entry {key}: {value}"
```

---

## Pattern 7: Parallel Tool Execution

### Multiple Tool Calls

For Claude models that support parallel tool execution:

```python
# Example tools that can be called in parallel
tools = [
    weather_tool,
    calendar_tool, 
    email_tool
]

response = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=4096,
    tools=tools,
    tool_choice={"type": "auto"},
    messages=[{"role": "user", "content": "Check my calendar for today, get the weather forecast, and send an email to my team about the meeting."}]
)

# Process multiple tool calls
for content_block in response.content:
    if content_block.type == "tool_use":
        tool_result = process_tool_call(content_block.name, content_block.input)
        print(f"Tool: {content_block.name}, Result: {tool_result}")
```

---

## Pattern 8: Vision + Tools Integration

### Image Analysis with Tools

```python
image_analysis_tool = {
    "name": "analyze_image_content",
    "description": "Analyzes and categorizes content found in images",
    "input_schema": {
        "type": "object",
        "properties": {
            "objects": {"type": "array", "items": {"type": "string"}, "description": "List of objects detected in the image"},
            "scene": {"type": "string", "description": "Overall scene description"},
            "text_detected": {"type": "string", "description": "Any text found in the image"},
            "dominant_colors": {"type": "array", "items": {"type": "string"}, "description": "Dominant colors in the image"}
        },
        "required": ["objects", "scene"]
    }
}

# Usage with image
response = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=4096,
    tools=[image_analysis_tool],
    tool_choice={"type": "tool", "name": "analyze_image_content"},
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Analyze this image and extract structured information using the analyze_image_content tool."},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image}}
        ]
    }]
)
```

---

## Best Practices

### Tool Design Principles

1. **Clear Naming**: Use descriptive tool names that indicate their purpose
   - ✅ `get_weather_forecast`
   - ❌ `weather_tool`

2. **Detailed Descriptions**: Include comprehensive descriptions
   ```python
   "description": "Retrieves current weather conditions and 5-day forecast for a specific location. Use when users ask about weather, temperature, precipitation, or atmospheric conditions."
   ```

3. **Specific Parameter Types**: Use appropriate data types and constraints
   ```python
   "temperature": {
       "type": "number",
       "minimum": -100,
       "maximum": 150,
       "description": "Temperature in Celsius"
   }
   ```

4. **Required vs Optional**: Clearly mark required parameters
   ```python
   "required": ["location"],  # Don't make everything required
   ```

### Prompt Engineering for Tools

1. **System Prompts**: Provide clear tool usage guidelines
   ```python
   system_prompt = """
   You have access to several tools. Use them when:
   - User asks for real-time information (use web_search)
   - User requests calculations (use calculator)
   - User needs data extraction (use appropriate extraction tool)
   
   Only use tools when necessary. If you can answer with your training data, do so.
   """
   ```

2. **Tool Choice Guidance**: Be explicit about when to use `auto` vs `tool` vs `any`
   - **`auto`**: Default for most conversational scenarios
   - **`tool`**: When you need guaranteed structured output
   - **`any`**: For action-required scenarios (chatbots, APIs)

3. **Error Handling**: Design tools to handle edge cases gracefully
   ```python
   def safe_calculator(expression):
       try:
           # Validate and sanitize input
           if not re.match(r'^[0-9+\-*/.() ]+$', expression):
               return "Error: Invalid characters in expression"
           result = eval(expression)
           return str(result)
       except:
           return "Error: Could not evaluate expression"
   ```

### Performance Optimization

1. **Tool Response Speed**: Keep tool functions lightweight
2. **Batch Operations**: Combine related operations when possible
3. **Caching**: Cache expensive operations (API calls, computations)
4. **Fallback Strategies**: Handle tool failures gracefully

---

## Common Use Cases

### 1. Data Extraction Pipeline
```python
# Chain: Web scraping → Entity extraction → Sentiment analysis → Summary
tools = [web_scraper_tool, entity_tool, sentiment_tool, summary_tool]
```

### 2. Customer Support Automation
```python
# Tools: Knowledge base search, ticket creation, user lookup, email sending
tools = [kb_search_tool, ticket_tool, user_lookup_tool, email_tool]
```

### 3. Content Analysis Workflow
```python
# Tools: PDF parser, text classifier, keyword extractor, report generator  
tools = [pdf_tool, classifier_tool, keyword_tool, report_tool]
```

### 4. Research Assistant
```python
# Tools: Web search, academic database, citation formatter, note taker
tools = [web_search_tool, academic_tool, citation_tool, notes_tool]
```

---

## Advanced Patterns

### Tool Chaining
```python
def chain_tools(user_input):
    # Step 1: Extract entities
    entities = call_tool("extract_entities", {"text": user_input})
    
    # Step 2: Enrich entities with additional data
    enriched_data = call_tool("enrich_entities", {"entities": entities})
    
    # Step 3: Generate final report
    report = call_tool("generate_report", {"data": enriched_data})
    
    return report
```

### Conditional Tool Selection
```python
def smart_tool_router(query, available_tools):
    # Analyze query to determine best tool
    if "weather" in query.lower():
        return ["weather_tool"]
    elif "calculate" in query.lower() or any(op in query for op in ["+", "-", "*", "/"]):
        return ["calculator_tool"]
    elif "search" in query.lower():
        return ["web_search_tool"]
    else:
        return available_tools  # Let Claude decide
```

### Error Recovery
```python
def robust_tool_call(tool_name, tool_input, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = process_tool_call(tool_name, tool_input)
            return result
        except Exception as e:
            if attempt == max_retries - 1:
                return f"Tool failed after {max_retries} attempts: {str(e)}"
            # Maybe modify input or try different approach
            time.sleep(1)
```

---

## Troubleshooting

### Common Issues

1. **Claude not using tools when expected**
   - Check tool descriptions are clear and specific
   - Verify system prompt encourages tool use
   - Consider using `tool_choice="any"` or specific tool forcing

2. **Tools being used unnecessarily**
   - Improve system prompt with clear guidelines
   - Use `tool_choice="auto"` with better context
   - Add examples of when NOT to use tools

3. **Malformed tool inputs**
   - Strengthen input schema validation
   - Add example inputs in tool descriptions
   - Use more specific parameter descriptions

4. **Tool execution errors**
   - Add comprehensive error handling
   - Validate inputs before processing
   - Provide meaningful error messages

### Debugging Tips

1. **Log tool calls**: Track which tools are called and with what inputs
2. **Test tool schemas**: Validate JSON schemas work as expected
3. **Monitor performance**: Track tool execution times and success rates
4. **A/B test prompts**: Compare different system prompts and tool descriptions

---

## Security Considerations

### Input Validation
```python
def safe_eval_calculator(expression):
    # Whitelist allowed characters
    if not re.match(r'^[0-9+\-*/.() ]+$', expression):
        return "Error: Invalid characters"
    
    # Limit expression length
    if len(expression) > 100:
        return "Error: Expression too long"
    
    # Use ast.literal_eval for safer evaluation when possible
    try:
        # For simple cases, use literal_eval
        result = ast.literal_eval(expression)
        return str(result)
    except:
        # Fall back to limited eval with timeout
        return safe_eval_with_timeout(expression)
```

### API Rate Limiting
```python
def rate_limited_tool(func, max_calls_per_minute=60):
    calls = []
    
    def wrapper(*args, **kwargs):
        now = time.time()
        # Remove calls older than 1 minute
        calls[:] = [t for t in calls if now - t < 60]
        
        if len(calls) >= max_calls_per_minute:
            return "Error: Rate limit exceeded"
        
        calls.append(now)
        return func(*args, **kwargs)
    
    return wrapper
```

---

## Integration Examples

### Flask API Integration
```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    user_message = request.json.get('message')
    
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=4096,
        tools=available_tools,
        tool_choice={"type": "auto"},
        messages=[{"role": "user", "content": user_message}]
    )
    
    # Process any tool calls
    if response.stop_reason == "tool_use":
        # Handle tool execution
        result = process_tools(response)
        return jsonify({"response": result})
    
    return jsonify({"response": response.content[0].text})
```

### Async Tool Processing
```python
import asyncio

async def async_tool_processor(tool_calls):
    tasks = []
    for tool_call in tool_calls:
        task = asyncio.create_task(
            process_tool_async(tool_call.name, tool_call.input)
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results
```

---

## Conclusion

Tool use with Claude enables powerful integrations and structured workflows. Key success factors:

1. **Clear tool definitions** with specific descriptions and schemas
2. **Appropriate tool choice** strategy for your use case
3. **Robust error handling** and input validation
4. **Performance optimization** for production environments
5. **Security considerations** for safe tool execution

Start with simple tools and gradually build complexity as you understand Claude's tool use patterns. The flexibility of the tool system allows for endless possibilities in creating intelligent, interactive applications.

---

## Additional Resources

- **Tool Choice Strategies**: Understanding when to use auto vs tool vs any
- **Structured JSON Extraction**: Advanced patterns for data extraction
- **Memory Management**: Building persistent context across conversations
- **Parallel Tool Execution**: Optimizing multi-tool workflows
- **Vision + Tools**: Combining image analysis with tool capabilities
- **Error Handling**: Robust tool execution patterns
- **Security**: Safe tool implementation practices

For more advanced examples and specific use cases, refer to the individual cookbook notebooks in the Anthropic cookbook repository.
