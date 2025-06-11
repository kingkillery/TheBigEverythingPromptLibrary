# Handling Function Calls with Reasoning Models

*Source: OpenAI Cookbook - https://cookbook.openai.com/examples/reasoning_function_calls*

## Overview

This guide demonstrates how to effectively combine function calling with OpenAI's reasoning models (like o3 and o4-mini) for complex, multi-step tasks involving external data sources. Reasoning models think before they answer, producing a long internal chain of thought before responding, making them particularly well-suited for complex workflows.

## Key Characteristics of Reasoning Models

### What Are Reasoning Models?
- Reasoning models are LLMs trained with reinforcement learning to perform reasoning
- They think before they answer, producing a long internal chain of thought before responding
- They excel at combining reasoning capabilities with agentic tool use
- Examples include o3 and o4-mini models

### Native Tool Use in Chain of Thought
- o3/o4-mini models are trained to use tools natively within their chain of thought (CoT)
- This unlocks improved reasoning capabilities around when and how to use tools
- They represent a significant step forward in tool calling capabilities

## Multi-Step Function Call Handling

### The Challenge
Reasoning models may produce a sequence of function calls that must be made in series, where some steps may depend on the results of previous ones.

### General Pattern for Complex Reasoning Workflows

At each step in the conversation, implement this pattern:

1. **Initialize a loop**
2. **Check for function calls**: If the response contains function calls, assume reasoning is ongoing
3. **Feed results back**: Feed function results and any intermediate reasoning back into the model
4. **Continue until completion**: If there are no function calls and you receive a Response.output with type 'message', the agent has finished

### Core Implementation Pattern

```python
def handle_reasoning_with_tools(messages, tools, tool_mapping):
    """
    Handle multi-step reasoning with function calls
    """
    conversation_history = messages.copy()
    
    while True:
        # Make API call
        response = client.responses.create(
            model="o4-mini",
            messages=conversation_history,
            tools=tools
        )
        
        # Check if response contains function calls
        if hasattr(response, 'function_calls') and response.function_calls:
            # Execute function calls
            function_results = invoke_functions_from_response(response, tool_mapping)
            
            # Add function call and results to conversation history
            conversation_history.extend(function_results)
            
            # Continue reasoning loop
            continue
        else:
            # No more function calls - reasoning complete
            return response
```

### Function Execution Helper

```python
def invoke_functions_from_response(response, tool_mapping):
    """
    Extract and execute function calls from response
    """
    results = []
    
    for function_call in response.function_calls:
        function_name = function_call.name
        function_args = json.loads(function_call.arguments)
        
        # Execute the function
        if function_name in tool_mapping:
            try:
                result = tool_mapping[function_name](**function_args)
                results.append({
                    "role": "tool",
                    "content": str(result),
                    "tool_call_id": function_call.id
                })
            except Exception as e:
                results.append({
                    "role": "tool",
                    "content": f"Error: {str(e)}",
                    "tool_call_id": function_call.id
                })
    
    return results
```

## Using the Responses API with Reasoning Models

### Preserving Reasoning History
It's essential to preserve any reasoning and function call responses in conversation history. This is how the model keeps track of what chain-of-thought steps it has run through.

### Previous Response ID Pattern
```python
def use_previous_response_pattern(previous_response_id, new_message):
    """
    Use previous_response_id to maintain reasoning context
    """
    response = client.responses.create(
        model="o4-mini",
        messages=[{"role": "user", "content": new_message}],
        previous_response_id=previous_response_id,  # Maintains reasoning context
        tools=tools
    )
    return response
```

### Explicit Reasoning Items
```python
def explicit_reasoning_items(messages, reasoning_items):
    """
    Explicitly include reasoning items in the conversation
    """
    enhanced_messages = messages.copy()
    
    # Add reasoning items to input
    for item in reasoning_items:
        enhanced_messages.append({
            "role": "assistant",
            "content": None,
            "reasoning": item["content"]
        })
    
    response = client.responses.create(
        model="o4-mini",
        messages=enhanced_messages,
        tools=tools
    )
    return response
```

## Best Practices for Function Descriptions

### Critical Clarity Requirements
Function description clarity becomes critical with reasoning models:

```python
# Good example - clear and specific
{
    "name": "check_transaction_eligibility",
    "description": "Check if a customer is eligible for a specific transaction type based on account status, balance, and transaction history. Use this when determining transaction permissions.",
    "parameters": {
        "type": "object",
        "properties": {
            "customer_id": {
                "type": "string",
                "description": "Unique identifier for the customer account"
            },
            "transaction_type": {
                "type": "string",
                "enum": ["withdrawal", "transfer", "payment"],
                "description": "Type of transaction to check eligibility for"
            },
            "amount": {
                "type": "number",
                "description": "Transaction amount in dollars"
            }
        },
        "required": ["customer_id", "transaction_type", "amount"]
    }
}

# Poor example - vague and overlapping
{
    "name": "check_account",
    "description": "Check account information",  # Too vague
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "ID"  # Unclear what kind of ID
            }
        }
    }
}
```

### Tool List Considerations
- **Size impact**: Longer tool lists can affect latency and reasoning depth
- **Performance**: While o3/o4-mini can handle extensive tool lists, performance can degrade if schema clarity isn't sharp
- **Organization**: Group related tools and use clear naming conventions

## Example Use Cases

### Customer Transaction Processing
```python
# Example: Multi-step customer eligibility check
tools = [
    {
        "name": "get_customer_info",
        "description": "Retrieve customer account details and status"
    },
    {
        "name": "check_transaction_limits",
        "description": "Verify transaction against account limits"
    },
    {
        "name": "fraud_detection_check",
        "description": "Run fraud detection analysis on transaction"
    },
    {
        "name": "process_transaction",
        "description": "Execute the approved transaction"
    }
]

# The reasoning model will:
# 1. Get customer info
# 2. Check transaction limits based on customer data
# 3. Run fraud detection if limits are OK
# 4. Process transaction if all checks pass
```

### Scientific Research Workflow
```python
# Example: Complex research query requiring multiple data sources
tools = [
    {
        "name": "search_pubmed",
        "description": "Search medical literature in PubMed database"
    },
    {
        "name": "query_clinical_trials",
        "description": "Search ongoing and completed clinical trials"
    },
    {
        "name": "analyze_data_trends",
        "description": "Perform statistical analysis on retrieved data"
    },
    {
        "name": "generate_summary_report",
        "description": "Create comprehensive research summary"
    }
]

# The reasoning model will:
# 1. Search relevant literature
# 2. Find related clinical trials
# 3. Analyze trends across data sources
# 4. Generate comprehensive report
```

## Performance Optimization

### Reasoning Items Management
```python
def optimize_reasoning_context(reasoning_items, max_items=10):
    """
    Keep only the most recent reasoning items to manage context size
    """
    if len(reasoning_items) > max_items:
        return reasoning_items[-max_items:]
    return reasoning_items
```

### Tool Selection Optimization
```python
def dynamic_tool_selection(query, all_tools):
    """
    Dynamically select relevant tools based on query content
    """
    relevant_tools = []
    query_lower = query.lower()
    
    for tool in all_tools:
        if any(keyword in query_lower for keyword in tool.get('keywords', [])):
            relevant_tools.append(tool)
    
    return relevant_tools[:5]  # Limit to 5 most relevant tools
```

## Error Handling and Fallbacks

### Robust Error Handling
```python
def safe_function_execution(function_name, function_args, tool_mapping):
    """
    Execute function with comprehensive error handling
    """
    try:
        if function_name not in tool_mapping:
            return f"Error: Function '{function_name}' not found"
        
        result = tool_mapping[function_name](**function_args)
        return result
        
    except TypeError as e:
        return f"Error: Invalid arguments for {function_name}: {str(e)}"
    except Exception as e:
        return f"Error executing {function_name}: {str(e)}"
```

### Fallback Strategies
```python
def implement_fallback_strategy(failed_function, query):
    """
    Implement fallback when function calls fail
    """
    fallback_response = client.responses.create(
        model="o4-mini",
        messages=[
            {
                "role": "system",
                "content": f"The {failed_function} tool is unavailable. Please provide the best response you can using your general knowledge."
            },
            {
                "role": "user",
                "content": query
            }
        ]
    )
    return fallback_response
```

## Key Takeaways

1. **Sequential Processing**: Reasoning models excel at multi-step workflows where each step depends on previous results
2. **Context Preservation**: Always maintain reasoning history and function call results in conversation context
3. **Clear Tool Descriptions**: Function clarity is critical for proper tool selection and usage
4. **Native Integration**: Reasoning models use tools within their chain of thought, leading to more intelligent tool usage
5. **Robust Error Handling**: Implement comprehensive error handling for production reliability
6. **Performance Considerations**: Balance tool list size with clarity and performance requirements

This approach enables sophisticated AI applications that can handle complex, multi-step reasoning tasks involving multiple external tools and data sources.