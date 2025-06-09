# Multi-Tool Orchestration with RAG using OpenAI's Responses API

## Description
Comprehensive guide for building dynamic, multi-tool workflows using OpenAI's Responses API with Retrieval-Augmented Generation (RAG) approach. This cookbook demonstrates how to implement intelligent query routing to appropriate tools, whether for general knowledge or accessing specific internal context from vector databases like Pinecone. Shows integration of function calls, web searches, and document retrieval to generate accurate, context-aware responses.

## Source
OpenAI Cookbook - Multi-Tool Orchestration with RAG approach using OpenAI's Responses API

## Overview

This guide showcases the flexibility of the Responses API, illustrating that beyond the internal `file_search` tool—which connects to an internal vector store—there is also the capability to easily connect to external vector databases. This allows for the implementation of a RAG approach in conjunction with hosted tooling, providing a versatile solution for various retrieval and generation tasks.

The approach demonstrates intelligent tool selection based on query type:
- **General knowledge queries**: Use built-in web search tools
- **Domain-specific queries**: Use external vector database search (Pinecone)
- **Mixed queries**: Sequential tool calling for comprehensive responses

---

## Key Components

### 1. Dataset Preparation
Uses medical reasoning dataset from Hugging Face, converting Question and Response columns into merged text for embedding and storage as metadata.

### 2. Vector Database Setup (Pinecone)
- Dynamic index creation based on embedding dimensionality
- Batch processing for efficient upserts
- Metadata preservation for context retrieval

### 3. Tool Definitions
Two primary tools for orchestration:

#### Web Search Preview Tool
```python
{
    "type": "web_search_preview",
    "user_location": {
        "type": "approximate",
        "country": "US",
        "region": "California", 
        "city": "SF"
    },
    "search_context_size": "medium"
}
```

#### Pinecone Search Tool
```python
{
    "type": "function",
    "name": "PineconeSearchDocuments",
    "description": "Search for relevant documents based on the medical question asked by the user that is stored within the vector database using a semantic query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The natural language query to search the vector database."
            },
            "top_k": {
                "type": "integer",
                "description": "Number of top results to return.",
                "default": 3
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}
```

---

## Implementation Pattern

### Basic Query Processing
```python
# Process each query dynamically
for item in queries:
    input_messages = [{"role": "user", "content": item["query"]}]
    
    # Call the Responses API with tools enabled
    response = client.responses.create(
        model="gpt-4o",
        input=[
            {"role": "system", "content": "When prompted with a question, select the right tool to use based on the question."},
            {"role": "user", "content": item["query"]}
        ],
        tools=tools,
        parallel_tool_calls=True
    )
    
    # Process tool calls and generate final response
    if response.output:
        tool_call = response.output[0]
        if tool_call.type in ["web_search_preview", "function_call"]:
            # Handle tool execution and response synthesis
```

### Sequential Tool Calling
For comprehensive responses requiring multiple information sources:

```python
response = client.responses.create(
    model="gpt-4o",
    input=[
        {"role": "system", "content": "Every time it's prompted with a question, first call the web search tool for results, then call `PineconeSearchDocuments` to find real examples in the internal knowledge base."},
        {"role": "user", "content": item}
    ],
    tools=tools,
    parallel_tool_calls=True
)
```

---

## System Prompts for Tool Selection

### Basic Tool Selection
```
When prompted with a question, select the right tool to use based on the question.
```

### Sequential Tool Usage
```
Every time it's prompted with a question, first call the web search tool for results, then call `PineconeSearchDocuments` to find real examples in the internal knowledge base.
```

### Context-Aware Selection
```
Analyze the user query and determine the most appropriate tool:
- For current events, general knowledge, or real-time information: use web_search_preview
- For domain-specific questions requiring internal knowledge base: use PineconeSearchDocuments
- For comprehensive analysis: use both tools sequentially
```

---

## Query Examples and Routing

### Example Query Types

#### General Knowledge (Web Search)
```
"Who won the cricket world cup in 1983?"
"What is the most common cause of death in the United States according to the internet?"
```

#### Domain-Specific (Vector DB Search)
```
"A 7-year-old boy with sickle cell disease is experiencing knee and hip pain, has been admitted for pain crises in the past, and now walks with a limp. His exam shows a normal, cool hip with decreased range of motion and pain with ambulation. What is the most appropriate next step in management according to the internal knowledge base?"
```

#### Mixed Context
```
"A 45-year-old man with a history of alcohol use presents with symptoms including confusion, ataxia, and ophthalmoplegia. What is the most likely diagnosis and the recommended treatment?"
```

---

## Vector Database Query Function

```python
def query_pinecone_index(client, index, model, query_text):
    # Generate an embedding for the query
    query_embedding = client.embeddings.create(input=query_text, model=model).data[0].embedding

    # Query the index and return top 5 matches
    res = index.query(vector=[query_embedding], top_k=5, include_metadata=True)
    print("Query Results:")
    for match in res['matches']:
        print(f"{match['score']:.2f}: {match['metadata'].get('Question', 'N/A')} - {match['metadata'].get('Answer', 'N/A')}")
    return res
```

---

## Response Generation with Retrieved Context

```python
# Retrieve and concatenate top 3 match contexts
matches = index.query(
    vector=[client.embeddings.create(input=query, model=MODEL).data[0].embedding],
    top_k=3,
    include_metadata=True
)['matches']

context = "\n\n".join(
    f"Question: {m['metadata'].get('Question', '')}\nAnswer: {m['metadata'].get('Answer', '')}"
    for m in matches
)

# Use the context to generate a final answer
response = client.responses.create(
    model="gpt-4o",
    input=f"Provide the answer based on the context: {context} and the question: {query} as per the internal knowledge base",
)
```

---

## Tool Call Processing Pattern

### Understanding Tool Call Structure
```python
# Examine tool call details
tool_calls = []
for i in response.output:
    tool_calls.append({
        "Type": i.type,
        "Call ID": i.call_id if hasattr(i, 'call_id') else i.id if hasattr(i, 'id') else "N/A",
        "Output": str(i.output) if hasattr(i, 'output') else "N/A",
        "Name": i.name if hasattr(i, 'name') else "N/A"
    })
```

### Appending Tool Results to Conversation
```python
# Append tool call and its output back into the conversation
input_messages.append(tool_call)
input_messages.append({
    "type": "function_call_output",
    "call_id": tool_call.call_id,
    "output": str(result)
})

# Get final answer incorporating the tool's result
final_response = client.responses.create(
    model="gpt-4o",
    input=input_messages,
)
```

---

## Best Practices

### Tool Design
1. **Clear naming**: Use descriptive tool names that indicate their purpose
2. **Detailed descriptions**: Include comprehensive descriptions in the "description" field
3. **Good parameter naming**: Use clear names and descriptions for each tool parameter
4. **API tools field**: Use API tools field exclusively rather than manually injecting tool descriptions

### Query Processing
1. **Intelligent routing**: Analyze query type to determine appropriate tool selection
2. **Context synthesis**: Combine results from multiple tools when beneficial
3. **Error handling**: Gracefully handle cases where no relevant information is found
4. **Performance optimization**: Use appropriate top_k values for vector searches

### Response Generation
1. **Context integration**: Seamlessly blend retrieved context with generated responses
2. **Source attribution**: Maintain traceability to original sources
3. **Quality control**: Validate responses against retrieved context
4. **User experience**: Provide clear, actionable responses

---

## Multi-Tool Orchestration Flow

### Sequential Processing Example
```python
# Step 1: Web Search Call
print("Step 1: Web Search Call - Initiating web search to gather initial information.")

# Step 2: Pinecone Search Call  
print("Step 2: Pinecone Search Call - Querying Pinecone to find relevant examples from the internal knowledge base.")

# Step 3: Response Synthesis
print("Step 3: Calling Responses API for Final Answer")
```

### Parallel Tool Execution
- Enable `parallel_tool_calls=True` for simultaneous execution
- Handle multiple tool results in response synthesis
- Combine complementary information sources

---

## Performance Considerations

### Embedding Model Selection
- **text-embedding-3-small**: Good balance of performance and cost
- **text-embedding-3-large**: Higher accuracy for complex queries
- Consider model consistency across indexing and query time

### Vector Database Optimization
- **Batch processing**: Use appropriate batch sizes for upserts (recommended: 32)
- **Index configuration**: Choose appropriate similarity metrics (dotproduct, cosine)
- **Metadata structure**: Design metadata for efficient retrieval and context synthesis

### Response API Usage
- **Model selection**: gpt-4o for optimal tool calling performance
- **Context management**: Monitor token usage in long conversations
- **Tool call limits**: Consider API rate limits for complex workflows

---

## Error Handling and Fallbacks

### Tool Call Failures
```python
if res["matches"]:
    best_match = res["matches"][0]["metadata"]
    result = f"**Question:** {best_match.get('Question', 'N/A')}\n**Answer:** {best_match.get('Answer', 'N/A')}"
else:
    result = "**No matching documents found in the index.**"
```

### Graceful Degradation
- Fallback to general knowledge when vector search returns no results
- Provide partial answers when only some tools succeed
- Clear communication about information limitations

---

## Advanced Patterns

### Custom Tool Orchestration
Create domain-specific tool combinations based on:
- Query complexity analysis
- Available information sources
- User context and preferences
- Historical performance data

### Dynamic Tool Selection
Implement intelligent routing based on:
- Query classification (intent detection)
- Available tool capabilities
- Expected response quality
- Performance metrics

### Contextual Tool Chaining
Design workflows where:
- Earlier tool results inform later tool selection
- Context accumulates across tool calls
- Final synthesis considers all available information

---

## Conclusion

This multi-tool orchestration approach with RAG enables:

1. **Flexible information retrieval** from multiple sources
2. **Intelligent tool selection** based on query characteristics  
3. **Comprehensive response generation** combining internal and external knowledge
4. **Scalable architecture** for complex information needs

The pattern demonstrates how OpenAI's Responses API can orchestrate different tools (web search for current information, vector databases for domain-specific knowledge) to provide accurate, contextually-aware responses. This approach is particularly valuable for applications requiring both real-time information and specialized domain knowledge.

By following these patterns and best practices, developers can build sophisticated AI systems that intelligently route queries to appropriate information sources and synthesize comprehensive, accurate responses.

---

## Implementation Notes

- **Empirical approach**: Build informative evaluations and iterate based on performance
- **Systematic testing**: Test prompt and tool combinations with representative queries
- **Performance monitoring**: Track tool selection accuracy and response quality
- **Continuous improvement**: Refine tool descriptions and orchestration logic based on usage patterns

Happy coding!
