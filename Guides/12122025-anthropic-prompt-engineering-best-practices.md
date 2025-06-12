# Anthropic Prompt Engineering Best Practices
*Based on official Anthropic documentation and Claude optimization techniques*

## Overview

This guide compiles the essential prompt engineering techniques from Anthropic's official documentation, specifically optimized for Claude but applicable to other advanced language models. These practices are proven to significantly improve output quality, consistency, and task completion rates.

## Core Methodology

### The Anthropic Approach Order
Anthropic recommends applying techniques in this specific sequence for maximum effectiveness:

1. **Be clear and direct**
2. **Use examples (multishot prompting)**
3. **Give Claude a role**
4. **Use XML tags**
5. **Chain complex prompts**
6. **Use long context effectively**

## 1. Clarity and Directness

### The "New Employee" Principle
Treat Claude like a brilliant but very new employee with amnesia. Provide all necessary context that you would give to someone completely unfamiliar with your domain.

#### Key Strategies:

**Provide Comprehensive Context**:
```
Instead of: "Analyze the quarterly results"

Use: "Analyze our Q3 2024 financial results for a SaaS company with 50 employees. 
Focus on revenue growth, customer acquisition costs, and churn rates. 
The audience is our board of directors who need insights for strategic planning."
```

**Be Specific About Outputs**:
```
Instead of: "Write some code"

Use: "Write Python code that connects to a PostgreSQL database and retrieves 
user data. Include error handling and return the results as a JSON object. 
Output only the code without explanations."
```

**Use Sequential Instructions**:
```
Please complete this analysis in the following order:
1. Review the provided data for completeness
2. Identify key trends and patterns
3. Calculate relevant metrics and ratios
4. Highlight significant changes from previous periods
5. Provide 3-5 actionable recommendations
```

### Context Checklist
Before submitting a prompt, ensure you've provided:
- [ ] Task purpose and objectives
- [ ] Target audience for the output
- [ ] Required format and structure
- [ ] Relevant background information
- [ ] Constraints and limitations
- [ ] Success criteria

## 2. Multishot Prompting (Examples)

### The Power of Examples
Including 3-5 diverse, relevant examples can dramatically improve output quality and reduce misinterpretation.

#### Example Structure Template:
```xml
<examples>
<example>
Input: [Example input 1]
Expected Output: [Desired output 1]
</example>

<example>
Input: [Example input 2 - different scenario]
Expected Output: [Desired output 2]
</example>

<example>
Input: [Example input 3 - edge case]
Expected Output: [Desired output 3]
</example>
</examples>
```

#### Example Quality Guidelines:
- **Relevance**: Examples should match your specific use case
- **Diversity**: Cover different scenarios and edge cases
- **Clarity**: Use clear, unambiguous examples
- **Consistency**: Maintain consistent format across examples
- **Quality**: Ensure examples represent your ideal output

#### Real Example - Customer Feedback Analysis:
```xml
<examples>
<example>
Input: "The product is okay but the customer service was terrible. Waited 2 hours for a response."
Output: 
Category: Mixed
Sentiment: Negative
Issues: Customer service response time
Positive: Product quality acceptable
Priority: High (service issue)
</example>

<example>
Input: "Love the new features! The UI is so much cleaner and faster now."
Output:
Category: Positive
Sentiment: Very Positive  
Issues: None
Positive: UI improvements, performance, new features
Priority: Low (positive feedback)
</example>

<example>
Input: "Billing charged me twice this month and I can't reach anyone to fix it."
Output:
Category: Complaint
Sentiment: Very Negative
Issues: Billing error, unreachable support
Positive: None
Priority: Very High (billing issue)
</example>
</examples>
```

### Pro Tip: Let Claude Help
Ask Claude to evaluate your examples:
- "Are these examples relevant and diverse enough?"
- "What edge cases should I add to my examples?"
- "Generate 2 more examples based on these patterns"

## 3. Role Assignment

### Expertise-Based Roles
Assigning a specific professional role significantly improves response quality and domain expertise.

#### Role Definition Framework:
```
You are a [SPECIFIC_ROLE] with [EXPERIENCE_LEVEL] of experience in [DOMAIN].

Your key qualifications include:
- [Relevant skill/certification 1]
- [Relevant skill/certification 2]  
- [Relevant skill/certification 3]

In this role, you are responsible for:
- [Primary responsibility 1]
- [Primary responsibility 2]
- [Primary responsibility 3]
```

#### Effective Role Examples:

**Marketing Strategist**:
```
You are a Senior Marketing Strategist with 12+ years of experience in B2B SaaS marketing. 

Your expertise includes:
- Growth marketing and demand generation
- Customer acquisition and retention strategies
- Marketing analytics and performance measurement
- Cross-channel campaign optimization

You are known for data-driven decision making and creating scalable marketing systems.
```

**Technical Architect**:
```
You are a Principal Software Architect with 15+ years of experience designing large-scale distributed systems.

Your expertise includes:
- Microservices architecture and containerization
- Cloud-native design patterns (AWS, Azure, GCP)
- Performance optimization and scalability
- Security architecture and compliance

You prioritize maintainability, scalability, and operational excellence in all designs.
```

### Role Specificity Guidelines:
- Use specific titles, not generic ones
- Include years of experience
- Mention relevant certifications or specializations
- Define decision-making authority level
- Establish professional perspective and priorities

## 4. XML Tag Structure

### Why XML Tags Work
XML tags provide clear structure that helps Claude parse and organize information more effectively, leading to higher accuracy and better-formatted responses.

#### Essential XML Tags:
- `<instructions>` - Core task definition
- `<context>` - Background information
- `<examples>` - Sample inputs/outputs
- `<constraints>` - Limitations and requirements
- `<thinking>` - Reasoning process
- `<output>` - Final response

#### Master Template:
```xml
<instructions>
[Clear, specific task description with objectives]
</instructions>

<context>
[Essential background information and domain knowledge]
</context>

<constraints>
[Specific limitations, requirements, and boundaries]
</constraints>

<examples>
[3-5 high-quality, diverse examples]
</examples>

<thinking>
Please work through this systematically:
1. [Analysis step 1]
2. [Analysis step 2]
3. [Synthesis step]
4. [Validation step]
</thinking>

<output>
[Specific format requirements for the final response]
</output>
```

### Advanced XML Patterns:

**Nested Structure for Complex Tasks**:
```xml
<task>
  <primary_objective>
    [Main goal]
  </primary_objective>
  
  <subtasks>
    <subtask id="1">
      <description>[Subtask 1 details]</description>
      <success_criteria>[How to evaluate completion]</success_criteria>
    </subtask>
    
    <subtask id="2">
      <description>[Subtask 2 details]</description>
      <success_criteria>[How to evaluate completion]</success_criteria>
    </subtask>
  </subtasks>
</task>
```

**Data Processing Structure**:
```xml
<data>
  <input_format>[Description of input data structure]</input_format>
  <processing_rules>
    <rule>[Processing rule 1]</rule>
    <rule>[Processing rule 2]</rule>
  </processing_rules>
  <output_format>[Required output structure]</output_format>
</data>
```

## 5. Chain of Thought (CoT) Reasoning

### Implementation Levels:

#### Basic CoT:
```
Think step-by-step before providing your final answer.
```

#### Guided CoT:
```
Please analyze this step-by-step:
1. First, identify the key components of the problem
2. Then, evaluate each component against our criteria
3. Next, consider potential risks and mitigation strategies
4. Finally, synthesize your findings into actionable recommendations
```

#### Structured CoT:
```xml
<thinking>
Let me work through this systematically:

Problem Analysis:
- [Break down the core issues]

Evaluation:
- [Assess options against criteria]

Risk Assessment:
- [Identify potential challenges]

Synthesis:
- [Combine insights into recommendations]
</thinking>

<answer>
[Final response based on the reasoning above]
</answer>
```

### When to Use CoT:
- Complex analytical tasks
- Multi-step problem solving
- Decision-making scenarios
- Research and synthesis tasks
- Mathematical or logical reasoning

### CoT Best Practices:
- Always show the thinking process
- Break complex problems into logical steps
- Validate reasoning at each step
- Connect thinking to final conclusions
- Use when accuracy is more important than speed

## 6. Prompt Chaining

### When to Chain Prompts:
- Tasks requiring multiple distinct skills
- Complex workflows with quality gates
- When single prompts become too complex
- For better error isolation and debugging

### Chain Types:

#### Sequential Chain:
```
Prompt 1: Research → [Results]
Prompt 2: Analyze [Results] → [Analysis]  
Prompt 3: Generate recommendations from [Analysis] → [Final Output]
```

#### Verification Chain:
```
Prompt 1: Generate initial response → [Draft]
Prompt 2: Review [Draft] for accuracy and completeness → [Review]
Prompt 3: Refine [Draft] based on [Review] → [Final Response]
```

#### Parallel Chain:
```
Prompt A: Technical analysis → [Tech Results]
Prompt B: Market analysis → [Market Results]
Prompt C: Financial analysis → [Finance Results]
Prompt D: Synthesize [Tech + Market + Finance Results] → [Final Report]
```

### Chain Design Principles:
- Each prompt should have a single, clear objective
- Use XML tags to pass data between prompts
- Include validation steps for quality assurance
- Design for modularity and reusability
- Plan error handling and recovery paths

## 7. Response Prefilling

### Strategic Prefill Uses:

#### Format Control:
```
Human: Generate a JSON response for the user data.
Assistant: {
```
*This forces Claude to start with JSON format*

#### Role Maintenance:
```
Human: Continue the conversation as Dr. Smith.
Assistant: [Dr. Smith]:
```
*This maintains character consistency*

#### Structure Enforcement:
```
Human: Provide a structured analysis.
Assistant: ## Executive Summary

Based on my analysis,
```
*This enforces specific formatting*

### Prefill Guidelines:
- Use sparingly and strategically
- Test prefill effectiveness
- Ensure prefills don't end with whitespace
- Consider impact on response quality
- Works only in non-extended thinking modes

## Common Patterns and Templates

### Analysis Pattern:
```xml
<role>
You are a Senior Business Analyst with expertise in [domain].
</role>

<instructions>
Analyze [subject] to determine [objective] for [audience].
</instructions>

<context>
[Relevant background information]
</context>

<analysis_framework>
1. Data review and validation
2. Pattern identification and trends
3. Root cause analysis
4. Impact assessment
5. Recommendation development
</analysis_framework>

<examples>
[2-3 relevant examples of similar analyses]
</examples>

<output_format>
- Executive Summary (2-3 sentences)
- Key Findings (3-5 bullet points)
- Detailed Analysis (structured sections)
- Recommendations (prioritized with rationale)
- Next Steps (specific actions with timelines)
</output_format>
```

### Creative Generation Pattern:
```xml
<role>
You are a Creative Director with [X] years of experience in [industry].
</role>

<instructions>
Generate [number] creative [output_type] for [purpose] targeting [audience].
</instructions>

<creative_brief>
Objective: [Clear creative goal]
Brand Voice: [Tone and personality]
Constraints: [Limitations and requirements]
Inspiration: [Reference styles or examples]
Success Metrics: [How to evaluate creative success]
</creative_brief>

<creative_process>
1. Brainstorm diverse concepts
2. Evaluate against objectives and constraints
3. Develop strongest concepts with detail
4. Refine for maximum impact and resonance
</creative_process>

<examples>
[Examples of similar high-quality creative work]
</examples>
```

## Quality Assurance

### Prompt Testing Checklist:
- [ ] Clear, specific instructions
- [ ] Adequate context provided
- [ ] Relevant examples included
- [ ] Appropriate role assigned
- [ ] Structured with XML tags
- [ ] Chain complexity when needed
- [ ] Success criteria defined

### Common Issues and Fixes:

**Issue**: Inconsistent outputs
**Fix**: Add more specific constraints and examples

**Issue**: Incomplete responses  
**Fix**: Break into subtasks or use chaining

**Issue**: Wrong format
**Fix**: Use XML structure and prefilling

**Issue**: Poor quality
**Fix**: Assign expert role and provide quality examples

**Issue**: Off-topic responses
**Fix**: Strengthen context and add explicit boundaries

## Conclusion

Anthropic's prompt engineering methodology emphasizes systematic application of proven techniques in a specific order. The key to success is:

1. **Start with clarity** - Make instructions unambiguous
2. **Add examples** - Show, don't just tell
3. **Assign roles** - Leverage expertise and perspective
4. **Structure with XML** - Organize for clarity and parsing
5. **Chain when complex** - Break down multi-step processes
6. **Test and iterate** - Continuously improve through empirical validation

These techniques, when applied systematically, can dramatically improve the quality, consistency, and effectiveness of your AI interactions.

---

*This guide represents current best practices based on Anthropic's official documentation. Continue to test and refine these approaches for your specific use cases.*