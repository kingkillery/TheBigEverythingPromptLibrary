# Prompt Engineering Quick Reference
*Essential patterns and templates for immediate use*

## Quick Start Checklist

### Before You Prompt:
- [ ] **Clear Goal**: What exactly do you want to achieve?
- [ ] **Context**: What background does the AI need?
- [ ] **Audience**: Who is the output for?
- [ ] **Format**: How should the response be structured?
- [ ] **Examples**: Do you have 2-3 good examples?

### Essential Elements:
- [ ] **Role**: Assign specific expertise
- [ ] **Instructions**: Clear, direct task definition
- [ ] **Context**: Necessary background information
- [ ] **Examples**: 2-5 relevant samples
- [ ] **Format**: Specific output requirements
- [ ] **Constraints**: Limitations and boundaries

## Universal Template

```xml
<role>
You are a [SPECIFIC_EXPERT] with [EXPERIENCE] in [DOMAIN].
</role>

<instructions>
[Clear, specific task with objective and audience]
</instructions>

<context>
[Essential background information]
</context>

<examples>
<example>
Input: [Example 1]
Output: [Desired result 1]
</example>

<example>
Input: [Example 2]
Output: [Desired result 2]
</example>
</examples>

<constraints>
- [Limitation 1]
- [Requirement 2]
- [Boundary 3]
</constraints>

<output_format>
[Specific structure and formatting requirements]
</output_format>

<thinking>
Please work through this step-by-step:
1. [Step 1]
2. [Step 2]
3. [Step 3]
</thinking>
```

## Pattern Library

### 1. Analysis Pattern

```xml
<role>Senior [Domain] Analyst</role>

<instructions>
Analyze [subject] to [objective] for [audience].
</instructions>

<analysis_framework>
1. Data review and validation
2. Pattern identification
3. Root cause analysis
4. Impact assessment
5. Recommendations
</analysis_framework>

<output_format>
- Executive Summary
- Key Findings (3-5 points)
- Detailed Analysis
- Recommendations (prioritized)
- Next Steps
</output_format>
```

### 2. Creative Generation Pattern

```xml
<role>Creative [Specialist] with [X] years experience</role>

<instructions>
Generate [number] [creative_output] for [purpose] targeting [audience].
</instructions>

<creative_brief>
- Objective: [Goal]
- Style: [Voice/tone]
- Constraints: [Limitations]
- Success Metrics: [Evaluation criteria]
</creative_brief>

<creative_process>
1. Brainstorm diverse concepts
2. Evaluate against objectives
3. Develop strongest ideas
4. Refine for maximum impact
</creative_process>
```

### 3. Problem-Solving Pattern

```xml
<role>Senior Problem-Solving Expert</role>

<instructions>
Solve [problem] considering [constraints] to achieve [outcome].
</instructions>

<problem_framework>
- Problem Definition: [Clear statement]
- Stakeholders: [Affected parties]
- Constraints: [Limitations]
- Success Criteria: [Measurement]
</problem_framework>

<solution_approach>
1. Problem analysis
2. Solution brainstorming
3. Feasibility assessment
4. Implementation planning
5. Risk mitigation
</solution_approach>
```

### 4. Comparison Pattern

```xml
<role>Subject Matter Expert in [domain]</role>

<instructions>
Compare [item_A] vs [item_B] across [criteria] for [decision_purpose].
</instructions>

<comparison_framework>
Evaluation Criteria:
- [Criterion 1]: [Weight/importance]
- [Criterion 2]: [Weight/importance]
- [Criterion 3]: [Weight/importance]
</comparison_framework>

<comparison_structure>
For each criterion:
1. Define evaluation standard
2. Assess Item A
3. Assess Item B
4. Compare strengths/weaknesses
5. Provide recommendation
</comparison_structure>
```

### 5. Content Creation Pattern

```xml
<role>[Content Type] Expert with expertise in [domain]</role>

<instructions>
Create [content_type] about [topic] for [audience] with [objective].
</instructions>

<content_strategy>
- Purpose: [Why this content exists]
- Audience: [Detailed target profile]
- Key Messages: [Core points to convey]
- Tone: [Voice and style]
- Success Metrics: [How to measure effectiveness]
</content_strategy>

<content_structure>
1. Hook: [Attention grabber]
2. Context: [Background/relevance]
3. Content: [Main information]
4. Conclusion: [Summary/key takeaways]
5. Call-to-Action: [Next steps]
</content_structure>
```

## Role Library

### Business Roles
- **Senior Marketing Strategist**: 12+ years B2B/B2C marketing
- **Business Development Manager**: 8+ years partnership/growth
- **Financial Analyst**: CPA with corporate finance expertise
- **Operations Manager**: Process optimization specialist
- **Product Manager**: User-focused product development expert

### Technical Roles
- **Senior Software Engineer**: 10+ years full-stack development
- **DevOps Engineer**: Cloud infrastructure and automation expert
- **Data Scientist**: ML/AI and statistical analysis specialist
- **Security Analyst**: Cybersecurity and risk assessment expert
- **UX Designer**: User research and interface design specialist

### Creative Roles
- **Creative Director**: Brand and campaign development expert
- **Content Strategist**: Editorial and content marketing specialist
- **Copywriter**: Persuasive writing and messaging expert
- **Visual Designer**: Brand identity and design systems expert

### Consulting Roles
- **Management Consultant**: Strategy and organizational expert
- **Technical Consultant**: Implementation and integration specialist
- **Change Management Consultant**: Organizational transformation expert

## Quick Fixes

### Problem: Inconsistent Outputs
**Fix**: Add specific constraints and examples
```xml
<constraints>
- Always use present tense
- Limit to exactly 3 recommendations
- Include quantifiable metrics when possible
- Follow the provided format exactly
</constraints>
```

### Problem: Too Generic
**Fix**: Add specific role and context
```xml
<role>
Senior Marketing Manager at a B2B SaaS company with 8+ years experience in demand generation and customer acquisition.
</role>

<context>
We're a 50-person startup selling project management software to mid-market companies. Our current CAC is $2,400 and LTV is $18,000.
</context>
```

### Problem: Wrong Format
**Fix**: Use explicit format specification
```xml
<output_format>
Use this exact structure:
## [Title]
- **Key Point 1**: [Description]
- **Key Point 2**: [Description]
- **Key Point 3**: [Description]

### Recommendation
[Specific action steps]
</output_format>
```

### Problem: Incomplete Response
**Fix**: Break into steps or use checklist
```xml
<completion_checklist>
Ensure your response includes:
- [ ] Executive summary (2-3 sentences)
- [ ] Analysis of each provided data point
- [ ] At least 3 specific recommendations
- [ ] Implementation timeline
- [ ] Success metrics
</completion_checklist>
```

## Power Phrases

### Thinking Triggers
- "Let's work through this step-by-step:"
- "First, let me understand the context..."
- "I need to consider multiple perspectives..."
- "Let me break this down systematically..."

### Quality Enhancers
- "Ensure accuracy by double-checking..."
- "Consider alternative approaches..."
- "Validate assumptions by..."
- "Support conclusions with specific evidence..."

### Format Controllers
- "Structure your response exactly as follows:"
- "Use this precise format:"
- "Organize information into these sections:"
- "Present findings using this template:"

### Constraint Enforcers
- "Focus only on..."
- "Exclude any mention of..."
- "Limit analysis to..."
- "Stay within the scope of..."

## Testing Templates

### A/B Test Setup
```
Version A (Control):
[Original prompt]

Version B (Test):
[Modified prompt with one change]

Success Metrics:
- Accuracy (1-10 scale)
- Completeness (% of requirements met)
- Consistency (variance across runs)
- Quality (subjective rating)
```

### Quality Evaluation
```
Response Quality Checklist:
- [ ] Addresses all parts of the request
- [ ] Follows specified format
- [ ] Maintains appropriate tone/voice
- [ ] Includes required elements
- [ ] Demonstrates expected expertise level
- [ ] Provides actionable insights
- [ ] Free of factual errors
```

## Emergency Debugging

### Response Too Long
**Fix**: Add length constraints
```
Limit your response to:
- Maximum 200 words
- Exactly 3 bullet points
- No more than 5 recommendations
```

### Response Too Short
**Fix**: Require elaboration
```
For each point, provide:
- Detailed explanation (2-3 sentences)
- Specific example or evidence
- Implementation steps
- Potential challenges
```

### Wrong Tone
**Fix**: Specify voice explicitly
```
Tone and Style:
- Professional but approachable
- Confident without being arrogant
- Data-driven and analytical
- Optimistic about solutions
```

### Missing Context
**Fix**: Provide domain background
```
<domain_context>
Industry: [Specific industry]
Company Stage: [Startup/Growth/Enterprise]
Current Challenges: [Key issues]
Success Criteria: [What good looks like]
</domain_context>
```

## Cheat Sheet Summary

1. **Always assign a specific role** with relevant expertise
2. **Provide 2-5 diverse examples** when possible
3. **Use XML structure** for complex prompts
4. **Be explicit about format** requirements
5. **Include step-by-step thinking** for complex tasks
6. **Test one variable at a time** when optimizing
7. **Break complex tasks** into smaller prompts
8. **Define success criteria** upfront

---

*Keep this guide handy for quick reference while crafting prompts. Focus on one improvement at a time for best results.*