# Claude Customer Service Agent: Comprehensive Implementation Guide

*Source: Anthropic Cookbook - Customer Service Agents*  
*Date: June 10, 2025*

## Overview

This comprehensive guide demonstrates how to build sophisticated customer service agents using Claude. Based on proven techniques from Anthropic's official cookbook, this guide covers everything from basic conversational interfaces to advanced multi-turn reasoning and escalation handling.

## Table of Contents

1. [Customer Service Agent Fundamentals](#customer-service-agent-fundamentals)
2. [Conversation Management](#conversation-management)
3. [Intent Recognition and Routing](#intent-recognition-and-routing)
4. [Knowledge Base Integration](#knowledge-base-integration)
5. [Escalation and Handoff](#escalation-and-handoff)
6. [Multi-Language Support](#multi-language-support)
7. [Performance Monitoring](#performance-monitoring)

## Customer Service Agent Fundamentals

### Core Agent Architecture

```python
import anthropic
from typing import List, Dict, Any, Optional
import json
import time
from dataclasses import dataclass
from enum import Enum

class ConversationState(Enum):
    GREETING = "greeting"
    PROBLEM_IDENTIFICATION = "problem_identification"
    INFORMATION_GATHERING = "information_gathering"
    SOLUTION_PROVIDING = "solution_providing"
    CONFIRMATION = "confirmation"
    ESCALATION = "escalation"
    RESOLUTION = "resolution"

@dataclass
class CustomerContext:
    customer_id: str
    name: Optional[str] = None
    email: Optional[str] = None
    account_type: Optional[str] = None
    previous_interactions: List[Dict] = None
    current_issue: Optional[str] = None
    sentiment: Optional[str] = None
    
    def __post_init__(self):
        if self.previous_interactions is None:
            self.previous_interactions = []

class CustomerServiceAgent:
    def __init__(self, anthropic_api_key: str):
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.conversation_history = []
        self.customer_context = None
        self.current_state = ConversationState.GREETING
        
    def initialize_conversation(self, customer_context: CustomerContext):
        """Initialize a new customer service conversation."""
        self.customer_context = customer_context
        self.conversation_history = []
        self.current_state = ConversationState.GREETING
        
        # Generate personalized greeting
        greeting = self.generate_greeting()
        self.conversation_history.append({
            "role": "assistant",
            "content": greeting,
            "timestamp": time.time(),
            "state": self.current_state.value
        })
        
        return greeting
    
    def generate_greeting(self) -> str:
        """Generate a personalized greeting based on customer context."""
        
        context_info = ""
        if self.customer_context.name:
            context_info += f"Customer name: {self.customer_context.name}\n"
        if self.customer_context.account_type:
            context_info += f"Account type: {self.customer_context.account_type}\n"
        if self.customer_context.previous_interactions:
            context_info += f"Previous interactions: {len(self.customer_context.previous_interactions)} interactions\n"
        
        prompt = f"""<context>
You are a helpful customer service agent for a technology company.
{context_info}
</context>

<task>
Generate a warm, professional greeting for this customer.
</task>

<guidelines>
1. Be friendly and professional
2. Personalize with the customer's name if available
3. Acknowledge returning customers appropriately
4. Set a helpful, solution-oriented tone
5. Keep it concise but welcoming
</guidelines>

Generate an appropriate greeting:"""

        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text.strip()
```

### Advanced Conversation Handling

```python
class AdvancedConversationManager:
    def __init__(self, claude_client):
        self.client = claude_client
        
    def process_customer_message(self, message: str, conversation_history: List[Dict], 
                                customer_context: CustomerContext) -> Dict[str, Any]:
        """Process customer message with advanced understanding."""
        
        # Analyze message for intent, sentiment, and urgency
        analysis = self.analyze_message(message, conversation_history, customer_context)
        
        # Generate contextual response
        response = self.generate_response(message, conversation_history, customer_context, analysis)
        
        return {
            "response": response,
            "analysis": analysis,
            "next_actions": self.determine_next_actions(analysis),
            "state_change": self.should_change_state(analysis)
        }
    
    def analyze_message(self, message: str, conversation_history: List[Dict], 
                       customer_context: CustomerContext) -> Dict[str, Any]:
        """Comprehensive message analysis."""
        
        history_text = self.format_conversation_history(conversation_history[-5:])  # Last 5 messages
        
        analysis_prompt = f"""<conversation_history>
{history_text}
</conversation_history>

<customer_context>
Name: {customer_context.name or 'Unknown'}
Account Type: {customer_context.account_type or 'Unknown'}
Previous Issues: {len(customer_context.previous_interactions)} interactions
Current Issue: {customer_context.current_issue or 'Not identified'}
</customer_context>

<current_message>
{message}
</current_message>

<task>
Analyze this customer message comprehensively and provide structured analysis.
</task>

<analysis_format>
INTENT: [primary intent - support_request, complaint, question, compliment, etc.]
SENTIMENT: [positive, neutral, negative, frustrated, angry]
URGENCY: [low, medium, high, critical]
ISSUE_TYPE: [technical, billing, account, product_info, etc.]
KEY_ENTITIES: [extract important entities like product names, error codes, dates]
ESCALATION_NEEDED: [yes/no with brief reason]
INFORMATION_NEEDED: [what additional info might be needed to help]
CONFIDENCE: [high/medium/low in this analysis]
</analysis_format>

Provide your analysis:"""

        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=400,
            messages=[{"role": "user", "content": analysis_prompt}]
        )
        
        return self.parse_analysis(response.content[0].text)
    
    def generate_response(self, message: str, conversation_history: List[Dict], 
                         customer_context: CustomerContext, analysis: Dict[str, Any]) -> str:
        """Generate contextually appropriate response."""
        
        history_text = self.format_conversation_history(conversation_history[-8:])
        
        response_prompt = f"""<role>
You are an expert customer service agent known for being helpful, empathetic, and solution-focused.
</role>

<conversation_history>
{history_text}
</conversation_history>

<customer_context>
Name: {customer_context.name or 'Unknown'}
Account Type: {customer_context.account_type or 'Unknown'}
Current Issue: {customer_context.current_issue or 'Not identified'}
</customer_context>

<message_analysis>
Intent: {analysis.get('intent', 'unknown')}
Sentiment: {analysis.get('sentiment', 'neutral')}
Urgency: {analysis.get('urgency', 'medium')}
Issue Type: {analysis.get('issue_type', 'general')}
</message_analysis>

<current_message>
{message}
</current_message>

<response_guidelines>
1. Acknowledge the customer's concern with empathy
2. Address their specific intent and issue type
3. Match the appropriate tone for their sentiment and urgency
4. Provide helpful, actionable information
5. Ask clarifying questions if needed
6. Offer clear next steps
7. Maintain professionalism while being personable
8. Use the customer's name naturally if available
</response_guidelines>

Generate an appropriate response:"""

        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=300,
            messages=[{"role": "user", "content": response_prompt}]
        )
        
        return response.content[0].text.strip()
    
    def format_conversation_history(self, history: List[Dict]) -> str:
        """Format conversation history for prompts."""
        formatted = ""
        for turn in history:
            role = "Customer" if turn["role"] == "user" else "Agent"
            formatted += f"{role}: {turn['content']}\n"
        return formatted
    
    def parse_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """Parse structured analysis from Claude's response."""
        analysis = {}
        
        for line in analysis_text.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                analysis[key] = value
        
        return analysis
```

## Intent Recognition and Routing

### Advanced Intent Classification

```python
class IntentClassifier:
    def __init__(self, claude_client):
        self.client = claude_client
        self.intent_categories = {
            "technical_support": {
                "description": "Technical issues, bugs, troubleshooting",
                "keywords": ["error", "bug", "not working", "broken", "crash", "slow"],
                "priority": "high",
                "routing": "technical_team"
            },
            "billing_inquiry": {
                "description": "Billing questions, payment issues, refunds",
                "keywords": ["bill", "charge", "payment", "refund", "invoice", "cost"],
                "priority": "medium",
                "routing": "billing_team"
            },
            "account_management": {
                "description": "Account changes, settings, access issues",
                "keywords": ["account", "login", "password", "settings", "profile"],
                "priority": "medium",
                "routing": "account_team"
            },
            "product_information": {
                "description": "Product features, specifications, availability",
                "keywords": ["features", "how to", "can I", "does it", "available"],
                "priority": "low",
                "routing": "sales_team"
            },
            "complaint": {
                "description": "Service complaints, negative feedback",
                "keywords": ["disappointed", "terrible", "awful", "complaint", "unhappy"],
                "priority": "high",
                "routing": "escalation_team"
            }
        }
    
    def classify_intent(self, message: str, conversation_context: List[Dict] = None) -> Dict[str, Any]:
        """Classify customer intent with high accuracy."""
        
        context_text = ""
        if conversation_context:
            context_text = "\n".join([f"{turn['role']}: {turn['content']}" 
                                    for turn in conversation_context[-3:]])
        
        intent_descriptions = "\n".join([
            f"- {intent}: {details['description']}"
            for intent, details in self.intent_categories.items()
        ])
        
        classification_prompt = f"""<task>
Classify the customer's intent based on their message and conversation context.
</task>

<intent_categories>
{intent_descriptions}
</intent_categories>

<conversation_context>
{context_text}
</conversation_context>

<customer_message>
{message}
</customer_message>

<instructions>
1. Analyze the customer's primary intent
2. Consider the conversation context
3. Look for emotional indicators
4. Assess urgency level
5. Provide confidence score
</instructions>

<response_format>
PRIMARY_INTENT: [intent_category]
SECONDARY_INTENT: [if applicable, otherwise "none"]
URGENCY: [low/medium/high/critical]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation]
SUGGESTED_ROUTING: [team or action]
EMOTIONAL_STATE: [calm/frustrated/angry/confused/etc.]
</response_format>

Classify this customer interaction:"""

        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=250,
            messages=[{"role": "user", "content": classification_prompt}]
        )
        
        return self.parse_classification(response.content[0].text)
    
    def parse_classification(self, classification_text: str) -> Dict[str, Any]:
        """Parse intent classification results."""
        result = {}
        
        for line in classification_text.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                if key == 'confidence':
                    try:
                        result[key] = float(value)
                    except ValueError:
                        result[key] = 0.5
                else:
                    result[key] = value
        
        return result
```

### Dynamic Routing System

```python
class CustomerServiceRouter:
    def __init__(self, claude_client):
        self.client = claude_client
        self.routing_rules = {
            "technical_support": {
                "conditions": ["urgency == 'high'", "confidence > 0.8"],
                "action": "route_to_specialist",
                "specialist_type": "technical"
            },
            "billing_inquiry": {
                "conditions": ["urgency in ['medium', 'high']"],
                "action": "route_to_specialist",
                "specialist_type": "billing"
            },
            "escalation_required": {
                "conditions": ["urgency == 'critical'", "emotional_state in ['angry', 'frustrated']"],
                "action": "immediate_escalation",
                "specialist_type": "supervisor"
            }
        }
    
    def determine_routing(self, intent_classification: Dict[str, Any], 
                         customer_context: CustomerContext) -> Dict[str, Any]:
        """Determine appropriate routing based on classification and context."""
        
        routing_prompt = f"""<customer_context>
Account Type: {customer_context.account_type or 'Standard'}
Previous Escalations: {len([i for i in customer_context.previous_interactions if i.get('escalated', False)])}
Current Issue Complexity: {intent_classification.get('primary_intent', 'unknown')}
</customer_context>

<current_classification>
Intent: {intent_classification.get('primary_intent', 'unknown')}
Urgency: {intent_classification.get('urgency', 'medium')}
Emotional State: {intent_classification.get('emotional_state', 'neutral')}
Confidence: {intent_classification.get('confidence', 0.5)}
</current_classification>

<task>
Determine the best routing action for this customer interaction.
</task>

<routing_options>
1. CONTINUE_WITH_AI: AI agent can handle this effectively
2. ROUTE_TO_HUMAN: Transfer to human specialist
3. ESCALATE_IMMEDIATELY: High-priority escalation needed
4. SCHEDULE_CALLBACK: Schedule expert callback
5. PROVIDE_SELF_SERVICE: Direct to self-service resources
</routing_options>

<decision_factors>
- Issue complexity and technical depth
- Customer emotional state and urgency
- Account type and history
- AI capability to resolve the issue
- Available human resources
</decision_factors>

<response_format>
ROUTING_DECISION: [routing_option]
SPECIALIST_TYPE: [if human routing needed]
PRIORITY_LEVEL: [low/medium/high/critical]
ESTIMATED_RESOLUTION_TIME: [time estimate]
REASONING: [brief explanation]
NEXT_ACTIONS: [specific steps to take]
</response_format>

Provide routing recommendation:"""

        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=300,
            messages=[{"role": "user", "content": routing_prompt}]
        )
        
        return self.parse_routing_decision(response.content[0].text)
    
    def parse_routing_decision(self, decision_text: str) -> Dict[str, Any]:
        """Parse routing decision from Claude's response."""
        decision = {}
        
        for line in decision_text.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                decision[key] = value
        
        return decision
```

## Knowledge Base Integration

### Intelligent Knowledge Retrieval

```python
import faiss
import numpy as np
from typing import List, Dict

class CustomerServiceKnowledgeBase:
    def __init__(self, claude_client, embedding_function):
        self.client = claude_client
        self.embed = embedding_function
        self.knowledge_index = None
        self.knowledge_articles = []
        self.article_metadata = []
        
    def build_knowledge_base(self, articles: List[Dict[str, Any]]):
        """Build searchable knowledge base from articles."""
        
        self.knowledge_articles = articles
        embeddings = []
        
        for article in articles:
            # Combine title and content for better retrieval
            searchable_text = f"{article['title']} {article['content']}"
            embedding = self.embed(searchable_text)
            embeddings.append(embedding)
            
            # Store metadata
            self.article_metadata.append({
                "title": article["title"],
                "category": article.get("category", "general"),
                "difficulty": article.get("difficulty", "beginner"),
                "last_updated": article.get("last_updated", "unknown"),
                "tags": article.get("tags", [])
            })
        
        # Build FAISS index
        embeddings_array = np.array(embeddings, dtype=np.float32)
        dimension = embeddings_array.shape[1]
        self.knowledge_index = faiss.IndexFlatIP(dimension)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_array)
        self.knowledge_index.add(embeddings_array)
    
    def retrieve_relevant_articles(self, query: str, customer_context: CustomerContext, 
                                 top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant knowledge base articles."""
        
        # Enhance query with customer context
        enhanced_query = self.enhance_query_with_context(query, customer_context)
        
        # Get embedding and search
        query_embedding = self.embed(enhanced_query)
        query_array = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_array)
        
        similarities, indices = self.knowledge_index.search(query_array, top_k)
        
        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx < len(self.knowledge_articles):
                article = self.knowledge_articles[idx].copy()
                article['relevance_score'] = float(similarity)
                article['metadata'] = self.article_metadata[idx]
                results.append(article)
        
        return results
    
    def enhance_query_with_context(self, query: str, customer_context: CustomerContext) -> str:
        """Enhance search query with customer context."""
        
        context_prompt = f"""<original_query>{query}</original_query>

<customer_context>
Account Type: {customer_context.account_type or 'Unknown'}
Previous Issues: {[i.get('issue_type') for i in customer_context.previous_interactions[-3:] if i.get('issue_type')]}
Current Issue: {customer_context.current_issue or 'Not specified'}
</customer_context>

<task>
Enhance the search query to better find relevant knowledge base articles.
Consider the customer's context and expand with related terms.
</task>

<guidelines>
1. Keep the original intent
2. Add relevant technical terms
3. Include synonyms and related concepts
4. Consider the customer's experience level
5. Maximum 2-3 sentences
</guidelines>

Enhanced search query:"""

        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=150,
            messages=[{"role": "user", "content": context_prompt}]
        )
        
        return response.content[0].text.strip()
    
    def generate_knowledge_based_response(self, query: str, relevant_articles: List[Dict], 
                                        customer_context: CustomerContext) -> str:
        """Generate response using retrieved knowledge articles."""
        
        articles_text = ""
        for i, article in enumerate(relevant_articles, 1):
            articles_text += f"""
Article {i}: {article['title']}
Category: {article['metadata']['category']}
Content: {article['content'][:500]}...
Relevance: {article['relevance_score']:.2f}

"""

        response_prompt = f"""<role>
You are a knowledgeable customer service agent with access to the company knowledge base.
</role>

<customer_query>
{query}
</customer_query>

<customer_context>
Name: {customer_context.name or 'Valued Customer'}
Account Type: {customer_context.account_type or 'Standard'}
Experience Level: Based on account type and previous interactions
</customer_context>

<relevant_knowledge_articles>
{articles_text}
</relevant_knowledge_articles>

<response_guidelines>
1. Use the knowledge articles to provide accurate information
2. Synthesize information from multiple articles if helpful
3. Adapt the complexity to the customer's experience level
4. Include specific steps or instructions when applicable
5. Mention if additional resources are available
6. Be clear about what you can and cannot help with
7. Maintain a helpful, professional tone
8. Offer to help with follow-up questions
</response_guidelines>

Generate a helpful response based on the knowledge base:"""

        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=400,
            messages=[{"role": "user", "content": response_prompt}]
        )
        
        return response.content[0].text.strip()
```

## Escalation and Handoff

### Intelligent Escalation System

```python
class EscalationManager:
    def __init__(self, claude_client):
        self.client = claude_client
        self.escalation_criteria = {
            "technical_complexity": {
                "threshold": 0.8,
                "indicators": ["API", "database", "server", "integration", "custom code"]
            },
            "emotional_escalation": {
                "threshold": 0.7,
                "indicators": ["angry", "frustrated", "disappointed", "terrible", "awful"]
            },
            "policy_exception": {
                "threshold": 0.9,
                "indicators": ["exception", "special case", "manager", "supervisor"]
            },
            "high_value_customer": {
                "threshold": 0.6,
                "indicators": ["enterprise", "premium", "VIP"]
            }
        }
    
    def assess_escalation_need(self, conversation_history: List[Dict], 
                             customer_context: CustomerContext, 
                             current_issue_analysis: Dict) -> Dict[str, Any]:
        """Assess whether escalation is needed."""
        
        conversation_text = "\n".join([
            f"{turn['role']}: {turn['content']}" 
            for turn in conversation_history[-10:]
        ])
        
        escalation_prompt = f"""<conversation_history>
{conversation_text}
</conversation_history>

<customer_context>
Account Type: {customer_context.account_type or 'Standard'}
Previous Escalations: {len([i for i in customer_context.previous_interactions if i.get('escalated', False)])}
Issue Complexity: {current_issue_analysis.get('primary_intent', 'unknown')}
Emotional State: {current_issue_analysis.get('emotional_state', 'neutral')}
</customer_context>

<escalation_criteria>
1. TECHNICAL_COMPLEXITY: Issue requires specialized technical knowledge
2. EMOTIONAL_ESCALATION: Customer is highly frustrated or angry
3. POLICY_EXCEPTION: Request requires management approval
4. REPEAT_ISSUE: Same issue reported multiple times
5. HIGH_VALUE_CUSTOMER: Premium customer with special needs
6. TIME_SENSITIVE: Urgent business impact
</escalation_criteria>

<task>
Evaluate whether this interaction should be escalated and to whom.
</task>

<assessment_format>
ESCALATION_NEEDED: [yes/no]
ESCALATION_TYPE: [technical/emotional/policy/managerial]
URGENCY_LEVEL: [low/medium/high/critical]
RECOMMENDED_SPECIALIST: [specific team or role]
CONFIDENCE: [0.0-1.0]
KEY_FACTORS: [main reasons for escalation]
HANDOFF_NOTES: [important context for receiving agent]
CUSTOMER_EXPECTATIONS: [what customer expects from escalation]
</assessment_format>

Provide escalation assessment:"""

        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=400,
            messages=[{"role": "user", "content": escalation_prompt}]
        )
        
        return self.parse_escalation_assessment(response.content[0].text)
    
    def generate_handoff_summary(self, conversation_history: List[Dict], 
                                customer_context: CustomerContext, 
                                escalation_assessment: Dict) -> str:
        """Generate comprehensive handoff summary for human agent."""
        
        handoff_prompt = f"""<task>
Create a comprehensive handoff summary for the human agent taking over this case.
</task>

<customer_information>
Name: {customer_context.name or 'Not provided'}
Account Type: {customer_context.account_type or 'Standard'}
Customer ID: {customer_context.customer_id}
Email: {customer_context.email or 'Not provided'}
</customer_information>

<issue_summary>
Primary Issue: {customer_context.current_issue or 'See conversation'}
Escalation Type: {escalation_assessment.get('escalation_type', 'General')}
Urgency: {escalation_assessment.get('urgency_level', 'Medium')}
</issue_summary>

<conversation_context>
Key points from conversation:
{self.extract_key_conversation_points(conversation_history)}
</conversation_context>

<handoff_requirements>
1. Provide clear, actionable summary
2. Highlight customer's main concerns
3. Note any promises made or expectations set
4. Include technical details if relevant
5. Suggest next steps
6. Mention customer's emotional state
7. Include any urgency factors
</handoff_requirements>

Generate handoff summary:"""

        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=500,
            messages=[{"role": "user", "content": handoff_prompt}]
        )
        
        return response.content[0].text.strip()
    
    def extract_key_conversation_points(self, conversation_history: List[Dict]) -> str:
        """Extract key points from conversation for handoff."""
        
        if len(conversation_history) <= 4:
            return "\n".join([f"- {turn['content']}" for turn in conversation_history])
        
        extraction_prompt = f"""<conversation>
{chr(10).join([f"{turn['role']}: {turn['content']}" for turn in conversation_history])}
</conversation>

<task>
Extract the 5-7 most important points from this conversation for handoff purposes.
</task>

<guidelines>
1. Focus on the customer's main issue and concerns
2. Include any specific technical details mentioned
3. Note customer preferences or requirements
4. Highlight any commitments made
5. Include emotional context if relevant
</guidelines>

Key conversation points:"""

        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=300,
            messages=[{"role": "user", "content": extraction_prompt}]
        )
        
        return response.content[0].text.strip()
    
    def parse_escalation_assessment(self, assessment_text: str) -> Dict[str, Any]:
        """Parse escalation assessment results."""
        assessment = {}
        
        for line in assessment_text.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                if key == 'confidence':
                    try:
                        assessment[key] = float(value)
                    except ValueError:
                        assessment[key] = 0.5
                elif key == 'escalation_needed':
                    assessment[key] = value.lower() in ['yes', 'true', '1']
                else:
                    assessment[key] = value
        
        return assessment
```

## Multi-Language Support

### Intelligent Language Detection and Response

```python
class MultiLanguageSupport:
    def __init__(self, claude_client):
        self.client = claude_client
        self.supported_languages = {
            "en": "English",
            "es": "Spanish", 
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "nl": "Dutch",
            "ru": "Russian",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese"
        }
    
    def detect_language(self, message: str) -> Dict[str, Any]:
        """Detect language of customer message."""
        
        detection_prompt = f"""<message>
{message}
</message>

<task>
Detect the language of this customer message and assess confidence.
</task>

<supported_languages>
{', '.join([f"{code}: {name}" for code, name in self.supported_languages.items()])}
</supported_languages>

<response_format>
LANGUAGE_CODE: [two-letter code]
LANGUAGE_NAME: [full language name]
CONFIDENCE: [0.0-1.0]
MIXED_LANGUAGE: [yes/no - if multiple languages detected]
</response_format>

Language detection result:"""

        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[{"role": "user", "content": detection_prompt}]
        )
        
        return self.parse_language_detection(response.content[0].text)
    
    def generate_multilingual_response(self, message: str, target_language: str, 
                                     customer_context: CustomerContext) -> str:
        """Generate customer service response in target language."""
        
        language_name = self.supported_languages.get(target_language, "English")
        
        multilingual_prompt = f"""<role>
You are a multilingual customer service agent fluent in {language_name}.
</role>

<customer_message>
{message}
</customer_message>

<customer_context>
Name: {customer_context.name or 'Valued Customer'}
Account Type: {customer_context.account_type or 'Standard'}
</customer_context>

<response_guidelines>
1. Respond naturally in {language_name}
2. Use appropriate cultural context and politeness levels
3. Maintain professional customer service tone
4. Address the customer's specific needs
5. Offer clear next steps
6. Use native speaker level fluency
7. Adapt formality to cultural expectations
</response_guidelines>

<task>
Generate a helpful customer service response in {language_name}.
</task>

Response in {language_name}:"""

        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=400,
            messages=[{"role": "user", "content": multilingual_prompt}]
        )
        
        return response.content[0].text.strip()
    
    def translate_for_escalation(self, conversation_history: List[Dict], 
                                source_language: str) -> str:
        """Translate conversation summary for English-speaking agents."""
        
        if source_language == "en":
            return self.format_conversation_for_escalation(conversation_history)
        
        conversation_text = "\n".join([
            f"{turn['role']}: {turn['content']}" 
            for turn in conversation_history
        ])
        
        translation_prompt = f"""<conversation_in_{source_language}>
{conversation_text}
</conversation_in_{source_language}>

<task>
Translate this customer service conversation to English and provide a clear summary for agent handoff.
</task>

<translation_guidelines>
1. Maintain the exact meaning and tone
2. Preserve customer service context
3. Note any cultural nuances
4. Include emotional indicators
5. Highlight key issues and requests
6. Provide clear, professional English
</translation_guidelines>

English translation and summary:"""

        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=600,
            messages=[{"role": "user", "content": translation_prompt}]
        )
        
        return response.content[0].text.strip()
    
    def parse_language_detection(self, detection_text: str) -> Dict[str, Any]:
        """Parse language detection results."""
        detection = {}
        
        for line in detection_text.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                if key == 'confidence':
                    try:
                        detection[key] = float(value)
                    except ValueError:
                        detection[key] = 0.5
                elif key == 'mixed_language':
                    detection[key] = value.lower() in ['yes', 'true', '1']
                else:
                    detection[key] = value
        
        return detection
```

## Performance Monitoring

### Comprehensive Analytics System

```python
import sqlite3
from datetime import datetime, timedelta
import json

class CustomerServiceAnalytics:
    def __init__(self, db_path: str = "customer_service_analytics.db"):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Initialize analytics database."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT,
                conversation_id TEXT,
                timestamp DATETIME,
                intent TEXT,
                sentiment TEXT,
                urgency TEXT,
                escalated BOOLEAN,
                resolved BOOLEAN,
                satisfaction_score REAL,
                response_time REAL,
                language TEXT,
                agent_type TEXT,
                resolution_category TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                turn_number INTEGER,
                role TEXT,
                content TEXT,
                timestamp DATETIME,
                intent_confidence REAL,
                sentiment_score REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def log_interaction(self, interaction_data: Dict[str, Any]):
        """Log customer interaction for analytics."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO interactions 
            (customer_id, conversation_id, timestamp, intent, sentiment, urgency, 
             escalated, resolved, satisfaction_score, response_time, language, 
             agent_type, resolution_category)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            interaction_data.get('customer_id'),
            interaction_data.get('conversation_id'),
            datetime.now(),
            interaction_data.get('intent'),
            interaction_data.get('sentiment'),
            interaction_data.get('urgency'),
            interaction_data.get('escalated', False),
            interaction_data.get('resolved', False),
            interaction_data.get('satisfaction_score'),
            interaction_data.get('response_time'),
            interaction_data.get('language', 'en'),
            interaction_data.get('agent_type', 'ai'),
            interaction_data.get('resolution_category')
        ))
        
        conn.commit()
        conn.close()
    
    def generate_performance_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Total interactions
        cursor.execute("""
            SELECT COUNT(*) FROM interactions 
            WHERE timestamp >= ? AND timestamp <= ?
        """, (start_date, end_date))
        total_interactions = cursor.fetchone()[0]
        
        # Resolution rate
        cursor.execute("""
            SELECT 
                COUNT(CASE WHEN resolved = 1 THEN 1 END) as resolved,
                COUNT(*) as total
            FROM interactions 
            WHERE timestamp >= ? AND timestamp <= ?
        """, (start_date, end_date))
        resolved_data = cursor.fetchone()
        resolution_rate = (resolved_data[0] / resolved_data[1]) if resolved_data[1] > 0 else 0
        
        # Escalation rate
        cursor.execute("""
            SELECT 
                COUNT(CASE WHEN escalated = 1 THEN 1 END) as escalated,
                COUNT(*) as total
            FROM interactions 
            WHERE timestamp >= ? AND timestamp <= ?
        """, (start_date, end_date))
        escalation_data = cursor.fetchone()
        escalation_rate = (escalation_data[0] / escalation_data[1]) if escalation_data[1] > 0 else 0
        
        # Average satisfaction
        cursor.execute("""
            SELECT AVG(satisfaction_score) 
            FROM interactions 
            WHERE timestamp >= ? AND timestamp <= ? 
            AND satisfaction_score IS NOT NULL
        """, (start_date, end_date))
        avg_satisfaction = cursor.fetchone()[0] or 0
        
        # Intent distribution
        cursor.execute("""
            SELECT intent, COUNT(*) as count
            FROM interactions 
            WHERE timestamp >= ? AND timestamp <= ?
            GROUP BY intent
            ORDER BY count DESC
        """, (start_date, end_date))
        intent_distribution = dict(cursor.fetchall())
        
        # Language distribution
        cursor.execute("""
            SELECT language, COUNT(*) as count
            FROM interactions 
            WHERE timestamp >= ? AND timestamp <= ?
            GROUP BY language
            ORDER BY count DESC
        """, (start_date, end_date))
        language_distribution = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days
            },
            "metrics": {
                "total_interactions": total_interactions,
                "resolution_rate": round(resolution_rate * 100, 2),
                "escalation_rate": round(escalation_rate * 100, 2),
                "average_satisfaction": round(avg_satisfaction, 2)
            },
            "distributions": {
                "intent": intent_distribution,
                "language": language_distribution
            },
            "generated_at": datetime.now().isoformat()
        }
    
    def get_trending_issues(self, days: int = 7) -> List[Dict[str, Any]]:
        """Identify trending customer issues."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        cursor.execute("""
            SELECT 
                intent,
                COUNT(*) as current_period_count,
                AVG(CASE WHEN escalated = 1 THEN 1.0 ELSE 0.0 END) as escalation_rate,
                AVG(satisfaction_score) as avg_satisfaction
            FROM interactions 
            WHERE timestamp >= ? AND timestamp <= ?
            GROUP BY intent
            HAVING COUNT(*) >= 5
            ORDER BY current_period_count DESC
        """, (start_date, end_date))
        
        trending_issues = []
        for row in cursor.fetchall():
            trending_issues.append({
                "intent": row[0],
                "volume": row[1],
                "escalation_rate": round(row[2] * 100, 2),
                "avg_satisfaction": round(row[3], 2) if row[3] else None
            })
        
        conn.close()
        return trending_issues
```

## Best Practices Summary

### Customer Service Excellence

1. **Empathy and Understanding**: Always acknowledge customer emotions and show genuine concern
2. **Clear Communication**: Use language appropriate to customer's technical level and cultural context
3. **Proactive Problem Solving**: Anticipate needs and offer comprehensive solutions
4. **Follow-up and Closure**: Ensure issues are fully resolved and customer satisfaction confirmed

### Technical Implementation

1. **Intent Classification**: Use multiple signals (keywords, context, sentiment) for accurate intent detection
2. **Context Preservation**: Maintain conversation state and customer history throughout interactions
3. **Escalation Logic**: Implement clear criteria for when and how to escalate to human agents
4. **Performance Monitoring**: Track key metrics and continuously improve based on data insights

### Scalability Considerations

1. **Caching Strategy**: Cache common responses and knowledge base queries for faster performance
2. **Load Balancing**: Distribute conversations across multiple agent instances
3. **Database Optimization**: Use appropriate indexing for customer data and conversation history
4. **API Rate Limiting**: Implement respectful rate limiting for external service calls

This comprehensive guide provides a complete framework for building sophisticated customer service agents with Claude, from basic conversation handling to advanced multi-language support and performance analytics.