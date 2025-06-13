"""
Chat API endpoints for AI Chat Interface
Provides backend functionality for the AI chat interface with inline editing capabilities.
"""

import os
import uuid
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

# AI Provider imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    provider: str = "openai"
    conversation_id: Optional[str] = None

class ArtifactEdit(BaseModel):
    artifact_id: str
    selected_text: str
    prompt: str
    full_content: str

class Artifact(BaseModel):
    id: str
    type: str
    title: str
    content: str
    language: Optional[str] = None
    created_at: datetime
    updated_at: datetime

# Database setup
DB_PATH = Path(__file__).parent / "chat.db"

def init_chat_db():
    """Initialize the chat database with required tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Conversations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSON
        )
    ''')
    
    # Messages table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT REFERENCES conversations(id),
            role TEXT CHECK(role IN ('user', 'assistant', 'system')),
            content TEXT,
            artifact_id TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSON
        )
    ''')
    
    # Artifacts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS artifacts (
            id TEXT PRIMARY KEY,
            conversation_id TEXT REFERENCES conversations(id),
            type TEXT,
            title TEXT,
            content TEXT,
            language TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Artifact versions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS artifact_versions (
            id TEXT PRIMARY KEY,
            artifact_id TEXT REFERENCES artifacts(id),
            content TEXT,
            changes JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_checkpoint BOOLEAN DEFAULT FALSE
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database on module load
init_chat_db()

# Router setup
router = APIRouter(prefix="/api", tags=["chat"])

class ChatProcessor:
    """Handles AI provider interactions and artifact generation."""
    
    def __init__(self):
        self.providers = {}
        self._setup_providers()
    
    def _setup_providers(self):
        """Initialize available AI providers."""
        if OPENAI_AVAILABLE:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.providers["openai"] = self._chat_openai
        
        if ANTHROPIC_AVAILABLE:
            self.anthropic_client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
            self.providers["claude"] = self._chat_claude
        
        if GEMINI_AVAILABLE:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.providers["gemini"] = self._chat_gemini
    
    async def process_message(self, message: str, provider: str, conversation_id: str) -> Dict[str, Any]:
        """Process a chat message and return response with potential artifact."""
        if provider not in self.providers:
            raise HTTPException(status_code=400, detail=f"Provider {provider} not available")
        
        # Check if message might need artifact creation
        needs_artifact = self._should_create_artifact(message)
        
        try:
            response = await self.providers[provider](message, needs_artifact)
            
            # Save conversation message
            self._save_message(conversation_id, "user", message)
            
            artifact = None
            if response.get("artifact"):
                artifact = self._create_artifact(
                    conversation_id=conversation_id,
                    artifact_type=response["artifact"]["type"],
                    title=response["artifact"]["title"],
                    content=response["artifact"]["content"],
                    language=response["artifact"].get("language")
                )
            
            # Save assistant message
            self._save_message(
                conversation_id, 
                "assistant", 
                response["message"], 
                artifact_id=artifact["id"] if artifact else None
            )
            
            return {
                "message": response["message"],
                "artifact": artifact,
                "provider": provider
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")
    
    def _should_create_artifact(self, message: str) -> bool:
        """Determine if a message should create an artifact."""
        artifact_keywords = [
            "write code", "create", "generate", "build", "make",
            "html", "css", "javascript", "python", "react",
            "document", "article", "essay", "story", "poem",
            "diagram", "chart", "visualization"
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in artifact_keywords)
    
    async def _chat_openai(self, message: str, needs_artifact: bool) -> Dict[str, Any]:
        """Handle OpenAI chat completion."""
        system_prompt = self._get_system_prompt(needs_artifact)
        
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        return self._parse_response(content)
    
    async def _chat_claude(self, message: str, needs_artifact: bool) -> Dict[str, Any]:
        """Handle Claude chat completion."""
        system_prompt = self._get_system_prompt(needs_artifact)
        
        response = self.anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4000,
            system=system_prompt,
            messages=[{"role": "user", "content": message}]
        )
        
        content = response.content[0].text
        return self._parse_response(content)
    
    async def _chat_gemini(self, message: str, needs_artifact: bool) -> Dict[str, Any]:
        """Handle Gemini chat completion."""
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        system_prompt = self._get_system_prompt(needs_artifact)
        
        response = model.generate_content(f"{system_prompt}\n\nUser: {message}")
        content = response.text
        return self._parse_response(content)
    
    def _get_system_prompt(self, needs_artifact: bool) -> str:
        """Get system prompt based on whether artifact creation is expected."""
        base_prompt = """You are an AI assistant integrated with an interactive canvas system. You can create artifacts (code, documents, visualizations) that users can edit inline."""
        
        if needs_artifact:
            return base_prompt + """
            
When creating content that would benefit from inline editing, format your response like this:

ARTIFACT_START
type: [code|text|html|react|diagram]
title: [Brief title for the artifact]
language: [programming language if applicable]
---
[The actual content here]
ARTIFACT_END

Then provide a brief explanation of what you created.
"""
        else:
            return base_prompt + "\n\nProvide helpful responses to user questions. Only create artifacts when specifically requested or when the content would benefit from inline editing."
    
    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse AI response to extract artifacts."""
        import re
        
        # Look for artifact markers
        artifact_pattern = r'ARTIFACT_START\s*\ntype:\s*(\w+)\s*\ntitle:\s*(.+?)\s*\n(?:language:\s*(\w+)\s*\n)?---\s*\n(.*?)\nARTIFACT_END'
        
        match = re.search(artifact_pattern, content, re.DOTALL)
        
        if match:
            artifact_type, title, language, artifact_content = match.groups()
            
            # Remove artifact from main message
            message = re.sub(artifact_pattern, "", content, flags=re.DOTALL).strip()
            
            return {
                "message": message,
                "artifact": {
                    "type": artifact_type,
                    "title": title.strip(),
                    "content": artifact_content.strip(),
                    "language": language
                }
            }
        
        return {"message": content}
    
    def _create_artifact(self, conversation_id: str, artifact_type: str, title: str, 
                        content: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Create a new artifact in the database."""
        artifact_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO artifacts (id, conversation_id, type, title, content, language)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (artifact_id, conversation_id, artifact_type, title, content, language))
        
        # Create initial version
        version_id = str(uuid.uuid4())
        cursor.execute('''
            INSERT INTO artifact_versions (id, artifact_id, content, is_checkpoint)
            VALUES (?, ?, ?, ?)
        ''', (version_id, artifact_id, content, True))
        
        conn.commit()
        conn.close()
        
        return {
            "id": artifact_id,
            "type": artifact_type,
            "title": title,
            "content": content,
            "language": language,
            "created_at": datetime.now().isoformat()
        }
    
    def _save_message(self, conversation_id: str, role: str, content: str, 
                     artifact_id: Optional[str] = None):
        """Save a message to the database."""
        message_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Ensure conversation exists
        cursor.execute('''
            INSERT OR IGNORE INTO conversations (id, title)
            VALUES (?, ?)
        ''', (conversation_id, "New Conversation"))
        
        cursor.execute('''
            INSERT INTO messages (id, conversation_id, role, content, artifact_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (message_id, conversation_id, role, content, artifact_id))
        
        conn.commit()
        conn.close()

# Initialize chat processor
chat_processor = ChatProcessor()

@router.post("/chat")
async def chat_endpoint(message: ChatMessage):
    """Main chat endpoint for processing user messages."""
    conversation_id = message.conversation_id or str(uuid.uuid4())
    
    response = await chat_processor.process_message(
        message.message,
        message.provider,
        conversation_id
    )
    
    response["conversation_id"] = conversation_id
    return response

@router.post("/edit-artifact")
async def edit_artifact_endpoint(edit_request: ArtifactEdit):
    """Edit a specific part of an artifact using AI."""
    try:
        # Get the original artifact
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM artifacts WHERE id = ?', (edit_request.artifact_id,))
        artifact_row = cursor.fetchone()
        
        if not artifact_row:
            raise HTTPException(status_code=404, detail="Artifact not found")
        
        # Use AI to process the edit
        edit_prompt = f"""
        Please improve the following selected text: "{edit_request.selected_text}"
        
        User request: {edit_request.prompt}
        
        Full context:
        {edit_request.full_content}
        
        Return only the improved version of the selected text, maintaining the same format and style.
        """
        
        # Use OpenAI for editing (could be configurable)
        if OPENAI_AVAILABLE:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that improves text while maintaining context and style."},
                    {"role": "user", "content": edit_prompt}
                ],
                temperature=0.3
            )
            
            improved_text = response.choices[0].message.content.strip()
            
            # Replace the selected text in the full content
            new_content = edit_request.full_content.replace(
                edit_request.selected_text, 
                improved_text
            )
            
            # Save new version
            version_id = str(uuid.uuid4())
            cursor.execute('''
                INSERT INTO artifact_versions (id, artifact_id, content)
                VALUES (?, ?, ?)
            ''', (version_id, edit_request.artifact_id, new_content))
            
            # Update artifact
            cursor.execute('''
                UPDATE artifacts SET content = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (new_content, edit_request.artifact_id))
            
            conn.commit()
            
            return {
                "success": True,
                "new_content": new_content,
                "improved_text": improved_text
            }
        
        else:
            raise HTTPException(status_code=500, detail="OpenAI not available for editing")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error editing artifact: {str(e)}")
    
    finally:
        conn.close()

@router.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str):
    """Get all messages for a conversation."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT m.*, a.type as artifact_type, a.title as artifact_title
        FROM messages m
        LEFT JOIN artifacts a ON m.artifact_id = a.id
        WHERE m.conversation_id = ?
        ORDER BY m.timestamp ASC
    ''', (conversation_id,))
    
    messages = []
    for row in cursor.fetchall():
        message = {
            "id": row[0],
            "role": row[2],
            "content": row[3],
            "timestamp": row[5],
            "artifact": {
                "id": row[4],
                "type": row[7],
                "title": row[8]
            } if row[4] else None
        }
        messages.append(message)
    
    conn.close()
    return {"messages": messages}

@router.get("/artifacts/{artifact_id}/versions")
async def get_artifact_versions(artifact_id: str):
    """Get version history for an artifact."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, content, created_at, is_checkpoint
        FROM artifact_versions
        WHERE artifact_id = ?
        ORDER BY created_at ASC
    ''', (artifact_id,))
    
    versions = []
    for row in cursor.fetchall():
        versions.append({
            "id": row[0],
            "content": row[1],
            "timestamp": row[2],
            "is_checkpoint": bool(row[3])
        })
    
    conn.close()
    return {"versions": versions}

@router.get("/conversations")
async def get_conversations():
    """Get list of all conversations."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT c.*, COUNT(m.id) as message_count
        FROM conversations c
        LEFT JOIN messages m ON c.id = m.conversation_id
        GROUP BY c.id
        ORDER BY c.updated_at DESC
    ''')
    
    conversations = []
    for row in cursor.fetchall():
        conversations.append({
            "id": row[0],
            "title": row[1],
            "created_at": row[2],
            "updated_at": row[3],
            "message_count": row[5]
        })
    
    conn.close()
    return {"conversations": conversations}

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation and all associated data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Delete artifact versions
        cursor.execute('''
            DELETE FROM artifact_versions 
            WHERE artifact_id IN (
                SELECT id FROM artifacts WHERE conversation_id = ?
            )
        ''', (conversation_id,))
        
        # Delete artifacts
        cursor.execute('DELETE FROM artifacts WHERE conversation_id = ?', (conversation_id,))
        
        # Delete messages
        cursor.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
        
        # Delete conversation
        cursor.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))
        
        conn.commit()
        return {"success": True}
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting conversation: {str(e)}")
    
    finally:
        conn.close()