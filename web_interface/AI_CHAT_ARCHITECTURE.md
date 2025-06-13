# AI Chat GUI with Inline Editing Architecture

## Overview
This document outlines the architecture for implementing an AI chat interface with inline editing capabilities, similar to Claude Artifacts, ChatGPT Canvas, and Google Gemini Canvas.

## Research Summary

### Existing Implementations Analysis

#### Claude Artifacts
- **Dual-pane interface**: Chat on left, artifact view on right
- **Inline editing**: Highlight-to-edit with floating toolbar
- **Version control**: Automatic versioning with navigation arrows
- **Content types**: Text, code, visualizations, apps
- **Real-time updates**: Changes reflected immediately
- **Features**: Copy, download, publish, code view toggle

#### ChatGPT Canvas
- **Collaborative workspace**: Side-by-side editing and chat
- **Built-in shortcuts**: Length adjustment, reading level, polish, emojis
- **Comment system**: Inline suggestions and feedback
- **Direct editing**: Click-to-edit functionality
- **Context awareness**: Automatic activation for appropriate tasks
- **Version control**: Back button for previous versions

#### Google Gemini Canvas
- **Real-time editing**: Changes appear instantly
- **Text selection**: Select-and-ask functionality
- **Quick adjustments**: Length and tone controls
- **Formatting tools**: Bold, italic, lists via toolbar
- **AI integration**: Add Gemini-powered features to apps
- **Preview capabilities**: Visual representation of code

## Proposed Architecture

### 1. Frontend Architecture

#### Component Structure
```
ai-chat-interface/
├── components/
│   ├── chat/
│   │   ├── ChatContainer.vue
│   │   ├── MessageList.vue
│   │   ├── MessageInput.vue
│   │   ├── TypingIndicator.vue
│   │   └── MessageBubble.vue
│   ├── canvas/
│   │   ├── CanvasContainer.vue
│   │   ├── ArtifactView.vue
│   │   ├── InlineEditor.vue
│   │   ├── FloatingToolbar.vue
│   │   ├── VersionControl.vue
│   │   └── FormatToolbar.vue
│   ├── shared/
│   │   ├── LoadingSpinner.vue
│   │   ├── ToastNotification.vue
│   │   └── ConfirmDialog.vue
│   └── layout/
│       ├── SplitPane.vue
│       ├── ResizablePanel.vue
│       └── TabContainer.vue
├── stores/
│   ├── chat.js
│   ├── canvas.js
│   ├── artifacts.js
│   └── providers.js
├── services/
│   ├── api/
│   │   ├── openai.js
│   │   ├── claude.js
│   │   ├── gemini.js
│   │   └── base.js
│   ├── storage/
│   │   ├── localStorage.js
│   │   ├── indexedDB.js
│   │   └── cache.js
│   └── utils/
│       ├── diff.js
│       ├── highlight.js
│       └── export.js
└── assets/
    ├── styles/
    └── icons/
```

#### Technology Stack
- **Framework**: Vue.js 3 (for existing integration)
- **State Management**: Pinia
- **Styling**: Tailwind CSS (already in use)
- **Text Editor**: CodeMirror 6 for code editing
- **Rich Text**: TipTap for document editing
- **Diff Engine**: Monaco Editor's diff capabilities
- **Real-time**: WebSocket integration

### 2. Core Features Implementation

#### 2.1 Dual-Pane Interface
```vue
<template>
  <div class="chat-interface flex h-screen">
    <!-- Chat Pane -->
    <ResizablePanel 
      :minWidth="300" 
      :defaultWidth="40"
      class="chat-pane border-r border-gray-200"
    >
      <ChatContainer />
    </ResizablePanel>
    
    <!-- Canvas Pane -->
    <ResizablePanel 
      :minWidth="400" 
      :defaultWidth="60"
      class="canvas-pane"
    >
      <CanvasContainer v-if="hasActiveArtifact" />
      <EmptyState v-else />
    </ResizablePanel>
  </div>
</template>
```

#### 2.2 Inline Editing System
```typescript
interface InlineEditingSystem {
  // Selection handling
  onTextSelection(range: Range): void;
  showFloatingToolbar(position: Position): void;
  hideFloatingToolbar(): void;
  
  // Edit operations
  editSelection(prompt: string): Promise<void>;
  explainSelection(): Promise<void>;
  improveSelection(type: 'grammar' | 'clarity' | 'style'): Promise<void>;
  
  // Version management
  createVersion(content: string): Version;
  restoreVersion(versionId: string): void;
  compareVersions(v1: string, v2: string): Diff[];
}
```

#### 2.3 Artifact Management
```typescript
interface Artifact {
  id: string;
  type: 'text' | 'code' | 'html' | 'react' | 'diagram';
  title: string;
  content: string;
  language?: string;
  versions: Version[];
  metadata: {
    createdAt: Date;
    updatedAt: Date;
    wordCount?: number;
    lineCount?: number;
  };
}

interface Version {
  id: string;
  content: string;
  timestamp: Date;
  changes: ChangeRecord[];
  isCheckpoint: boolean;
}
```

### 3. Backend Architecture

#### 3.1 API Endpoints
```python
# New endpoints to add to existing Flask app

@app.route('/api/chat/conversations', methods=['POST'])
def create_conversation():
    """Create new chat conversation"""
    pass

@app.route('/api/chat/conversations/<conversation_id>/messages', methods=['POST'])
def send_message(conversation_id):
    """Send message and get AI response"""
    pass

@app.route('/api/artifacts', methods=['POST'])
def create_artifact():
    """Create new artifact"""
    pass

@app.route('/api/artifacts/<artifact_id>/edit', methods=['POST'])
def edit_artifact(artifact_id):
    """Edit specific part of artifact"""
    pass

@app.route('/api/artifacts/<artifact_id>/versions', methods=['GET'])
def get_artifact_versions(artifact_id):
    """Get version history"""
    pass

@app.route('/api/providers/<provider>/chat', methods=['POST'])
def chat_with_provider(provider):
    """Route to specific AI provider"""
    pass
```

#### 3.2 Database Schema
```sql
-- New tables to add to existing SQLite database

CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON
);

CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT REFERENCES conversations(id),
    role TEXT CHECK(role IN ('user', 'assistant', 'system')),
    content TEXT,
    artifact_id TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON
);

CREATE TABLE artifacts (
    id TEXT PRIMARY KEY,
    conversation_id TEXT REFERENCES conversations(id),
    type TEXT,
    title TEXT,
    content TEXT,
    language TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE artifact_versions (
    id TEXT PRIMARY KEY,
    artifact_id TEXT REFERENCES artifacts(id),
    content TEXT,
    changes JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_checkpoint BOOLEAN DEFAULT FALSE
);
```

### 4. Integration Strategy

#### 4.1 With Existing Web Interface
- **Shared header/navigation**: Maintain existing Artificial Garden branding
- **Prompt library integration**: Quick access to existing prompts
- **Settings persistence**: Use existing localStorage patterns
- **Styling consistency**: Extend current Tailwind theme

#### 4.2 Progressive Enhancement
1. **Phase 1**: Basic chat interface
2. **Phase 2**: Artifact creation and viewing
3. **Phase 3**: Inline editing capabilities
4. **Phase 4**: Multi-provider support
5. **Phase 5**: Collaboration features

### 5. Technical Implementation Details

#### 5.1 Text Selection and Editing
```javascript
class InlineEditor {
  constructor(container) {
    this.container = container;
    this.setupSelectionHandlers();
  }
  
  setupSelectionHandlers() {
    this.container.addEventListener('mouseup', this.handleSelection.bind(this));
    this.container.addEventListener('keyup', this.handleSelection.bind(this));
  }
  
  handleSelection(event) {
    const selection = window.getSelection();
    if (selection.rangeCount > 0 && !selection.isCollapsed) {
      const range = selection.getRangeAt(0);
      const rect = range.getBoundingClientRect();
      this.showFloatingToolbar(rect);
    } else {
      this.hideFloatingToolbar();
    }
  }
  
  showFloatingToolbar(rect) {
    const toolbar = document.getElementById('floating-toolbar');
    toolbar.style.left = `${rect.left + rect.width / 2}px`;
    toolbar.style.top = `${rect.top - 50}px`;
    toolbar.classList.remove('hidden');
  }
}
```

#### 5.2 Real-time Updates
```javascript
class ArtifactSync {
  constructor(artifactId) {
    this.artifactId = artifactId;
    this.debounceTimer = null;
    this.setupAutoSave();
  }
  
  setupAutoSave() {
    this.editor.on('change', (content) => {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = setTimeout(() => {
        this.saveVersion(content);
      }, 1000);
    });
  }
  
  async saveVersion(content) {
    try {
      await fetch(`/api/artifacts/${this.artifactId}/versions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content, timestamp: Date.now() })
      });
    } catch (error) {
      console.error('Failed to save version:', error);
    }
  }
}
```

#### 5.3 Version Control System
```javascript
class VersionManager {
  constructor(artifactId) {
    this.artifactId = artifactId;
    this.versions = [];
    this.currentIndex = 0;
  }
  
  async loadVersions() {
    const response = await fetch(`/api/artifacts/${this.artifactId}/versions`);
    this.versions = await response.json();
    this.currentIndex = this.versions.length - 1;
  }
  
  canUndo() {
    return this.currentIndex > 0;
  }
  
  canRedo() {
    return this.currentIndex < this.versions.length - 1;
  }
  
  undo() {
    if (this.canUndo()) {
      this.currentIndex--;
      return this.versions[this.currentIndex];
    }
  }
  
  redo() {
    if (this.canRedo()) {
      this.currentIndex++;
      return this.versions[this.currentIndex];
    }
  }
}
```

### 6. Security Considerations

#### 6.1 Input Sanitization
- Sanitize all user inputs before processing
- Validate artifact content for malicious code
- Implement CSP headers for iframe rendering

#### 6.2 API Security
- Rate limiting for AI provider requests
- API key encryption and secure storage
- Session management and CSRF protection

#### 6.3 Content Safety
- Filter harmful content in generated artifacts
- Validate code execution in sandboxed environments
- Implement user permission levels

### 7. Performance Optimizations

#### 7.1 Client-side
- Virtual scrolling for long conversations
- Lazy loading of artifact content
- Debounced auto-saving
- Efficient diff algorithms

#### 7.2 Server-side
- Caching of AI responses
- Database query optimization
- Async processing for long-running tasks
- Connection pooling for AI providers

### 8. Accessibility Features

- Keyboard navigation support
- Screen reader compatibility
- High contrast mode
- Font size adjustments
- Focus management for dynamic content

### 9. Mobile Responsiveness

- Touch-friendly interface
- Responsive layout adjustments
- Swipe gestures for navigation
- Mobile-optimized toolbars

## Implementation Timeline

### Week 1: Foundation
- Set up component structure
- Create basic chat interface
- Implement message handling

### Week 2: Canvas Integration
- Build artifact container
- Add basic viewing capabilities
- Implement version control

### Week 3: Inline Editing
- Text selection handling
- Floating toolbar implementation
- Edit operations

### Week 4: AI Integration
- Connect to AI providers
- Implement artifact generation
- Add error handling

### Week 5: Enhancement
- Advanced editing features
- Export/import functionality
- Performance optimization

### Week 6: Testing & Polish
- Cross-browser testing
- Accessibility audit
- User experience refinement

## Conclusion

This architecture provides a comprehensive foundation for building a sophisticated AI chat interface with inline editing capabilities. The modular design allows for incremental implementation while maintaining integration with the existing prompt library system.