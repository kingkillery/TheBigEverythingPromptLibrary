# Jules Coding Agent - Awesome Prompts Collection

*Curated prompts for Jules, an async coding agent from Google Labs. These prompts are designed to work with Jules' specific capabilities and workflow.*

## About Jules

Jules is Google Labs' async coding agent that excels at understanding repository context and making precise, contextual changes. Visit [jules.google.com](https://jules.google.com) to learn more.

---

## Everyday Development Tasks

**Refactoring & Modernization**
- `// Refactor {a specific} file from {x} to {y}...` - General-purpose, applies to any language or repo
- `// Convert these commonJS modules to ES modules...` - JS/TS projects modernizing legacy code
- `// Turn this callback-based code into async/await...` - JavaScript or Python codebases improving async logic
- `// Implement a data class for this dictionary structure...` - Python projects moving towards structured data handling with dataclasses or Pydantic

**Testing & Quality**
- `// Add a test suite...` - Useful for repos lacking test coverage
- `// Add type hints to {a specific} Python function...` - Python codebases transitioning to typed code
- `// Generate mock data for {a specific} schema...` - APIs, frontends, or test-heavy environments

---

## Debugging & Troubleshooting

**Error Diagnosis**
- `// Help me fix {a specific} error...` - For any repo where you're stuck on a runtime or build error
- `// Why is {this specific snippet of code} slow?` - Performance profiling for loops, functions, or queries
- `// Trace why this value is undefined...` - Frontend and backend JS/TS bugs
- `// Diagnose this memory leak...` - Server-side apps or long-running processes

**Debugging Tools**
- `// Add logging to help debug this issue...` - Useful when troubleshooting silent failures
- `// Find race conditions in this async code` - Concurrent systems in JS, Python, Go, etc.
- `// Add print statements to trace the execution flow of this Python script...` - For debugging complex Python scripts

---

## Documentation & Code Quality

**Documentation Generation**
- `// Write a README for this project` - Any repo lacking a basic project overview
- `// Add comments to this code` - Improves maintainability of complex logic
- `// Write API docs for this endpoint` - REST or GraphQL backends
- `// Generate Sphinx-style docstrings for this Python module/class/function...` - Python projects using Sphinx

---

## Testing Strategies

**Test Implementation**
- `// Add integration tests for this API endpoint` - Express, FastAPI, Django, Flask apps
- `// Write a test that mocks fetch` - Browser-side fetch or axios logic
- `// Convert this test from Mocha to Jest` - JS test suite migrations
- `// Generate property-based tests for this function` - Functional or logic-heavy code

**Advanced Testing**
- `// Simulate slow network conditions in this test suite` - Web and mobile apps
- `// Write a test to ensure backward compatibility for this function` - Library or SDK maintainers
- `// Write a Pytest fixture to mock this external API call...` - Python projects using Pytest and robust mocking

---

## Package & Dependency Management

**Maintenance & Updates**
- `// Upgrade my linter and autofix breaking config changes` - JS/TS repos using ESLint or Prettier
- `// Show me the changelog for React 19` - Web frontend apps using React
- `// Which dependencies can I safely remove?` - Bloated or legacy codebases
- `// Check if these packages are still maintained` - Security-conscious or long-term projects
- `// Set up Renovate or Dependabot for auto-updates` - Best for active projects with CI/CD

---

## AI-Native Development Tasks

**Repository Analysis**
- `// Analyze this repo and generate 3 feature ideas` - Vision-stage or greenfield products
- `// Identify tech debt in this file` - Codebases with messy or fragile logic
- `// Find duplicate logic across files` - Sprawling repos lacking DRY practices
- `// Cluster related functions and suggest refactors` - Projects with lots of utils or helpers

**Task Optimization**
- `// Help me scope this issue so Jules can solve it` - For working with Jules on real issues
- `// Convert this function into a reusable plugin/module` - Componentizing logic-heavy code
- `// Refactor this Python function to be more amenable to parallel processing...` - Optimizing performance in computationally intensive Python applications

---

## Context & Communication

**Project Management**
- `// Write a status update based on recent commits` - Managerial and async communication
- `// Summarize all changes in the last 7 days` - Catching up after time off

---

## Fun & Experimental Features

**User Experience Enhancements**
- `// Add a confetti animation when {a specific} action succeeds` - Frontend web apps with user delight moments
- `// Inject a developer joke when {a specific} build finishes` - Personal projects or team tools
- `// Build a mini CLI game that runs in the terminal` - For learning or community fun
- `// Add a dark mode Easter egg to this UI` - Design-heavy frontend projects
- `// Turn this tool into a GitHub App` - Reusable, platform-integrated tools

---

## Project Initialization

**Starting New Projects**
- `// What's going on in this repo?` - Great for legacy repos or onboarding onto unfamiliar code
- `// Initialize a new Express app with CORS enabled` - Web backend projects using Node.js and Express
- `// Set up a monorepo using Turborepo and PNPM` - Multi-package JS/TS projects with shared dependencies
- `// Bootstrap a Python project with Poetry and Pytest` - Python repos aiming for clean dependency and test setup
- `// Create a starter template for a Chrome extension` - Browser extension development
- `// I want to build a web scraper—start me off` - Data scraping or automation tools using Python/Node

---

## Usage Tips

**Best Practices for Jules:**
- Be specific about file names and locations when possible
- Use descriptive context about your project type and goals
- Reference existing patterns in your codebase for consistency
- Break complex tasks into smaller, focused prompts when needed

**Prompt Customization:**
- Replace `{a specific}` placeholders with actual file names, functions, or components
- Add project-specific context to improve relevance
- Combine prompts for complex multi-step operations

---

*Original collection curated for Jules coding agent. These prompts leverage Jules' async capabilities and repository understanding for maximum effectiveness.*
