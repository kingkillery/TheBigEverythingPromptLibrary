# Developer ChatGPT Prompts Collection

*High-quality developer-focused prompts from PickleBoxer's personal collection. These prompts are designed specifically for software engineers, covering code generation, refactoring, debugging, and development workflows.*

## Code Refactoring & Improvement

### **Multi-Prompt Code Refactoring Workflow**
```
Step 1 - Re-write for Modern Standards:
Review the following code and re-write it to modern es6 programming standards and formatting:
[insert code here]

Step 2 - Security & Logic Review:
Review your provided code 'tempFunction' for any logical or security concerns and provide a list of recommendations.

Step 3 - Validate Recommendations:
Review your above recommendations. Tell me why you were wrong and if any recommendations were overlooked or incorrectly added?

Step 4 - Final Implementation:
Re-write the 'tempFunction' function based off your review and recommendations.

Step 5 - Create Tests:
Create two [define technology] tests for the above 'tempFunction' function. One that is expected to pass and one that is expected to fail.
```

### **Modernize Legacy Code**
```
Refactor the following code to modern es6 programming standards:
[INSERT YOUR CODE HERE]
```

### **Split Large Functions**
```
Refactor the following code into multiple methods to improve readability and maintainability:
[INSERT YOUR CODE HERE]
```

### **Performance Optimization**
```
Refactor the following code to improve performance:
[INSERT YOUR CODE HERE]
```

### **Apply Coding Best Practices**
```
Review the following code and refactor it to make it more DRY and adopt the SOLID programming principles.
[INSERT YOUR CODE HERE]
```

### **Follow Style Guidelines**
```
Rewrite the code below following the Google style guidelines for javascript.
[INSERT YOUR CODE HERE]
```

## Code Generation & Boilerplate

### **Generate Boilerplate Functions**
```
Context: I'm creating a software to manage projects
Technologies: Go, PostgreSQL
Description: It's a function that let me find users by its email or username and returns the structure type "Member"
You have to: create the function for me
```

### **Create Dockerfile**
```
Write a Dockerfile for:
[FRAMEWORK]
```

### **Generate Class from JSON**
```
Create a [PLATFORM] class from this JSON object
[JSON]
```

### **Software Architecture Spec**
```
You are a world-class software engineer.
I need you to draft a technical software spec for building the following:
[ DESCRIPTION ]
Think through how you would build it step by step.
Then, respond with the complete spec as a well-organized markdown file.
I will then reply with "build," and you will proceed to implement the exact spec, writing all of the code needed. I will periodically interject with "continue" to prompt you to keep going. Continue until complete.
```

### **Real-time Communication Implementation**
```
I need a piece of code in [INSERT YOUR TECHNOLOGIES HERE] to implement [real-time communication]
```

## Debugging & Error Handling

### **Find and Fix Bugs**
```
I'm developing software in [INSERT YOUR TECHNOLOGIES HERE] and I need you help me to find and fix all the errors in my code, following the best practices. I'll provide you my code and you'll give me the code with all the corrections explained line by line
```

### **Debug React Component**
```
Please find and fix the bug in the [component name] component that is causing [describe the issue].
[INSERT YOUR CODE HERE]
```

### **Error Analysis**
```
I wrote this code [CODE] I got this error [ERROR] How can I fix it? or What does this error mean?
```

### **Improve Error Handling**
```
How can I improve the error handling in my [LANGUAGE] code? [CODE]
```

## Code Documentation & Explanation

### **Explain Code for Non-Technical Users**
```
I don't know how to code, but I want to understand how this works. Explain the following code to me in a way that a non-technical person can understand. Always use Markdown with nice formatting to make it easier to follow. Organize it by sections with headers. Include references to the code as markdown code blocks in each section. The code:
[insert code here]
```

### **Comprehensive Documentation**
```
Please add comprehensive documentation for [file or module name], including clear and concise explanations of its purpose, design, and implementation. Consider including examples of how to use the module, as well as any relevant diagrams or flow charts to help illustrate its workings. Ensure that the documentation is easily accessible to other developers and is updated as the module evolves. Consider using documentation tools such as inline comments, markdown files, or a documentation generator to simplify the process.
[insert code here]
```

### **Code Understanding for New Developers**
```
Context: I'm starting a new position as backend developer and I have to start to understand how some functions are working
Technologies: [INSERT YOUR TECHNOLOGIES HERE]
You have to: explain me the code line by line
[INSERT YOUR CODE HERE]
```

### **Add Code Comments**
```
Add comments to the following code:
[INSERT YOUR CODE HERE]
```

## Architecture & Design

### **Create Architecture Diagram**
```
Write the Mermaid code for an architecture diagram for this solution [DESCRIBE SOLUTION]

Example:
graph TD;
A[Client] -->|HTTP Request| B(API Gateway);
B -->|HTTP Request| C[Service 1];
B -->|HTTP Request| D[Service 2];
C -->|Database Query| E[Database];
D -->|Database Query| E;
```

### **Entity Relationship Diagram**
```
Write the Mermaid code for an entity relationship diagram for these classes [INSERT CLASSES]
```

## Testing & Quality Assurance

### **Generate Unit Tests**
```
Please write unit tests for [file or module name] to ensure its proper functioning
[insert code here]
```

### **Create Test Cases**
```
Create 2 unit tests for the provided code. One for a successful condition and one for failure.
```

## Code Conversion & Migration

### **Language Conversion**
```
Rewrite the following code in Rust:
[INSERT YOUR CODE HERE]
```

### **Get Code Alternatives**
```
I'll provide you with a piece of code that I made and I need you give me alternatives to do the same in other way:
[INSERT YOUR CODE HERE]
```

## Frontend & UI Development

### **Responsive Design Implementation**
```
Please implement responsive design for the [component name] component to ensure that it looks and functions correctly on different screen sizes and devices. Consider using [responsive design technique or library] to achieve this.
[insert code here]
```

### **Internationalization**
```
Please implement internationalization for the [component name] component to ensure that it can be used by users in multiple languages. Consider using [internationalization library or technique] to achieve this.
```

## Utility & Helper Prompts

### **Regular Expression Generator**
```
Write a regular expression that matches / Write a RegEx pattern for:
[REQUEST]
```

### **Generate Documentation**
```
Generate documentation for the code below. You should include detailed instructions to allow a developer to run it on a local machine, explain what the code does, and list vulnerabilities that exist in this code.
[enter code]
```

### **Create Terms and Services**
```
Create terms and services for my website about an [AI tool] called [name].
```

### **Write Technical Blog Post**
```
Write a detailed blog on How to build a [COVID tracker] using React with proper structuring of code.
```

### **Create Cheat Sheet**
```
Write a cheat sheet for [markdown formatting].
```

## Code Review & Analysis

### **Comprehensive Code Review**
```
I'm working on a [LANGUAGE] project and I need you to review my code and suggest improvements. [CODE]
```

### **Security & Vulnerability Review**
```
Review this code for errors and refactor to fix any issues:
[INSERT YOUR CODE HERE]
```

## Prompt Optimization

### **ChatGPT Prompt Optimizer**
```
I'll provide a chatGPT prompt. You'll ask questions to understand the audience and goals, then optimize the prompt for effectiveness and relevance using the principle of specificity.
```

### **Enhanced Descriptive Prompts**
```
[your prompt]
Re-write the above text to be more verbose and include a lot of superfluous description about each thing, use very painting language.
```

---

## Developer Tips & Best Practices

### **Multi-Prompt Approach (Prompt Chaining)**
- **Split Complex Tasks**: Break prompts into multiple steps for better results
- **Single Responsibility**: Keep each prompt focused on one specific outcome
- **Sequential Processing**: Use outputs from one prompt as inputs for the next

### **Effective Prompting Techniques**
1. **Be Specific**: List exactly what you want, what you know, and what to exclude
2. **Provide Examples**: Include expected inputs, data, and desired outputs
3. **Add Context**: Include technology stack, project type, and constraints
4. **Use Reflection**: Ask "Why were you wrong?" to improve accuracy
5. **Request Alternatives**: Ask for multiple approaches to the same problem

### **Code Quality Workflow**
1. **Re-write** → Modern standards and formatting
2. **Review** → Security and logic concerns  
3. **Validate** → Check recommendations for accuracy
4. **Implement** → Final refactored code
5. **Test** → Create unit tests for validation

---

*This collection focuses on practical, workflow-oriented prompts that help developers work more efficiently with AI assistance. Each prompt is designed to produce production-ready code and solutions.*
