# Repository Management Scripts

Python tools for maintaining The Big Everything Prompt Library.

## Available Scripts

### idxtool.py
GPT indexing and searching tool for managing custom instructions.

```bash
# Rebuild table of contents
python idxtool.py --toc

# Find GPT by ID or URL
python idxtool.py --find-gpt 3rtbLUIUO

# Rename GPT files to include ID prefix
python idxtool.py --rename

# Create template from ChatGPT URL
python idxtool.py --template https://chat.openai.com/g/g-example
```

### maintenance.py
Repository health checker and link validator.

```bash
# Run comprehensive health check
python maintenance.py
```

**Features:**
- Validates markdown links in README files
- Identifies directories missing README files
- Generates repository statistics
- Reports broken links and organizational issues

### gptparser.py
Core parsing module for GPT markdown files.

**Supported GPT fields:**
- URL, Title, Description
- Instructions, Actions
- Knowledge Base Files
- Protected status

## Usage Notes

- All scripts assume Python 3.11+
- Run from repository root directory
- Scripts maintain existing file organization patterns
- Use for quality assurance and maintenance tasks

## File Naming Conventions

**GPT Custom Instructions:** `{GPTID}_{Name}.md` or `{GPTID}_{Name}[version].md`