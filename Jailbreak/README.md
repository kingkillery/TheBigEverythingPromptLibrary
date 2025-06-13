# Jailbreak Research & Documentation

This directory contains documented jailbreak techniques for various LLM systems. This content is maintained for **educational and security research purposes only**, to help developers understand attack vectors and improve system defenses.

## ⚠️ Important Disclaimer

- **Educational Purpose Only**: These techniques are documented for security research and defense development
- **Responsible Use**: Do not use these techniques to circumvent safety measures in production systems
- **Attribution**: Original researchers are credited where known

## 📊 By Attack Method

### Emoji-Based Attacks
- **[GPT-4O Emoji Jailbreak](./OpenAI/gpt4o-via-emojis-06082024.md)** - Hyper-token-efficient adversarial emoji attacks (elder_plinius, 06/08/2024)

### Role-Playing/Identity Manipulation
- **[AGI Impersonation](./OpenAI/gpt4o-agi_db-10232024.md)** - "Fooled by AGI" technique targeting GPT-4O (Unknown author, 10/23/2024)

### Platform-Specific Exploits
- **[Meta.ai Jailbreak](./Meta.ai/elder_plinius_04182024.md)** - Platform-specific technique (elder_plinius, 04/18/2024)
- **[Cohere Command R+](./Cohere/CommandR_Plus_04112024.md)** - Command R+ specific technique (04/11/2024)

## 🏢 By Platform

### OpenAI (GPT-4O)
- [AGI Database Simulation](./OpenAI/gpt4o-agi_db-10232024.md) - Role-playing as advanced AGI system
- [Elder Plinius Method](./OpenAI/gpt4o-plinius-05132024.md) - Early GPT-4O jailbreak
- [Emoji Attack Vector](./OpenAI/gpt4o-via-emojis-06082024.md) - Ultra-short emoji-based prompts

### Cohere
- [Command R+ Bypass](./Cohere/CommandR_Plus_04112024.md) - Platform-specific technique

### Meta.ai
- [Elder Plinius Technique](./Meta.ai/elder_plinius_04182024.md) - Meta.ai specific approach

## 📈 Timeline
- **04/11/2024**: Cohere Command R+ technique documented
- **04/18/2024**: Meta.ai jailbreak by elder_plinius
- **05/13/2024**: Early GPT-4O jailbreak by elder_plinius  
- **06/08/2024**: Emoji-based attack on GPT-4O
- **10/23/2024**: AGI impersonation technique

## 🔗 Related Resources

- **[Security Documentation](../Security/README.md)** - Defense techniques and protection methods
- **[System Prompts](../SystemPrompts/README.md)** - Official system prompts with security implementations

## 🌱 Share New Jailbreak Findings
Discovered a novel jailbreak or red-team prompt? Use the Prompt Garden's **Plant** flow.  Once it passes safety review (we block disallowed content but allow ethical research examples) it will be merged here automatically.
