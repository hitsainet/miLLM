# Claude Code Custom Command Standards & Conventions

## Overview

This document establishes standardized rules and conventions for creating maintainable, flexible, and conflict-free custom slash commands for Claude Code. These standards ensure consistency across team members and make commands easy to discover, use, and maintain.

## 1. File Structure & Organization

### 1.1 Directory Structure
```
.claude/
├── commands/
│   ├── project/           # Project management commands
│   │   ├── init.md
│   │   ├── setup.md
│   │   └── status.md
│   ├── dev/              # Development workflow commands
│   │   ├── code-review.md
│   │   ├── refactor.md
│   │   └── debug.md
│   ├── test/             # Testing commands
│   │   ├── run-tests.md
│   │   ├── generate-tests.md
│   │   └── coverage.md
│   ├── deploy/           # Deployment commands
│   │   ├── prepare-release.md
│   │   ├── staging.md
│   │   └── production.md
│   ├── security/         # Security-related commands
│   │   ├── audit.md
│   │   ├── scan.md
│   │   └── validate.md
│   ├── docs/             # Documentation commands
│   │   ├── generate.md
│   │   ├── update.md
│   │   └── validate.md
│   └── util/             # Utility commands
│       ├── cleanup.md
│       ├── backup.md
│       └── optimize.md
```

### 1.2 Scope Guidelines
- **Project Commands**: `.claude/commands/` - Shared with team via git
- **User Commands**: `~/.claude/commands/` - Personal commands across all projects
- **Namespacing**: Use subdirectories for logical grouping

## 2. Naming Conventions

### 2.1 Command Naming Rules
- Use **kebab-case** for all command files: `code-review.md`, `run-tests.md`
- Use **descriptive, action-oriented** names that clearly indicate purpose
- Keep names **concise but explicit**: prefer `deploy-staging` over `deploy-to-staging-environment`
- Use **consistent verb patterns**:
  - `create-*` for generation commands
  - `run-*` for execution commands
  - `validate-*` for verification commands
  - `generate-*` for code/content generation

### 2.2 Namespace Conventions
Commands are invoked as `/namespace:command-name` where:
- **namespace** = directory name
- **command-name** = filename without `.md`

Examples:
- `/project:init` → `.claude/commands/project/init.md`
- `/dev:code-review` → `.claude/commands/dev/code-review.md`
- `/test:run-unit` → `.claude/commands/test/run-unit.md`

### 2.3 Reserved Namespace Avoidance
Avoid these prefixes to prevent conflicts with built-in commands:
- `claude`, `anthropic`, `system`
- `help`, `clear`, `model`, `agents`
- `hooks`, `permissions`, `compact`
- `install`, `config`, `debug`

## 3. Command File Structure

### 3.1 Frontmatter Standards
All commands must include structured frontmatter:

```yaml
---
description: "Brief one-line description of what the command does"
argument-hint: "[optional-arg] <required-arg>"
allowed-tools: Bash(git add:*, git status:*), FileEdit, GrepTool
model: claude-3-5-sonnet-20241022
category: "development|testing|deployment|documentation|security|utility"
version: "1.0.0"
author: "team-name or individual"
last-updated: "2025-01-15"
---
```

### 3.2 Command Content Structure
```markdown
# Command Title

Brief description of what this command accomplishes.

## Purpose
Explain the specific problem this command solves.

## Usage
```
/namespace:command-name [arguments]
```

## Parameters
- `$ARGUMENTS` - Description of expected arguments
- Additional parameter documentation if needed

## Instructions

### Pre-execution Checks
1. Verify prerequisites
2. Validate current state
3. Check for required tools/files

### Main Workflow
1. **Step 1**: Clear action with expected outcome
2. **Step 2**: Next action with validation
3. **Step 3**: Final verification

### Post-execution
1. Verify results
2. Update documentation if needed
3. Commit changes if appropriate

## Examples
- `/namespace:command-name simple-example`
- `/namespace:command-name complex-example with-options`

## Dependencies
- List required tools (git, npm, etc.)
- Required files or configurations
- Expected project structure

## Notes
- Important considerations
- Common pitfalls to avoid
- Related commands
```

## 4. Parameter Handling Standards

### 4.1 Argument Processing
- Use `$ARGUMENTS` placeholder for dynamic input
- Validate arguments at the start of command execution
- Provide clear error messages for invalid input
- Support both single arguments and space-separated multiple arguments

### 4.2 Parameter Documentation
- Always document expected argument format in `argument-hint`
- Use standard notation:
  - `<required>` for mandatory arguments
  - `[optional]` for optional arguments
  - `...` for variable number of arguments

Examples:
```yaml
argument-hint: "<issue-number>"           # Single required
argument-hint: "[branch-name]"            # Single optional  
argument-hint: "<command> [options...]"   # Required + optional multiple
argument-hint: "[file-pattern] [--flag]"  # Multiple optional
```

## 5. Help System Standards

### 5.1 Built-in Help Integration
Commands automatically appear in `/help` with their description from frontmatter.

### 5.2 Self-Documenting Commands
Create help commands for complex command suites:
```markdown
# .claude/commands/namespace/help.md
---
description: "Show help for all namespace commands"
---

# Namespace Commands Help

## Available Commands

### Development Commands
- `/namespace:command1` - Description
- `/namespace:command2` - Description

### Usage Examples
...
```

## 6. Error Handling & Validation

### 6.1 Input Validation Patterns
```markdown
### Pre-execution Checks
1. **Validate Arguments**: 
   - If `$ARGUMENTS` is empty, show usage and exit
   - If invalid format, provide correction example
2. **Check Prerequisites**:
   - Verify required tools are available
   - Check current git status if needed
   - Validate project structure
```

### 6.2 Graceful Failure
- Always provide actionable error messages
- Suggest fixes for common problems
- Fail early with clear explanations
- Never leave the project in an inconsistent state

## 7. Tool Integration Standards

### 7.1 Allowed Tools Declaration
Be explicit about required tools in frontmatter:
```yaml
allowed-tools: 
  - Bash(git add:*, git commit:*, git push:*)
  - FileEdit
  - GrepTool
  - WebSearchTool
```

### 7.2 Tool Usage Patterns
- **Git Operations**: Always check status before making changes
- **File Operations**: Verify file exists before editing
- **External Commands**: Check command availability first
- **Web Operations**: Handle network failures gracefully

## 8. Security & Safety Standards

### 8.1 Safe Command Practices
- Never include hardcoded secrets or credentials
- Validate external input before using in commands
- Use allowlists for bash commands when possible
- Avoid destructive operations without confirmation

### 8.2 Permission Patterns
```yaml
allowed-tools: 
  - Bash(npm install:*, npm run:build|test|lint)  # Specific commands only
  - Bash(git add:*, git status:*)                 # Safe git operations
  - FileEdit                                      # File editing allowed
```

## 9. Documentation Standards

### 9.1 Command Documentation
- Include examples for all common use cases
- Document dependencies and prerequisites
- Explain the reasoning behind complex workflows
- Keep documentation updated with command changes

### 9.2 Team Documentation
Create a team-specific command reference:
```markdown
# .claude/commands/help.md
---
description: "Team command reference and standards"
---

# Team Claude Code Commands

## Command Categories
- `/project:*` - Project management
- `/dev:*` - Development workflows  
- `/test:*` - Testing operations

## Usage Guidelines
...
```

## 10. Versioning & Maintenance

### 10.1 Version Management
- Include version in frontmatter for tracking changes
- Use semantic versioning (1.0.0, 1.1.0, 2.0.0)
- Document breaking changes in command comments
- Maintain backward compatibility when possible

### 10.2 Regular Maintenance
- Review commands quarterly for relevance
- Update tool integrations as they evolve
- Refactor common patterns into shared utilities
- Archive or remove obsolete commands

## 11. Testing Standards

### 11.1 Command Testing
- Test commands in clean environments
- Verify all documented examples work
- Test edge cases and error conditions
- Ensure commands work across different project states

### 11.2 Integration Testing
- Test command combinations and workflows
- Verify team member can use commands without setup
- Test on different operating systems if applicable
- Validate with different Claude Code versions

## 12. Best Practices Summary

### DO:
✅ Use clear, descriptive names with consistent patterns  
✅ Include comprehensive frontmatter metadata  
✅ Document all parameters and provide examples  
✅ Validate inputs and handle errors gracefully  
✅ Use namespacing for logical organization  
✅ Test commands thoroughly before sharing  
✅ Keep commands focused on single responsibilities  
✅ Use standard markdown formatting  

### DON'T:
❌ Use generic names like `command1` or `temp`  
❌ Skip documentation or examples  
❌ Hardcode project-specific paths or values  
❌ Create commands without error handling  
❌ Use reserved namespace prefixes  
❌ Make commands that require manual intervention  
❌ Create overly complex single commands  
❌ Forget to update documentation when changing commands  

## 13. Migration Strategy

### 13.1 Existing Command Updates
When updating existing commands to meet these standards:
1. Add proper frontmatter to existing commands
2. Reorganize into appropriate namespaces
3. Update documentation and examples
4. Test all changes thoroughly
5. Communicate changes to team

### 13.2 New Command Development
For all new commands:
1. Follow naming conventions from the start
2. Use the standard template structure
3. Include comprehensive testing
4. Document thoroughly before sharing
5. Get team review for complex workflows

---

**Version**: 1.0.0  
**Last Updated**: 2025-08-22  
**Maintenance**: Review quarterly, update as Claude Code evolves
