# XCC Framework

## Quick Setup: From Template to Your Project

### Prerequisites
- Git installed
- GitHub account
- **VS Code with Claude Code extension** (Primary development environment)
- Node.js (for optional automation scripts and MCP servers)

---

## Features

### 🗂️ Organized Structure
All framework files are organized under the `0xcc/` directory:
- Separation between framework and project files
- Navigation with `0` prefix sorting to top of file explorer
- Portable framework structure

### 🔍 Research Integration
The framework includes contextual research suggestions via MCP server integration:
- Context-aware research queries tailored to your project phase
- Workflow integration using `/mcp ref search` commands
- Decision making support

### 🏠 Session Management
Automated session management and context preservation:
- Session tracking with state preservation
- Context cleanup when conversations get too large
- Transcript capture for learning and team collaboration
- Session resumption

---

## Step-by-Step Setup

### 1. Clone the Template
```bash
git clone https://github.com/Onegaishimas/xcc_lattice.git
```

### 2. Remove Template Git History
```bash
rm -rf xcc_lattice/.git
```

### 3. Rename to Your Project
```bash
mv xcc_lattice your-project-name
cd your-project-name
```

### 4. Initialize XCC Framework
```bash
# Create complete 0xcc framework structure
mkdir -p 0xcc/adrs 0xcc/docs 0xcc/instruct 0xcc/prds 0xcc/tasks 0xcc/tdds 0xcc/tids
mkdir -p 0xcc/transcripts 0xcc/checkpoints 0xcc/scripts

# Initialize session state
echo '{"sessionNumber": 0, "currentPhase": "setup", "totalSessionTime": "0 hours"}' > 0xcc/session_state.json

# Initialize research context
echo '{"projectContext": {}, "researchHistory": [], "researchPatterns": {}}' > 0xcc/research_context.json

# Optional: Add transcripts to .gitignore if you want to keep them private
echo "0xcc/transcripts/" >> .gitignore
```

### 5. Create New GitHub Repository
1. Go to [github.com](https://github.com)
2. Click **"+"** → **"New repository"**
3. Repository name: `your-project-name`
4. Set to **Public** or **Private**
5. **DO NOT** check "Add a README file"
6. Click **"Create repository"**

### 6. Initialize New Git Repository
```bash
git init
git add .
git commit -m "Initial commit: XCC Framework with 0xcc organization"
git branch -M main
```

### 7. Connect to GitHub and Push
```bash
git remote add origin https://github.com/yourusername/your-project-name.git
git push -u origin main
```

**Authentication:** GitHub will prompt for:
- **Username:** `yourusername`
- **Password:** Use your Personal Access Token (not your GitHub password)

---

## MCP Server Setup (Optional but Recommended)

The XCC framework can be enhanced with MCP servers for research capabilities:

### Install ref MCP Server
```bash
# Install globally for easy access across projects
npm install -g @modelcontextprotocol/server-ref

# Verify installation
ref --version
```

### Using MCP in Claude Code
Claude Code has built-in MCP support via the `/mcp` command:
```bash
# Basic usage
/mcp ref search "your search query"

# Example research query
/mcp ref search "React vs Vue comparison 2024"
```

### Test MCP Integration
```bash
# In Claude Code chat window, test the connection:
/mcp ref search "test query"

# Should return search results if properly configured
```

---

## Claude Code Integration

### 1. Open Project in VS Code
```bash
code .
```

### 2. Start Claude Code Session
- Open Command Palette: `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
- Type: **"Claude Code: Start Chat"**
- Or use the Claude Code icon in the Activity Bar

### 3. Initialize XCC Framework Context
```bash
# In Claude Code chat:
@CLAUDE.md

@0xcc/instruct/001_create-project-prd.md

I want to build [describe your project idea in detail]

# The framework will guide you through strategic questions with research options
```

### 4. Using Research Integration
When you see research options like:
```
B) 🔍 Research first: Use /mcp ref search "MVP vs full application development approaches"
```

Select this option to get:
- **Contextual research suggestions** tailored to your project type and phase
- **Guided research questions** to focus your investigation  
- **Structured application** of research findings back to your project decisions
- **Seamless workflow continuation** with informed choices

### 5. Housekeeping Commands for Claude Code

#### Session Management
```bash
# Create comprehensive checkpoint (save current state)
"Please create a checkpoint"

# Resume previous session with full context
"Please help me resume where I left off"

# Clean up context when it gets too large
"My context is getting too large, please clean up"

# Save current session transcript for learning
"Please save the session transcript"

# Check overall project status and progress
"Please show me the current project status"
```

#### Quick Context Loading
```bash
# Standard session start sequence
@CLAUDE.md
@0xcc/session_state.json

# Load current work area based on phase
@0xcc/prds/     # For PRD work
@0xcc/tdds/     # For TDD work
@0xcc/tids/     # For TID work  
@0xcc/tasks/    # For task execution

# Research context when needed
@0xcc/research_context.json
```

---

## Workflow Process

### Phase 1: Project Foundation
```bash
# Session 1: Project Vision with Research Support
@0xcc/instruct/001_create-project-prd.md
# Use research options to inform project scope and user decisions
# Output: 0xcc/prds/000_PPRD|[project-name].md

# Session 2: Technical Foundation with Research
@0xcc/instruct/002_create-adr.md
@0xcc/prds/000_PPRD|[project-name].md
# Research technology choices before making architectural decisions
# Output: 0xcc/adrs/000_PADR|[project-name].md
# Action: Copy Project Standards section to CLAUDE.md
```

### Phase 2: Feature Development (For each feature)
```bash
# Feature Requirements with Research Support
@0xcc/instruct/003_create-feature-prd.md
@0xcc/prds/000_PPRD|[project-name].md
@0xcc/adrs/000_PADR|[project-name].md
# Research feature patterns, user stories, and security requirements
# Output: 0xcc/prds/[###]_FPRD|[feature-name].md

# Technical Design with Architecture Research
@0xcc/instruct/004_create-tdd.md  
@0xcc/prds/[###]_FPRD|[feature-name].md
# Research architecture patterns, data design, and component organization
# Output: 0xcc/tdds/[###]_FTDD|[feature-name].md

# Implementation Planning with Best Practices Research
@0xcc/instruct/005_create-tid.md
@0xcc/prds/[###]_FPRD|[feature-name].md
@0xcc/tdds/[###]_FTDD|[feature-name].md
# Research implementation patterns, coding standards, and optimization strategies
# Output: 0xcc/tids/[###]_FTID|[feature-name].md

# Task Generation with Planning Research
@0xcc/instruct/006_generate-tasks.md
@0xcc/prds/[###]_FPRD|[feature-name].md
# Optional: Reference TDD and TID for enhanced context
# Output: 0xcc/tasks/[###]_FTASKS|[feature-name].md

# Implementation with Progress Tracking
@0xcc/instruct/007_process-task-list.md
@0xcc/tasks/[###]_FTASKS|[feature-name].md
# Execute tasks with automatic progress tracking and checkpointing
```

---

## What You Get

### XCC Lattice Project Structure
```
your-project-name/
├── 0xcc/                           # Core XCC Framework
│   ├── adrs/                       # Architecture Decision Records
│   │   └── 000_PADR|Project_Name.md
│   ├── docs/                       # Additional framework documentation
│   ├── instruct/                   # XCC Framework instruction files
│   │   ├── 000_README.md
│   │   ├── 001_create-project-prd.md
│   │   ├── 002_create-adr.md
│   │   ├── 003_create-feature-prd.md
│   │   ├── 004_create-tdd.md
│   │   ├── 005_create-tid.md
│   │   ├── 006_generate-tasks.md
│   │   └── 007_process-task-list.md
│   ├── prds/                       # Product Requirements Documents
│   │   ├── 000_PPRD|Project_Name.md
│   │   ├── 001_FPRD|Feature_A.md
│   │   └── 002_FPRD|Feature_B.md
│   ├── tasks/                      # Task Lists with progress tracking
│   │   ├── 001_FTASKS|Feature_A.md
│   │   └── 002_FTASKS|Feature_B.md
│   ├── tdds/                       # Technical Design Documents
│   │   ├── 001_FTDD|Feature_A.md
│   │   └── 002_FTDD|Feature_B.md
│   ├── tids/                       # Technical Implementation Documents
│   │   ├── 001_FTID|Feature_A.md
│   │   └── 002_FTID|Feature_B.md
│   ├── transcripts/                # Session transcripts for learning
│   │   ├── session_001.md
│   │   └── research_log.md
│   ├── checkpoints/                # Automated state backups
│   ├── scripts/                    # Optional automation scripts
│   ├── session_state.json          # Current session tracking
│   └── research_context.json       # Research history and context
├── .claude/                        # Claude Extensions System
│   ├── commands/                   # Command definitions
│   │   ├── analyze.md              # /analyze command
│   │   ├── collaborate.md          # /collaborate command
│   │   ├── feature.md              # /feature command
│   │   ├── health.md               # /health command
│   │   ├── review.md               # /review command
│   │   └── smart-clear.md          # /smart-clear command
│   ├── context/                    # Context management
│   │   ├── agents/                 # Expert agent contexts
│   │   │   ├── product_engineer.md
│   │   │   ├── qa_engineer.md
│   │   │   ├── architect.md
│   │   │   └── test_engineer.md
│   │   ├── health/                 # Health monitoring
│   │   │   ├── dashboard.md
│   │   │   └── assessment_template.md
│   │   ├── project/                # Project information
│   │   │   ├── health_summary.md
│   │   │   └── setup_info.md
│   │   └── sessions/               # Session tracking
│   │       └── session_template.md
│   ├── prompts/                    # Analysis templates
│   │   ├── analysis/               # Expert analysis prompts
│   │   │   ├── architecture/       # Architecture assessment
│   │   │   ├── integration/        # Integration review
│   │   │   ├── product/            # Product context analysis
│   │   │   ├── quality/            # Code quality review
│   │   │   └── testing/            # Testing strategy analysis
│   │   └── usage_guide.md          # Analysis templates guide
│   └── quick_reference.md          # Extensions usage guide
├── CLAUDE.md                       # Enhanced project memory system
├── src/                            # Your actual project code
├── tests/                          # Your project tests
├── package.json                    # Your project dependencies (if applicable)
└── README.md                       # This file
```

### Key Benefits

#### 🗂️ **Organization**
- Framework isolation: All XCC files in `0xcc/` directory
- Navigation: `0` prefix sorts framework to top in file explorers
- Clear boundaries: Framework vs project code separation
- Portable updates: Framework can be updated independently

#### 🔍 **Research Integration**
- Contextual research suggestions for major decision points
- MCP integration for research without workflow interruption
- Research tracking in session transcripts and context
- Decision making with current best practices and data

#### 🏠 **Session Management**  
- Automatic state tracking with progress monitoring
- Context cleanup when conversations get too large
- Transcript capture for learning, team collaboration, and knowledge transfer
- Session resumption that preserves research findings and decision context
- Progress analytics across sessions and development phases

#### 📊 **Productivity Features**
- Time tracking across sessions with estimation improvements
- Decision consistency through documented research and standards
- Reduced context switching with integrated research capabilities
- Team collaboration through comprehensive session documentation

---

## Example Usage Patterns

### Starting a New Session
```bash
# Claude Code chat
@CLAUDE.md
"Please help me resume where I left off"

# Claude will:
# - Check 0xcc/session_state.json for last position
# - Load current document context  
# - Show progress and next actions
# - Present any blockers or research findings
# - Provide specific resume guidance
```

### Research Integration
```bash
# When you see research options in the workflow:
"I'll choose option B to research this topic first"

# Claude provides:
# - Specific MCP search commands: /mcp ref search "context-aware query"
# - Research focus areas and key questions to investigate
# - Framework for applying research findings to your specific project
# - Structured way to resume with informed choice
```

### Mid-Session Checkpoint
```bash
"Please create a checkpoint before I take a break"

# Claude will:
# - Update 0xcc/session_state.json with current progress
# - Save transcript with key decisions and research findings
# - Commit progress with structured message including research context
# - Provide specific resume commands for next session
```

### Context Management
```bash
"My context is getting too large, please clean up"

# Claude will:
# - Create comprehensive checkpoint preserving all research and decisions
# - Clean conversation context while preserving essential information
# - Reload only critical project information and current work context
# - Resume exactly where you left off with clean, focused context
```

### Research Integration Example
```bash
# During ADR creation, when choosing backend technology:
"I'll research the backend options first"

# Claude responds with:
/mcp ref search "Python web framework comparison FastAPI Django Flask 2024"

# After reviewing research results:
"Based on this research, FastAPI seems best for our API-first SaaS platform because..."
# Decision gets documented in both ADR and research context
```

---

## Additional Features

### Research Integration
The framework captures and leverages research throughout development:
- Decision tracking with research basis and confidence levels
- Pattern recognition from research queries across projects
- Knowledge accumulation for future decision making
- Team knowledge sharing through research documentation

### Session Analytics and Learning
- Time estimation improvements based on actual session data
- Decision quality metrics correlating research with outcomes
- Process optimization from transcript analysis and pattern recognition
- Framework evolution based on usage patterns and feedback

### Team Collaboration
- Context handoffs with complete research and decision history
- Knowledge preservation across team changes and project pauses
- Onboarding acceleration through comprehensive project documentation
- Decision transparency for stakeholders and team alignment

### Framework Portability
- Easy updates: Replace `0xcc/` directory to update framework
- Project templates: Copy `0xcc/` structure to new projects
- Custom extensions: Add organization-specific instructions
- Version control: Track framework evolution separately from project code

---

## Troubleshooting

### If Context Gets Lost
```bash
# Emergency context recovery
@CLAUDE.md
@0xcc/session_state.json
@0xcc/prds/000_PPRD|[project-name].md
@0xcc/adrs/000_PADR|[project-name].md

# Then ask: "Please help me understand where I am in the workflow"
```

### If Session State is Corrupted
```bash
# Manual session state recreation
@CLAUDE.md
@0xcc/research_context.json
git log --oneline -10

# Ask: "Please help me recreate the session state based on recent commits and research history"
```

### If MCP Research Isn't Working
```bash
# Test MCP connection
/mcp ref search "test query"

# If no response, check MCP server installation:
npm list -g @modelcontextprotocol/server-ref

# Reinstall if needed:
npm install -g @modelcontextprotocol/server-ref
```

### If You Need to Start a Phase Over
```bash
# Clean restart while preserving research and decisions
"Please create a final checkpoint and help me restart the [current phase] with a clean approach"

# Claude will preserve research findings while providing a fresh start
```

---

## Migration from Old Structure

If you have an existing XCC project without the `0xcc/` structure:

### Quick Migration Script
```bash
#!/bin/bash
echo "Migrating XCC framework to 0xcc directory..."

# Create new structure
mkdir -p 0xcc

# Move existing directories
[ -d "adrs" ] && mv adrs 0xcc/
[ -d "docs" ] && mv docs 0xcc/
[ -d "instruct" ] && mv instruct 0xcc/
[ -d "prds" ] && mv prds 0xcc/
[ -d "tasks" ] && mv tasks 0xcc/
[ -d "tdds" ] && mv tdds 0xcc/
[ -d "tids" ] && mv tids 0xcc/

# Create missing directories
mkdir -p 0xcc/transcripts 0xcc/checkpoints 0xcc/scripts

# Initialize session state
echo '{"sessionNumber": 0, "currentPhase": "migrated", "totalSessionTime": "0 hours"}' > 0xcc/session_state.json

echo "Migration complete! Update your CLAUDE.md file references to use 0xcc/ paths"
```

### Update File References
After migration, update your CLAUDE.md file references:
- Change `@prds/` to `@0xcc/prds/`
- Change `@adrs/` to `@0xcc/adrs/`
- Update all other framework paths to include `0xcc/` prefix

---

**Setup Time:** ~5 minutes  
**Features:** Organized structure, research integration, session management  
**Ready to Code:** Start with enhanced Project PRD creation!

The XCC Framework provides an organized, research-informed, and automatically managed development experience within Claude Code and VS Code. The `0xcc/` structure ensures framework files are organized, updatable, and separated from project code.
# miLLM
