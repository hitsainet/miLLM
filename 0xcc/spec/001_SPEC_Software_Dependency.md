# miStudio Software Dependency Specification

## Overview

This document provides a comprehensive inventory of all software dependencies, frameworks, libraries, and tools referenced across the miStudio (Mechanistic Interpretability Toolkit) documentation and requirements. Each entry includes the source file(s) where the software was referenced and its intended usage context.

---

## Core Programming Languages & Frameworks

### Python Ecosystem
- **Python** - Primary programming language for AI/ML development
  - **Sources**: All PRD files (001-009_PRD), main toolkit specifications
  - **Usage**: Core language for machine learning, data processing, API development
  - **Criticality**: Essential

- **FastAPI** - Modern async web framework for building APIs
  - **Sources**: Integration APIs (006_PRD), Platform Infrastructure (002_PRD), Real-time Monitoring (007_PRD), all service PRDs
  - **Usage**: REST API development, async request handling, OpenAPI documentation
  - **Criticality**: Essential

- **Pydantic** - Data validation and settings management using Python type annotations
  - **Sources**: All service PRDs for data model definitions
  - **Usage**: Request/response validation, configuration management, type safety
  - **Criticality**: Essential

- **Click** - Command line interface creation toolkit
  - **Sources**: Main toolkit PRD (001_PRD) for CLI interfaces
  - **Usage**: Building user-friendly command-line tools and interfaces
  - **Criticality**: High

- **Uvicorn** - ASGI web server implementation for Python
  - **Sources**: Platform Infrastructure (002_PRD)
  - **Usage**: High-performance ASGI server for FastAPI applications
  - **Criticality**: Essential

### Machine Learning & AI Libraries

- **PyTorch** - Open-source machine learning framework
  - **Sources**: All AI-related PRDs, main toolkit (001_PRD), SAE Training (004_PRD)
  - **Usage**: Deep learning model implementation, tensor operations, GPU acceleration
  - **Criticality**: Essential

- **Transformers (Hugging Face)** - State-of-the-art machine learning for PyTorch, TensorFlow, and JAX
  - **Sources**: Main toolkit (001_PRD), Data Preparation (003_PRD), SAE Training (004_PRD)
  - **Usage**: Pre-trained transformer models, tokenization, model inference
  - **Criticality**: Essential

- **NumPy** - Fundamental package for scientific computing with Python
  - **Sources**: Main toolkit (001_PRD), implied across all ML components
  - **Usage**: Numerical computations, array operations, mathematical functions
  - **Criticality**: Essential

- **nvidia-ml-py** - Python bindings for NVIDIA Management Library
  - **Sources**: Platform Infrastructure (002_PRD)
  - **Usage**: GPU monitoring, memory management, device information
  - **Criticality**: Essential (for Jetson deployment)

---

## Development & AI Tools

### AI Development Platforms

- **Cursor** - AI-powered code editor
  - **Sources**: README.md, instruct/*.md files
  - **Usage**: AI-assisted development, structured coding workflows
  - **Criticality**: Development tool (optional)

- **Claude Code** - Anthropic's AI coding assistant CLI
  - **Sources**: README.md, instruct files for structured development workflows
  - **Usage**: AI-guided feature development, code generation
  - **Criticality**: Development tool (optional)

- **Windsurf** - AI coding assistant
  - **Sources**: README.md as compatible development tool
  - **Usage**: Alternative AI-assisted development platform
  - **Criticality**: Development tool (optional)

### Version Control & Development

- **Git** - Distributed version control system
  - **Sources**: Task processing files (process-task-list.md), README.md
  - **Usage**: Version control, change tracking, collaboration
  - **Criticality**: Essential

- **GitHub** - Web-based Git repository hosting service
  - **Sources**: README.md for repository management and cloning
  - **Usage**: Code repository hosting, collaboration, issue tracking
  - **Criticality**: High

- **Docker** - Containerization platform
  - **Sources**: All infrastructure PRDs, main toolkit (001_PRD), Platform Infrastructure (002_PRD)
  - **Usage**: Application containerization, deployment consistency
  - **Criticality**: Essential

- **Docker Compose** - Tool for defining and running multi-container Docker applications
  - **Sources**: Platform Infrastructure (002_PRD)
  - **Usage**: Multi-service orchestration, development environments
  - **Criticality**: Essential

---

## Infrastructure & Deployment

### Container & Orchestration

- **Ubuntu 22.04** - Linux distribution serving as base operating system
  - **Sources**: Platform Infrastructure (002_PRD)
  - **Usage**: Container base image, consistent runtime environment
  - **Criticality**: Essential

- **NVIDIA Container Toolkit** - Tools for building and running GPU accelerated Docker containers
  - **Sources**: Platform Infrastructure (002_PRD)
  - **Usage**: GPU access within containers, CUDA runtime integration
  - **Criticality**: Essential (for GPU acceleration)

- **NVIDIA Jetson** - Edge AI computing platform (Nano, Orin Nano, Orin NX variants)
  - **Sources**: All PRD files as primary target deployment platform
  - **Usage**: Edge computing hardware, GPU acceleration, embedded deployment
  - **Criticality**: Essential (target hardware)

### Databases & Storage

- **Redis** - In-memory data structure store
  - **Sources**: Integration APIs (006_PRD) for caching and task queue management
  - **Usage**: Caching, session storage, task queues, real-time data
  - **Criticality**: High

- **InfluxDB** - Time-series database designed for high-performance time-series data
  - **Sources**: Real-time Monitoring (007_PRD)
  - **Usage**: Time-series metrics storage, monitoring data persistence
  - **Criticality**: Medium

- **TimescaleDB** - Time-series database built on PostgreSQL
  - **Sources**: Real-time Monitoring (007_PRD)
  - **Usage**: Alternative time-series storage with SQL compatibility
  - **Criticality**: Medium

- **FAISS** - Library for efficient similarity search and clustering of dense vectors
  - **Sources**: Feature Discovery (005_PRD) for vector similarity search
  - **Usage**: Feature search, similarity matching, vector indexing
  - **Criticality**: High

---

## Web Technologies & Visualization

### Frontend Frameworks

- **React** - JavaScript library for building user interfaces
  - **Sources**: Visualization & Reporting (008_PRD)
  - **Usage**: Interactive dashboard development, component-based UI
  - **Criticality**: High

- **Vue.js** - Progressive JavaScript framework for building UIs
  - **Sources**: Visualization & Reporting (008_PRD)
  - **Usage**: Alternative frontend framework for dashboard development
  - **Criticality**: Medium

- **D3.js** - JavaScript library for data-driven document manipulation
  - **Sources**: Visualization & Reporting (008_PRD)
  - **Usage**: Custom data visualizations, interactive charts, SVG graphics
  - **Criticality**: High

- **Plotly** - Interactive graphing library for Python and JavaScript
  - **Sources**: Visualization & Reporting (008_PRD)
  - **Usage**: Scientific plotting, interactive visualizations, dashboard components
  - **Criticality**: High

- **Bokeh** - Interactive visualization library for Python
  - **Sources**: Visualization & Reporting (008_PRD)
  - **Usage**: Web-based interactive visualizations, dashboard creation
  - **Criticality**: Medium

### Backend & API Technologies

- **WebSocket** - Communication protocol for full-duplex communication
  - **Sources**: Integration APIs (006_PRD), Real-time Monitoring (007_PRD), Visualization (008_PRD)
  - **Usage**: Real-time updates, live data streaming, interactive dashboards
  - **Criticality**: High

- **OpenAPI/Swagger** - API specification format and documentation tools
  - **Sources**: Integration APIs (006_PRD), all service PRDs
  - **Usage**: API documentation, client SDK generation, specification compliance
  - **Criticality**: Essential

- **GraphQL** - Query language and runtime for APIs
  - **Sources**: Integration APIs (006_PRD)
  - **Usage**: Flexible data querying, complex query operations
  - **Criticality**: Medium

---

## Data Processing & Analytics

### Stream Processing

- **Apache Kafka** - Distributed event streaming platform
  - **Sources**: Real-time Monitoring (007_PRD)
  - **Usage**: Event streaming, real-time data pipelines, message queuing
  - **Criticality**: Medium

- **Redis Streams** - Stream processing capabilities within Redis
  - **Sources**: Real-time Monitoring (007_PRD)
  - **Usage**: Lightweight stream processing, real-time analytics
  - **Criticality**: Medium

- **Celery** - Distributed task queue for Python
  - **Sources**: Integration APIs (006_PRD)
  - **Usage**: Background task processing, workflow orchestration
  - **Criticality**: High

### Data Science Libraries

- **Pandas** - Data manipulation and analysis library
  - **Sources**: Implied across data processing components
  - **Usage**: Data manipulation, CSV/JSON processing, data analysis
  - **Criticality**: High

- **Scikit-learn** - Machine learning library for Python
  - **Sources**: Implied for clustering and statistical analysis
  - **Usage**: Clustering algorithms, statistical analysis, feature processing
  - **Criticality**: High

---

## Document & Report Generation

### Document Processing

- **Jinja2** - Modern and designer-friendly templating language for Python
  - **Sources**: Visualization & Reporting (008_PRD)
  - **Usage**: Report template generation, dynamic content creation
  - **Criticality**: High

- **ReportLab** - Python library for generating PDFs and graphics
  - **Sources**: Visualization & Reporting (008_PRD)
  - **Usage**: PDF report generation, professional document formatting
  - **Criticality**: High

- **WeasyPrint** - Visual rendering engine for HTML and CSS to PDF
  - **Sources**: Visualization & Reporting (008_PRD)
  - **Usage**: HTML to PDF conversion, CSS-styled document generation
  - **Criticality**: Medium

- **python-docx** - Python library for creating and updating Microsoft Word documents
  - **Sources**: Visualization & Reporting (008_PRD)
  - **Usage**: Word document generation, collaborative document formats
  - **Criticality**: Medium

---

## Testing & Quality Assurance

### Testing Frameworks

- **pytest** - Testing framework for Python
  - **Sources**: Task processing files, all service PRDs for testing requirements
  - **Usage**: Unit testing, integration testing, test automation
  - **Criticality**: Essential

- **Jest** - JavaScript testing framework
  - **Sources**: Task generation files (generate-tasks.md)
  - **Usage**: Frontend testing, JavaScript unit tests
  - **Criticality**: High (if using JavaScript frontend)

---

## Security & Compliance

### Cryptography & Security

- **RSA/ECDSA** - Digital signature algorithms
  - **Sources**: Feature Steering (009_PRD) for audit log integrity
  - **Usage**: Digital signatures, audit trail verification, non-repudiation
  - **Criticality**: Essential (for compliance)

- **AES-256** - Advanced Encryption Standard
  - **Sources**: Feature Steering (009_PRD) for sensitive data encryption
  - **Usage**: Data encryption, secure storage, confidentiality
  - **Criticality**: Essential (for compliance)

- **JWT (JSON Web Tokens)** - Standard for securely transmitting information
  - **Sources**: Platform Infrastructure (002_PRD)
  - **Usage**: Authentication, authorization, secure API access
  - **Criticality**: Essential

---

## Monitoring & Observability

### Monitoring Systems

- **Prometheus** - Open-source monitoring and alerting toolkit
  - **Sources**: Platform Infrastructure (002_PRD), Real-time Monitoring (007_PRD)
  - **Usage**: Metrics collection, monitoring, alerting rules
  - **Criticality**: High

- **Grafana** - Open-source analytics and monitoring solution
  - **Sources**: Integration APIs (006_PRD)
  - **Usage**: Metrics visualization, monitoring dashboards, alerting
  - **Criticality**: Medium

- **OpenTelemetry** - Observability framework for cloud-native software
  - **Sources**: Real-time Monitoring (007_PRD)
  - **Usage**: Distributed tracing, metrics collection, observability
  - **Criticality**: Medium

---

## External Integrations

### Communication & Notifications

- **Slack** - Business communication platform
  - **Sources**: Integration APIs (006_PRD), Visualization (008_PRD)
  - **Usage**: Team notifications, alert delivery, integration workflows
  - **Criticality**: Low (optional integration)

- **Microsoft Teams** - Collaboration and communication platform
  - **Sources**: Visualization (008_PRD)
  - **Usage**: Enterprise communication, notification delivery
  - **Criticality**: Low (optional integration)

- **Email/SMS Systems** - Electronic communication protocols
  - **Sources**: Real-time Monitoring (007_PRD), Feature Steering (009_PRD)
  - **Usage**: Alert notifications, system communications, incident response
  - **Criticality**: Medium

- **Webhooks** - HTTP callbacks for system integration
  - **Sources**: Multiple PRDs for external system communication
  - **Usage**: Event-driven integrations, external system notifications
  - **Criticality**: High

### Business Intelligence & External Tools

- **Tableau** - Business intelligence and analytics platform
  - **Sources**: Visualization & Reporting (008_PRD)
  - **Usage**: Advanced analytics, enterprise reporting, data visualization
  - **Criticality**: Low (optional integration)

- **Power BI** - Business analytics solution by Microsoft
  - **Sources**: Visualization & Reporting (008_PRD)
  - **Usage**: Enterprise reporting, business intelligence dashboards
  - **Criticality**: Low (optional integration)

- **Google Sheets** - Cloud-based spreadsheet application
  - **Sources**: Visualization & Reporting (008_PRD)
  - **Usage**: Data export, collaborative analysis, simple reporting
  - **Criticality**: Low (optional integration)

- **Microsoft Excel** - Spreadsheet application
  - **Sources**: Visualization & Reporting (008_PRD)
  - **Usage**: Data export, offline analysis, enterprise reporting
  - **Criticality**: Low (optional integration)

### Development & Documentation Tools

- **Jupyter Notebooks** - Web-based interactive computing environment
  - **Sources**: Visualization & Reporting (008_PRD)
  - **Usage**: Interactive analysis, research documentation, exploratory data analysis
  - **Criticality**: Medium

- **GitHub/GitLab** - Git repository hosting platforms
  - **Sources**: Integration APIs (006_PRD), Visualization (008_PRD)
  - **Usage**: Code hosting, CI/CD integration, documentation publishing
  - **Criticality**: High

---

## Dependency Priority Classification

### Essential (System Cannot Function Without)
- Python, PyTorch, FastAPI, Docker, NVIDIA Jetson platform
- Core ML libraries (Transformers, NumPy)
- Container runtime (Ubuntu, NVIDIA Container Toolkit)
- Authentication & security (JWT, cryptographic libraries)
- Version control (Git)

### High Priority (Core Functionality Depends On)
- Visualization libraries (React, D3.js, Plotly)
- Data processing (Redis, FAISS)
- Testing frameworks (pytest)
- API documentation (OpenAPI/Swagger)
- Real-time communication (WebSocket)

### Medium Priority (Enhanced Features)
- Alternative databases (InfluxDB, TimescaleDB)
- Stream processing (Kafka, Redis Streams)
- Advanced visualization (Bokeh, alternative frontend frameworks)
- Monitoring systems (Prometheus, Grafana)

### Low Priority (Optional Integrations)
- External BI tools (Tableau, Power BI)
- Communication platforms (Slack, Teams)
- Document formats (Excel, Google Sheets)

---

## Hardware Dependencies

### NVIDIA Jetson Variants
- **Jetson Nano**: 4GB RAM, basic GPU capabilities
- **Jetson Orin Nano**: 8GB RAM, enhanced AI performance  
- **Jetson Orin NX**: 16GB RAM, high-performance AI computing

### CUDA & GPU Requirements
- CUDA-compatible GPU for tensor operations
- NVIDIA drivers and CUDA toolkit
- GPU memory management for constrained environments

---

## Development Environment Requirements

### Minimum Development Setup
- Python 3.10+
- Docker and Docker Compose
- Git version control
- Modern web browser (for dashboard development)
- NVIDIA GPU with CUDA support (recommended)

### Recommended Development Tools
- AI-assisted development environment (Cursor, Claude Code, or similar)
- Integrated development environment with Python support
- Container orchestration familiarity
- Knowledge of FastAPI and modern web frameworks

---

This specification serves as the authoritative reference for all software dependencies across the miStudio project, enabling accurate dependency management, license compliance, and deployment planning.
