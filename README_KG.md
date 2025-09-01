# 🧠 Knowledge Graph Agent

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Build and explore knowledge graphs through intelligent agent-driven learning

The **Knowledge Graph Agent** is a sophisticated distributed system that creates personalized learning paths through iterative research and knowledge graph construction. It uses parallel processing, vector search, and AI-driven content generation to build comprehensive knowledge structures.

## 📑 Table of Contents

- [🚀 Quick Start](#quick-start)
- [🏗️ Infrastructure Requirements](#infrastructure-requirements)
- [🎛️ Configuration](#configuration)
- [🌟 Features](#features)
- [💡 Usage Examples](#usage-examples)
- [🐛 Debug Modes](#debug-modes)
- [📊 Session Status Types](#session-status-types)
- [❓ FAQ](#faq)

## Quick Start

### Infrastructure Requirements

The Knowledge Graph Agent requires several components:

#### 1. **Database Systems**
```bash
# Neo4j (Knowledge Graph Storage)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# PostgreSQL (Educational Content & Checkpoints)
LANGGRAPH_CHECKPOINT_DB_URL=postgresql://user:pass@localhost:5432/kg_db
```

#### 2. **Celery Task Queue**
```bash
# Install Redis or RabbitMQ as message broker
# Redis (recommended)
redis-server

# Set Celery broker URL
CELERY_BROKER_URL=redis://localhost:6379/0
```

#### 3. **Model Configuration**
Create `conf.yaml` file (see [Configuration Guide](docs/configuration_guide.md)):
```yaml
BASIC_MODEL:
  model: "gpt-4o"  # or qwen-max-latest, gemini-2.0-flash, etc.
  api_key: "your_api_key"
```

#### 4. **Environment Variables**
```bash
# Required APIs
GEMINI_API_KEY=your_gemini_api_key  # Or other model API key
EMBEDDING_PROVIDER=openai  # For vector search
SEARCH_API=tavily  # For research phase
TAVILY_API_KEY=your_tavily_key

# Optional but recommended
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=kg_agent_project
```

### Installation & Setup

1. **Install Dependencies**:
   ```bash
   uv sync
   ```

2. **Start Infrastructure**:
   ```bash
   # Start Redis (for Celery)
   redis-server
   
   # Start Neo4j
   neo4j start
   
   # Start PostgreSQL
   pg_ctl start
   ```

3. **Start Celery Workers** (Required for parallel processing):
   ```bash
   # In separate terminal
   celery -A src.orchestrator.session worker --loglevel=info
   ```

4. **Run the Agent**:
   ```bash
   # Interactive mode
   python demo.py --interactive
   
   # Direct command
   python demo.py "Learn Python web scraping fundamentals"
   ```

## Configuration

### Core Settings (`conf.yaml`)

The agent requires model configuration. Key settings:

```yaml
# Model Configuration (Required)
BASIC_MODEL:
  model: "gpt-4o"
  api_key: "your_api_key"
  
# Knowledge Graph Settings
max_iteration_main: 5              # Max research iterations
max_parallel_inner_loops: 5       # Parallel concept processing
max_focus_concepts_per_iteration: 5
reflection_confidence_threshold: 0.8
min_confidence_threshold: 0.7

# Educational Content Generation
enable_educational_content_generation: true
educational_content_timeout: 600   # 10 minutes per concept
educational_content_max_plan_iterations: 2

# Research Settings
max_search_queries: 2
max_extract_urls: 2
```

### Environment Variables

```bash
# Core Requirements
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
LANGGRAPH_CHECKPOINT_DB_URL=postgresql://user:pass@localhost:5432/kg_db
CELERY_BROKER_URL=redis://localhost:6379/0

# Model APIs (choose one or more)
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
QWEN_API_KEY=your_qwen_key

# Research & Embeddings
EMBEDDING_PROVIDER=openai
SEARCH_API=tavily
TAVILY_API_KEY=your_tavily_key

# Optional Monitoring
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=kg_agent_project
```

## Features

### 🏗️ Distributed Architecture
- **Celery-based Parallel Processing**: Concepts researched simultaneously across workers
- **Batch Processing**: Configurable parallel execution limits
- **Task Timeouts**: 5-minute timeouts for concept research, 10-minute for content generation
- **Fault Tolerance**: Graceful handling of failed tasks and partial completions

### 🔬 Iterative Research Process
- **Vector Search**: Uses embeddings to find related concepts in existing knowledge
- **Confidence-based Decisions**: Multiple confidence thresholds for research quality
- **Cycle Detection**: Prevents circular prerequisite relationships
- **AWG Consolidation**: Merges research results into coherent knowledge structures

### 🎯 Personalized Learning Paths
- **Goal Identification**: Vector search to find or create learning objectives
- **Prerequisite Discovery**: Automatic identification of learning dependencies
- **Topological Ordering**: Optimal learning sequence generation
- **Content Adaptation**: Tailored to knowledge level and learning preferences

### 📊 Session Management
- **Multiple Completion States**: Success, partial completion, or failure modes
- **Educational Content Generation**: Separate phase after knowledge graph construction
- **Checkpointing**: PostgreSQL-based session persistence
- **Rich Monitoring**: LangSmith integration for debugging and analytics

## Architecture

The system implements a sophisticated iterative research and construction pipeline:

```mermaid
graph TD
    A[User Query Context] --> B[Session Orchestrator KG1]
    B --> C[Goal Identification]
    C --> D[Criteria Check KG2]
    D --> E{Continue Research?}
    E -->|Yes| F[Parallel Inner Loops KG3]
    F --> G[Celery Task Queue]
    G --> H[Concept Research Workers]
    H --> I[AWG Consolidation KG4]
    I --> J[PKG Interface]
    J --> K[Neo4j Knowledge Graph]
    I --> D
    E -->|No| L[Topological Ordering]
    L --> M[Educational Content Generation]
    M --> N[PostgreSQL Storage]
    K --> O[Debug Visualizations]
    N --> P[Learning Materials]
```

### Core Components

1. **Session Orchestrator (KG1)**: Main control loop managing iterations and coordination
2. **Criteria Check (KG2)**: Determines research continuation and selects focus concepts
3. **Inner Loop Processor (KG3)**: Celery tasks for parallel concept research and definition
4. **AWG Consolidator (KG4)**: Merges research results and commits to persistent storage
5. **PKG Interface**: Neo4j client handling knowledge graph operations
6. **Educational Content Generator**: Creates learning materials from completed graphs

## Usage Examples

### Interactive Mode (Recommended)

```bash
python demo.py --interactive
```

Provides guided prompts for:
- Language selection (English/中文)
- Learning goal specification
- Knowledge level assessment
- Learning preferences configuration

### Command Line Mode

```bash
# Basic usage
python demo.py "Learn Python web scraping fundamentals"

# With specific parameters
python demo.py "Master cloud architecture" \
  --topic "microservices, containers, orchestration" \
  --knowledge-level intermediate \
  --focus-areas "scalability, security" \
  --learning-style hands-on \
  --debug-mode rich

# Debug with visualizations
python demo.py "Learn machine learning" \
  --debug-mode interactive \
  --enable-rich-output \
  --show-interactive
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `goal` | Learning objective (required) | - |
| `--topic` | Specific topic focus | None |
| `--knowledge-level` | beginner/intermediate/advanced | beginner |
| `--focus-areas` | Comma-separated focus areas | None |
| `--learning-style` | visual/hands-on/theoretical/mixed | None |
| `--debug-mode` | basic/rich/interactive | basic |
| `--enable-rich-output` | Colored console output | false |
| `--show-interactive` | Browser visualizations | false |

## Debug Modes

### Basic Mode (Default)
```bash
python demo.py "Learn Python" --debug-mode basic
```
- Simple text logging only
- Minimal overhead for production
- Standard session completion reporting

### Rich Mode
```bash
python demo.py "Learn Python" --debug-mode rich --enable-rich-output
```
- Colored console output with progress bars
- Real-time iteration tracking
- Enhanced error reporting with context
- Session start/complete callbacks
- AWG update notifications

### Interactive Mode
```bash
python demo.py "Learn Python" --debug-mode interactive --enable-rich-output --show-interactive
```
- All rich mode features plus:
- Browser-based graph visualizations
- Real-time AWG statistics tables
- Interactive concept research tracking
- Educational content generation progress
- Export capabilities for knowledge graphs

## Session Status Types

The session orchestrator returns different completion statuses:

### Success Status
- **`SUCCESS_PREREQUISITES_MET`**: All learning prerequisites identified and knowledge graph complete

### Partial Completion Status
- **`PARTIAL_MAX_ITERATIONS`**: Reached iteration limit but made progress
- **`PARTIAL_NO_PROGRESS`**: No new concepts found but goal incomplete  
- **`PARTIAL_EDUCATIONAL_CONTENT_ISSUES`**: Knowledge graph complete but content generation had failures

### Failure Status
- **`FAILURE_ERROR`**: Critical error during research process
- **`FAILURE_EDUCATIONAL_CONTENT`**: Complete failure in content generation
- **`FAILURE_CRITICAL_ERROR`**: System-level failure

### Decision Criteria
The system uses these criteria for research continuation:
- **`CONTINUE_RESEARCH`**: More concepts need definition
- **`STOP_PREREQUISITES_MET`**: Learning path complete
- **`STOP_MAX_ITERATIONS`**: Iteration limit reached
- **`STOP_NO_PROGRESS`**: No actionable concepts remaining
- **`STOP_ERROR`**: Error requiring termination

## Programmatic Usage

### Core Session Orchestrator

```python
from src.orchestrator.session import session_orchestrator
from src.orchestrator.models import UserQueryContext

# Create user context
uqc = UserQueryContext(
    goal_string="Learn Python web scraping",
    raw_topic_string="BeautifulSoup, Scrapy, requests",
    prior_knowledge_level="beginner",
    preferences={"learning_style": "hands-on"}
)

# Run session (returns session summary)
result = await session_orchestrator(uqc.model_dump())
print(f"Status: {result['additional_data']['overall_status']}")
```

### Demo Wrapper (Simplified)

```python
from demo import ask_kg

# Synchronous wrapper with debug features
ask_kg(
    goal_string="Master cloud architecture", 
    raw_topic_string="microservices, containers",
    prior_knowledge_level="intermediate",
    preferences={"focus_areas": "security, scalability"},
    debug_mode="rich",
    enable_rich_output=True
)
```

### Session Configuration

```python
from src.config import Configuration

# Access configuration settings
config = Configuration()
print(f"Max iterations: {config.max_iteration_main}")
print(f"Parallel workers: {config.max_parallel_inner_loops}")
print(f"Content generation enabled: {config.enable_educational_content_generation}")
```

## FAQ

### Infrastructure Setup

**Q: What infrastructure do I need to run this?**
A: You need a distributed setup: Neo4j database, PostgreSQL database, Redis/RabbitMQ message broker, and Celery workers. This is not a simple standalone application.

**Q: Why do I need Celery workers?**
A: The system processes multiple concepts in parallel using Celery for distributed task execution. Without workers, the session will hang waiting for task completion.

**Q: Can I run this without PostgreSQL?**
A: No, PostgreSQL is required for educational content storage and LangGraph checkpointing. The system expects the `educational_reports` table for content persistence.

**Q: Do I need both Neo4j AND PostgreSQL?**
A: Yes. Neo4j stores the knowledge graph structure and relationships. PostgreSQL stores educational content, session data, and checkpoints. They serve different purposes.

### Configuration Issues

**Q: I get "Configuration incomplete" errors**
A: Ensure you have both `.env` file and `conf.yaml` file properly configured:
```bash
# .env - Environment variables
NEO4J_URI=bolt://localhost:7687
LANGGRAPH_CHECKPOINT_DB_URL=postgresql://user:pass@localhost:5432/kg_db
CELERY_BROKER_URL=redis://localhost:6379/0

# conf.yaml - Model configuration  
BASIC_MODEL:
  model: "gpt-4o"
  api_key: "your_api_key"
```

**Q: Sessions fail with "task timeout" errors**
A: This is normal - tasks have 5-minute timeouts for concept research and 10-minute timeouts for content generation. Adjust `educational_content_timeout` in configuration if needed.

### Performance & Scaling

**Q: How do I control parallel processing?**
A: Configure in `conf.yaml`:
```yaml
max_parallel_inner_loops: 5        # Parallel concept processing
max_focus_concepts_per_iteration: 5 # Concepts per iteration
max_iteration_main: 5               # Total research iterations
```

**Q: The system seems to run forever**
A: The system uses iterative research with stopping criteria. Check session status - it may be `PARTIAL_MAX_ITERATIONS` or `PARTIAL_NO_PROGRESS` indicating completion.

**Q: How do I monitor what's happening?**
A: Use rich debug mode (`--debug-mode rich --enable-rich-output`) to see real-time progress, or enable LangSmith tracing for detailed monitoring.

### Content Generation

**Q: Educational content generation fails**
A: This is a separate phase after knowledge graph construction. Check:
- `enable_educational_content_generation: true` in config
- Celery workers are running and can handle content generation tasks
- Sufficient timeout settings (default 600 seconds per concept)

**Q: Can I disable content generation?**
A: Yes, set `enable_educational_content_generation: false` in `conf.yaml` to only build knowledge graphs without educational materials.

### Integration & Development

**Q: How do I integrate this into my application?**
A: Use the core `session_orchestrator()` function directly rather than the demo wrapper. Ensure your application can handle the async nature and has access to the required infrastructure.

**Q: Can I run this in production?**
A: Yes, but ensure proper infrastructure setup: managed databases, Redis cluster, proper Celery worker deployment, and monitoring. The system is designed for production use with proper setup.

## Contributing

The Knowledge Graph Agent is part of the DeerFlow project. To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues and questions:

- 📧 Open an issue on GitHub
- 💬 Join our community discussions
- 📖 Check the main [DeerFlow documentation](README.md)

---

**Happy Learning! 🚀**
