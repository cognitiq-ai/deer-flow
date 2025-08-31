# Database Modules Documentation

This directory contains modular database clients and utilities for deer-flow.

## Overview

The database layer is organized into the following components:

- **Client Classes**: Connection management and high-level operations
- **Schema Classes**: Database schema creation and management  
- **Repository Classes**: Business logic and CRUD operations

## PostgreSQL Components

### PostgresClient

Central client for PostgreSQL operations with connection management.

```python
from src.db import PostgresClient

# Initialize with environment variable
client = PostgresClient()

# Or with explicit database URL
client = PostgresClient(database_url="postgresql://user:pass@localhost:5432/dbname")

# Use as context manager
with PostgresClient() as client:
    results = client.execute_query("SELECT * FROM educational_reports WHERE goal_id = %s", ("goal_123",))
    
# Initialize schema
client.init_schema()
```

### PostgresSchema

Manages database schema creation and verification.

```python
from src.db import PostgresClient, PostgresSchema

with PostgresClient() as client:
    schema = PostgresSchema(client.connection)
    schema.setup_schema()  # Create all tables and indexes
    success = schema.verify_schema()  # Check if everything exists
```

### EducationalReportsRepository  

High-level repository for educational content operations.

```python
from src.db import EducationalReportsRepository

# Use as context manager (recommended)
with EducationalReportsRepository() as repo:
    # Create a new educational report
    report_id = repo.create_report(
        concept_id="concept_123",
        concept_name="Linear Algebra",
        goal_id="goal_456", 
        content={"report": "content"},
        learning_objectives=["Understand vectors", "Solve systems"],
        summary="Introduction to linear algebra concepts",
        position_in_sequence=1,
        total_concepts=10
    )
    
    # Retrieve reports
    report = repo.get_report_by_id(report_id)
    concept_report = repo.get_report_by_concept_id("concept_123")
    goal_reports = repo.get_reports_by_goal_id("goal_456", ordered=True)
    
    # Get learning sequence summary
    summary = repo.get_learning_sequence_summary("goal_456")
    print(f"Completion: {summary['completion_percentage']}%")
```

## Neo4j Components

### Neo4jClient

Client for Neo4j knowledge graph operations.

```python
from src.db import Neo4jClient

# Initialize with environment variables (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
with Neo4jClient() as client:
    missing_components = client.init_schema()
    if missing_components:
        print("Missing schema components:", missing_components)
```

### Neo4jSchema

Manages Neo4j database schema including constraints, indexes, and vector indexes.

```python
from src.db import Neo4jClient, Neo4jSchema

with Neo4jClient() as client:
    schema = Neo4jSchema(client.driver)
    schema.setup_schema()
    missing = schema.verify_schema()
```

## Environment Variables

The database modules use the following environment variables:

### PostgreSQL
- `LANGGRAPH_CHECKPOINT_DB_URL`: PostgreSQL connection URL (required)

### Neo4j  
- `NEO4J_URI`: Neo4j database URI (required)
- `NEO4J_USERNAME`: Neo4j username (required)
- `NEO4J_PASSWORD`: Neo4j password (required)

Example `.env` file:
```bash
# PostgreSQL for educational reports and checkpoints
LANGGRAPH_CHECKPOINT_DB_URL=postgresql://user:password@localhost:5432/deer_flow_db

# Neo4j for knowledge graph
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

## CLI Usage

Both clients provide CLI tools for schema initialization:

```bash
# Initialize PostgreSQL schema
python -m src.db.postgres_client

# Initialize Neo4j schema  
python -m src.db.neo4j_client
```

## Error Handling

All database operations use custom exception classes:

- `PostgresConnectionError`: PostgreSQL connection or operation failures
- `Neo4jConnectionError`: Neo4j connection or operation failures

These exceptions provide clear error messages and chain the underlying database errors.

## Best Practices

1. **Use Context Managers**: Always use `with` statements for automatic connection cleanup
2. **Initialize Schema**: Call `init_schema()` before first use in production
3. **Handle Exceptions**: Catch and handle database-specific exceptions appropriately
4. **Environment Variables**: Use environment variables for database configuration
5. **Repository Pattern**: Use repository classes for business logic rather than direct client access
