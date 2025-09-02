"""Database client modules for deer-flow.

This package provides database clients and utilities for various database systems
used by deer-flow, including Neo4j for knowledge graphs and PostgreSQL for
application data and checkpoints.
"""

from .db_interface import EducationalReportsRepository
from .neo4j_client import Neo4jClient, Neo4jConnectionError
from .neo4j_schema import Neo4jSchema
from .postgres_client import PostgresClient, PostgresConnectionError
from .postgres_schema import PostgresSchema

__all__ = [
    # Neo4j
    "Neo4jClient",
    "Neo4jConnectionError",
    "Neo4jSchema",
    # PostgreSQL
    "PostgresClient",
    "PostgresConnectionError",
    "PostgresSchema",
    # Repositories
    "EducationalReportsRepository",
]
