"""Neo4j database schema definitions and setup.

This module defines the Neo4j database schema including constraints,
indexes, and provides utilities for schema initialization.
"""

from enum import Enum
from typing import List, Tuple

from neo4j import GraphDatabase

from src.config.configuration import Configuration
from src.llms.llm import get_embedding_model


class RelationshipType(str, Enum):
    """Defines the types of relationships in the Neo4j graph."""

    HAS_PREREQUISITE = "HAS_PREREQUISITE"
    PART_OF_GOAL = "PART_OF_GOAL"
    IS_DUPLICATE_OF = "IS_DUPLICATE_OF"
    IS_A = "IS_A"  # For concept hierarchy, e.g., "Python IS_A ProgrammingLanguage"
    PART_OF = "PART_OF"  # For mereology, e.g., "Loop PART_OF ControlFlow"
    RELATED_TO = "RELATED_TO"  # Generic relationship
    # Add other relationship types as needed


class Neo4jSchema:
    """Neo4j database schema management."""

    def __init__(self, driver: GraphDatabase.driver):
        """Initialize schema manager.

        Args:
            driver: Neo4j driver instance.
        """
        self.driver = driver
        config = Configuration()
        self.embedding_model = get_embedding_model(config.embedding_provider)
        self.embedding_dimension = self.embedding_model.dimension

    def create_constraints(self) -> None:
        """Create all required constraints."""
        constraints = [
            # Concept node constraints
            "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS NOT NULL",  # noqa: E501
        ]

        with self.driver.session() as session:
            for constraint in constraints:
                session.run(constraint)

    def create_indexes(self) -> None:
        """Create all required indexes."""
        indexes = [
            # Concept node indexes
            "CREATE INDEX concept_difficulty IF NOT EXISTS FOR (c:Concept) ON (c.difficulty_level)",
            "CREATE INDEX concept_created_at IF NOT EXISTS FOR (c:Concept) ON (c.created_at)",
            "CREATE INDEX concept_last_updated IF NOT EXISTS "
            "FOR (c:Concept) ON (c.last_updated_by_agent_at)",
        ]

        with self.driver.session() as session:
            for index in indexes:
                session.run(index)

    def create_vector_index(self) -> None:
        """Create vector search indexes for concept embeddings."""
        vector_indexes = [
            f"""
            CREATE VECTOR INDEX name_embedding_index IF NOT EXISTS
            FOR (c:Concept)
            ON (c.name_embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {self.embedding_dimension},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """,
            f"""
            CREATE VECTOR INDEX topic_embedding_index IF NOT EXISTS
            FOR (c:Concept)
            ON (c.topic_embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {self.embedding_dimension},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """,
            f"""
            CREATE VECTOR INDEX definition_embedding_index IF NOT EXISTS
            FOR (c:Concept)
            ON (c.definition_embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {self.embedding_dimension},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """,
        ]

        with self.driver.session() as session:
            for vector_index in vector_indexes:
                session.run(vector_index)

    def setup_schema(self) -> None:
        """Set up complete database schema."""
        self.create_constraints()
        self.create_indexes()
        self.create_vector_index()

    def verify_schema(self) -> List[Tuple[str, str]]:
        """Verify schema setup and return any missing components.

        Returns:
            List of tuples containing (component_type, component_name) for any
            missing schema components.
        """
        missing_components = []

        with self.driver.session() as session:
            # Check constraints
            constraints = session.run("SHOW CONSTRAINTS").data()
            expected_constraints = {"concept_id", "concept_name"}
            existing_constraints = {c["name"] for c in constraints}
            missing_constraints = expected_constraints - existing_constraints
            missing_components.extend([("constraint", c) for c in missing_constraints])

            # Check indexes
            indexes = session.run("SHOW INDEXES").data()
            expected_indexes = {
                "concept_difficulty",
                "concept_created_at",
                "concept_last_updated",
                "name_embedding_index",
                "topic_embedding_index",
                "definition_embedding_index",
            }
            existing_indexes = {i["name"] for i in indexes}
            missing_indexes = expected_indexes - existing_indexes
            missing_components.extend([("index", i) for i in missing_indexes])

            # Note: Vector index configuration verification is skipped as some Neo4j versions
            # don't return detailed options in SHOW INDEXES. The indexes are verified by existence only.

        return missing_components
