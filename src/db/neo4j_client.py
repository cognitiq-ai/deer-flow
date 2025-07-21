"""Neo4j client module for graph database operations.

This module provides a client interface for Neo4j database operations,
including configuration, connection management, and schema initialization.
"""

import os
import sys
from typing import List, Optional, Tuple, Type

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from src.db.schema import Neo4jSchema


class Neo4jConnectionError(Exception):
    """Exception raised when Neo4j connection fails."""


class Neo4jClient:
    """Client for Neo4j database operations."""

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        driver_class: Type = GraphDatabase,
    ):
        """Initialize Neo4j client.

        Args:
            uri: Neo4j database URI. Defaults to NEO4J_URI environment variable.
            username: Neo4j username. Defaults to NEO4J_USERNAME environment variable.
            password: Neo4j password. Defaults to NEO4J_PASSWORD environment variable.
            driver_class: Class to use for creating the driver. Defaults to GraphDatabase.

        Raises:
            ValueError: If configuration is incomplete.
        """
        self.uri = uri or os.getenv("NEO4J_URI")
        self.username = username or os.getenv("NEO4J_USERNAME")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        self._driver_class = driver_class

        if not all([self.uri, self.username, self.password]):
            raise ValueError(
                "Neo4j configuration incomplete. Please provide URI, username, and password "
                "either directly or through environment variables."
            )

        self._driver = None

    def _verify_connectivity(self, driver) -> None:
        """Verify connectivity to Neo4j database.

        Args:
            driver: Neo4j driver instance to verify.

        Raises:
            Neo4jConnectionError: If connection verification fails.
        """
        try:
            driver.verify_connectivity()
        except ServiceUnavailable as e:
            raise Neo4jConnectionError(
                "Failed to connect to Neo4j database. Please check your configuration."
            ) from e

    @property
    def driver(self) -> GraphDatabase.driver:
        """Get or create Neo4j driver instance.

        Returns:
            Neo4j driver instance.

        Raises:
            Neo4jConnectionError: If connection to Neo4j fails.
        """
        if self._driver is None:
            self._driver = self._driver_class.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_lifetime=3600,  # 1 hour
                max_connection_pool_size=50,
                connection_acquisition_timeout=60,  # 60 seconds
            )
            self._verify_connectivity(self._driver)
        return self._driver

    def close(self) -> None:
        """Close Neo4j driver connection."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def init_schema(self) -> List[Tuple[str, str]]:
        """Initialize Neo4j database schema.

        Returns:
            List of tuples containing missing schema components (type, name).
            Empty list if all components are present.

        Raises:
            Neo4jConnectionError: If schema initialization fails.
        """
        try:
            schema = Neo4jSchema(self.driver)
            schema.setup_schema()
            return schema.verify_schema()
        except Exception as e:
            raise Neo4jConnectionError(f"Error initializing schema: {str(e)}") from e

    def __enter__(self) -> "Neo4jClient":
        """Context manager entry.

        Returns:
            Self for use in with statement.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit.

        Ensures driver is closed when exiting context.
        """
        self.close()


def init_schema_cli() -> None:
    """Command-line interface for initializing Neo4j database schema."""
    try:
        with Neo4jClient() as client:
            missing_components = client.init_schema()
            if missing_components:
                print("Warning: Some schema components are missing:")
                for component_type, name in missing_components:
                    print(f"  - {component_type}: {name}")
            else:
                print("Schema setup completed successfully!")
    except Exception as e:
        print(f"Error initializing schema: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    init_schema_cli()
