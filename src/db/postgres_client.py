"""PostgreSQL client module for database operations.

This module provides a client interface for PostgreSQL database operations,
including configuration, connection management, and schema initialization.
"""

import os
import sys
from typing import Optional, Type

import psycopg
from psycopg.rows import dict_row

from src.db.postgres_schema import PostgresSchema


class PostgresConnectionError(Exception):
    """Exception raised when PostgreSQL connection fails."""


class PostgresClient:
    """Client for PostgreSQL database operations."""

    def __init__(
        self,
        database_url: Optional[str] = None,
        connection_class: Type = psycopg,
    ):
        """Initialize PostgreSQL client.

        Args:
            database_url: PostgreSQL database URL. Defaults to LANGGRAPH_CHECKPOINT_DB_URL environment variable.
            connection_class: Class to use for creating connections. Defaults to psycopg.

        Raises:
            ValueError: If configuration is incomplete.
        """
        self.database_url = database_url or os.getenv("LANGGRAPH_CHECKPOINT_DB_URL")
        self._connection_class = connection_class

        if not self.database_url:
            raise ValueError(
                "PostgreSQL configuration incomplete. Please provide database_url "
                "either directly or through LANGGRAPH_CHECKPOINT_DB_URL environment variable."
            )

        if not self.database_url.startswith(("postgresql://", "postgres://")):
            raise ValueError(
                "Database URL must be a PostgreSQL connection string starting with 'postgresql://' or 'postgres://'"
            )

        self._connection = None

    def _verify_connectivity(self, connection) -> None:
        """Verify connectivity to PostgreSQL database.

        Args:
            connection: PostgreSQL connection instance to verify.

        Raises:
            PostgresConnectionError: If connection verification fails.
        """
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
        except Exception as e:
            raise PostgresConnectionError(
                "Failed to connect to PostgreSQL database. Please check your configuration."
            ) from e

    @property
    def connection(self) -> psycopg.Connection:
        """Get or create PostgreSQL connection instance.

        Returns:
            PostgreSQL connection instance.

        Raises:
            PostgresConnectionError: If connection to PostgreSQL fails.
        """
        if self._connection is None or self._connection.closed:
            self._connection = self._connection_class.connect(
                self.database_url, row_factory=dict_row, autocommit=True
            )
            self._verify_connectivity(self._connection)
        return self._connection

    def close(self) -> None:
        """Close PostgreSQL connection."""
        if self._connection is not None and not self._connection.closed:
            self._connection.close()
            self._connection = None

    def init_schema(self) -> bool:
        """Initialize PostgreSQL database schema.

        Returns:
            True if schema initialization was successful, False otherwise.

        Raises:
            PostgresConnectionError: If schema initialization fails.
        """
        try:
            schema = PostgresSchema(self.connection)
            schema.setup_schema()
            return schema.verify_schema()
        except Exception as e:
            raise PostgresConnectionError(f"Error initializing schema: {str(e)}") from e

    def execute_query(self, query: str, params: Optional[tuple] = None) -> list:
        """Execute a query and return results.

        Args:
            query: SQL query to execute
            params: Optional query parameters

        Returns:
            List of result dictionaries

        Raises:
            PostgresConnectionError: If query execution fails.
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()
        except Exception as e:
            raise PostgresConnectionError(f"Error executing query: {str(e)}") from e

    def execute_update(self, query: str, params: Optional[tuple] = None) -> int:
        """Execute an update query and return affected rows.

        Args:
            query: SQL query to execute
            params: Optional query parameters

        Returns:
            Number of affected rows

        Raises:
            PostgresConnectionError: If query execution fails.
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                return cursor.rowcount
        except Exception as e:
            raise PostgresConnectionError(f"Error executing update: {str(e)}") from e

    def __enter__(self) -> "PostgresClient":
        """Context manager entry.

        Returns:
            Self for use in with statement.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit.

        Ensures connection is closed when exiting context.
        """
        self.close()


def init_schema_cli() -> None:
    """Command-line interface for initializing PostgreSQL database schema."""
    try:
        with PostgresClient() as client:
            success = client.init_schema()
            if success:
                print("PostgreSQL schema setup completed successfully!")
            else:
                print(
                    "Warning: Some schema components may be missing or failed to create."
                )
    except Exception as e:
        print(f"Error initializing PostgreSQL schema: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    init_schema_cli()
