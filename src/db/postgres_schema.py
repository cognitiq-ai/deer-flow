"""PostgreSQL database schema definitions and setup.

This module defines the PostgreSQL database schema including tables,
indexes, and provides utilities for schema initialization.
"""

import logging
from typing import List

import psycopg


class PostgresSchema:
    """PostgreSQL database schema management."""

    def __init__(self, connection: psycopg.Connection):
        """Initialize schema manager.

        Args:
            connection: PostgreSQL connection instance.
        """
        self.connection = connection
        self.logger = logging.getLogger(__name__)

    def create_extensions(self) -> None:
        """Create necessary PostgreSQL extensions."""
        extensions = [
            "uuid-ossp",  # For UUID generation in educational_reports
            "pg_trgm",  # For text similarity searches (optional but useful)
        ]

        try:
            with self.connection.cursor() as cursor:
                for extension in extensions:
                    cursor.execute(f'CREATE EXTENSION IF NOT EXISTS "{extension}";')
                self.logger.info("PostgreSQL extensions created/verified successfully")
        except Exception as e:
            self.logger.error(f"Failed to create extensions: {e}")
            raise

    def create_educational_reports_table(self) -> None:
        """Create the educational_reports table if it doesn't exist."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS educational_reports (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            concept_id VARCHAR(255) NOT NULL,
            concept_name VARCHAR(500) NOT NULL,
            goal_id VARCHAR(255) NOT NULL,
            content JSONB NOT NULL,
            learning_objectives TEXT[],
            summary TEXT,
            position_in_sequence INTEGER,
            total_concepts INTEGER,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        """

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(create_table_sql)
                self.logger.info(
                    "Educational reports table created/verified successfully"
                )
        except Exception as e:
            self.logger.error(f"Failed to create educational_reports table: {e}")
            raise

    def create_educational_reports_indexes(self) -> None:
        """Create indexes for the educational_reports table."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_educational_reports_concept_id ON educational_reports(concept_id);",
            "CREATE INDEX IF NOT EXISTS idx_educational_reports_goal_id ON educational_reports(goal_id);",
            "CREATE INDEX IF NOT EXISTS idx_educational_reports_sequence ON educational_reports(goal_id, position_in_sequence);",
            "CREATE INDEX IF NOT EXISTS idx_educational_reports_created_at ON educational_reports(created_at);",
            "CREATE INDEX IF NOT EXISTS idx_educational_reports_updated_at ON educational_reports(updated_at);",
        ]

        try:
            with self.connection.cursor() as cursor:
                for index_sql in indexes:
                    cursor.execute(index_sql)
                self.logger.info(
                    "Educational reports indexes created/verified successfully"
                )
        except Exception as e:
            self.logger.error(f"Failed to create educational_reports indexes: {e}")
            raise

    def setup_schema(self) -> None:
        """Set up complete database schema."""
        self.create_extensions()
        self.create_educational_reports_table()
        self.create_educational_reports_indexes()

    def verify_schema(self) -> List[str]:
        """Verify schema setup and return list of missing components.

        Returns:
            List of missing component names. Empty list if all components exist.
        """
        missing_components = []

        try:
            with self.connection.cursor() as cursor:
                # Check if required extensions exist
                extension_check_sql = """
                SELECT extname FROM pg_extension 
                WHERE extname IN ('uuid-ossp', 'pg_trgm');
                """
                cursor.execute(extension_check_sql)
                existing_extensions = {row["extname"] for row in cursor.fetchall()}

                expected_extensions = {"uuid-ossp", "pg_trgm"}
                missing_extensions = expected_extensions - existing_extensions
                if missing_extensions:
                    missing_components.extend(
                        [f"extension:{ext}" for ext in missing_extensions]
                    )

                # Check if educational_reports table exists
                table_check_sql = """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'educational_reports'
                );
                """
                cursor.execute(table_check_sql)
                table_exists = cursor.fetchone()["exists"]

                if not table_exists:
                    missing_components.append("table:educational_reports")
                else:
                    # Check if required indexes exist
                    index_check_sql = """
                    SELECT indexname FROM pg_indexes 
                    WHERE tablename = 'educational_reports'
                    AND schemaname = 'public';
                    """
                    cursor.execute(index_check_sql)
                    existing_indexes = {row["indexname"] for row in cursor.fetchall()}

                    expected_indexes = {
                        "idx_educational_reports_concept_id",
                        "idx_educational_reports_goal_id",
                        "idx_educational_reports_sequence",
                        "idx_educational_reports_created_at",
                        "idx_educational_reports_updated_at",
                    }

                    missing_indexes = expected_indexes - existing_indexes
                    if missing_indexes:
                        missing_components.extend(
                            [f"index:{idx}" for idx in missing_indexes]
                        )

                if not missing_components:
                    self.logger.info("All schema components verified successfully")
                else:
                    self.logger.warning(f"Missing components: {missing_components}")

                return missing_components

        except Exception as e:
            self.logger.error(f"Error verifying schema: {e}")
            return ["error:verification_failed"]

    def get_table_info(self, table_name: str = "educational_reports") -> List[dict]:
        """Get information about a table's structure.

        Args:
            table_name: Name of the table to inspect

        Returns:
            List of column information dictionaries
        """
        try:
            column_info_sql = """
            SELECT 
                column_name,
                data_type,
                is_nullable,
                column_default
            FROM information_schema.columns 
            WHERE table_schema = 'public' 
            AND table_name = %s
            ORDER BY ordinal_position;
            """

            with self.connection.cursor() as cursor:
                cursor.execute(column_info_sql, (table_name,))
                return cursor.fetchall()

        except Exception as e:
            self.logger.error(f"Error getting table info for {table_name}: {e}")
            return []

    def drop_educational_reports_table(self) -> None:
        """Drop the educational_reports table. Use with caution!"""
        drop_table_sql = "DROP TABLE IF EXISTS educational_reports CASCADE;"

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(drop_table_sql)
                self.logger.info("Educational reports table dropped successfully")
        except Exception as e:
            self.logger.error(f"Failed to drop educational_reports table: {e}")
            raise
