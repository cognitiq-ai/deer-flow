"""Educational Reports Repository for database operations.

This module provides a repository interface for educational reports,
including CRUD operations, queries, and business logic for educational content storage.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from psycopg.types.json import Jsonb

from src.db.postgres_client import PostgresClient, PostgresConnectionError


class EducationalReportsRepository:
    """Repository for educational reports database operations."""

    def __init__(self, postgres_client: Optional[PostgresClient] = None):
        """Initialize repository.

        Args:
            postgres_client: PostgreSQL client instance. If None, creates a new one.
        """
        self.client = postgres_client or PostgresClient()
        self.logger = logging.getLogger(__name__)

    def create_report(
        self,
        concept_id: str,
        concept_name: str,
        goal_id: str,
        content: Dict[str, Any],
        learning_objectives: List[str],
        summary: str,
        position_in_sequence: int,
        total_concepts: int,
    ) -> str:
        """Create a new educational report.

        Args:
            concept_id: ID of the concept this report is for
            concept_name: Name of the concept
            goal_id: ID of the overall learning goal
            content: Full educational content as JSON
            learning_objectives: List of learning objectives
            summary: Brief summary of the content
            position_in_sequence: Position in the learning sequence
            total_concepts: Total number of concepts in the sequence

        Returns:
            ID of the created report

        Raises:
            PostgresConnectionError: If database operation fails.
        """
        try:
            insert_sql = """
            INSERT INTO educational_reports (
                concept_id, concept_name, goal_id, content, learning_objectives,
                summary, position_in_sequence, total_concepts
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """

            with self.client.connection.cursor() as cursor:
                cursor.execute(
                    insert_sql,
                    (
                        concept_id,
                        concept_name,
                        goal_id,
                        Jsonb(content),
                        learning_objectives,
                        summary,
                        position_in_sequence,
                        total_concepts,
                    ),
                )

                result = cursor.fetchone()
                report_id = str(result["id"]) if result else None

                if not report_id:
                    raise PostgresConnectionError("Failed to create educational report")

                self.logger.info(f"Created educational report with ID: {report_id}")
                return report_id

        except Exception as e:
            self.logger.error(f"Error creating educational report: {e}")
            raise PostgresConnectionError(
                f"Failed to create educational report: {str(e)}"
            ) from e

    def get_report_by_id(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Get an educational report by ID.

        Args:
            report_id: ID of the report to retrieve

        Returns:
            Report data as dictionary, or None if not found

        Raises:
            PostgresConnectionError: If database operation fails.
        """
        try:
            select_sql = """
            SELECT * FROM educational_reports 
            WHERE id = %s
            """

            with self.client.connection.cursor() as cursor:
                cursor.execute(select_sql, (report_id,))
                result = cursor.fetchone()

                if result:
                    self.logger.debug(f"Retrieved educational report: {report_id}")

                return result

        except Exception as e:
            self.logger.error(f"Error retrieving educational report {report_id}: {e}")
            raise PostgresConnectionError(
                f"Failed to retrieve educational report: {str(e)}"
            ) from e

    def get_report_by_concept_id(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Get an educational report by concept ID.

        Args:
            concept_id: ID of the concept to find report for

        Returns:
            Report data as dictionary, or None if not found

        Raises:
            PostgresConnectionError: If database operation fails.
        """
        try:
            select_sql = """
            SELECT * FROM educational_reports 
            WHERE concept_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """

            with self.client.connection.cursor() as cursor:
                cursor.execute(select_sql, (concept_id,))
                result = cursor.fetchone()

                if result:
                    self.logger.debug(
                        f"Retrieved educational report for concept: {concept_id}"
                    )

                return result

        except Exception as e:
            self.logger.error(
                f"Error retrieving educational report for concept {concept_id}: {e}"
            )
            raise PostgresConnectionError(
                f"Failed to retrieve educational report: {str(e)}"
            ) from e

    def get_reports_by_goal_id(
        self, goal_id: str, ordered: bool = True
    ) -> List[Dict[str, Any]]:
        """Get all educational reports for a goal.

        Args:
            goal_id: ID of the goal to find reports for
            ordered: If True, returns reports ordered by position_in_sequence

        Returns:
            List of report data dictionaries

        Raises:
            PostgresConnectionError: If database operation fails.
        """
        try:
            order_clause = (
                "ORDER BY position_in_sequence ASC"
                if ordered
                else "ORDER BY created_at DESC"
            )
            select_sql = f"""
            SELECT * FROM educational_reports 
            WHERE goal_id = %s
            {order_clause}
            """

            with self.client.connection.cursor() as cursor:
                cursor.execute(select_sql, (goal_id,))
                results = cursor.fetchall()

                self.logger.debug(
                    f"Retrieved {len(results)} educational reports for goal: {goal_id}"
                )
                return results

        except Exception as e:
            self.logger.error(
                f"Error retrieving educational reports for goal {goal_id}: {e}"
            )
            raise PostgresConnectionError(
                f"Failed to retrieve educational reports: {str(e)}"
            ) from e

    def update_report(
        self,
        report_id: str,
        content: Optional[Dict[str, Any]] = None,
        learning_objectives: Optional[List[str]] = None,
        summary: Optional[str] = None,
    ) -> bool:
        """Update an educational report.

        Args:
            report_id: ID of the report to update
            content: Updated content (optional)
            learning_objectives: Updated learning objectives (optional)
            summary: Updated summary (optional)

        Returns:
            True if update was successful, False otherwise

        Raises:
            PostgresConnectionError: If database operation fails.
        """
        try:
            update_fields = []
            params = []

            if content is not None:
                update_fields.append("content = %s")
                params.append(Jsonb(content))

            if learning_objectives is not None:
                update_fields.append("learning_objectives = %s")
                params.append(learning_objectives)

            if summary is not None:
                update_fields.append("summary = %s")
                params.append(summary)

            if not update_fields:
                self.logger.warning("No fields to update in educational report")
                return False

            update_fields.append("updated_at = %s")
            params.append(datetime.now())

            params.append(report_id)  # For WHERE clause

            update_sql = f"""
            UPDATE educational_reports 
            SET {", ".join(update_fields)}
            WHERE id = %s
            """

            affected_rows = self.client.execute_update(update_sql, tuple(params))

            if affected_rows > 0:
                self.logger.info(f"Updated educational report: {report_id}")
                return True
            else:
                self.logger.warning(f"No educational report found with ID: {report_id}")
                return False

        except Exception as e:
            self.logger.error(f"Error updating educational report {report_id}: {e}")
            raise PostgresConnectionError(
                f"Failed to update educational report: {str(e)}"
            ) from e

    def delete_report(self, report_id: str) -> bool:
        """Delete an educational report.

        Args:
            report_id: ID of the report to delete

        Returns:
            True if deletion was successful, False otherwise

        Raises:
            PostgresConnectionError: If database operation fails.
        """
        try:
            delete_sql = "DELETE FROM educational_reports WHERE id = %s"

            affected_rows = self.client.execute_update(delete_sql, (report_id,))

            if affected_rows > 0:
                self.logger.info(f"Deleted educational report: {report_id}")
                return True
            else:
                self.logger.warning(f"No educational report found with ID: {report_id}")
                return False

        except Exception as e:
            self.logger.error(f"Error deleting educational report {report_id}: {e}")
            raise PostgresConnectionError(
                f"Failed to delete educational report: {str(e)}"
            ) from e

    def get_learning_sequence_summary(self, goal_id: str) -> Dict[str, Any]:
        """Get a summary of the learning sequence for a goal.

        Args:
            goal_id: ID of the goal

        Returns:
            Dictionary with sequence summary information

        Raises:
            PostgresConnectionError: If database operation fails.
        """
        try:
            summary_sql = """
            SELECT 
                COUNT(*) as total_reports,
                MIN(position_in_sequence) as first_position,
                MAX(position_in_sequence) as last_position,
                MAX(total_concepts) as expected_total,
                MIN(created_at) as first_created,
                MAX(updated_at) as last_updated
            FROM educational_reports 
            WHERE goal_id = %s
            """

            with self.client.connection.cursor() as cursor:
                cursor.execute(summary_sql, (goal_id,))
                result = cursor.fetchone()

                if result:
                    summary = {
                        "goal_id": goal_id,
                        "total_reports": result["total_reports"],
                        "first_position": result["first_position"],
                        "last_position": result["last_position"],
                        "expected_total": result["expected_total"],
                        "completion_percentage": (
                            (result["total_reports"] / result["expected_total"] * 100)
                            if result["expected_total"] and result["expected_total"] > 0
                            else 0
                        ),
                        "first_created": result["first_created"],
                        "last_updated": result["last_updated"],
                    }

                    self.logger.debug(
                        f"Generated learning sequence summary for goal: {goal_id}"
                    )
                    return summary
                else:
                    return {
                        "goal_id": goal_id,
                        "total_reports": 0,
                        "completion_percentage": 0,
                    }

        except Exception as e:
            self.logger.error(
                f"Error getting learning sequence summary for goal {goal_id}: {e}"
            )
            raise PostgresConnectionError(
                f"Failed to get learning sequence summary: {str(e)}"
            ) from e

    def close(self) -> None:
        """Close the database connection."""
        if self.client:
            self.client.close()

    def __enter__(self) -> "EducationalReportsRepository":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
