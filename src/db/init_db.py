#!/usr/bin/env python3
"""
PostgreSQL Database Initialization Script

This script initializes the PostgreSQL database with the required setup including:
- Required extensions installation
- Application schema creation (tables, indexes)
- Verification of all components

Run this script once after setting up your PostgreSQL server.
"""

import os
import sys
from urllib.parse import urlparse

from dotenv import load_dotenv

# Add project root to path so we can import our modules
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

try:
    from src.db.postgres_client import PostgresClient, PostgresConnectionError
    from src.db.postgres_schema import PostgresSchema
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running this script from the project root directory.")
    sys.exit(1)


def main():
    """Initialize PostgreSQL database."""
    print("🚀 CognitIQ Agent - PostgreSQL Database Initialization")
    print("=" * 55)

    # Load environment variables
    load_dotenv()

    # Check required environment variables
    db_url = os.getenv("LANGGRAPH_CHECKPOINT_DB_URL")
    if not db_url:
        print("❌ Missing required environment variable: LANGGRAPH_CHECKPOINT_DB_URL")
        print("Please check your .env file and ensure PostgreSQL URL is set.")
        print("\nExample:")
        print(
            "LANGGRAPH_CHECKPOINT_DB_URL=postgresql://user:password@localhost:5432/kg_db"
        )
        sys.exit(1)

    if not db_url.startswith(("postgresql://", "postgres://")):
        print("❌ LANGGRAPH_CHECKPOINT_DB_URL must be a PostgreSQL connection string")
        print("Example: postgresql://user:password@localhost:5432/kg_db")
        sys.exit(1)

    parsed = urlparse(db_url)
    target_database = parsed.path.lstrip("/") if parsed.path else None
    if not target_database:
        print("❌ No database name found in LANGGRAPH_CHECKPOINT_DB_URL")
        print("Please provide a database in the connection string path.")
        sys.exit(1)

    # Step 1: Connect to the exact database provided in the URL
    print("\n🔧 Setting up database...")
    print(f"   • Target database from URL: {target_database}")
    print("\n🔌 Connecting to PostgreSQL database from LANGGRAPH_CHECKPOINT_DB_URL...")
    try:
        # Always use the exact connection string passed by the user/config.
        postgres_client = PostgresClient(database_url=db_url)

        # Test connection
        with postgres_client.connection.cursor() as cursor:
            cursor.execute("SELECT current_database() as current_db, 1 as test")
            result = cursor.fetchone()
            assert result["test"] == 1
            if result.get("current_db") != target_database:
                print(
                    "⚠️  Connected database does not match connection string path: "
                    f"expected '{target_database}', got '{result.get('current_db')}'."
                )

        print(
            "✅ Successfully connected to PostgreSQL database "
            f"'{result.get('current_db', target_database)}'"
        )

    except PostgresConnectionError as e:
        print(f"❌ Failed to connect to PostgreSQL: {e}")
        print(
            "Please check your PostgreSQL credentials and ensure the server is running."
        )
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error connecting to PostgreSQL: {e}")
        sys.exit(1)

    try:
        # Step 2: Initialize schema manager in the same target database.
        print("\n📋 Initializing schema manager...")
        schema = PostgresSchema(postgres_client.connection)
        print("✅ Schema manager initialized")

        # Set up complete schema
        print("\n🔧 Setting up database schema...")
        print("   • Installing extensions...")
        schema.create_extensions()
        print("   • Creating tables...")
        schema.create_educational_reports_table()
        print("   • Creating indexes...")
        schema.create_educational_reports_indexes()
        print("✅ Database schema setup complete")

        # Verify schema
        print("\n🔍 Verifying schema setup...")
        missing_components = schema.verify_schema()

        if missing_components:
            print("⚠️  Some schema components are missing or misconfigured:")
            for component in missing_components:
                component_type, component_name = component.split(":", 1)
                print(f"   • {component_type}: {component_name}")
            print(
                "\nPlease check the PostgreSQL logs for any errors during schema creation."
            )
            return False
        else:
            print("✅ All schema components verified successfully")

        # Display schema summary
        print("\n📊 Schema Summary:")
        with postgres_client.connection.cursor() as cursor:
            # Count extensions
            cursor.execute("""
                SELECT COUNT(*) as count FROM pg_extension 
                WHERE extname IN ('uuid-ossp', 'pg_trgm')
            """)
            extension_count = cursor.fetchone()["count"]
            print(f"   • Extensions installed: {extension_count}")

            # Count tables in public schema
            cursor.execute("""
                SELECT COUNT(*) as count FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            """)
            table_count = cursor.fetchone()["count"]
            print(f"   • Tables created: {table_count}")

            # Count indexes on our table
            cursor.execute("""
                SELECT COUNT(*) as count FROM pg_indexes 
                WHERE tablename = 'educational_reports' AND schemaname = 'public'
            """)
            index_count = cursor.fetchone()["count"]
            print(f"   • Indexes created: {index_count}")

        print("\n🎉 PostgreSQL database initialization completed successfully!")
        print("\nYour database is now ready for:")
        print("  • Educational content storage")
        print("  • LangGraph checkpoint persistence")
        print("  • Knowledge graph operations")
        print("\nYou can now run the application and start building knowledge graphs.")

        return True

    except Exception as e:
        print(f"\n❌ Schema setup failed: {e}")
        return False

    finally:
        # Clean up
        postgres_client.close()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
