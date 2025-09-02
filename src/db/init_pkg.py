#!/usr/bin/env python3
"""
Neo4j Database Initialization Script

This script initializes the Neo4j database with the required schema including:
- Constraints for data integrity
- Indexes for performance
- Vector indexes for embedding searches

Run this script once after setting up your Neo4j database.
"""

import os
import sys

from dotenv import load_dotenv

# Add project root to path so we can import our modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from .neo4j_client import Neo4jClient
    from .neo4j_schema import Neo4jSchema
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running this script from the project root directory.")
    sys.exit(1)


def main():
    """Initialize Neo4j database schema."""
    print("🚀 CognitIQ Agent - Neo4j Database Initialization")
    print("=" * 50)

    # Load environment variables
    load_dotenv()

    # Check required environment variables
    required_vars = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file and ensure all Neo4j credentials are set.")
        sys.exit(1)

    # Initialize Neo4j client
    print("🔌 Connecting to Neo4j database...")
    try:
        neo4j_client = Neo4jClient()

        # Test connection
        with neo4j_client.driver.session() as session:
            result = session.run("RETURN 1 as test")
            assert result.single()["test"] == 1

        print("✅ Successfully connected to Neo4j database")

    except Exception as e:
        print(f"❌ Failed to connect to Neo4j: {e}")
        print("Please check your Neo4j credentials and ensure the database is running.")
        sys.exit(1)

    try:
        # Initialize schema manager
        print("\n📋 Initializing schema manager...")
        schema = Neo4jSchema(neo4j_client.driver)
        print("✅ Schema manager initialized")

        # Set up complete schema
        print("\n🔧 Setting up database schema...")
        print("   • Creating constraints...")
        schema.create_constraints()
        print("   • Creating indexes...")
        schema.create_indexes()
        print("   • Creating vector indexes...")
        schema.create_vector_index()
        print("✅ Database schema setup complete")

        # Verify schema
        print("\n🔍 Verifying schema setup...")
        missing_components = schema.verify_schema()

        if missing_components:
            print("⚠️  Some schema components are missing or misconfigured:")
            for component_type, component_name in missing_components:
                print(f"   • {component_type}: {component_name}")
            print(
                "\nPlease check the Neo4j logs for any errors during schema creation."
            )
            return False
        else:
            print("✅ All schema components verified successfully")

        # Display schema summary
        print("\n📊 Schema Summary:")
        with neo4j_client.driver.session() as session:
            # Count constraints
            constraints = session.run("SHOW CONSTRAINTS").data()
            print(f"   • Constraints: {len(constraints)}")

            # Count indexes (including vector indexes)
            indexes = session.run("SHOW INDEXES").data()
            regular_indexes = [idx for idx in indexes if idx.get("type") != "VECTOR"]
            vector_indexes = [idx for idx in indexes if idx.get("type") == "VECTOR"]
            print(f"   • Regular indexes: {len(regular_indexes)}")
            print(f"   • Vector indexes: {len(vector_indexes)}")

        print("\n🎉 Neo4j database initialization completed successfully!")
        print("\nYour database is now ready for the CognitIQ Agent.")
        print("You can now run the application and start building knowledge graphs.")

        return True

    except Exception as e:
        print(f"\n❌ Schema setup failed: {e}")
        return False

    finally:
        # Clean up
        neo4j_client.close()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
