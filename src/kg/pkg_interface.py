"""Neo4j Persistent Knowledge Graph Interface (PKGInterface).

This module provides the PKGInterface class that handles interactions with
the Neo4j knowledge graph following the agent framework specification.
"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from src.kg.models import (
    AgentWorkingGraph,
    ConceptNode,
    ConceptNodeStatus,
    Relationship,
    RelationshipType,
)
from src.db.neo4j_client import Neo4jClient

# Type alias for concept data
ConceptData = Dict[str, Union[str, float, List[str], Dict[str, float]]]


class PrerequisiteCycleException(Exception):
    """Exception raised when a relationship would create a cycle.

    Note: This exception is used for all cycle-prone relationship types,
    not just prerequisites. The name is maintained for backward compatibility.
    """

    def __init__(
        self,
        source_id: str,
        target_id: str,
        path: List[str],
        relationship_type: str = "HAS_PREREQUISITE",
    ):
        """Initialize the exception.

        Args:
            source_id: ID of the source node in the relationship
            target_id: ID of the target node in the relationship
            path: The path that forms the cycle
            relationship_type: Type of relationship that would create the cycle
        """
        self.source_id = source_id
        self.target_id = target_id
        self.path = path
        self.relationship_type = relationship_type
        self.offending_relationship_details = f"{source_id} -> {target_id}"
        super().__init__(
            f"Adding {relationship_type} from {source_id} to {target_id} would create a cycle: {path}"
        )


class PKGInterface:
    """
    Persistent Knowledge Graph Interface.

    Provides methods to interact with Neo4j for knowledge graph operations.
    """

    def __init__(self, neo4j_client: Optional[Neo4jClient] = None):
        """Initialize the PKGInterface with Neo4j connection.

        Args:
            neo4j_client: Optional Neo4jClient instance. If not provided, a new one will be created.
        """
        self.neo4j_client = neo4j_client or Neo4jClient()

    def _parse_node_attributes(self, attributes: Any) -> Dict[str, Any]:
        """Parse node attributes from Neo4j, handling JSON strings and empty values.

        Args:
            attributes: Raw attributes value from Neo4j

        Returns:
            Parsed attributes dictionary
        """
        if not attributes:
            return {}

        if isinstance(attributes, str):
            try:
                return json.loads(attributes)
            except json.JSONDecodeError:
                return {}

        if isinstance(attributes, dict):
            return attributes

        return {}

    def _parse_node_data(self, node: Any) -> Dict[str, Any]:
        """Parse Neo4j node data into a standardized dictionary.

        Args:
            node: Neo4j node object

        Returns:
            Dictionary with node properties
        """
        if not node:
            return {}

        # Handle different ways Neo4j might return node data
        if hasattr(node, "get"):
            # Node object with get method
            node_data = {
                "id": node.get("id"),
                "name": node.get("name"),
                "topic": node.get("topic", ""),
                "definition": node.get("definition"),
                "node_type": node.get("node_type", "concept"),
                "exists_in_pkg": node.get("exists_in_pkg", False),
                "discovery_count_search": node.get("discovery_count_search", 0),
                "discovery_count_llm_inference": node.get(
                    "discovery_count_llm_inference", 0
                ),
                "research_results": node.get("research_results", []),
                "pages_analyzed_count": node.get("pages_analyzed_count", 0),
                "definition_confidence_llm": node.get("definition_confidence_llm", 0.0),
                "last_updated_timestamp": node.get(
                    "last_updated_timestamp", datetime.now()
                ),
                "status": node.get("status", ConceptNodeStatus.STUB.value),
                "name_embedding": node.get("name_embedding"),
                "topic_embedding": node.get("topic_embedding"),
                "definition_embedding": node.get("definition_embedding"),
            }
        elif hasattr(node, "__getitem__"):
            # Dictionary-like access
            try:
                node_data = {
                    "id": node.get("id") if hasattr(node, "get") else node["id"],
                    "name": (
                        node.get("name") if hasattr(node, "get") else node.get("name")
                    ),
                    "topic": (
                        node.get("topic", "")
                        if hasattr(node, "get")
                        else node.get("topic", "")
                    ),
                    "definition": (
                        node.get("definition")
                        if hasattr(node, "get")
                        else node.get("definition")
                    ),
                    "node_type": (
                        node.get("node_type", "concept")
                        if hasattr(node, "get")
                        else node.get("node_type", "concept")
                    ),
                    "exists_in_pkg": (
                        node.get("exists_in_pkg", False)
                        if hasattr(node, "get")
                        else node.get("exists_in_pkg", False)
                    ),
                    "discovery_count_search": (
                        node.get("discovery_count_search", 0)
                        if hasattr(node, "get")
                        else node.get("discovery_count_search", 0)
                    ),
                    "discovery_count_llm_inference": (
                        node.get("discovery_count_llm_inference", 0)
                        if hasattr(node, "get")
                        else node.get("discovery_count_llm_inference", 0)
                    ),
                    "research_results": (
                        node.get("research_results", [])
                        if hasattr(node, "get")
                        else node.get("research_results", [])
                    ),
                    "pages_analyzed_count": (
                        node.get("pages_analyzed_count", 0)
                        if hasattr(node, "get")
                        else node.get("pages_analyzed_count", 0)
                    ),
                    "definition_confidence_llm": (
                        node.get("definition_confidence_llm", 0.0)
                        if hasattr(node, "get")
                        else node.get("definition_confidence_llm", 0.0)
                    ),
                    "last_updated_timestamp": (
                        node.get("last_updated_timestamp", datetime.now())
                        if hasattr(node, "get")
                        else node.get("last_updated_timestamp", datetime.now())
                    ),
                    "status": (
                        node.get("status", ConceptNodeStatus.STUB.value)
                        if hasattr(node, "get")
                        else node.get("status", ConceptNodeStatus.STUB.value)
                    ),
                    "name_embedding": (
                        node.get("name_embedding")
                        if hasattr(node, "get")
                        else node.get("name_embedding")
                    ),
                    "topic_embedding": (
                        node.get("topic_embedding")
                        if hasattr(node, "get")
                        else node.get("topic_embedding")
                    ),
                    "definition_embedding": (
                        node.get("definition_embedding")
                        if hasattr(node, "get")
                        else node.get("definition_embedding")
                    ),
                }
            except (KeyError, TypeError):
                return {}
        else:
            # Fallback for unexpected node types
            return {}

        return node_data

    def _create_concept_node(self, node_data: Dict[str, Any]) -> Optional[ConceptNode]:
        """Create a ConceptNode from parsed node data.

        Args:
            node_data: Dictionary with node properties

        Returns:
            ConceptNode instance or None if data is invalid
        """
        try:
            if not node_data or not node_data.get("id"):
                return None

            return ConceptNode(
                id=node_data["id"],
                name=node_data.get("name", ""),
                topic=node_data.get("topic", ""),
                definition=node_data.get("definition"),
                node_type=node_data.get("node_type", "concept"),
                exists_in_pkg=node_data.get("exists_in_pkg", False),
                definition_research=node_data.get("definition_research", []),
                definition_confidence_llm=node_data.get(
                    "definition_confidence_llm", 0.0
                ),
                last_updated_timestamp=node_data.get(
                    "last_updated_timestamp", datetime.now()
                ),
                defined_status=ConceptNodeStatus(
                    node_data.get("status", ConceptNodeStatus.STUB.value)
                ),
                name_embedding=node_data.get("name_embedding"),
                topic_embedding=node_data.get("topic_embedding"),
                definition_embedding=node_data.get("definition_embedding"),
            )
        except (ValueError, TypeError):
            # Log the error or handle it appropriately
            return None

    def get_node_by_id(self, node_id: str) -> Optional[ConceptNode]:
        """
        Get a node from Neo4j by ID.

        Args:
            node_id: The ID of the node to retrieve

        Returns:
            ConceptNode if found, None otherwise
        """
        query = """
        MATCH (n:Concept {id: $id})
        RETURN n
        """

        try:
            with self.neo4j_client.driver.session() as session:
                record = session.run(query, {"id": node_id}).single()

                if not record:
                    return None

                node_data = self._parse_node_data(record["n"])
                return self._create_concept_node(node_data)
        except Exception:
            return None

    def find_or_create_node(self, node_data: ConceptNode) -> ConceptNode:
        """
        Find or create a node in Neo4j.

        Args:
            node_data: The node data to find or create

        Returns:
            The found or created ConceptNode with updated ID
        """
        # If node has no ID, generate one
        if node_data.id is None:
            node_data.id = str(uuid.uuid4())

        # Check if node exists
        existing_node = self.get_node_by_id(node_data.id)

        if existing_node:
            # Node exists, return it
            return existing_node

        # Node doesn't exist, create it
        query = """
        CREATE (n:Concept $node_props)
        RETURN n
        """

        # Convert node data to properties dict
        node_props = node_data.model_dump()

        # Convert enum values to strings for Neo4j
        node_props["status"] = node_data.status.value

        # Convert research sources into yaml for Neo4j
        node_props["definition_research"] = [
            research_output.to_yaml()
            for research_output in node_data.definition_research
        ]
        node_props["prerequisites_research"] = [
            research_output.to_yaml()
            for research_output in node_data.prerequisites_research
        ]

        # Convert datetime to ISO format for Neo4j
        if isinstance(node_props.get("last_updated_timestamp"), datetime):
            node_props["last_updated_timestamp"] = node_props[
                "last_updated_timestamp"
            ].isoformat()

        # Handle None embedding vectors by converting to empty lists
        for embedding_field in [
            "name_embedding",
            "topic_embedding",
            "definition_embedding",
        ]:
            if embedding_field not in node_props or node_props[embedding_field] is None:
                node_props[embedding_field] = []

        # Update node_data.exists_in_pkg to True
        node_props["exists_in_pkg"] = True

        with self.neo4j_client.driver.session() as session:
            record = session.run(query, {"node_props": node_props}).single()

            if not record:
                raise Exception("Failed to create node")

            # Return the original node_data with ID
            return node_data

    def detect_relationship_cycle(
        self, source_id: str, target_id: str, relationship_type: RelationshipType
    ) -> Tuple[bool, Optional[List[str]]]:
        """
        Check if creating a relationship from source to target would create a cycle for the given relationship type.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship_type: Type of relationship to check for cycles

        Returns:
            A tuple (has_cycle, path). If has_cycle is True, path contains the nodes in the cycle.
        """
        # Only check cycles for directional relationships that could form meaningful cycles
        cycle_prone_types = [
            RelationshipType.HAS_PREREQUISITE,
            RelationshipType.IS_TYPE_OF,
            RelationshipType.IS_PART_OF,
        ]

        if relationship_type not in cycle_prone_types:
            return False, None

        # Check if there's already a path from target back to source (which would create a cycle)
        rel_type_str = relationship_type.value
        query = f"""
        MATCH path = (target:Concept {{id: $target_id}})-[:{rel_type_str}*]->(source:Concept {{id: $source_id}})
        RETURN [node IN nodes(path) | node.id] AS cycle_path
        """

        try:
            with self.neo4j_client.driver.session() as session:
                record = session.run(
                    query, {"source_id": source_id, "target_id": target_id}
                ).single()

                if record and record.get("cycle_path"):
                    # Return the path of the cycle
                    return True, record["cycle_path"]

                return False, None
        except Exception:
            return False, None

    def detect_prerequisites_cycle(
        self, source_id: str, target_id: str
    ) -> Tuple[bool, Optional[List[str]]]:
        """
        Check if creating a HAS_PREREQUISITE relationship from source to target would create a cycle.

        This method is maintained for backward compatibility.

        Args:
            source_id: Source node ID
            target_id: Target node ID

        Returns:
            A tuple (has_cycle, path). If has_cycle is True, path contains the nodes in the cycle.
        """
        return self.detect_relationship_cycle(
            source_id, target_id, RelationshipType.HAS_PREREQUISITE
        )

    def find_or_create_relationship(self, rel_data: Relationship) -> Relationship:
        """
        Find or create a relationship in Neo4j.

        Args:
            rel_data: The relationship data to find or create

        Returns:
            The found or created Relationship with updated ID

        Raises:
            PrerequisiteCycleException: If adding a HAS_PREREQUISITE relationship would create a cycle
        """
        # If relationship has no ID, generate one
        if rel_data.id is None:
            rel_data.id = str(uuid.uuid4())

        # Check if source and target nodes exist
        source_node = self.get_node_by_id(rel_data.source_node_id)
        target_node = self.get_node_by_id(rel_data.target_node_id)

        if not source_node or not target_node:
            missing = "source" if not source_node else "target"
            raise ValueError(f"{missing} node does not exist")

        # Check for potential cycles for cycle-prone relationship types
        rel_type_enum = (
            rel_data.type
            if isinstance(rel_data.type, RelationshipType)
            else RelationshipType(rel_data.type)
        )

        # Check for cycles in directional relationships
        if rel_type_enum in [
            RelationshipType.HAS_PREREQUISITE,
            RelationshipType.IS_TYPE_OF,
            RelationshipType.IS_PART_OF,
        ]:
            has_cycle, cycle_path = self.detect_relationship_cycle(
                rel_data.source_node_id, rel_data.target_node_id, rel_type_enum
            )
            if has_cycle:
                raise PrerequisiteCycleException(
                    rel_data.source_node_id,
                    rel_data.target_node_id,
                    cycle_path,
                    rel_type_enum.value,
                )

        # Check if a relationship of the same type already exists between these nodes
        rel_type = (
            rel_data.type.value
            if isinstance(rel_data.type, RelationshipType)
            else rel_data.type
        )

        query = """
        MATCH (source:Concept {id: $source_id})-[r]->(target:Concept {id: $target_id})
        WHERE type(r) = $rel_type
        RETURN r
        """

        try:
            with self.neo4j_client.driver.session() as session:
                # Check if relationship of same type already exists
                record = session.run(
                    query,
                    {
                        "source_id": rel_data.source_node_id,
                        "target_id": rel_data.target_node_id,
                        "rel_type": rel_type,
                    },
                ).single()

                if record:
                    # Relationship exists, merge the data
                    existing_rel_dict = dict(record["r"].items())

                    # Create existing relationship object
                    existing_rel = Relationship(
                        id=existing_rel_dict.get("id"),
                        source_node_id=rel_data.source_node_id,
                        target_node_id=rel_data.target_node_id,
                        type=rel_type,
                        discovery_count_search=existing_rel_dict.get(
                            "discovery_count_search", 0
                        ),
                        discovery_count_llm_inference=existing_rel_dict.get(
                            "discovery_count_llm_inference", 0
                        ),
                        source_urls=existing_rel_dict.get("source_urls", []),
                        type_confidence_llm=existing_rel_dict.get(
                            "type_confidence_llm", 0.0
                        ),
                        existence_confidence_llm=existing_rel_dict.get(
                            "existence_confidence_llm", 0.0
                        ),
                        last_updated_timestamp=existing_rel_dict.get(
                            "last_updated_timestamp", datetime.now()
                        ),
                    )

                    # Merge new data into existing relationship
                    merged_rel = self._merge_relationships(existing_rel, rel_data)

                    # Update the relationship in Neo4j
                    self._update_relationship(merged_rel)

                    return merged_rel

                # Relationship doesn't exist, create it
                rel_props = rel_data.model_dump()

                # Remove source_node_id, target_node_id, and type from properties
                rel_props.pop("source_node_id", None)
                rel_props.pop("target_node_id", None)
                rel_props.pop("type", None)

                # Convert datetime to ISO format for Neo4j
                if isinstance(rel_props.get("last_updated_timestamp"), datetime):
                    rel_props["last_updated_timestamp"] = rel_props[
                        "last_updated_timestamp"
                    ].isoformat()

                create_query = f"""
                MATCH (source:Concept {{id: $source_id}})
                MATCH (target:Concept {{id: $target_id}})
                CREATE (source)-[r:{rel_type} $rel_props]->(target)
                RETURN r
                """

                record = session.run(
                    create_query,
                    {
                        "source_id": rel_data.source_node_id,
                        "target_id": rel_data.target_node_id,
                        "rel_props": rel_props,
                    },
                ).single()

                if not record:
                    raise Exception("Failed to create relationship")

                # Return the original rel_data with ID
                return rel_data
        except Exception as e:
            if isinstance(e, PrerequisiteCycleException):
                raise
            raise Exception(f"Failed to create or find relationship: {str(e)}")

    def _merge_relationships(
        self, existing_rel: Relationship, new_rel: Relationship
    ) -> Relationship:
        """
        Merge a new relationship into an existing relationship, taking higher confidence values.

        Args:
            existing_rel: The existing relationship in the database
            new_rel: The new relationship data to merge

        Returns:
            Merged relationship with updated values
        """
        # Create a copy of the existing relationship to avoid modifying the original
        merged_rel = Relationship(
            id=existing_rel.id,
            source_node_id=existing_rel.source_node_id,
            target_node_id=existing_rel.target_node_id,
            type=existing_rel.type,
            discovery_count_search=existing_rel.discovery_count_search,
            discovery_count_llm_inference=existing_rel.discovery_count_llm_inference,
            source_urls=existing_rel.source_urls.copy(),
            type_confidence_llm=existing_rel.type_confidence_llm,
            existence_confidence_llm=existing_rel.existence_confidence_llm,
            last_updated_timestamp=existing_rel.last_updated_timestamp,
        )

        # Use the relationship's built-in merge method
        merged_rel.merge_relationship(new_rel)

        return merged_rel

    def _update_relationship(self, relationship: Relationship) -> None:
        """
        Update an existing relationship in Neo4j with new properties.

        Args:
            relationship: The relationship to update
        """
        rel_type = (
            relationship.type.value
            if isinstance(relationship.type, RelationshipType)
            else relationship.type
        )

        # Prepare properties for update
        rel_props = relationship.model_dump()

        # Remove fields that shouldn't be updated
        rel_props.pop("id", None)
        rel_props.pop("source_node_id", None)
        rel_props.pop("target_node_id", None)
        rel_props.pop("type", None)

        # Convert datetime to ISO format for Neo4j
        if isinstance(rel_props.get("last_updated_timestamp"), datetime):
            rel_props["last_updated_timestamp"] = rel_props[
                "last_updated_timestamp"
            ].isoformat()

        query = f"""
        MATCH (source:Concept {{id: $source_id}})-[r:{rel_type}]->(target:Concept {{id: $target_id}})
        WHERE r.id = $rel_id
        SET r += $rel_props
        RETURN r
        """

        try:
            with self.neo4j_client.driver.session() as session:
                session.run(
                    query,
                    {
                        "source_id": relationship.source_node_id,
                        "target_id": relationship.target_node_id,
                        "rel_id": relationship.id,
                        "rel_props": rel_props,
                    },
                )
        except Exception as e:
            raise Exception(f"Failed to update relationship: {str(e)}")

    def commit_changes(
        self,
        nodes_to_upsert: List[Union[ConceptNode, Dict[str, Any]]],
        edges_to_upsert: List[Union[Relationship, Dict[str, Any]]],
        nodes_to_delete: Optional[List[Union[ConceptNode, Dict[str, Any]]]] = None,
    ) -> Dict[str, List[Union[str, Dict[str, str]]]]:
        """
        Commit changes to Neo4j.

        Args:
            nodes_to_upsert: List of nodes to create or update (ConceptNode objects or dictionaries)
            edges_to_upsert: List of relationships to create or update (Relationship objects or dictionaries)
            nodes_to_delete: Optional list of nodes to delete (ConceptNode objects or dictionaries)

        Returns:
            Dictionary with lists of committed and rejected changes
        """
        result = {
            "committed_nodes": [],
            "committed_edges": [],
            "rejected_edges": [],
            "deleted_nodes": [],
            "errors": [],
        }

        # Process nodes first
        for node_data in nodes_to_upsert:
            try:
                # Convert dictionary to ConceptNode if needed
                if isinstance(node_data, dict):
                    node = ConceptNode(**node_data)
                else:
                    node = node_data

                committed_node = self.find_or_create_node(node)
                result["committed_nodes"].append(committed_node.id)
            except Exception as e:
                node_id = (
                    node_data.get("id")
                    if isinstance(node_data, dict)
                    else (node_data.id if hasattr(node_data, "id") else "unknown")
                )
                result["errors"].append({"node_id": node_id, "error": str(e)})

        # Process edges
        for edge_data in edges_to_upsert:
            try:
                # Convert dictionary to Relationship if needed
                if isinstance(edge_data, dict):
                    edge = Relationship(**edge_data)
                else:
                    edge = edge_data

                # Check for cycles in cycle-prone relationship types
                rel_type_enum = (
                    edge.type
                    if isinstance(edge.type, RelationshipType)
                    else RelationshipType(edge.type)
                )
                rel_type_str = rel_type_enum.value

                if rel_type_enum in [
                    RelationshipType.HAS_PREREQUISITE,
                    RelationshipType.IS_TYPE_OF,
                    RelationshipType.IS_PART_OF,
                ]:
                    has_cycle, cycle_path = self.detect_relationship_cycle(
                        edge.source_node_id, edge.target_node_id, rel_type_enum
                    )
                    if has_cycle:
                        result["rejected_edges"].append(
                            {
                                "source_id": edge.source_node_id,
                                "target_id": edge.target_node_id,
                                "type": rel_type_str,
                                "reason": f"Would create a cycle: {cycle_path}",
                            }
                        )
                        continue

                # Create or update the relationship
                committed_edge = self.find_or_create_relationship(edge)
                result["committed_edges"].append(committed_edge.id)

            except PrerequisiteCycleException as e:
                edge_source = (
                    edge_data.get("source_node_id")
                    if isinstance(edge_data, dict)
                    else edge_data.source_node_id
                )
                edge_target = (
                    edge_data.get("target_node_id")
                    if isinstance(edge_data, dict)
                    else edge_data.target_node_id
                )
                edge_type = (
                    edge_data.get("type")
                    if isinstance(edge_data, dict)
                    else (
                        edge_data.type.value
                        if isinstance(edge_data.type, RelationshipType)
                        else edge_data.type
                    )
                )

                result["rejected_edges"].append(
                    {
                        "source_id": edge_source,
                        "target_id": edge_target,
                        "type": edge_type,
                        "reason": f"Would create a cycle: {e.path}",
                    }
                )

            except Exception as e:
                edge_source = (
                    edge_data.get("source_node_id")
                    if isinstance(edge_data, dict)
                    else edge_data.source_node_id
                )
                edge_target = (
                    edge_data.get("target_node_id")
                    if isinstance(edge_data, dict)
                    else edge_data.target_node_id
                )
                edge_type = (
                    edge_data.get("type")
                    if isinstance(edge_data, dict)
                    else (
                        edge_data.type.value
                        if isinstance(edge_data.type, RelationshipType)
                        else edge_data.type
                    )
                )

                result["errors"].append(
                    {
                        "edge": {
                            "source_id": edge_source,
                            "target_id": edge_target,
                            "type": edge_type,
                        },
                        "error": str(e),
                    }
                )

        # Process node deletions
        if nodes_to_delete:
            for node_data in nodes_to_delete:
                try:
                    # Extract node ID from either ConceptNode object or dictionary
                    if isinstance(node_data, dict):
                        node_id = node_data.get("id")
                    else:
                        node_id = node_data.id if hasattr(node_data, "id") else None

                    if node_id:
                        # Delete the node and all its relationships
                        success = self._delete_node_by_id(node_id)
                        if success:
                            result["deleted_nodes"].append(node_id)
                        else:
                            result["errors"].append(
                                {"node_id": node_id, "error": "Failed to delete node"}
                            )
                    else:
                        result["errors"].append(
                            {
                                "node_id": "unknown",
                                "error": "Node ID not found for deletion",
                            }
                        )
                except Exception as e:
                    node_id = (
                        node_data.get("id")
                        if isinstance(node_data, dict)
                        else (node_data.id if hasattr(node_data, "id") else "unknown")
                    )
                    result["errors"].append(
                        {"node_id": node_id, "error": f"Error deleting node: {str(e)}"}
                    )

        return result

    def fetch_subgraph(self, node_ids: List[str], depth: int = 2) -> AgentWorkingGraph:
        """
        Fetch a relevant subgraph from Neo4j.

        Args:
            node_ids: List of node IDs to start from
            depth: How many hops to traverse

        Returns:
            An AgentWorkingGraph containing the subgraph
        """
        working_graph = AgentWorkingGraph()

        # Fetch relevant nodes and their relationships
        query = """
        MATCH path = (n:Concept)-[*0..{depth}]-(related:Concept)
        WHERE n.id IN $node_ids
        RETURN path
        """.format(
            depth=depth
        )

        try:
            with self.neo4j_client.driver.session() as session:
                result = session.run(query, {"node_ids": node_ids})

                # Process each path in the result
                for record in result:
                    path = record["path"]

                    # Process nodes in the path
                    for node in path.nodes:
                        node_data = self._parse_node_data(node)
                        concept_node = self._create_concept_node(node_data)

                        if concept_node and concept_node.id not in working_graph.nodes:
                            working_graph.add_node(concept_node)

                    # Process relationships in the path
                    for rel in path.relationships:
                        rel_id = rel.get("id")
                        if rel_id and rel_id not in working_graph.relationships:
                            relationship = Relationship(
                                id=rel_id,
                                source_node_id=rel.start_node.get("id"),
                                target_node_id=rel.end_node.get("id"),
                                type=rel.type,
                                discovery_count_search=rel.get(
                                    "discovery_count_search", 0
                                ),
                                discovery_count_llm_inference=rel.get(
                                    "discovery_count_llm_inference", 0
                                ),
                                source_urls=rel.get("source_urls", []),
                                type_confidence_llm=rel.get("type_confidence_llm", 0.0),
                                existence_confidence_llm=rel.get(
                                    "existence_confidence_llm", 0.0
                                ),
                                last_updated_timestamp=rel.get(
                                    "last_updated_timestamp", datetime.now()
                                ),
                            )
                            working_graph.add_relationship(relationship)
        except Exception:
            # Return empty graph on error
            pass

        return working_graph

    def _delete_node_by_id(self, node_id: str) -> bool:
        """
        Delete a node and all its relationships from Neo4j.

        Args:
            node_id: The ID of the node to delete

        Returns:
            True if successful, False otherwise
        """
        query = """
        MATCH (n:Concept {id: $node_id})
        DETACH DELETE n
        """

        try:
            with self.neo4j_client.driver.session() as session:
                result = session.run(query, {"node_id": node_id})
                # Check if any nodes were deleted
                summary = result.consume()
                return summary.counters.nodes_deleted > 0
        except Exception:
            return False

    def update_node_name_embedding(
        self, node_id: str, embedding_vector: List[float]
    ) -> bool:
        """
        Update the name embedding vector for a node in Neo4j.

        Args:
            node_id: The ID of the node to update
            embedding_vector: The name embedding vector to store

        Returns:
            True if successful, False otherwise
        """
        query = """
        MATCH (n:Concept {id: $node_id})
        SET n.name_embedding = $embedding_vector
        RETURN n
        """

        try:
            with self.neo4j_client.driver.session() as session:
                result = session.run(
                    query, {"node_id": node_id, "embedding_vector": embedding_vector}
                )
                return result.single() is not None
        except Exception:
            return False

    def update_node_topic_embedding(
        self, node_id: str, embedding_vector: List[float]
    ) -> bool:
        """
        Update the topic embedding vector for a node in Neo4j.

        Args:
            node_id: The ID of the node to update
            embedding_vector: The topic embedding vector to store

        Returns:
            True if successful, False otherwise
        """
        query = """
        MATCH (n:Concept {id: $node_id})
        SET n.topic_embedding = $embedding_vector
        RETURN n
        """

        try:
            with self.neo4j_client.driver.session() as session:
                result = session.run(
                    query, {"node_id": node_id, "embedding_vector": embedding_vector}
                )
                return result.single() is not None
        except Exception:
            return False

    def update_node_definition_embedding(
        self, node_id: str, embedding_vector: List[float]
    ) -> bool:
        """
        Update the definition embedding vector for a node in Neo4j.

        Args:
            node_id: The ID of the node to update
            embedding_vector: The definition embedding vector to store

        Returns:
            True if successful, False otherwise
        """
        query = """
        MATCH (n:Concept {id: $node_id})
        SET n.definition_embedding = $embedding_vector
        RETURN n
        """

        try:
            with self.neo4j_client.driver.session() as session:
                result = session.run(
                    query, {"node_id": node_id, "embedding_vector": embedding_vector}
                )
                return result.single() is not None
        except Exception:
            return False

    def vector_search_definition(
        self,
        definition_embedding: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.7,
    ) -> AgentWorkingGraph:
        """
        Fetch a relevant subgraph from Neo4j using vector similarity.

        Args:
            definition_embedding: Vector embedding to search with
            limit: Maximum number of similar nodes to return
            similarity_threshold: Threshold for vector similarity

        Returns:
            An AgentWorkingGraph containing the subgraph
        """
        working_graph = AgentWorkingGraph()

        # Fetch nodes by vector similarity
        query = """
        CALL db.index.vector.queryNodes('definition_embedding_index', $limit, $embedding)
        YIELD node, score
        WHERE node.node_type = 'concept' AND score >= $similarity_threshold
        RETURN node, score
        ORDER BY score DESC
        """

        try:
            with self.neo4j_client.driver.session() as session:
                result = session.run(
                    query,
                    {
                        "embedding": definition_embedding,
                        "similarity_threshold": similarity_threshold,
                        "limit": limit,
                    },
                )

                # Collect node IDs to find relationships
                node_ids = []

                # Process each node in the result
                for record in result:
                    node = record["node"]
                    node_data = self._parse_node_data(node)
                    concept_node = self._create_concept_node(node_data)

                    if concept_node:
                        node_ids.append(concept_node.id)
                        working_graph.add_node(concept_node)

                # If we found nodes, fetch relationships between them
                if node_ids:
                    rel_query = """
                    MATCH (n:Concept)-[r]->(m:Concept)
                    WHERE n.id IN $node_ids AND m.id IN $node_ids
                    RETURN r, n, m
                    """

                    rel_result = session.run(rel_query, {"node_ids": node_ids})

                    # Process each relationship
                    for record in rel_result:
                        rel = record["r"]
                        source_node = record["n"]
                        target_node = record["m"]

                        rel_id = rel.get("id")
                        if rel_id and rel_id not in working_graph.relationships:
                            relationship = Relationship(
                                id=rel_id,
                                source_node_id=source_node.get("id"),
                                target_node_id=target_node.get("id"),
                                type=rel.type,
                                discovery_count_search=rel.get(
                                    "discovery_count_search", 0
                                ),
                                discovery_count_llm_inference=rel.get(
                                    "discovery_count_llm_inference", 0
                                ),
                                source_urls=rel.get("source_urls", []),
                                type_confidence_llm=rel.get("type_confidence_llm", 0.0),
                                existence_confidence_llm=rel.get(
                                    "existence_confidence_llm", 0.0
                                ),
                                last_updated_timestamp=rel.get(
                                    "last_updated_timestamp", datetime.now()
                                ),
                            )
                            working_graph.add_relationship(relationship)
        except Exception:
            # Return empty graph on error
            pass

        return working_graph

    def vector_search_name(
        self,
        name_embedding: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.8,
        node_type: Optional[str] = None,
    ) -> List[ConceptNode]:
        """
        Find a goal node using vector similarity search.

        Args:
            name_embedding: Vector embedding of the name string
            similarity_threshold: Threshold for considering a match

        Returns:
            List of ConceptNode matches
        """
        query = (
            """
            CALL db.index.vector.queryNodes('name_embedding_index', $limit, $embedding)
            YIELD node, score
            WHERE node.node_type = $node_type AND score >= $similarity_threshold
            RETURN node, score
            ORDER BY score DESC
            LIMIT 1
            """
            if node_type
            else """CALL db.index.vector.queryNodes('name_embedding_index', $limit, $embedding)
            YIELD node, score
            WHERE score >= $similarity_threshold
            RETURN node, score
            ORDER BY score DESC
            LIMIT 1
            """
        )

        try:
            with self.neo4j_client.driver.session() as session:
                result = (
                    session.run(
                        query,
                        {
                            "embedding": name_embedding,
                            "similarity_threshold": similarity_threshold,
                            "limit": limit,
                            "node_type": node_type,
                        },
                    )
                    if node_type
                    else session.run(
                        query,
                        {
                            "embedding": name_embedding,
                            "similarity_threshold": similarity_threshold,
                            "limit": limit,
                        },
                    )
                )

                # Collect nodes
                nodes = []

                # Process each node in the result
                for record in result:
                    node = record["node"]
                    node_data = self._parse_node_data(node)
                    concept_node = self._create_concept_node(node_data)

                    if concept_node:
                        nodes.append(concept_node)

                return nodes
        except Exception:
            return []

    def vector_search_topic(
        self,
        topic_embedding: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.7,
    ) -> List[ConceptNode]:
        """
        Find concepts using topic embedding vector search.

        Args:
            topic_embedding: Vector embedding of the concept topic
            limit: Maximum number of similar nodes to return
            similarity_threshold: Threshold for vector similarity

        Returns:
            List of ConceptNode matches
        """
        query = """
        CALL db.index.vector.queryNodes('topic_embedding_index', $limit, $embedding)
        YIELD node, score
        WHERE node.node_type = 'concept' AND score >= $similarity_threshold
        RETURN node, score
        ORDER BY score DESC
        """

        concepts = []
        try:
            with self.neo4j_client.driver.session() as session:
                result = session.run(
                    query,
                    {
                        "embedding": topic_embedding,
                        "similarity_threshold": similarity_threshold,
                        "limit": limit,
                    },
                )

                for record in result:
                    node = record["node"]
                    node_data = self._parse_node_data(node)
                    concept = self._create_concept_node(node_data)

                    if concept:
                        concepts.append(concept)
        except Exception:
            # Return empty list on error
            pass

        return concepts
