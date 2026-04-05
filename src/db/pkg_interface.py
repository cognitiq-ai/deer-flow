"""
Neo4j Persistent Knowledge Graph Interface (PKGInterface).

This module provides the PKGInterface class that handles interactions with
the Neo4j knowledge graph following the agent framework specification.
"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from src.db.neo4j_client import Neo4jClient
from src.kg.agent_working_graph import AgentWorkingGraph
from src.kg.base_models import ConceptNode, Relationship, RelationshipType

# Type alias for concept data
ConceptData = Dict[str, Union[str, float, List[str], Dict[str, float]]]


class PrerequisiteCycleException(Exception):
    """Exception raised when a relationship would create a cycle."""

    def __init__(
        self,
        source_id: str,
        target_id: str,
        path: List[str],
        relationship_type: str = "HAS_PREREQUISITE",
    ):
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
        self.neo4j_client = neo4j_client or Neo4jClient()

    def _parse_attributes(self, attributes: Any) -> Dict[str, Any]:
        """Parse node attributes from Neo4j, handling JSON strings and empty values."""

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

    def _create_concept_node(self, node_data: Any) -> Optional[ConceptNode]:
        """Create a ConceptNode from parsed node data."""

        try:
            if not node_data or not node_data.get("id"):
                return None

            profile_data = self._parse_attributes(node_data.get("profile"))
            evaluation_data = self._parse_attributes(node_data.get("evaluation"))

            return ConceptNode(
                id=node_data["id"],
                name=node_data.get("name"),
                topic=node_data.get("topic", ""),
                profile=profile_data or None,
                evaluation=evaluation_data or None,
                node_type=node_data.get("node_type", "concept"),
                exists_in_pkg=True,
                session_disposition=None,
                updated_at=node_data.get("updated_at", datetime.fromtimestamp(0)),
                name_embedding=node_data.get("name_embedding", []),
                topic_embedding=node_data.get("topic_embedding", []),
                definition_embedding=node_data.get("definition_embedding", []),
            )
        except (ValueError, TypeError):
            return None

    def _parse_relationship_type(self, relationship_type: Any) -> RelationshipType:
        """Parse relationship type values from Neo4j/model payload."""
        if isinstance(relationship_type, RelationshipType):
            return relationship_type

        if isinstance(relationship_type, str):
            try:
                return RelationshipType[relationship_type]
            except KeyError:
                return RelationshipType(relationship_type)

        raise ValueError(f"Unsupported relationship type: {relationship_type}")

    def _create_relationship(self, rel_data: Any) -> Optional[Relationship]:
        """Create a Relationship from parsed relation data."""
        try:
            if not rel_data:
                return None

            if (
                not rel_data.get("id")
                or not rel_data.get("source_node_id")
                or not rel_data.get("target_node_id")
            ):
                return None

            profile_data = self._parse_attributes(rel_data.get("profile"))
            rel_type = self._parse_relationship_type(rel_data.get("type"))

            updated_at = rel_data.get("updated_at", datetime.now())
            if isinstance(updated_at, str):
                try:
                    updated_at = datetime.fromisoformat(updated_at)
                except ValueError:
                    updated_at = datetime.now()

            return Relationship(
                id=rel_data.get("id"),
                source_node_id=rel_data.get("source_node_id"),
                target_node_id=rel_data.get("target_node_id"),
                type=rel_type,
                discovery_count=rel_data.get("discovery_count", 0),
                profile=profile_data or None,
                updated_at=updated_at,
            )
        except (ValueError, TypeError):
            return None

    def _serialize_relationship_props(self, rel_data: Relationship) -> Dict[str, Any]:
        """Serialize relationship fields for Neo4j persistence."""
        rel_props = rel_data.model_dump()

        # Persist nested objects as JSON strings for stable Neo4j storage.
        rel_props["profile"] = json.dumps(rel_props.get("profile"))

        if isinstance(rel_props.get("updated_at"), datetime):
            rel_props["updated_at"] = rel_props["updated_at"].isoformat()

        return rel_props

    def _serialize_node_props(self, node_data: ConceptNode) -> Dict[str, Any]:
        """Serialize ConceptNode fields for Neo4j persistence."""
        node_props = node_data.model_dump()
        # Runtime-only field; never persist session disposition in canonical PKG.
        node_props.pop("session_disposition", None)

        if isinstance(node_props.get("updated_at"), datetime):
            node_props["updated_at"] = node_props["updated_at"].isoformat()

        # Persist nested objects as JSON strings for stable Neo4j storage.
        node_props["profile"] = json.dumps(node_props.get("profile"))
        node_props["evaluation"] = json.dumps(node_props.get("evaluation"))

        for embedding_field in [
            "name_embedding",
            "topic_embedding",
            "definition_embedding",
        ]:
            if embedding_field not in node_props or node_props[embedding_field] is None:
                node_props[embedding_field] = []

        node_props["exists_in_pkg"] = True
        return node_props

    def _update_node(self, node_data: ConceptNode) -> ConceptNode:
        """Update an existing node in Neo4j and return refreshed node."""
        query = """
        MATCH (n:Concept {id: $id})
        SET n += $node_props
        RETURN n
        """

        node_props = self._serialize_node_props(node_data)
        with self.neo4j_client.driver.session() as session:
            record = session.run(
                query,
                {"id": node_data.id, "node_props": node_props},
            ).single()

            if not record:
                raise Exception("Failed to update node")

            updated_node = self._create_concept_node(record["n"])
            if not updated_node:
                raise Exception("Failed to parse updated node")
            return updated_node

    def get_node_by_id(self, node_id: str) -> Optional[ConceptNode]:
        """Get a node from Neo4j by ID."""

        query = """
        MATCH (n:Concept {id: $id})
        RETURN n
        """

        try:
            with self.neo4j_client.driver.session() as session:
                record = session.run(query, {"id": node_id}).single()

                if not record:
                    return None

                return self._create_concept_node(record["n"])
        except Exception:
            return None

    def find_or_create_node(self, node_data: ConceptNode) -> ConceptNode:
        """Find or create a node in Neo4j (upsert semantics)."""

        # Check if node exists
        existing_node = self.get_node_by_id(node_data.id)

        if existing_node:
            # Node exists, update persisted fields (including profile/evaluation)
            return self._update_node(node_data)

        # Node doesn't exist, create it
        query = """
        CREATE (n:Concept $node_props)
        RETURN n
        """

        # Convert node data to properties dict
        node_props = self._serialize_node_props(node_data)

        with self.neo4j_client.driver.session() as session:
            record = session.run(query, {"node_props": node_props}).single()

            if not record:
                raise Exception("Failed to create node")

            created_node = self._create_concept_node(record["n"])
            if not created_node:
                raise Exception("Failed to parse created node")
            return created_node

    def detect_relationship_cycle(
        self, source_id: str, target_id: str, relationship_type: RelationshipType
    ) -> Tuple[bool, Optional[List[str]]]:
        """Check if creating a relationship from source to target would create a cycle for the given relationship type."""

        # Only check cycles for directional relationships that could form meaningful cycles
        cycle_prone_types = [
            RelationshipType.HAS_PREREQUISITE,
            RelationshipType.IS_TYPE_OF,
            RelationshipType.IS_PART_OF,
        ]

        if relationship_type not in cycle_prone_types:
            return False, None

        # Check if there's already a path from target back to source (which would create a cycle)
        rel_type_str = relationship_type.code
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

    def find_or_create_relationship(self, rel_data: Relationship) -> Relationship:
        """Find or create a relationship in Neo4j."""

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
                    rel_type_enum.code,
                )

        # Check if a relationship of the same type already exists between these nodes
        rel_type = (
            rel_data.type.code
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
                    existing_dict = dict(record["r"].items())
                    existing_dict["source_node_id"] = rel_data.source_node_id
                    existing_dict["target_node_id"] = rel_data.target_node_id
                    existing_dict["type"] = rel_type
                    existing_dict["updated_at"] = existing_dict.get(
                        "updated_at", datetime.now()
                    )

                    # Create existing relationship object
                    existing_rel = self._create_relationship(existing_dict)
                    if not existing_rel:
                        raise Exception("Failed to parse existing relationship")

                    # Merge new data into existing relationship
                    merged_rel = self._merge_relationships(existing_rel, rel_data)

                    # Update the relationship in Neo4j
                    self._update_relationship(merged_rel)

                    return merged_rel

                # Relationship doesn't exist, create it
                rel_props = self._serialize_relationship_props(rel_data)

                # Remove source_node_id, target_node_id, and type from properties
                rel_props.pop("source_node_id", None)
                rel_props.pop("target_node_id", None)
                rel_props.pop("type", None)

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
        """Merge a new relationship into an existing relationship, taking higher confidence values."""

        # Create a copy of the existing relationship to avoid modifying the original
        merged_rel = Relationship(
            id=existing_rel.id,
            source_node_id=existing_rel.source_node_id,
            target_node_id=existing_rel.target_node_id,
            type=existing_rel.type,
            discovery_count=existing_rel.discovery_count,
            profile=existing_rel.profile.model_copy() if existing_rel.profile else None,
        )

        # Use the relationship's built-in merge method
        merged_rel.merge_relationship(new_rel)

        return merged_rel

    def _update_relationship(self, relationship: Relationship) -> None:
        """Update an existing relationship in Neo4j with new properties."""

        rel_type = (
            relationship.type.code
            if isinstance(relationship.type, RelationshipType)
            else relationship.type
        )

        # Prepare properties for update
        rel_props = self._serialize_relationship_props(relationship)

        # Remove fields that shouldn't be updated
        rel_props.pop("id", None)
        rel_props.pop("source_node_id", None)
        rel_props.pop("target_node_id", None)
        rel_props.pop("type", None)

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
        upsert_nodes: List[Union[ConceptNode, Dict[str, Any]]],
        upsert_edges: List[Union[Relationship, Dict[str, Any]]],
        delete_nodes: Optional[List[Union[ConceptNode, Dict[str, Any]]]] = None,
    ) -> Dict[str, List[Union[str, Dict[str, str]]]]:
        """Commit changes to Neo4j."""

        result = {
            "committed_nodes": [],
            "committed_edges": [],
            "rejected_edges": [],
            "deleted_nodes": [],
            "errors": [],
        }

        # Process nodes first
        for node_data in upsert_nodes:
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
        for edge_data in upsert_edges:
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
                rel_type_str = rel_type_enum.code

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
                        edge_data.type.code
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
                        edge_data.type.code
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
        if delete_nodes:
            for node_data in delete_nodes:
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

    def fetch_subgraph(
        self,
        node_ids: List[str],
        depth: int = 2,
        node_type_filter: Optional[Union[str, List[str]]] = None,
        edge_type_filter: Optional[List[str]] = None,
    ) -> AgentWorkingGraph:
        """Fetch a relevant subgraph from Neo4j."""

        working_graph = AgentWorkingGraph()

        normalized_node_types = None
        if node_type_filter:
            if isinstance(node_type_filter, str):
                node_type_values = [node_type_filter]
            else:
                node_type_values = list(node_type_filter)

            normalized_node_types = [
                str(node_type).strip().lower()
                for node_type in node_type_values
                if isinstance(node_type, str) and str(node_type).strip()
            ]

        normalized_edge_types = None
        if edge_type_filter:
            if isinstance(edge_type_filter, str):
                edge_type_values = [edge_type_filter]
            else:
                edge_type_values = list(edge_type_filter)

            normalized_edge_types = []
            for edge_type in edge_type_values:
                if isinstance(edge_type, RelationshipType):
                    normalized_edge_types.append(edge_type.code.lower())
                elif isinstance(edge_type, str) and edge_type.strip():
                    normalized_edge_types.append(edge_type.strip().lower())

            if not normalized_edge_types:
                normalized_edge_types = None

        # Fetch relevant nodes and their relationships
        query = """
        MATCH path = (n:Concept)-[*0..{depth}]-(related:Concept)
        WHERE n.id IN $node_ids
        """.format(depth=depth)

        params = {"node_ids": node_ids}

        if normalized_node_types:
            query += (
                " AND all(node_in_path IN nodes(path) "
                "WHERE toLower(node_in_path.node_type) IN $node_type_filters)"
            )
            params["node_type_filters"] = normalized_node_types

        if normalized_edge_types:
            query += (
                " AND all(rel_in_path IN relationships(path) "
                "WHERE toLower(type(rel_in_path)) IN $edge_type_filters)"
            )
            params["edge_type_filters"] = normalized_edge_types

        query += " RETURN path"

        try:
            with self.neo4j_client.driver.session() as session:
                result = session.run(query, params)

                # Process each path in the result
                for record in result:
                    path = record["path"]

                    # Process nodes in the path
                    for node in path.nodes:
                        concept_node = self._create_concept_node(node)

                        if concept_node and concept_node.id not in working_graph.nodes:
                            working_graph.add_node(concept_node)

                    # Process relationships in the path
                    for rel in path.relationships:
                        rel_id = rel.get("id")
                        if rel_id and rel_id not in working_graph.relationships:
                            relationship = self._create_relationship(
                                {
                                    "id": rel_id,
                                    "source_node_id": rel.start_node.get("id"),
                                    "target_node_id": rel.end_node.get("id"),
                                    "type": rel.type,
                                    "discovery_count": rel.get("discovery_count", 0),
                                    "profile": rel.get("profile"),
                                    "updated_at": rel.get("updated_at", datetime.now()),
                                }
                            )
                            if not relationship:
                                continue
                            working_graph.add_relationship(relationship)
        except Exception:
            # Return empty graph on error
            pass

        return working_graph

    def _delete_node_by_id(self, node_id: str) -> bool:
        """Delete a node and all its relationships from Neo4j."""

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

    def _vector_search(
        self,
        embedding: List[float],
        embedding_type: str,
        limit: int = 10,
        similarity_threshold: float = 0.7,
        node_type_filter: Optional[Union[str, List[str]]] = None,
        edge_type_filter: Optional[List[str]] = None,
        return_graph: bool = False,
        max_results: Optional[int] = None,
    ) -> Union[List[ConceptNode], AgentWorkingGraph]:
        """
        Generic helper function for vector similarity search.

        Args:
            embedding: The embedding vector to search with
            embedding_type: The type of embedding ('name', 'topic', or 'definition')
            limit: Maximum number of nodes to fetch from vector index
            similarity_threshold: Minimum similarity score required
            node_type_filter: Optional filter for node type (e.g., 'concept')
            edge_type_filter: Optional list of relationship types to include in returned graph
            return_graph: If True, return AgentWorkingGraph with relationships; if False, return List[ConceptNode]
            max_results: Optional hard limit on number of results returned (applied after similarity filtering)

        Returns:
            Either List[ConceptNode] or AgentWorkingGraph depending on return_graph parameter
        """

        # Map embedding type to index name
        index_map = {
            "name": "name_embedding_index",
            "topic": "topic_embedding_index",
            "definition": "definition_embedding_index",
        }

        if embedding_type not in index_map:
            raise ValueError(
                f"Invalid embedding_type: {embedding_type}. Must be one of {list(index_map.keys())}"
            )

        index_name = index_map[embedding_type]

        normalized_node_types = None
        if node_type_filter:
            if isinstance(node_type_filter, str):
                node_type_values = [node_type_filter]
            else:
                node_type_values = list(node_type_filter)

            normalized_node_types = [
                str(node_type).strip().lower()
                for node_type in node_type_values
                if isinstance(node_type, str) and str(node_type).strip()
            ]

        normalized_edge_types = None
        if edge_type_filter:
            if isinstance(edge_type_filter, str):
                edge_type_values = [edge_type_filter]
            else:
                edge_type_values = list(edge_type_filter)

            normalized_edge_types = []
            for edge_type in edge_type_values:
                if isinstance(edge_type, RelationshipType):
                    normalized_edge_types.append(edge_type.code.lower())
                elif isinstance(edge_type, str) and edge_type.strip():
                    normalized_edge_types.append(edge_type.strip().lower())

            if not normalized_edge_types:
                normalized_edge_types = None

        # Build query with optional node type filter
        base_query = f"""
        CALL db.index.vector.queryNodes('{index_name}', $limit, $embedding)
        YIELD node, score
        WHERE score >= $similarity_threshold
        """
        params = {
            "embedding": embedding,
            "similarity_threshold": similarity_threshold,
            "limit": limit,
        }

        if normalized_node_types:
            base_query += " AND toLower(node.node_type) IN $node_type_filters"
            params["node_type_filters"] = normalized_node_types

        base_query += """
        RETURN node, score
        ORDER BY score DESC
        """

        if max_results:
            base_query += f" LIMIT {max_results}"

        concepts = []
        try:
            with self.neo4j_client.driver.session() as session:
                result = session.run(base_query, params)

                # Process each node in the result
                for record in result:
                    node = record["node"]
                    concept_node = self._create_concept_node(node)

                    if concept_node:
                        concepts.append(concept_node)

                # If return_graph is True, build AgentWorkingGraph with relationships
                if return_graph:
                    working_graph = AgentWorkingGraph()

                    if not concepts:
                        return working_graph

                    # Add all found nodes to the graph
                    for concept in concepts:
                        working_graph.add_node(concept)

                    # Fetch relationships between the found nodes
                    node_ids = [c.id for c in concepts]
                    rel_query = """
                    MATCH (n:Concept)-[r]->(m:Concept)
                    WHERE n.id IN $node_ids AND m.id IN $node_ids
                    """

                    rel_params = {"node_ids": node_ids}

                    if normalized_edge_types:
                        rel_query += " AND toLower(type(r)) IN $edge_type_filters"
                        rel_params["edge_type_filters"] = normalized_edge_types

                    rel_query += """
                    RETURN r, n, m
                    """

                    rel_result = session.run(rel_query, rel_params)

                    # Process each relationship
                    for record in rel_result:
                        rel = record["r"]
                        source_node = record["n"]
                        target_node = record["m"]

                        rel_id = rel.get("id")
                        if rel_id and rel_id not in working_graph.relationships:
                            relationship = self._create_relationship(
                                {
                                    "id": rel_id,
                                    "source_node_id": source_node.get("id"),
                                    "target_node_id": target_node.get("id"),
                                    "type": rel.type,
                                    "discovery_count": rel.get("discovery_count", 0),
                                    "profile": rel.get("profile"),
                                    "updated_at": rel.get("updated_at", datetime.now()),
                                }
                            )
                            if not relationship:
                                continue
                            working_graph.add_relationship(relationship)

                    return working_graph

                # Return just the list of concepts
                return concepts

        except Exception:
            # Return empty result on error
            return [] if not return_graph else AgentWorkingGraph()

    def vector_search_definition(
        self,
        definition_embedding: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.7,
        node_type_filter: Optional[Union[str, List[str]]] = None,
        edge_type_filter: Optional[List[str]] = None,
    ) -> AgentWorkingGraph:
        """Fetch a relevant subgraph from Neo4j using vector similarity."""
        if node_type_filter is None:
            node_type_filter = ["concept"]

        return self._vector_search(
            embedding=definition_embedding,
            embedding_type="definition",
            limit=limit,
            similarity_threshold=similarity_threshold,
            node_type_filter=node_type_filter,
            edge_type_filter=edge_type_filter,
            return_graph=True,
        )

    def vector_search_definition_nodes(
        self,
        definition_embedding: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.7,
        node_type_filter: Optional[Union[str, List[str]]] = None,
    ) -> List[ConceptNode]:
        """Find concept nodes using definition embedding vector similarity."""
        if node_type_filter is None:
            node_type_filter = ["concept"]

        return self._vector_search(
            embedding=definition_embedding,
            embedding_type="definition",
            limit=limit,
            similarity_threshold=similarity_threshold,
            node_type_filter=node_type_filter,
            return_graph=False,
        )

    def find_nodes_by_exact_name(
        self,
        name: str,
        limit: int = 5,
        node_type_filter: Optional[Union[str, List[str]]] = None,
    ) -> List[ConceptNode]:
        """Find concept nodes with exact normalized name match."""
        if not isinstance(name, str) or not name.strip():
            return []

        normalized_name = " ".join(name.strip().lower().split())
        if not normalized_name:
            return []

        normalized_node_types = None
        if node_type_filter:
            if isinstance(node_type_filter, str):
                node_type_values = [node_type_filter]
            else:
                node_type_values = list(node_type_filter)

            normalized_node_types = [
                str(node_type).strip().lower()
                for node_type in node_type_values
                if isinstance(node_type, str) and str(node_type).strip()
            ]

            if not normalized_node_types:
                normalized_node_types = None

        query = """
        MATCH (n:Concept)
        WHERE toLower(trim(replace(n.name, '  ', ' '))) = $normalized_name
        """
        params: Dict[str, Any] = {"normalized_name": normalized_name, "limit": limit}

        if normalized_node_types:
            query += " AND toLower(n.node_type) IN $node_type_filters"
            params["node_type_filters"] = normalized_node_types

        query += """
        RETURN n
        ORDER BY n.updated_at DESC
        LIMIT $limit
        """

        concepts: List[ConceptNode] = []
        try:
            with self.neo4j_client.driver.session() as session:
                result = session.run(query, params)
                for record in result:
                    node = self._create_concept_node(record["n"])
                    if node:
                        concepts.append(node)
        except Exception:
            return []

        return concepts

    def vector_search_name(
        self,
        name_embedding: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.8,
        node_type: Optional[str] = None,
    ) -> List[ConceptNode]:
        """Find a goal node using vector similarity search."""
        return self._vector_search(
            embedding=name_embedding,
            embedding_type="name",
            limit=limit,
            similarity_threshold=similarity_threshold,
            node_type_filter=node_type,
            return_graph=False,
            max_results=1,
        )

    def vector_search_topic(
        self,
        topic_embedding: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.7,
    ) -> List[ConceptNode]:
        """Find concepts using topic embedding vector search."""
        return self._vector_search(
            embedding=topic_embedding,
            embedding_type="topic",
            limit=limit,
            similarity_threshold=similarity_threshold,
            node_type_filter="concept",
            return_graph=False,
        )
