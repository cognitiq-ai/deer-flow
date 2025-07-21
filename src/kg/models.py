"""Pydantic models for knowledge graph entities."""

import hashlib
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import networkx as nx
import yaml
from pydantic import BaseModel, Field

try:
    import gravis as gv

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


class ConceptNodeStatus(str, Enum):
    """Status of a concept node in the knowledge graph."""

    STUB = "stub"  # Initial placeholder with minimal information
    DEFINED_LOW_CONFIDENCE = "defined_low_confidence"  # Researched but low confidence
    DEFINED_HIGH_CONFIDENCE = (
        "defined_high_confidence"  # Well-researched with high confidence
    )
    CANONICAL = "canonical"  # Verified and canonical definition
    DEPRECATED = "deprecated"  # No longer valid or relevant


class RelationshipType(str, Enum):
    """Type of relationship between concept nodes."""

    HAS_PREREQUISITE = "HAS_PREREQUISITE"  # Target is a prerequisite for source
    FULFILS_GOAL = "FULFILS_GOAL"  # Source fulfils target goal
    IS_TYPE_OF = "IS_TYPE_OF"  # Source is a type of target (hierarchical relationship)
    IS_PART_OF = "IS_PART_OF"  # Source is part of target (compositional relationship)
    IS_DUPLICATE_OF = "IS_DUPLICATE_OF"  # Source is a duplicate of target
    NO_RELATIONSHIP = "NO_RELATIONSHIP"  # No relationship exists between the concepts

    def __str__(self) -> str:
        return self.value

    @property
    def description(self) -> str:
        """Return the description of the relationship type."""
        if self == RelationshipType.HAS_PREREQUISITE:
            return "A prerequiste relationship exists between the concepts"
        elif self == RelationshipType.FULFILS_GOAL:
            return "A goal fulfillment relationship exists between the concepts"
        elif self == RelationshipType.IS_TYPE_OF:
            return (
                "A type/subtype (hierarchical) relationship exists between the concepts"
            )
        elif self == RelationshipType.IS_PART_OF:
            return "A part/component (compositional) relationship exists between the concepts"
        elif self == RelationshipType.IS_DUPLICATE_OF:
            return "A duplicate relationship exists between the concepts"
        else:
            return "None of the above relationships exist between these two concepts"


class ResearchSource(BaseModel):
    """
    Represents the sources of the research process.
    """

    url: str
    title: Optional[str] = None
    snippet: Optional[str] = None
    content: Optional[str] = None

    def __hash__(self) -> int:
        return int(self.id, 16)

    @property
    def id(self) -> str:
        """
        Return the ID of the source.
        """
        return hashlib.sha256(self.url.encode()).hexdigest()

    def merge(self, other: "ResearchSource") -> None:
        """
        Merge another source into this source.
        """
        if self.url == other.url:
            self.title = self.title or other.title
            self.snippet = self.snippet or other.snippet
            self.content = self.content or other.content


class ResearchQA(BaseModel):
    """
    (Question, Answer) tuple research output.
    """

    query: str
    result_summary: str
    confidence: Optional[float] = None


class ResearchOutput(BaseModel):
    """
    (Question, Summary) tuple research output with sources.
    Summary with inline citations [1], [2], ..
    corresponding to the source list index.
    Sources are the sources of the summary.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query_result_summary: ResearchQA
    sources: List[ResearchSource]

    def __hash__(self) -> int:
        return self.id.__hash__()

    def to_yaml(self) -> str:
        """
        Return the YAML representation of the research output.
        """
        return yaml.dump(self.model_dump(), sort_keys=False)


class ConsolidatedResearchOutput(BaseModel):
    """
    Represents the consolidated research output.
    """

    queries_answers: List[ResearchQA] = Field(default_factory=list)
    sources: List[ResearchSource] = Field(default_factory=list)

    def merge(self, other: ResearchOutput) -> None:
        """
        Merge a research output into this one.
        Remaps other's answer citations into its sources.
        """
        self.queries_answers.append(other.query_result_summary)
        self.sources = list(set(self.sources).union(set(other.sources)))

    def update_sources(self, source: ResearchSource) -> None:
        """
        Update the sources of the research output.
        """
        for s in self.sources:
            if s.url == source.url:
                s.merge(source)
                return

        self.sources.append(source)


class ConceptNode(BaseModel):
    """
    Represents a concept node in the knowledge graph.

    A concept can be a skill, topic, idea, or any other entity that can be learned.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str  # web scraper
    topic: str = ""  # Python
    definition: Optional[str] = None
    node_type: str = "concept"  # Default type
    exists_in_pkg: bool = False

    # Discovery metadata
    definition_research: List[ResearchOutput] = Field(default_factory=list)
    prerequisites_research: List[ResearchOutput] = Field(default_factory=list)

    # Confidence scores
    definition_confidence_llm: float = 0.0

    # Embedding vectors for vector search
    name_embedding: Optional[List[float]] = None
    topic_embedding: Optional[List[float]] = None
    definition_embedding: Optional[List[float]] = None

    # Status and timestamps
    last_updated_timestamp: datetime = Field(default_factory=datetime.now)
    defined_status: Optional[ConceptNodeStatus] = None

    def __hash__(self) -> int:
        return self.id.__hash__()

    def get_status(self) -> ConceptNodeStatus:
        """Determine concept status based on definition and confidence."""
        if not self.definition or self.confidence < 0.3:
            return ConceptNodeStatus.STUB
        elif self.confidence < 0.7:
            return ConceptNodeStatus.DEFINED_LOW_CONFIDENCE
        else:
            return ConceptNodeStatus.DEFINED_HIGH_CONFIDENCE

    @property
    def status(self) -> ConceptNodeStatus:
        """
        Return the status of the concept.
        """
        return self.defined_status or self.get_status()

    @property
    def confidence(self) -> float:
        """
        Return the confidence of the concept.
        """
        return self.definition_confidence_llm

    @property
    def definition_research_yaml(self) -> str:
        """
        Return the YAML representation of the definition research.
        """
        return yaml.dump(self.definition_research, sort_keys=False)

    def merge_node(self, node: "ConceptNode") -> None:
        """
        Merge this node into incoming node.
        Updating with higher confidence values
        This node's `self.id` is retained.

        Args:
            node: Another node to merge into this one
        """

        # Merge information (taking higher confidence values, merging lists)
        if node.confidence > self.confidence:
            self.definition = node.definition
            self.definition_confidence_llm = node.definition_confidence_llm
            self.definition_embedding = node.definition_embedding

        # Merge research results (ensure uniqueness)
        existing_results = set(self.definition_research)
        new_results = set(node.definition_research)
        self.definition_research = list(existing_results.union(new_results))

        # Update status if the new node has a better status
        if (
            node.status != ConceptNodeStatus.STUB
            and self.status == ConceptNodeStatus.STUB
        ):
            self.defined_status = node.status

        # Update timestamp
        self.last_updated_timestamp = max(
            self.last_updated_timestamp, node.last_updated_timestamp
        )


class Relationship(BaseModel):
    """
    Represents a relationship between concept nodes in the knowledge graph.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_node_id: str
    target_node_id: str
    type: RelationshipType

    # Discovery metadata
    research_results: List[ResearchOutput] = Field(default_factory=list)
    discovery_count_search: int = 0
    discovery_count_llm_inference: int = 0
    source_urls: List[str] = Field(default_factory=list)

    # Confidence scores
    type_confidence_llm: float = 0.0
    existence_confidence_llm: float = 0.0

    # Timestamps
    last_updated_timestamp: datetime = Field(default_factory=datetime.now)

    def __hash__(self) -> int:
        return self.id.__hash__()

    @property
    def confidence(self) -> float:
        """
        Return the confidence of the relationship.
        """
        return self.existence_confidence_llm

    def merge_relationship(self, other: "Relationship") -> None:
        """
        Merge this relationship into incoming relationship.
        Updating with higher confidence values.
        This relationship's `self.id` is retained.

        Args:
            other: Another relationship to merge into this one
        """
        # Only merge if they're the same type and between the same nodes
        if (
            self.source_node_id != other.source_node_id
            or self.target_node_id != other.target_node_id
            or self.type != other.type
        ):
            return

        # Sum discovery counts
        self.discovery_count_search += other.discovery_count_search
        self.discovery_count_llm_inference += other.discovery_count_llm_inference

        # Merge source URLs (ensure uniqueness)
        existing_urls = set(self.source_urls)
        new_urls = set(other.source_urls)
        self.source_urls = list(existing_urls.union(new_urls))

        # Take higher confidence values
        self.type_confidence_llm = max(
            self.type_confidence_llm, other.type_confidence_llm
        )
        self.existence_confidence_llm = max(
            self.existence_confidence_llm, other.existence_confidence_llm
        )

        # Update timestamp to the most recent
        self.last_updated_timestamp = max(
            self.last_updated_timestamp, other.last_updated_timestamp
        )


class AgentWorkingGraph(BaseModel):
    """
    Represents the working graph that the agent is currently using.

    This is a temporary structure used during the agent's operation.
    """

    nodes: Dict[str, ConceptNode] = Field(default_factory=dict)
    relationships: Dict[str, Relationship] = Field(default_factory=dict)

    def add_node(self, node: ConceptNode) -> None:
        """Add a node to the working graph."""
        if node.id is not None:
            self.nodes[node.id] = node

    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship to the working graph."""
        if relationship.id is not None:
            self.relationships[relationship.id] = relationship

    def get_node(self, node_id: str) -> Optional[ConceptNode]:
        """Get a node from the working graph by ID."""
        return self.nodes.get(node_id)

    def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        """Get a relationship from the working graph by ID."""
        return self.relationships.get(relationship_id)

    def get_relationships_by_source(self, source_node_id: str) -> List[Relationship]:
        """Get all relationships where the given node is the source."""
        return [
            r for r in self.relationships.values() if r.source_node_id == source_node_id
        ]

    def get_relationships_by_target(self, target_node_id: str) -> List[Relationship]:
        """Get all relationships where the given node is the target."""
        return [
            r for r in self.relationships.values() if r.target_node_id == target_node_id
        ]

    def delete_node(self, node_id: str) -> None:
        """Delete a node from the working graph."""
        if node_id in self.nodes:
            del self.nodes[node_id]
        # Also delete all relationships with node_id as source or target
        for rel in self.relationships.values():
            if rel.source_node_id == node_id or rel.target_node_id == node_id:
                del self.relationships[rel.id]

    def delete_relationship(self, relationship_id: str) -> None:
        """Delete a relationship from the working graph."""
        if relationship_id in self.relationships:
            del self.relationships[relationship_id]

    def to_networkx_graph(self) -> nx.DiGraph:
        """Convert the working graph to a NetworkX directed graph."""
        G = nx.DiGraph()

        # Add nodes with their data
        for node_id, node in self.nodes.items():
            G.add_node(
                node_id,
                type=node.node_type,
                status=node.status,
                weight=node.confidence,
            )

        # Add edges with their data
        for rel_id, relationship in self.relationships.items():
            G.add_edge(
                relationship.source_node_id,
                relationship.target_node_id,
                id=relationship.id,
                type=relationship.type,
                weight=relationship.confidence,
            )

        return G

    def create_visualization(self, title: str = "Knowledge Graph Visualization") -> Any:
        """
        Create an interactive Gravis visualization of the working graph.

        Args:
            title: Title for the visualization

        Returns:
            Gravis Figure object for interactive visualization
        """
        if not VISUALIZATION_AVAILABLE:
            print("Gravis not available for visualization")
            return None

        if not self.nodes:
            print("No nodes to display")
            return None

        # Create a clean NetworkX graph for Gravis
        G = self.to_networkx_graph()

        # Define node shape mappings
        def get_node_shape(node: ConceptNode) -> str:
            if node.node_type == "goal":
                return "hexagon"
            elif node.exists_in_pkg:
                return "circle"
            elif node.status == ConceptNodeStatus.DEFINED_LOW_CONFIDENCE:
                return "circle"
            elif node.status == ConceptNodeStatus.DEFINED_HIGH_CONFIDENCE:
                return "circle"
            else:
                return "rectangle"

        # Define edge colors mapping
        def get_edge_color(edge: Relationship) -> str:
            if edge.type == RelationshipType.HAS_PREREQUISITE:  # Red
                return "#e74c3c"
            elif edge.type == RelationshipType.FULFILS_GOAL:  # Purple
                return "#9b59b6"
            elif edge.type == RelationshipType.IS_TYPE_OF:  # Blue
                return "#3498db"
            elif edge.type == RelationshipType.IS_PART_OF:  # Green
                return "#27ae60"
            elif edge.type == RelationshipType.IS_DUPLICATE_OF:  # Orange
                return "#f39c12"
            else:
                return "#000000"

        # Define community colors mapping
        community_colors = [
            "red",
            "blue",
            "green",
            "orange",
            "pink",
            "brown",
            "yellow",
            "cyan",
            "magenta",
            "violet",
        ]

        # Set graph-level properties
        G.graph.update(
            {
                "node_opacity": 0.7,
                "edge_opacity": 0.8,
                "node_label_size": 8,
                "edge_label_size": 8,
            }
        )

        # Centrality measures
        node_centralities = nx.algorithms.centrality.degree_centrality(G)

        # Community detection
        communities = nx.algorithms.community.greedy_modularity_communities(G)

        # Node properties
        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            concept_node = self.get_node(node_id)

            # Goal node
            if concept_node.node_type == "goal":
                node_data["color"] = "#27ae60"
                node_data["size"] = 30
                node_data["shape"] = "hexagon"

            else:
                # Opacity by status
                if concept_node.status == ConceptNodeStatus.STUB:
                    node_data["opacity"] = 0.1
                else:
                    node_data["opacity"] = 0.7

                # Size by centrality
                node_data["size"] = 10 + node_centralities[node_id] * 20

                # Color by community
                for community_counter, community_members in enumerate(communities):
                    if node_id in community_members:
                        break
                node_data["color"] = community_colors[
                    community_counter % len(community_colors)
                ]

                # Shape by status
                node_data["shape"] = get_node_shape(concept_node)

                # Label and hover text
                node_data["label"] = (
                    concept_node.name[:20] + "..."
                    if len(concept_node.name) > 20
                    else concept_node.name
                )

                # Create detailed hover text
                status_name = concept_node.status.value
                definition_preview = ""
                if concept_node.definition:
                    definition_preview = concept_node.definition

                hover_lines = [
                    f"<b>{concept_node.name}</b>",
                    f"Topic: {concept_node.topic}" if concept_node.topic else "",
                    f"Status: {status_name}",
                    f"Confidence: {concept_node.confidence:.2f}",
                    f"Type: {concept_node.node_type.title()}",
                ]

                if concept_node.exists_in_pkg:
                    hover_lines.append("✓ Exists in PKG")

                if definition_preview:
                    hover_lines.append(f"Definition: {definition_preview}")

                node_data["hover"] = "<br>".join(filter(None, hover_lines))

        # Edge properties
        for source, target, edge_data in G.edges(data=True):
            rel_id = edge_data.get("id")
            relationship = self.get_relationship(rel_id)

            # Color by relationship type
            edge_data["color"] = get_edge_color(relationship)

            # Label for relationship type
            edge_data["label"] = relationship.type.value

            # Hover text for edge
            source_node = self.get_node(source)
            target_node = self.get_node(target)

            hover_lines = [
                f"<b>{relationship.type.value}</b>",
                f"{source_node.name if source_node else 'Unknown'} → {target_node.name if target_node else 'Unknown'}",
                f"Confidence: {relationship.confidence:.2f}",
            ]

            if relationship.source_urls:
                hover_lines.append(
                    f"Sources: {len(relationship.source_urls)} references"
                )

            edge_data["hover"] = "<br>".join(hover_lines)

        # Create the visualization
        fig = gv.d3(
            G,
            # Graph layout and display
            graph_height=600,
            zoom_factor=2.0,
            layout_algorithm_active=True,
            # Node settings
            show_node=True,
            node_size_data_source="size",
            node_drag_fix=True,
            node_hover_neighborhood=True,
            node_hover_tooltip=True,
            use_node_size_normalization=True,
            node_size_normalization_min=10,
            node_size_normalization_max=30,
            # Node labels
            show_node_label=True,
            node_label_data_source="label",
            node_label_size_factor=0.5,
            show_node_label_border=True,
            # Edge settings
            show_edge=True,
            edge_curvature=0.3,
            edge_hover_tooltip=True,
            use_edge_size_normalization=True,
            # Edge labels
            show_edge_label=False,
            # Interactive features
            show_menu=True,
            show_details=False,
            # Force simulation settings for better layout
            use_many_body_force=True,
            many_body_force_strength=-100,
            use_links_force=True,
            links_force_distance=80,
            links_force_strength=0.7,
            use_collision_force=True,
            collision_force_radius=25,
            use_centering_force=True,
        )

        return fig

    def show_interactive_graph(
        self, title: str = "Interactive Knowledge Graph"
    ) -> None:
        """Show the interactive Gravis graph in the browser."""
        if not VISUALIZATION_AVAILABLE:
            print("Gravis not available for interactive visualization")
            return

        fig = self.create_visualization(title)
        if fig:
            fig.display()
            print("🌐 Interactive graph opened in browser")

    def save_interactive_graph(
        self,
        filename: str = "kg_visualization.html",
        title: str = "Knowledge Graph Visualization",
    ) -> None:
        """Save the interactive Gravis graph to an HTML file."""
        if not VISUALIZATION_AVAILABLE:
            print("Gravis not available for saving interactive visualization")
            return

        fig = self.create_visualization(title)
        if fig:
            fig.export_html(filename)
            print(f"💾 Interactive graph saved to {filename}")

    def find_prerequisites_path(self, goal_node_id: str) -> List[str]:
        """
        Find all nodes that are prerequisites for the goal node.

        Args:
            goal_node_id: ID of the goal node

        Returns:
            List of node IDs that are prerequisites (in topological order if possible)
        """
        G = self.to_networkx_graph()

        if goal_node_id not in G:
            return []

        # Get all nodes reachable from goal via HAS_PREREQUISITE relationships
        prerequisite_nodes = []

        # DFS to find all prerequisites
        def dfs_prerequisites(node_id: str, visited: set) -> None:
            if node_id in visited:
                return
            visited.add(node_id)

            # Look for outgoing HAS_PREREQUISITE edges
            for _, target_id, edge_data in G.edges(node_id, data=True):
                if edge_data["type"] == RelationshipType.HAS_PREREQUISITE:
                    prerequisite_nodes.append(target_id)
                    dfs_prerequisites(target_id, visited)

        dfs_prerequisites(goal_node_id, set())
        return prerequisite_nodes

    def find_unresolved_stubs(
        self, confidence_threshold: float = 0.7
    ) -> List[ConceptNode]:
        """
        Find all nodes that are stubs or have low confidence.

        Args:
            confidence_threshold: Minimum confidence threshold for considering a node resolved

        Returns:
            List of ConceptNode objects that need further research
        """
        unresolved = []

        for node in self.nodes.values():
            if (
                node.status == ConceptNodeStatus.STUB
                or node.confidence < confidence_threshold
                or not node.definition
            ):
                unresolved.append(node)

        return unresolved

    def merge_node(self, node: ConceptNode) -> None:
        """
        Merge a node into the working graph.
        Updating with higher confidence values.

        Args:
            node: ConceptNode to merge
        """
        if node.id is None:
            return

        existing_node = self.nodes.get(node.id)
        if existing_node:
            existing_node.merge_node(node)
            self.nodes[node.id] = existing_node
        else:
            # Add new node
            self.nodes[node.id] = node.model_copy()

    def merge_relationship(self, relationship: Relationship) -> None:
        """
        Merge a relationship into the working graph.
        Updating with higher confidence values.

        Args:
            relationship: Relationship to merge
        """
        if relationship.id is None:
            return

        existing_rel = self.relationships.get(relationship.id)
        if existing_rel:
            existing_rel.merge_relationship(relationship)
            self.relationships[relationship.id] = existing_rel
        else:
            # Add new relationship
            self.relationships[relationship.id] = relationship

    def deep_copy(self) -> "AgentWorkingGraph":
        """Create a deep copy of the working graph."""
        # Create new instance with copied data
        new_graph = AgentWorkingGraph()

        # Deep copy nodes
        for node_id, node in self.nodes.items():
            new_graph.nodes[node_id] = ConceptNode(**node.model_dump())

        # Deep copy relationships
        for rel_id, relationship in self.relationships.items():
            new_graph.relationships[rel_id] = Relationship(**relationship.model_dump())

        return new_graph

    def merge_awg(self, other_awg: "AgentWorkingGraph") -> None:
        """
        Merge another AgentWorkingGraph into this one.

        Args:
            other_awg: Another AgentWorkingGraph to merge
        """
        # Merge all nodes from other AWG
        for node_id, node in other_awg.nodes.items():
            self.merge_node(node)

        # Merge all relationships from other AWG
        for rel_id, relationship in other_awg.relationships.items():
            self.merge_relationship(relationship)

    def merge_concepts(self, concept1_id: str, concept2_id: str) -> ConceptNode:
        """
        Merge two concepts in the AWG.
        This concept's `self.id` is retained.

        Args:
            concept1_id: The ID of the first concept
            concept2_id: The ID of the second concept

        Returns:
            The merged concept node
        """

        concept1, concept2 = self.get_node(concept1_id), self.get_node(concept2_id)
        if concept1 is None or concept2 is None:
            return None

        # Merge concept2 into concept1
        concept1.merge_node(concept2)

        # Update the node in the AWG
        self.nodes[concept1.id] = concept1

        # Update relationships of concept2 to concept1
        for _, rel in self.relationships.items():
            if rel.source_node_id == concept2.id:
                rel.source_node_id = concept1.id
            elif rel.target_node_id == concept2.id:
                rel.target_node_id = concept1.id

        # Remove concept2 if it exists
        if concept2.id in self.nodes:
            del self.nodes[concept2.id]

        return concept1

    def is_cyclic_relationship(
        self, new_relationship: Relationship, relationship_type: RelationshipType
    ) -> bool:
        """
        Check if adding a new relationship would create a cycle for a specific relationship type.

        Args:
            new_relationship: The relationship to check
            relationship_type: The type of relationship to check for cycles

        Returns:
            True if adding the relationship would create a cycle
        """
        # Create a temporary graph with only the specified relationship type
        G = nx.DiGraph()

        # Add all nodes
        for node_id in self.nodes.keys():
            G.add_node(node_id)

        # Add existing relationships of the specified type
        for rel in self.relationships.values():
            if rel.type == relationship_type:
                G.add_edge(rel.source_node_id, rel.target_node_id)

        # Add the new relationship temporarily
        G.add_edge(new_relationship.source_node_id, new_relationship.target_node_id)

        # Check if this creates a cycle
        try:
            # If the graph is acyclic, this will succeed
            list(nx.topological_sort(G))
            return False
        except (nx.NetworkXError, nx.NetworkXUnfeasible):
            # A cycle exists
            return True

    def get_relationships(
        self, relationship_type: RelationshipType
    ) -> List[Relationship]:
        """
        Get all relationships of a specific type sorted by confidence (lowest first).

        Args:
            relationship_type: The type of relationship to get

        Returns:
            List of relationships sorted by confidence (lowest first)
        """
        relationships = [
            rel for rel in self.relationships.values() if rel.type == relationship_type
        ]
        return sorted(relationships, key=lambda r: r.existence_confidence_llm)

    def remove_cycle_relationship(self, relationship_type: RelationshipType) -> bool:
        """
        Remove the lowest confidence relationship that causes a cycle.

        Args:
            relationship_type: The type of relationship to check

        Returns:
            True if a relationship was removed, False otherwise
        """
        # Get all relationships of this type sorted by confidence
        sorted_relationships = self.get_relationships(relationship_type)

        for rel in sorted_relationships:
            # Temporarily remove this relationship
            temp_rel = self.relationships.pop(rel.id, None)
            if temp_rel is None:
                continue

            # Check if removing it breaks the cycle
            G = nx.DiGraph()
            for node_id in self.nodes.keys():
                G.add_node(node_id)

            for remaining_rel in self.relationships.values():
                if remaining_rel.type == relationship_type:
                    G.add_edge(
                        remaining_rel.source_node_id, remaining_rel.target_node_id
                    )

            try:
                # If no cycle exists now, we found the culprit
                list(nx.topological_sort(G))
                return True  # Successfully removed cycle-causing relationship
            except (nx.NetworkXError, nx.NetworkXUnfeasible):
                # Still has cycle, restore the relationship and try next one
                self.relationships[rel.id] = temp_rel

        return False  # Could not resolve cycle by removing relationships

    def resolve_cycles(self) -> List[str]:
        """
        Preemptively resolve potential cycles in the AWG by removing lowest confidence relationships.

        This method checks for cycles within the current AWG and removes the lowest confidence
        relationships that cause cycles, as suggested in the specification.

        Returns:
            List of relationship IDs that were removed
        """
        removed_rel_ids = []
        cycle_prone_types = [
            RelationshipType.HAS_PREREQUISITE,
            RelationshipType.IS_TYPE_OF,
            RelationshipType.IS_PART_OF,
        ]

        for rel_type in cycle_prone_types:
            # Keep trying to remove cycle-causing relationships until no more cycles exist
            max_attempts = 10  # Prevent infinite loops
            attempt = 0

            while attempt < max_attempts:
                # Check if there are cycles for this relationship type
                G = nx.DiGraph()
                for node_id in self.nodes.keys():
                    G.add_node(node_id)

                for rel in self.relationships.values():
                    if rel.type == rel_type:
                        G.add_edge(rel.source_node_id, rel.target_node_id)

                try:
                    # If this succeeds, no cycle exists
                    list(nx.topological_sort(G))
                    break  # No cycle, move to next relationship type
                except (nx.NetworkXError, nx.NetworkXUnfeasible):
                    # Cycle exists, try to remove the lowest confidence relationship
                    # First, get all relationship IDs before removal
                    before_removal_ids = set(self.relationships.keys())

                    removed = self.remove_cycle_relationship(rel_type)
                    if removed:
                        # Find which relationship was removed
                        after_removal_ids = set(self.relationships.keys())
                        removed_this_time = before_removal_ids - after_removal_ids
                        removed_rel_ids.extend(list(removed_this_time))
                    else:
                        # Could not resolve cycle, stop trying
                        break

                attempt += 1

        return removed_rel_ids

    def get_definitions(self) -> Dict[str, str]:
        """
        Get the definitions of all concepts
        """
        definitions = {}
        for node in self.nodes.values():
            if node.definition:
                definitions[node.name] = node.definition
        return definitions

    def to_incident_encoding(self, relationship_type: RelationshipType) -> str:
        """
        Convert the working graph to text-based incident encoding representation
        for a specific relationship type.

        Args:
            relationship_type: The type of relationship to encode

        Returns:
            String representation in incident encoding format
        """
        if not self.nodes:
            return "Empty graph - no nodes found."

        # Build the text representation
        lines = []

        # First, list all nodes with numbers
        lines.append("G describes a graph among nodes:")
        node_list = list(self.nodes.values())
        for i, node in enumerate(node_list, 1):
            lines.append(f"{i}. {node.name}")

        lines.append("")  # Empty line

        # Then, describe relationships of the specified type
        lines.append(f"Relationships ({relationship_type.value}):")

        # Group relationships by source node
        source_to_targets = {}
        for rel in self.relationships.values():
            if rel.type == relationship_type:
                source_node = self.get_node(rel.source_node_id)
                target_node = self.get_node(rel.target_node_id)

                if source_node and target_node:
                    if source_node.name not in source_to_targets:
                        source_to_targets[source_node.name] = []
                    source_to_targets[source_node.name].append(target_node.name)

        # Format the relationships
        for source_name, target_names in source_to_targets.items():
            targets_str = ", ".join(target_names)
            lines.append(f"{source_name} {relationship_type.value} {targets_str}.")

        return "\n".join(lines)
