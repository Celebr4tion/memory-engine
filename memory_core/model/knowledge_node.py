"""
Knowledge node module for storing semantic information as graph nodes.

This module defines the structure for knowledge nodes in the memory engine.
"""

from typing import Optional, Dict, Any, ClassVar
import time


class KnowledgeNode:
    """
    Represents a node in the knowledge graph.

    A KnowledgeNode stores semantic information and can be connected to other nodes
    through relationships. Each node has associated metadata like ratings and source.
    """

    def __init__(
        self,
        content: str,
        source: str,
        creation_timestamp: Optional[float] = None,
        rating_richness: float = 0.5,
        rating_truthfulness: float = 0.5,
        rating_stability: float = 0.5,
        node_id: Optional[str] = None,
    ):
        """
        Initialize a KnowledgeNode with the provided attributes.

        Args:
            content: The textual data contained in the node
            source: The source of the information (e.g., 'Wikipedia', 'UserInput')
            creation_timestamp: When the node was created (defaults to current time)
            rating_richness: Rating for information richness (0.0 to 1.0)
            rating_truthfulness: Rating for truthfulness (0.0 to 1.0)
            rating_stability: Rating for stability over time (0.0 to 1.0)
            node_id: Unique identifier for the node (None if not yet saved)
        """
        self.node_id = node_id
        self.content = content
        self.source = source
        self.creation_timestamp = (
            creation_timestamp if creation_timestamp is not None else time.time()
        )
        self.rating_richness = rating_richness
        self.rating_truthfulness = rating_truthfulness
        self.rating_stability = rating_stability

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the KnowledgeNode to a dictionary representation.

        Returns:
            Dictionary containing all node attributes
        """
        return {
            "node_id": self.node_id,
            "content": self.content,
            "source": self.source,
            "creation_timestamp": self.creation_timestamp,
            "rating_richness": self.rating_richness,
            "rating_truthfulness": self.rating_truthfulness,
            "rating_stability": self.rating_stability,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeNode":
        """
        Create a KnowledgeNode from a dictionary representation.

        Args:
            data: Dictionary containing node attributes

        Returns:
            A new KnowledgeNode instance
        """
        return cls(
            content=data["content"],
            source=data["source"],
            creation_timestamp=data["creation_timestamp"],
            rating_richness=data["rating_richness"],
            rating_truthfulness=data["rating_truthfulness"],
            rating_stability=data["rating_stability"],
            node_id=data.get("node_id"),  # node_id may be None
        )

    def __eq__(self, other):
        """
        Compare this KnowledgeNode with another for equality.

        Two nodes are considered equal if they have the same attributes.

        Args:
            other: Another KnowledgeNode to compare with

        Returns:
            True if the nodes are equal, False otherwise
        """
        if not isinstance(other, KnowledgeNode):
            return False

        return (
            self.node_id == other.node_id
            and self.content == other.content
            and self.source == other.source
            and self.creation_timestamp == other.creation_timestamp
            and self.rating_richness == other.rating_richness
            and self.rating_truthfulness == other.rating_truthfulness
            and self.rating_stability == other.rating_stability
        )

    def __repr__(self):
        """
        Return a string representation of the KnowledgeNode.

        Returns:
            String representation of the node
        """
        node_id_str = f"'{self.node_id}'" if self.node_id else "None"
        return (
            f"KnowledgeNode(node_id={node_id_str}, "
            f"content='{self.content[:30]}{'...' if len(self.content) > 30 else ''}', "
            f"source='{self.source}', "
            f"timestamp={self.creation_timestamp:.2f}, "
            f"richness={self.rating_richness:.2f}, "
            f"truthfulness={self.rating_truthfulness:.2f}, "
            f"stability={self.rating_stability:.2f})"
        )
