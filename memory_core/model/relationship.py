"""
Relationship module for storing connections between knowledge nodes.

This module defines the structure for relationships in the knowledge graph.
"""
from typing import Dict, Any, Optional
import time


class Relationship:
    """
    Represents a directed relationship between two nodes in the knowledge graph.
    
    A Relationship connects two KnowledgeNodes with a specific relation type
    and contains metadata about the connection.
    """
    
    def __init__(
        self,
        from_id: str,
        to_id: str,
        relation_type: str,
        timestamp: Optional[float] = None,
        confidence_score: float = 0.5,
        version: int = 1,
        edge_id: Optional[str] = None
    ):
        """
        Initialize a Relationship with the provided attributes.
        
        Args:
            from_id: ID of the source node
            to_id: ID of the target node
            relation_type: Type of relationship (e.g., 'is_a', 'part_of', 'causes')
            timestamp: When the relationship was created (defaults to current time)
            confidence_score: Confidence in the relationship (0.0 to 1.0)
            version: Version number of the relationship
            edge_id: Unique identifier for the relationship (None if not yet saved)
        """
        self.edge_id = edge_id
        self.from_id = from_id
        self.to_id = to_id
        self.relation_type = relation_type
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.confidence_score = confidence_score
        self.version = version
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Relationship to a dictionary representation.
        
        Returns:
            Dictionary containing all relationship attributes
        """
        return {
            'edge_id': self.edge_id,
            'from_id': self.from_id,
            'to_id': self.to_id,
            'relation_type': self.relation_type,
            'timestamp': self.timestamp,
            'confidence_score': self.confidence_score,
            'version': self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Relationship':
        """
        Create a Relationship from a dictionary representation.
        
        Args:
            data: Dictionary containing relationship attributes
            
        Returns:
            A new Relationship instance
        """
        return cls(
            from_id=data['from_id'],
            to_id=data['to_id'],
            relation_type=data['relation_type'],
            timestamp=data['timestamp'],
            confidence_score=data['confidence_score'],
            version=data['version'],
            edge_id=data.get('edge_id')  # edge_id may be None
        )
    
    def __eq__(self, other):
        """
        Compare this Relationship with another for equality.
        
        Two relationships are considered equal if they have the same attributes.
        
        Args:
            other: Another Relationship to compare with
            
        Returns:
            True if the relationships are equal, False otherwise
        """
        if not isinstance(other, Relationship):
            return False
        
        return (
            self.edge_id == other.edge_id and
            self.from_id == other.from_id and
            self.to_id == other.to_id and
            self.relation_type == other.relation_type and
            self.timestamp == other.timestamp and
            self.confidence_score == other.confidence_score and
            self.version == other.version
        )
    
    def __repr__(self):
        """
        Return a string representation of the Relationship.
        
        Returns:
            String representation of the relationship
        """
        edge_id_str = f"'{self.edge_id}'" if self.edge_id else "None"
        return (
            f"Relationship(edge_id={edge_id_str}, "
            f"from_id='{self.from_id}', "
            f"to_id='{self.to_id}', "
            f"relation_type='{self.relation_type}', "
            f"timestamp={self.timestamp:.2f}, "
            f"confidence_score={self.confidence_score:.2f}, "
            f"version={self.version})"
        ) 