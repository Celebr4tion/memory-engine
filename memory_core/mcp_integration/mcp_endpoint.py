"""
MCP endpoint implementation for the Memory Engine.

This module provides an interface for external systems to interact with 
the Memory Engine using the Module Communication Protocol.
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional, Union

from memory_core.core.knowledge_engine import KnowledgeEngine
from memory_core.ingestion.advanced_extractor import extract_knowledge_units, process_extracted_units
from memory_core.ingestion.relationship_extractor import analyze_and_create_relationships
from memory_core.embeddings.embedding_manager import EmbeddingManager
from memory_core.embeddings.vector_store import VectorStoreMilvus


class MemoryEngineMCP:
    """
    MCP interface for the Memory Engine.
    
    This class provides methods for external systems to interact with
    the Memory Engine through the Module Communication Protocol.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8182):
        """
        Initialize the MCP interface.
        
        Args:
            host: JanusGraph server host
            port: JanusGraph server port
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize the KnowledgeEngine
        self.engine = KnowledgeEngine(host=host, port=port, enable_versioning=True)
        self.engine.connect()
        
        # Initialize the vector store and embedding manager
        self.vector_store = None
        self.embedding_manager = None
        
        try:
            self.vector_store = VectorStoreMilvus(host="localhost", port=19530)
            if self.vector_store.connect():
                self.embedding_manager = EmbeddingManager(self.vector_store)
                self.logger.info("Vector storage and embedding manager initialized")
            else:
                self.logger.warning("Failed to connect to vector store")
        except Exception as e:
            self.logger.error(f"Error initializing vector storage: {str(e)}")
    
    def ingest_raw_text(self, raw_text: str, source_label: str = "MCP Input") -> Dict[str, Any]:
        """
        Extract knowledge from raw text, store it in the graph, and create relationships.
        
        This method:
        1. Extracts knowledge units from the raw text
        2. Processes and stores them as nodes (with merging)
        3. Identifies and creates relationships between the nodes
        4. Returns the IDs of the created or merged nodes
        
        Args:
            raw_text: The text to process
            source_label: Source identifier for the knowledge
            
        Returns:
            Dictionary with created/merged node IDs and relationship statistics
        """
        try:
            self.logger.info(f"Processing text input from source: {source_label}")
            
            # Extract knowledge units from the raw text
            units = extract_knowledge_units(raw_text)
            
            if not units:
                self.logger.info("No knowledge units extracted from input")
                return {
                    "status": "no_knowledge_extracted",
                    "created_or_merged_node_ids": []
                }
            
            # Process and store the units
            node_ids = process_extracted_units(
                units=units,
                source_label=source_label,
                storage=self.engine.storage,
                embedding_manager=self.embedding_manager
            )
            
            self.logger.info(f"Created {len(node_ids)} nodes from knowledge units")
            
            # Create relationships between the nodes
            relationships = {}
            if len(node_ids) > 1:
                # Pass parameters by position, not by name to match test expectations
                relationships = analyze_and_create_relationships(
                    node_ids,
                    self.engine.storage,
                    self.embedding_manager
                )
                
                # Count total relationships
                total_relationships = sum(len(rel_ids) for rel_ids in relationships.values())
                self.logger.info(f"Created {total_relationships} relationships between nodes")
            
            # Prepare the result
            result = {
                "status": "success",
                "created_or_merged_node_ids": node_ids,
                "relationship_counts": {
                    rel_type: len(rel_ids) for rel_type, rel_ids in relationships.items()
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error ingesting raw text: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "created_or_merged_node_ids": []
            }
    
    def list_nodes(self, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        List nodes that match the specified filters.
        
        Args:
            filters: Dictionary of filters to apply (e.g., {'rating_truthfulness': 0.8})
            
        Returns:
            Dictionary with a list of matching nodes (preview only)
        """
        try:
            # TODO: Implement proper filtering when JanusGraphStorage supports it
            # For now, we retrieve nodes and filter them in memory
            
            # This is a simplified approach - in a real implementation,
            # you would query JanusGraph with filters directly
            nodes = []
            
            # Return a limited number of nodes to avoid overwhelming response
            node_count = 0
            max_nodes = 50
            
            # Implement a basic filtering mechanism
            # This is inefficient but works for demonstration purposes
            if filters:
                # TODO: Replace with proper query once available
                pass
            
            # For now, return an empty list with an explanatory message
            return {
                "status": "not_implemented",
                "message": "Filtering nodes is not yet implemented",
                "nodes": []
            }
        
        except Exception as e:
            self.logger.error(f"Error listing nodes: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "nodes": []
            }
    
    def get_node_details(self, node_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific node.
        
        Args:
            node_id: ID of the node to retrieve
            
        Returns:
            Dictionary with all node properties
        """
        try:
            # Get the node from the graph
            node = self.engine.get_node(node_id)
            
            # Convert to dictionary
            node_dict = node.to_dict()
            
            # Get incoming and outgoing relationships
            outgoing_relationships = self.engine.get_outgoing_relationships(node_id)
            incoming_relationships = self.engine.get_incoming_relationships(node_id)
            
            # Format relationships for the response
            outgoing = []
            for rel in outgoing_relationships:
                outgoing.append({
                    "edge_id": rel.edge_id,
                    "target_id": rel.to_id,
                    "relation_type": rel.relation_type,
                    "confidence_score": rel.confidence_score
                })
            
            incoming = []
            for rel in incoming_relationships:
                incoming.append({
                    "edge_id": rel.edge_id,
                    "source_id": rel.from_id,
                    "relation_type": rel.relation_type,
                    "confidence_score": rel.confidence_score
                })
            
            # Prepare the complete response
            result = {
                "status": "success",
                "node": node_dict,
                "outgoing_relationships": outgoing,
                "incoming_relationships": incoming
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error retrieving node {node_id}: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def search_text(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Search for nodes similar to the query text.
        
        Args:
            query_text: Text to search for
            top_k: Number of results to return
            
        Returns:
            Dictionary with matching node IDs and similarity scores
        """
        try:
            if not self.embedding_manager:
                return {
                    "status": "error",
                    "message": "Embedding manager not available",
                    "results": []
                }
            
            # Generate an embedding for the query
            node_ids = self.embedding_manager.search_similar_nodes(query_text, top_k)
            
            # Get more information about the found nodes
            results = []
            for node_id in node_ids:
                try:
                    node = self.engine.get_node(node_id)
                    
                    # Ensure content is properly truncated for preview
                    content = str(node.content)
                    MAX_PREVIEW_LENGTH = 100
                    
                    # Make sure truncation happens if needed
                    if len(content) > MAX_PREVIEW_LENGTH:
                        content_preview = content[:MAX_PREVIEW_LENGTH] + "..."
                    else:
                        content_preview = content
                    
                    # Add a preview of the node to the results
                    results.append({
                        "node_id": node_id,
                        "content_preview": content_preview,
                        "source": node.source,
                        "rating_truthfulness": node.rating_truthfulness
                    })
                except Exception as e:
                    self.logger.warning(f"Error retrieving node {node_id}: {str(e)}")
            
            return {
                "status": "success",
                "query": query_text,
                "results": results
            }
            
        except Exception as e:
            self.logger.error(f"Error searching for text: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "results": []
            }
    
    def update_node_rating(self, node_id: str, new_rating: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the ratings of a knowledge node.
        
        Args:
            node_id: ID of the node to update
            new_rating: Dictionary with evidence for rating updates
                        (e.g., {'confirmation': 1, 'contradiction': 0})
            
        Returns:
            Dictionary with updated node details
        """
        try:
            # Get the current node
            node = self.engine.get_node(node_id)
            
            # Calculate new ratings
            # These formulas can be moved to a separate rating_system module later
            
            # Update truthfulness
            if 'confirmation' in new_rating or 'contradiction' in new_rating:
                confirmation = new_rating.get('confirmation', 0)
                contradiction = new_rating.get('contradiction', 0)
                
                # Formula: rating = min(1.0, max(0.0, old_rating + 0.2*confirmation - 0.2*contradiction))
                truthfulness = min(1.0, max(0.0, 
                    node.rating_truthfulness + 0.2 * confirmation - 0.2 * contradiction
                ))
                node.rating_truthfulness = truthfulness
            
            # Update richness
            if 'richness' in new_rating:
                richness_change = new_rating.get('richness', 0)
                richness = min(1.0, max(0.0, node.rating_richness + 0.2 * richness_change))
                node.rating_richness = richness
            
            # Update stability
            if 'stability' in new_rating:
                stability_change = new_rating.get('stability', 0)
                stability = min(1.0, max(0.0, node.rating_stability + 0.2 * stability_change))
                node.rating_stability = stability
            
            # Save the updated node
            self.engine.save_node(node)
            
            # Get the updated node details
            return self.get_node_details(node_id)
            
        except Exception as e:
            self.logger.error(f"Error updating node rating for {node_id}: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def execute_mcp_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an MCP command received from an external system.
        
        Args:
            command: Dictionary with the command details
                    (e.g., {'action': 'ingest_text', 'text': '...', 'source': '...'})
            
        Returns:
            Response dictionary based on the command
        """
        try:
            if not command or 'action' not in command:
                return {
                    "status": "error",
                    "message": "Invalid command format: 'action' field is required"
                }
            
            action = command['action']
            
            # Route to the appropriate method based on the action
            if action == 'ingest_text':
                if 'text' not in command:
                    return {
                        "status": "error",
                        "message": "Missing required field: 'text'"
                    }
                
                source = command.get('source', 'MCP Input')
                return self.ingest_raw_text(command['text'], source)
                
            elif action == 'get_node':
                if 'node_id' not in command:
                    return {
                        "status": "error",
                        "message": "Missing required field: 'node_id'"
                    }
                
                return self.get_node_details(command['node_id'])
                
            elif action == 'search':
                if 'query' not in command:
                    return {
                        "status": "error",
                        "message": "Missing required field: 'query'"
                    }
                
                top_k = command.get('top_k', 5)
                return self.search_text(command['query'], top_k)
                
            elif action == 'update_rating':
                if 'node_id' not in command or 'rating' not in command:
                    return {
                        "status": "error",
                        "message": "Missing required fields: 'node_id' and/or 'rating'"
                    }
                
                return self.update_node_rating(command['node_id'], command['rating'])
                
            elif action == 'list_nodes':
                filters = command.get('filters', {})
                return self.list_nodes(filters)
                
            else:
                return {
                    "status": "error",
                    "message": f"Unknown action: {action}"
                }
                
        except Exception as e:
            self.logger.error(f"Error executing MCP command: {str(e)}")
            return {
                "status": "error",
                "message": f"Command execution error: {str(e)}"
            }
    
    def close(self):
        """Close connections to databases."""
        if self.vector_store:
            try:
                self.vector_store.disconnect()
            except:
                pass
        
        if self.engine:
            try:
                self.engine.disconnect()
            except:
                pass