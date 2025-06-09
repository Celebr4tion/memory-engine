"""
Knowledge Agent implementation using Google Agent Development Kit.

This module provides an agent implementation that uses the Memory Engine
for knowledge extraction, storage, and retrieval.
"""

import logging
from typing import Dict, List, Any, Optional, Union

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.tools import FunctionTool

from memory_core.core.knowledge_engine import KnowledgeEngine
from memory_core.ingestion.advanced_extractor import extract_knowledge_units
from memory_core.model.knowledge_node import KnowledgeNode
from memory_core.model.relationship import Relationship


class KnowledgeAgent(BaseAgent):
    """
    An agent for extracting, storing, and retrieving knowledge using the Memory Engine.
    
    This agent integrates with Google ADK and provides tools for interacting
    with the knowledge graph.
    """
    
    # Field declarations for Pydantic
    knowledge_engine: KnowledgeEngine
    llm_agent: LlmAgent
    logger: logging.Logger
    
    # Pydantic configuration
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(
        self,
        knowledge_engine: KnowledgeEngine,
        model: str = "gemini-2.5-pro-exp-03-25",  # Updated to latest model
        name: str = "knowledge_agent",
        description: str = "An agent that can extract, store, and retrieve knowledge",
        **kwargs
    ):
        """
        Initialize the knowledge agent.
        
        Args:
            knowledge_engine: The KnowledgeEngine instance to use
            model: The LLM model to use (default: gemini-2.5-pro-exp-03-25)
            name: The name of the agent
            description: The description of the agent
            **kwargs: Additional arguments to pass to BaseAgent
        """
        # Connect to the knowledge engine
        if not knowledge_engine.storage.g:
            knowledge_engine.connect()
            
        # Define tools for interacting with the knowledge graph
        tools = [
            FunctionTool(self.extract_and_store_knowledge),
            FunctionTool(self.retrieve_knowledge),
            FunctionTool(self.create_relationship)
        ]
        
        # Create internal LLM agent
        llm_agent = LlmAgent(
            name=f"{name}_llm",
            model=model,
            description="LLM agent for knowledge processing",
            tools=tools,
            instruction="""
            You are a knowledge agent that helps users extract, store, and retrieve knowledge.
            You can process text to extract structured knowledge, store it in a knowledge graph,
            and retrieve relevant information based on user queries.
            
            When extracting knowledge, analyze the text carefully to identify distinct
            pieces of information, their relationships, and metadata.
            
            When retrieving knowledge, provide the most relevant information
            and explain how the pieces of knowledge are related.
            """,
        )
        
        # Initialize logger
        logger = logging.getLogger(__name__)
        
        # Initialize the BaseAgent with custom fields
        super().__init__(
            name=name,
            description=description,
            knowledge_engine=knowledge_engine,
            llm_agent=llm_agent,
            logger=logger,
            sub_agents=[llm_agent],
            **kwargs
        )
    
    async def _run_async_impl(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the knowledge agent's main logic.
        
        This method routes requests to the appropriate sub-methods based on the input.
        """
        return await self.llm_agent._run_async_impl(input_data)
    
    async def extract_and_store_knowledge(self, text: str) -> Dict[str, Any]:
        """
        Extract knowledge units from text and store them in the knowledge graph.
        
        Args:
            text: The text to extract knowledge from
            
        Returns:
            A dictionary with node_ids and a summary of the extracted knowledge
        """
        try:
            self.logger.info("Extracting knowledge units from text")
            knowledge_units = extract_knowledge_units(text)
            
            if not knowledge_units:
                return {
                    "status": "no_knowledge_extracted",
                    "message": "No knowledge units could be extracted from the provided text.",
                    "node_ids": []
                }
            
            # Store each knowledge unit as a node
            node_ids = []
            for unit in knowledge_units:
                # Create a knowledge node from the unit
                node = KnowledgeNode(
                    content=unit["content"],
                    source=unit.get("source", {}).get("reference", "Knowledge extraction"),
                    rating_richness=unit.get("metadata", {}).get("importance", 0.5),
                    rating_truthfulness=unit.get("metadata", {}).get("confidence_level", 0.8),
                    rating_stability=0.7  # Default value for stability
                )
                
                # Save the node to the graph
                node_id = self.knowledge_engine.save_node(node)
                node_ids.append({
                    "id": node_id,
                    "content": unit["content"],
                    "tags": unit.get("tags", [])
                })
                
                # If the unit has metadata about domain, store it as a relationship
                if "metadata" in unit and "domain" in unit["metadata"]:
                    domain = unit["metadata"]["domain"]
                    # Create a domain node if it doesn't exist
                    # This is a simplified approach; in a real implementation,
                    # you would check if the domain node already exists
                    domain_node = KnowledgeNode(
                        content=f"Domain: {domain}",
                        source="Knowledge domain categorization",
                        rating_richness=0.7,
                        rating_truthfulness=0.9,
                        rating_stability=0.9
                    )
                    domain_id = self.knowledge_engine.save_node(domain_node)
                    
                    # Create a relationship between the knowledge node and domain node
                    relationship = Relationship(
                        from_id=node_id,
                        to_id=domain_id,
                        relation_type="BELONGS_TO_DOMAIN",
                        confidence_score=0.9
                    )
                    self.knowledge_engine.save_relationship(relationship)
            
            return {
                "status": "success",
                "message": f"Extracted and stored {len(node_ids)} knowledge units.",
                "node_ids": node_ids
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting knowledge: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to extract knowledge: {str(e)}",
                "node_ids": []
            }
    
    async def retrieve_knowledge(self, query: str) -> Dict[str, Any]:
        """
        Retrieve knowledge related to a query from the knowledge graph.
        
        Args:
            query: The query to search for
            
        Returns:
            A dictionary with relevant knowledge nodes and their relationships
        """
        try:
            # For now, we're using vector similarity search via the embedding manager
            # This assumes the embedding manager is configured and connected
            embedding_manager = self.knowledge_engine.embedding_manager
            
            if not embedding_manager:
                return {
                    "status": "error",
                    "message": "Embedding manager is not available for semantic search.",
                    "results": []
                }
            
            # Search for similar nodes
            similar_node_ids = embedding_manager.search_similar_nodes(query, top_k=5)
            
            if not similar_node_ids:
                return {
                    "status": "no_results",
                    "message": "No relevant knowledge found.",
                    "results": []
                }
            
            # Retrieve the node data for each matching ID
            results = []
            for node_id in similar_node_ids:
                try:
                    node = self.knowledge_engine.get_node(node_id)
                    
                    # Get outgoing relationships
                    outgoing = []
                    try:
                        rel_outgoing = self.knowledge_engine.graph.get_outgoing_relationships(node_id)
                        for rel in rel_outgoing:
                            target_node = self.knowledge_engine.get_node(rel.to_id)
                            outgoing.append({
                                "relation_type": rel.relation_type,
                                "target_content": target_node.content,
                                "confidence": rel.confidence_score
                            })
                    except Exception as e:
                        self.logger.warning(f"Error getting outgoing relationships: {str(e)}")
                    
                    # Get incoming relationships
                    incoming = []
                    try:
                        rel_incoming = self.knowledge_engine.graph.get_incoming_relationships(node_id)
                        for rel in rel_incoming:
                            source_node = self.knowledge_engine.get_node(rel.from_id)
                            incoming.append({
                                "relation_type": rel.relation_type,
                                "source_content": source_node.content,
                                "confidence": rel.confidence_score
                            })
                    except Exception as e:
                        self.logger.warning(f"Error getting incoming relationships: {str(e)}")
                    
                    results.append({
                        "id": node_id,
                        "content": node.content,
                        "source": node.source,
                        "outgoing_relationships": outgoing,
                        "incoming_relationships": incoming
                    })
                except Exception as e:
                    self.logger.warning(f"Error retrieving node {node_id}: {str(e)}")
            
            return {
                "status": "success",
                "message": f"Found {len(results)} relevant knowledge nodes.",
                "results": results
            }
            
        except Exception as e:
            self.logger.error(f"Error retrieving knowledge: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to retrieve knowledge: {str(e)}",
                "results": []
            }
    
    async def create_relationship(
        self, 
        from_id: str, 
        to_id: str, 
        relation_type: str, 
        confidence_score: float = 0.8
    ) -> Dict[str, Any]:
        """
        Create a relationship between two knowledge nodes.
        
        Args:
            from_id: The ID of the source node
            to_id: The ID of the target node
            relation_type: The type of relationship
            confidence_score: The confidence score for this relationship
            
        Returns:
            A dictionary with the relationship ID and status
        """
        try:
            # Verify that both nodes exist
            try:
                self.knowledge_engine.get_node(from_id)
                self.knowledge_engine.get_node(to_id)
            except ValueError as e:
                return {
                    "status": "error",
                    "message": f"One or both nodes not found: {str(e)}",
                    "edge_id": None
                }
            
            # Create the relationship
            relationship = Relationship(
                from_id=from_id,
                to_id=to_id,
                relation_type=relation_type.upper(),
                confidence_score=confidence_score
            )
            
            edge_id = self.knowledge_engine.save_relationship(relationship)
            
            return {
                "status": "success",
                "message": f"Created relationship of type {relation_type} between nodes {from_id} and {to_id}.",
                "edge_id": edge_id
            }
            
        except Exception as e:
            self.logger.error(f"Error creating relationship: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to create relationship: {str(e)}",
                "edge_id": None
            }


def create_knowledge_agent(
    host: str = "localhost", 
    port: int = 8182,
    model: str = "gemini-2.5-pro-exp-03-25",
    enable_versioning: bool = True
) -> KnowledgeAgent:
    """
    Create a knowledge agent with a configured knowledge engine.
    
    Args:
        host: JanusGraph host
        port: JanusGraph port
        model: LLM model to use
        enable_versioning: Whether to enable versioning in the knowledge engine
        
    Returns:
        A configured KnowledgeAgent instance
    """
    # Create and configure the knowledge engine
    knowledge_engine = KnowledgeEngine(
        host=host,
        port=port,
        enable_versioning=enable_versioning
    )
    
    # Create the knowledge agent
    agent = KnowledgeAgent(
        knowledge_engine=knowledge_engine,
        model=model
    )
    
    return agent