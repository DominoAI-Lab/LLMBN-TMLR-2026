"""
Base interface for Bayesian Network structure generators.

Defines common API for both LLM-based and traditional 
approaches to Bayesian Network structure generation.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd
import networkx as nx

from llmbn.llm import LLMClient
from llmbn.errors.generation_error import ParseResponseError, MaxRetriesError


class BaseGenerator(ABC):
    """
    Abstract base class for Bayesian Network structure generators.
    
    Attributes:
        model: LLM model name (for LLM-based generators)
        logger: Logger instance for output
        client: LLM client (for LLM-based generators)
        messages: Message history for LLM (for LLM-based generators)
    """
    
    def __init__(
        self, 
        model: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the generator.
        
        Args:
            model: LLM model name (required for LLM-based generators)
            logger: Logger instance for output
        """
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize LLM client only if model is provided
        if model is not None:
            self.client = LLMClient.from_model(model, self.logger)
            self.messages = [
                {"role": "system", "content": "You are a helpful assistant that constructs Bayesian Networks."},
            ]
            self.prompt = None
        else:
            self.client = None
            self.messages = None
            self.prompt = None
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the generator implementation."""
        raise NotImplementedError("Subclasses must implement .name property")
    
    @property
    def is_llm_based(self) -> bool:
        """Whether this generator uses an LLM for structure generation."""
        return self.model is not None
    
    @abstractmethod
    def run(
        self,
        desc_variables: str,
        dag_variables: list[str],
        observation: Optional[pd.DataFrame] = None,
        generations: Optional[list] = None,
        **kwargs
    ) -> tuple[int, Dict[str, Any]]:
        """
        Generate a Bayesian Network structure.
        
        Args:
            desc_variables: Variable descriptions in string format
            dag_variables: List of variable names
            observation: Observed data for structure learning (required for traditional generators)
            generations: Previous generations (for compatibility)
            **kwargs: Additional generator-specific parameters
            
        Returns:
            Tuple containing:
            - validation_status: 1 if valid DAG, 0 otherwise
            - results_dict: Dictionary with generated structure and metrics
              including 'Generation' (structure representation) and 'Matrix'
              (adjacency matrix)
              
        Raises:
            ValueError: If required inputs are missing or invalid
        """
        raise NotImplementedError("Subclasses must implement .run()")
    
    # LLM-specific methods (only used by LLM-based generators)
    def prepare_messages(self, desc_variables: str, dag_variables: list[str]) -> list[dict]:
        """
        Prepare messages for LLM generation.
        
        Args:
            desc_variables: Variable descriptions in string format
            dag_variables: List of variable names
            
        Returns:
            list[dict]: Messages formatted for LLM API
            
        Raises:
            NotImplementedError: If called on non-LLM generator or not implemented
        """
        if not self.is_llm_based:
            raise NotImplementedError("prepare_messages() only available for LLM-based generators")
        raise NotImplementedError("Subclasses must implement .prepare_messages()")
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response into structured BN representation.
        
        Args:
            response: Raw text response from LLM
            
        Returns:
            Dict[str, Any]: Structured BN representation
            
        Raises:
            NotImplementedError: If called on non-LLM generator or not implemented
            ParseResponseError: If response cannot be parsed
        """
        if not self.is_llm_based:
            raise NotImplementedError("parse_response() only available for LLM-based generators")
        raise NotImplementedError("Subclasses must implement .parse_response()")
    
    def generate(self, max_retries: int = 3) -> Dict[str, Any]:
        """
        Generate a Bayesian Network structure using the language model.
        
        Args:
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dict[str, Any]: Generated BN structure
            
        Raises:
            NotImplementedError: If called on non-LLM generator
            ParseResponseError: If response cannot be parsed
            MaxRetriesError: If max retries reached
        """
        if not self.is_llm_based:
            raise NotImplementedError("generate() only available for LLM-based generators")
        
        for attempt in range(1, max_retries + 1):
            self.logger.debug(f"[{self.name}] Generating... ({attempt}/{max_retries})")
            response = self.client.chat(messages=self.messages)
            self.logger.debug(f"[{self.name}] Raw response:\n{response}")
            try:
                generation = self.parse_response(response)
                self.logger.debug(f"[{self.name}] JSON output:\n{generation}")
                return generation
            except ParseResponseError as e:
                self.logger.exception(f"[{self.name}] Parse Response Error on attempt {attempt}: {e}")
                raise e
            except Exception as e:
                self.logger.exception(f"[{self.name}] Exception on attempt {attempt}: {e}; Retrying...")
                continue
        raise MaxRetriesError("Max retries reached")
    
    def construct_matrix(
        self,
        generation: Dict[str, Any],
        dag_variables: list[str],
    ) -> pd.DataFrame:
        """
        Construct adjacency matrix from generation dictionary.
        
        Args:
            generation: Generated BN structure dictionary
            dag_variables: List of variable names
            
        Returns:
            pd.DataFrame: Adjacency matrix representation
            
        Raises:
            NotImplementedError: If called on non-LLM generator or not implemented
        """
        if not self.is_llm_based:
            raise NotImplementedError("construct_matrix() only available for LLM-based generators")
        raise NotImplementedError("Subclasses must implement .construct_matrix()")
    
    # Utility methods (available to all generators)
    def _validate_inputs(
        self,
        observation: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Validate input data.
        
        Args:
            observation: Input data to validate (required for traditional generators)
            
        Raises:
            ValueError: If input data is invalid
        """
        if not self.is_llm_based and (observation is None or observation.empty):
            raise ValueError("Observation data is required for traditional generators")
        
        if observation is not None and len(observation.columns) < 2:
            raise ValueError("At least 2 variables are required for structure learning")
        
        if observation is not None:
            self.logger.debug(f"[{self.name}] Input validation passed: {len(observation.columns)} variables, {len(observation)} samples")
    
    def _log_results(
        self,
        results: Dict[str, Any],
    ) -> None:
        """
        Log generator results.
        
        Args:
            results: Results dictionary from run()
        """
        self.logger.info(f"[{self.name}] Generator completed")
        
        if 'Matrix' in results:
            matrix = results['Matrix']
            if hasattr(matrix, 'shape'):
                self.logger.info(f"[{self.name}] Generated structure: {matrix.shape[0]}x{matrix.shape[1]} adjacency matrix")
        
        if 'Generation' in results:
            self.logger.debug(f"[{self.name}] Structure generation completed successfully")
    
    def _dag_to_adjacency_matrix(
        self,
        dag: nx.DiGraph,
        dag_variables: list[str],
    ) -> pd.DataFrame:
        """
        Convert a DAG to adjacency matrix format.
        
        Args:
            dag: NetworkX DiGraph or similar object with edges() method
            dag_variables: List of variable names
            
        Returns:
            pd.DataFrame: Adjacency matrix representation
        """
        matrix = pd.DataFrame(0, index=dag_variables, columns=dag_variables)
        
        for edge in dag.edges():
            parent, child = edge
            if parent in dag_variables and child in dag_variables:
                matrix.loc[parent, child] = 1
        
        return matrix
    
    def _dag_to_generation_dict(
        self,
        dag: nx.DiGraph,
        dag_variables: list[str],
    ) -> Dict[str, Any]:
        """
        Convert a DAG to generation dict format for compatibility.
        
        Args:
            dag: NetworkX DiGraph or similar object with edges() method
            dag_variables: List of variable names
            
        Returns:
            Dict[str, Any]: BN structure in standard format
        """
        nodes = []
        edges = []
        
        for node in dag_variables:
            parents = [p for p in dag.predecessors(node)]
            nodes.append({
                "node_id": dag_variables.index(node),
                "node_name": node,
                "parents": parents,
                "description": f"Variable {node}",
                "distribution": "discrete",
                "conditional_probability_table": "To be estimated"
            })
        
        for edge in dag.edges():
            edges.append({
                "from": edge[0],
                "to": edge[1],
                "justification": f"Learned from data using {self.name}"
            })
        
        return {
            "bn": {
                "nodes": nodes,
                "edges": edges,
                "network_summary": f"Bayesian network learned using {self.name} with {len(nodes)} variables and {len(edges)} edges"
            }
        }