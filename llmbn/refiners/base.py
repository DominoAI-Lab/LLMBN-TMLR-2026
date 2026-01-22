"""
Base interface for Bayesian Network structure refiners.

Defines the common API for both LLM-enhanced and traditional refiners that optimize existing BN structures using
observation data.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd


class BaseRefiner(ABC):
    """
    Abstract base class for Bayesian Network structure refiners.
    
    Attributes:
        logger: Logger instance for output
    """
    
    def __init__(
        self,
        model: str,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize refiner with optional logger.
        
        Args:
            logger: Logger instance for output
        """
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the refiner implementation."""
        raise NotImplementedError("Subclasses must implement .name()")
    
    @abstractmethod
    def run(
        self,
        desc_variables: str,
        dag_variables: list[str],
        dag: pd.DataFrame,
        observation: pd.DataFrame,
        init_generation: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Refine an existing Bayesian Network structure using observation data.
        
        Args:
            desc_variables: Variable descriptions in string format
            dag_variables: List of variable names
            dag: True DAG structure as adjacency matrix
            observation: Observed data for refinement
            init_generation: Initial BN structure to refine
            **kwargs: Additional refiner-specific parameters
            
        Returns:
            Dict with results including:
            - 'Matrix': Adjacency matrix of refined structure
            - 'Graph': NetworkX graph representation
            - 'Score': Final score achieved
            - Additional refiner-specific metrics
        """
    
    def _validate_inputs(
        self, 
        observation: pd.DataFrame, 
        init_generation: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Validate observation data and initial generation.
        
        Args:
            observation: Observation data to validate
            init_generation: Initial structure to validate
            
        Raises:
            ValueError: If observation data is invalid or insufficient
        """
        if observation is None or observation.empty:
            raise ValueError("Observation data cannot be None or empty")
        
        if len(observation.columns) < 2:
            raise ValueError("At least 2 variables are required for structure refinement")
        
        if init_generation is None:
            self.logger.warning(
                "No initial generation provided, starting from empty structure"
            )
        
        self.logger.info(
            "Input validation passed: %d variables, %d samples",
            len(observation.columns),
            len(observation)
        )
    
    def _log_results(
        self,
        results: Dict[str, Any],
    ) -> None:
        """
        Log refinement results to the configured logger.
        
        Args:
            results: Results dictionary containing metrics and structures
        """
        self.logger.info("Refiner %s completed", self.name)
        self.logger.info("Final score: %s", results.get('Score', 'N/A'))
        
        if 'Matrix' in results:
            matrix = results['Matrix']
            if hasattr(matrix, 'shape'):
                self.logger.info(
                    "Refined structure: %dx%d adjacency matrix",
                    matrix.shape[0],
                    matrix.shape[1],
                )
        
        if 'Graph' in results:
            graph = results['Graph']
            if hasattr(graph, 'edges'):
                self.logger.info(
                    "Number of edges in refined structure: %d",
                    len(graph.edges),
                )
