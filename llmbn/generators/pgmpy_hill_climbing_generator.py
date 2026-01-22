"""
Traditional hill climbing generator using pgmpy.

Implements structure generation using pgmpy's HillClimbSearch with BIC scoring.
"""

import logging
from typing import Optional, Dict, Any
import pandas as pd
from pgmpy.estimators import HillClimbSearch, BIC

from .base import BaseGenerator


class PgmpyHillClimbingGenerator(BaseGenerator):
    """
    Traditional hill climbing generator using pgmpy's HillClimbSearch.
    
    Generates Bayesian network structures from observation data using
    hill climbing search with BIC scoring.
    
    Attributes:
        logger: Logger instance for output
        max_iter: Maximum iterations for hill climbing
    """
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        max_iter: int = 20,
    ) -> None:
        """
        Initialize the hill climbing generator.
        
        Args:
            logger: Logger instance for output
            max_iter: Maximum number of iterations for hill climbing
        """
        super().__init__(model=None, logger=logger)
        self.max_iter = max_iter
    
    @property
    def name(self) -> str:
        """Name of the generator implementation."""
        return "PgmpyHillClimbingGenerator"
    
    def run(
        self,
        desc_variables: str,
        dag_variables: list[str],
        observation: Optional[pd.DataFrame] = None,
        generations: Optional[list] = None,
        **kwargs
    ) -> tuple[int, Dict[str, Any]]:
        """
        Generate a Bayesian Network structure using hill climbing.
        
        Args:
            desc_variables: Variable descriptions (unused, for compatibility)
            dag_variables: List of variable names
            observation: Observed data for structure learning
            generations: Previous generations (unused, for compatibility)
            **kwargs: Additional parameters (unused)
            
        Returns:
            Tuple containing:
            - validation_status: 1 if valid DAG, 0 otherwise
            - results_dict: Dictionary with generated structure and metrics
              including 'Generation' (structure representation) and 'Matrix'
              (adjacency matrix)
              
        Raises:
            ValueError: If observation data is missing or invalid
        """
        if observation is None:
            raise ValueError("Observation data is required for traditional generators")
        
        self._validate_inputs(observation)
        
        try:
            # Set up the hill climbing search
            hc = HillClimbSearch(observation)
            scoring_method = BIC(observation)
            
            # Perform hill climbing search
            learned_model = hc.estimate(
                scoring_method=scoring_method,
                max_indegree=4,
                max_iter=self.max_iter
            )
            
            # Convert to adjacency matrix
            matrix = self._dag_to_adjacency_matrix(
                dag=learned_model,
                dag_variables=dag_variables,
            )
            
            # Create a generation dict for compatibility
            generation = self._dag_to_generation_dict(
                dag=learned_model,
                dag_variables=dag_variables,
            )
            
            results = {
                'Generation': generation,
                'Matrix': matrix
            }
            
            self._log_results(results)
            return 1, results  # Always return 1 for valid DAG
            
        except Exception as e:
            self.logger.error("Error in hill climbing generation: %s", e)
            return 0, {'Generation': None, 'Matrix': None}
    