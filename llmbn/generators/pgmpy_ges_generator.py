"""
Greedy Equivalence Search (GES) generator using pgmpy.

Implements structure generation using pgmpy's GES algorithm.
"""

import logging
from typing import Optional, Dict, Any
import pandas as pd
from pgmpy.estimators import GES

from .base import BaseGenerator


class PgmpyGESGenerator(BaseGenerator):
    """
    GES generator using pgmpy's GES estimator.
    
    Generates Bayesian network structures from observation data using
    the Greedy Equivalence Search (GES) algorithm, a score-based method that
    works in three phases: forward phase (adding edges), backward phase (removing edges),
    and edge flipping phase (flipping edge orientations).
    
    Attributes:
        logger: Logger instance for output
        min_improvement: Minimum score improvement threshold for operations
        scoring_method: Scoring method for evaluating networks
    """
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        min_improvement: float = 1e-6,
        scoring_method: str = "bic-d",
    ) -> None:
        """
        Initialize the GES generator.
        
        Args:
            logger: Logger instance for output
            min_improvement: Minimum score improvement threshold for operations
            scoring_method: Scoring method for evaluating networks
                Supported: k2, bdeu, bds, bic-d, aic-d, ll-g, aic-g, bic-g, ll-cg, aic-cg, bic-cg
        """
        super().__init__(model=None, logger=logger)
        self.min_improvement = min_improvement
        self.scoring_method = scoring_method
    
    @property
    def name(self) -> str:
        """Name of the generator implementation."""
        return "PgmpyGESGenerator"
    
    def run(
        self,
        desc_variables: str,
        dag_variables: list[str],
        observation: Optional[pd.DataFrame] = None,
        generations: Optional[list] = None,
        **kwargs
    ) -> tuple[int, Dict[str, Any]]:
        """
        Generate a Bayesian Network structure using GES algorithm.
        
        Args:
            desc_variables: Variable descriptions (unused, for compatibility)
            dag_variables: List of variable names
            observation: Observed data for structure learning
            generations: Previous generations (unused, for compatibility)
            **kwargs: Additional parameters:
                - scoring_method: Override default scoring method
                - min_improvement: Override default min improvement threshold
            
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
        
        # Override defaults with kwargs if provided
        scoring_method = kwargs.get('scoring_method', self.scoring_method)
        min_improvement = kwargs.get('min_improvement', self.min_improvement)
        
        try:
            # Set up the GES estimator
            ges = GES(observation)
            
            # Perform GES estimation
            learned_model = ges.estimate(
                scoring_method=scoring_method,
                min_improvement=min_improvement,
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
            self.logger.error("Error in GES generation: %s", e)
            return 0, {'Generation': None, 'Matrix': None}
