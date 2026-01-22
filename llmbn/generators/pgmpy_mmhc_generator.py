"""
MMHC generator using pgmpy.

Implements structure generation using pgmpy's Max-Min Hill Climbing algorithm.
"""

import logging
from typing import Optional, Dict, Any
import pandas as pd
from pgmpy.estimators import MmhcEstimator

from .base import BaseGenerator


class PgmpyMMHCGenerator(BaseGenerator):
    """
    MMHC generator using pgmpy's MmhcEstimator.
    
    Generates Bayesian network structures from observation data using
    Max-Min Hill Climbing algorithm, which combines constraint-based
    and score-based methods.
    
    Attributes:
        logger: Logger instance for output
    """
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the MMHC generator.
        
        Args:
            logger: Logger instance for output
        """
        super().__init__(model=None, logger=logger)
    
    @property
    def name(self) -> str:
        """Name of the generator implementation."""
        return "PgmpyMMHCGenerator"
    
    def run(
        self,
        desc_variables: str,
        dag_variables: list[str],
        observation: Optional[pd.DataFrame] = None,
        generations: Optional[list] = None,
        **kwargs
    ) -> tuple[int, Dict[str, Any]]:
        """
        Generate a Bayesian Network structure using MMHC.
        
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
            # Set up the MMHC estimator
            mmhc = MmhcEstimator(observation)
            
            # Perform MMHC estimation
            learned_model = mmhc.estimate()
            
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
            self.logger.error("Error in MMHC generation: %s", e)
            return 0, {'Generation': None, 'Matrix': None}
    