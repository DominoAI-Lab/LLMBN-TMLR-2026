import json
import logging
from typing import Any, Dict, Optional
import pandas as pd

from ..utils.graph_utils import construct_matrix_from_nodes
from .base import BaseGenerator
from ..errors.generation_error import ParseResponseError, MaxRetriesError

BASELINE_PROMPT = """
You are an expert in building Bayesian Networks. You will receive variables in table format with columns [node, var_name, var_description, var_distribution]. Your task is to construct a Bayesian Network as a Directed Acyclic Graph (DAG) based on the table following the instructions below.

[Instructions]
1. Parse the table to understand:
   - The node id (node).
   - The variable name (var_name).
   - The semantic meaning of each variable (var_description).
   - The distribution type (var_distribution).

2. Construct a Bayesian Network as a Directed Acyclic Graph (DAG) by:
   - Proposing parents (if any) for each node based on relevant domain knowledge or any clues from the table.
   - Ensuring no cycles exist.

3. Provide a strict JSON object with the following structure:
   {{
     "bn": {{
       "nodes": [
         {{
           "node_id": <integer>,
           "node_name": <string>,
           "parents": [<string>, ...],
           "description": <string>,
           "distribution": <string>,
           "conditional_probability_table": <string>
         }},
         ...
       ],
       "edges": [
         {{
           "from": <string>,
           "to": <string>,
           "justification": <string>
         }},
         ...
       ],
       "network_summary": <string>
     }}
   }}


[Output Format]
1. In "nodes":
   - "node_id": A unique ID or index for each node.
   - "node_name": The variable name from var_name column.
   - "parents": Array of node_name values for parent nodes.
   - "description": Short text from var_description.
   - "distribution": Data from var_distribution column.
   - "conditional_probability_table": e.g. "P(tub | asia)".

2. In "edges":
   - "from" and "to": References to the "node_name" fields, indicating parent-child relationships.
   - "justification": A concise reason for why this relationship exists.

3. "network_summary": A concise explanation of how the Bayesian Network structure was derived.

4. Output ONLY the valid JSON object, with no additional commentary or text.

[Variables]
{desc_variables}
"""

class PromptBNGenerator(BaseGenerator):
    """
    LLM-based Bayesian Network generator using prompt engineering.
    
    Creates networks from variable descriptions without observation data
    (data-free generation) by leveraging LLM domain knowledge.
    
    Attributes:
        model: Name of the LLM model to use
        logger: Logger instance for output
        prompt: Template prompt for BN generation
        messages: Message history for LLM
    """

    def __init__(
        self,
        model: str,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize PromptBN generator with model and logger.
        
        Args:
            model: Name of the LLM model to use
            logger: Logger instance for output
        """
        super().__init__(model, logger)
        self.prompt = BASELINE_PROMPT
        self.messages = []
        self.logger.debug("[Model] %s", self.model)
        self.logger.debug("[Client] %s", type(self.client))
    
    @property
    def name(self) -> str:
        """Name of the generator implementation."""
        return "PromptBNGenerator"

    def prepare_user_prompt(self, desc_variables: str, dag_variables: list[str]) -> str:
        """
        Format the user prompt with variable descriptions.
        
        Args:
            desc_variables: Variable descriptions in string format
            dag_variables: List of variable names (unused)
            
        Returns:
            str: Formatted prompt for LLM
        """
        prompt = self.prompt.format(desc_variables=desc_variables)
        return prompt

    def prepare_messages(
        self,
        desc_variables: str,
        dag_variables: list[str],
    ) -> list[dict]:
        """
        Prepare the message list for the language model.
        
        Args:
            desc_variables: Variable descriptions in string format
            dag_variables: List of variable names
            
        Returns:
            list[dict]: Messages formatted for LLM API
        """
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant that constructs Bayesian Networks."},
            {"role": "user", "content": self.prepare_user_prompt(desc_variables, dag_variables)}
        ]
        self.logger.debug("[Prompt] %s", self.messages[-1]['content'])
        return self.messages

    def construct_matrix(
        self,
        generation: Dict[str, Any],
        dag_variables: list[str],
    ) -> pd.DataFrame:
        return construct_matrix_from_nodes(
            generation=generation,
            dag_variables=dag_variables,
        )

    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the response from the language model into structured BN representation.
        
        Args:
            response: Raw text response from LLM
            
        Returns:
            Dict[str, Any]: Structured BN representation
            
        Raises:
            ParseResponseError: If response cannot be parsed as JSON
        """
        # Remove wrapping triple backticks (with or without 'json')
        content = response.replace("```json\n", "").replace("\n```", "").strip()
        try:
            return json.loads(content, strict=False)
        except json.JSONDecodeError as e:
            self.logger.debug("Unjsonified Raw Generation:\n```json\n%s\n```", response)
            self.logger.exception("JSON Decode Error: %s", e)
            raise ParseResponseError(f"JSON Decode Error: {e}") from e
    
    def run(
        self,
        desc_variables: str,
        dag_variables: list[str],
        observation: Optional[pd.DataFrame] = None,
        generations: Optional[list] = None,
        **kwargs
    ) -> tuple[int, Dict[str, Any]]:
        """
        Generate a Bayesian Network structure using LLM.
        
        Args:
            desc_variables: Variable descriptions in string format
            dag_variables: List of variable names
            observation: Observed data (unused for LLM-based generation)
            generations: Previous generations (unused)
            **kwargs: Additional parameters (unused)
            
        Returns:
            Tuple containing:
            - validation_status: 1 if valid DAG, 0 otherwise
            - results_dict: Dictionary with generated structure and metrics
              including 'Generation' (structure representation) and 'Matrix'
              (adjacency matrix)
              
        Raises:
            ParseResponseError: If LLM response cannot be parsed
            MaxRetriesError: If max retries reached
        """
        self.messages = self.prepare_messages(desc_variables, dag_variables)
        try:    
            generation = self.generate()
        except ParseResponseError as e:
            self.logger.exception("Parse Response Error: %s", e)
            raise e
        except MaxRetriesError as e:
            self.logger.exception("Max Retries Error: %s; Retrying...", e)
            raise e
        
        try:
            matrix = self.construct_matrix(generation, dag_variables)
        except Exception as e:
            self.logger.exception("Invalid DAG Error: %s", e)
            raise e
        else:
            self.logger.debug("Valid DAG, Raw generation:\n%s", generation)
            self.logger.debug("Constructed Matrix:\n%s", matrix)
        
        results = {
            "Generation": generation,
            "Matrix": matrix
        }
        return 1, results