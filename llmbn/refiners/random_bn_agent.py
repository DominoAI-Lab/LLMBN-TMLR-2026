import logging
import random
from collections import deque
from typing import Optional, Tuple
import networkx as nx
import numpy as np
import pandas as pd
import copy
import itertools
from joblib import Parallel, delayed

from pgmpy.estimators import BIC, BDeu
from pgmpy.models import BayesianNetwork

from llmbn.errors.agent_error import StateUpdateError
from llmbn.utils.eval_utils import evaluate_generation
from llmbn.utils.graph_utils import (
    construct_discrete_bn_from_bn_dict,
    initialize_empty_graph,
    adjacency_df_to_bn,
    evaluate_operation,
)
from llmbn.utils.score_utils import LocalScoreCache
from .base import BaseRefiner   
from .react_bn_agent import ReActBNAgent

THRESHOLD = 1e-6


class RandomBNAgent(ReActBNAgent):
    """
    Random Move Bayesian Network refiner
    --------------------------------------------------------
    A baseline refiner that uses purely random decisions instead
    of LLM-based reasoning. It mirrors the entire ReActBNAgent workflow
    (candidate generation, scoring, tabu constraints, update_state, SHD/NHD
    logging, etc.) but replaces the LLM with a random selector.
    
    All structure, scoring, Tabu constraints, state updates, SHD/NHD calculations
    remain consistent with the ReAct-LLM version. The only difference is that
    "the next action is completely randomly selected", including random selection
    of termination actions.
    """

    def __init__(self, model: str, logger: logging.Logger = None):
        super().__init__(model=model, logger=logger)

    @property
    def name(self):
        return "RandomBNAgent"

"""
LLM-enhanced Bayesian Network refiner using ReAct framework.

Implements the ReAct (Reason + Act) approach for BN structure refinement,
combining LLM reasoning with traditional search to make intelligent
decisions about graph modifications based on observation data.
"""

import logging
import re
import json
import copy
import pandas as pd
import random
import pprint
import networkx as nx
import itertools
import numpy as np
from pgmpy.base import DAG
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BIC
from pgmpy.estimators import BDeu
from typing import Optional, Tuple
import itertools
from collections import deque
from joblib import Parallel, delayed

from llmbn.llm import LLMClient
from llmbn.errors.agent_error import StateUpdateError
from llmbn.utils.eval_utils import evaluate_generation
from llmbn.utils.graph_utils import (
    construct_discrete_bn_from_bn_dict,
    initialize_empty_graph,
    adjacency_df_to_bn,
    evaluate_operation,
)
from llmbn.utils.score_utils import LocalScoreCache

from .react_bn_agent import ReActBNAgent


class RandomBNAgent(ReActBNAgent):

    def __init__(
        self,
        model: str,
        logger: logging.Logger = None,
    ) -> None:
        super().__init__(model, logger)
    
    @property
    def name(self) -> str:
        return "RandomBNAgent"

    def random_act(
        self,
        state: dict,
        action_space: list,
    ) -> Tuple[str, int, float]:
        """
        Use random action selector to reason and act.
        """
        k = len(action_space)
        choices = list(range(-1, k))  # include termination
        action_idx = random.choice(choices)

        fake_reasoning = f"Randomly selected action index {action_idx}."

        return fake_reasoning, action_idx, 1.0

    def run(
        self,
        desc_variables: str,
        dag_variables: list[str],
        dag: pd.DataFrame,
        observation: pd.DataFrame,
        init_generation: dict = None,
        tabu_length: int = 100,
        epsilon: float = 1e-4,
        max_iter: int = 20,
        max_indegree: int = None,
        forbidden_edges: list = None,
        required_edges: list = None,
        scoring_method: str = 'bic',
        top_k: int = 10,
    ) -> dict:
        """
        Refine BN structure using LLM-guided search with ReAct framework.
        
        Args:
            desc_variables: Variable descriptions in string format
            dag_variables: List of variable names
            dag: True DAG structure as adjacency matrix
            observation: Observed data for refinement
            init_generation: Initial BN structure to refine
            tabu_length: Length of tabu list to prevent cycles
            epsilon: Minimum score improvement threshold
            max_iter: Maximum number of iterations
            max_indegree: Maximum number of parents per node
            forbidden_edges: List of (from, to) tuples that cannot be added
            required_edges: List of (from, to) tuples that cannot be removed
            scoring_method: Scoring function ('bic' or 'bdeu')
            top_k: Number of top candidates to present to LLM
            
        Returns:
            Dict with results including:
            - 'Matrix': Adjacency matrix of refined structure
            - 'Graph': NetworkX graph representation
            - 'Score': Final score achieved
            - 'ActionHistory': List of actions taken
            - 'MetricsHistory': Metrics at each iteration
        """
        self.history.clear()  # Clear action history
        self.nodes = dag_variables
        self.ref_graph = adjacency_df_to_bn(dag, dag_variables)
        
        # Algorithm parameters
        self._load_scorer(scoring_method, observation)  # Load scorer
        forbidden_edges = set(forbidden_edges) if forbidden_edges else set()
        required_edges = set(required_edges) if required_edges else set()
        tabu_list = deque(maxlen=tabu_length)

        # Initialize graph
        if init_generation:
            current_graph = construct_discrete_bn_from_bn_dict(
                generation_dict=init_generation,
                dag_variables=dag_variables,
                logger=self.logger,
            )
            for node in self.nodes:
                current_graph.add_node(node)
            self.logger.info(f"#Egdes in the initial graph: {len(current_graph.edges())}")
        else:
            current_graph = initialize_empty_graph(dag_variables)
            self.logger.info(f"Initialized empty graph with {len(dag_variables)} nodes")

        # Initialize score cache
        cache = LocalScoreCache()

        # Initialize score
        current_score = 0.0
        for node in self.nodes:
            parents = set(current_graph.predecessors(node))
            current_score += cache.get_local_score(node, parents, self.scorer)
        
        # Initialize state
        state = self._initialize_state(current_graph, current_score,desc_variables)
        
        # Initialize iteration conditions
        iteration = 0
        done = False
        best_op = None
        best_score_delta = None
        self.logger.info(
            "Starting ReActBN with SHD=%s, NHD=%s, score=%s",
            state['shd'], state['nhd'], state['score']
        )
        metrics_history = []
        
        # --- Main loop: Hill climbing ---
        while not done and iteration < max_iter:
            iteration += 1
            # --- Main loop: Generate all legal operations ---
            legal_ops = self._generate_legal_operations(
                state['graph'], self.nodes, tabu_list, max_indegree, forbidden_edges, required_edges
            )
            
            # --- Main loop: Evaluate all legal operations in parallel using joblib ---
            def evaluate_op(op):
                return evaluate_operation(
                    current_graph=state['graph'],
                    operation=op,
                    cache=cache,
                    scorer=self.scorer,
                    current_score=state['score'],
                    logger=self.logger,
                )
            if legal_ops:
                results = Parallel(n_jobs=-1, prefer='threads')(
                    delayed(evaluate_op)(op) for op in legal_ops
                )
                self.logger.info(f"#Legal operations: {len(legal_ops)}")
                self.logger.info(f"#Results: {len(results)}")
                # self.logger.info(
                #     "Sample of Operation results: %s", "\n".join([str(res) for res in legal_ops[:2]])
                # )
                # self.logger.info(
                #     "Sample of Op evaluation results: %s", "\n".join([str(res) for res in results[:2]])
                # )
                op_results = list(zip(legal_ops, results))
                
                op_results.sort(key=lambda x: x[1], reverse=True)
                top_k_ops = op_results[:top_k]
                action_space = [
                    {
                        "type": op[0],
                        "from": op[1][0],
                        "to": op[1][1],
                        "score_delta": res - current_score
                    }
                    for op, res in top_k_ops
                ]
                state_for_llm = {
                    "graph": state['graph'],
                    "score": state['score'],
                    "desc_variables": state['desc_variables'],
                    "history": list(self.history),
                }
                self.logger.info(
                    "Action Space: %s",
                    json.dumps(action_space, indent=4)
                )
                reasoning, action_idx, confidence = self.random_act(
                    state_for_llm,
                    action_space,
                )
                self.logger.info(f"Action index: {action_idx}, Confidence: {confidence}")
                self.logger.info(f"Reasoning: {reasoning}")
                
                # Enforce a minimum number of iterations (3) before allowing termination
                min_iterations_before_termination = 3
                
                # If no positive score deltas, allow termination regardless of iteration count
                if action_idx == -1 and iteration >= min_iterations_before_termination:
                    self.logger.info(
                        f"Iteration {iteration}: LLM called for termination (Action index: -1). Confidence: {confidence}"
                    )
                    self.logger.debug(f"Reasoning: {reasoning}")
                    current_nhd, current_shd = self._compute_and_log_metrics(
                        state['graph'], iteration, state['score'], None,
                    )
                    reward = 0.0
                    self._log_step(
                        metrics_history=metrics_history,
                        action_history=self.history,
                        iteration=iteration,
                        current_score=state['score'],
                        current_nhd=current_nhd,
                        current_shd=current_shd,
                        action=None, action_idx=None, reward=None, confidence=None, reasoning=None, legal_ops_len=0,
                        terminated=True,
                    )
                    done = True
                    action = None
                    break
                elif action_idx == -1 and iteration < min_iterations_before_termination:
                    self.logger.info(
                        f"Iteration {iteration}: LLM called for termination (Action index: -1) before minimum iterations. Forcing continuation by selecting best action"
                    )
                    # Force selection of the best action (index 0) instead of terminating
                    action_idx = 0
                elif action_idx >= len(top_k_ops):
                    self.logger.info(
                        f"Invalid action index: {action_idx}. Skipping this iteration."
                    )
                    self._log_step(
                        metrics_history=metrics_history,
                        action_history=self.history,
                        iteration=iteration,
                        current_score=state['score'],
                        current_nhd=current_nhd,
                        current_shd=current_shd,
                        action=None, action_idx=None, reward=None, confidence=None, reasoning="LLM responded with an invalid action index. Skip this iteration.", legal_ops_len=len(legal_ops),
                        terminated=False,
                    )
                    continue
                
                chosen_op, chosen_result = top_k_ops[action_idx]
                self.logger.info(f"Chosen op: {chosen_op}")
                self.logger.info(f"Chosen result: {chosen_result}")
                
                best_op = chosen_op
                best_new_score = chosen_result
                best_score_delta = chosen_result - current_score
                self.logger.info(
                    f"Iteration {iteration}: selected_op={best_op}, selected_action_idx={action_idx}, score_delta={best_score_delta}, confidence={confidence}"
                )
            else:
                best_op = None
                best_score_delta = 0.0
                best_new_score = current_score
                reward = 0.0
                self.logger.info(f"Iteration {iteration}: no legal ops found")
                current_nhd, current_shd = self._compute_and_log_metrics(
                    state['graph'], iteration, state['score'], None,
                )
                self._log_step(
                    metrics_history=metrics_history,
                    action_history=self.history,
                    iteration=iteration,
                    current_score=state['score'],
                    current_nhd=current_nhd,
                    current_shd=current_shd,
                    action=None, action_idx=None, reward=0.0, confidence=1.0, reasoning="No legal ops found. Terminating.", legal_ops_len=0, terminated=True,
                )
                break

            # --- Main loop: Stopping condition ---
            if best_op is None or best_score_delta < epsilon:
                self.logger.info(
                    f"Stopping: no legal best operation found or score delta {best_score_delta} is less than epsilon ({epsilon})."
                )
                done = True
                action = None
                reward = 0.0
                break
            
            # --- Main loop: Apply best operation in-place ---
            self._apply_operation_in_place(state, best_op, tabu_list)
            
            # --- Main loop: Update state ---
            op_type, (parent, child) = best_op
            action = {'type': op_type, 'from': parent, 'to': child}
            for node in self.nodes:
                if node not in state['graph']:
                    state['graph'].add_node(node)
            current_nhd, current_shd = self._compute_and_log_metrics(
                state['graph'], iteration, best_new_score, confidence,
            )
            action_result = {
                'graph': state['graph'],
                'score': best_new_score,
                'shd': current_shd,
                'nhd': current_nhd,
            }
            reward = best_score_delta
            state = self.update_state(
                state,
                action,
                action_result,
                reward,
            )
            # --- Main loop: Log step ---
            self._log_step(
                metrics_history=metrics_history,
                action_history=self.history,
                iteration=iteration,
                current_score=state['score'],
                current_nhd=current_nhd,
                current_shd=current_shd,
                action=action,
                action_idx=action_idx,
                reward=reward,
                confidence=confidence,
                reasoning=reasoning,
                legal_ops_len=len(legal_ops),
                terminated=False,
            )
        
        self.logger.info(
            f"Hill climbing finished after {iteration} iterations. Final score: {state['score']}")
        self.logger.info(
            f"Local score cache: {cache.hits} hits, {cache.misses} misses.")
        result = self._finalize_result(
            state, action, state['graph'], state['score'], state['shd'], state['nhd'], iteration, reward, metrics_history,
        )
        result["ActionHistory"] = self.history
        return result

    
        """
        Computes the reward as the change in score after taking the action.
        """
        new_score = action_result["score"]
        current_score = state["score"]
        return new_score - current_score

        """
        Returns all valid add/delete/reverse actions for the given node, each with the resulting parent set and local score.
        Does not modify the original graph.
        """
        actions = []
        current_parents = set(graph.get_parents(node))
        all_nodes = set(graph.nodes())
        # Try adding a parent
        for potential_parent in all_nodes - {node} - current_parents:
            new_graph = copy.deepcopy(graph)
            new_graph.add_edge(potential_parent, node)
            if nx.is_directed_acyclic_graph(new_graph):
                new_parents = set(current_parents | {potential_parent})
                local_score = self.scorer.local_score(node, list(new_parents))
                actions.append({
                    "type": "add",
                    "from": potential_parent,
                    "to": node,
                    "local_score": local_score,
                    "new_parents": list(new_parents),
                })
        # Try deleting a parent
        for parent in current_parents:
            new_graph = copy.deepcopy(graph)
            new_graph.remove_edge(parent, node)
            new_parents = set(current_parents - {parent})
            local_score = self.scorer.local_score(node, list(new_parents))
            actions.append({
                "type": "delete",
                "from": parent,
                "to": node,
                "local_score": local_score,
                "new_parents": list(new_parents),
            })
        # Try reversing edges (if allowed)
        for parent in current_parents:
            if graph.has_edge(node, parent):
                continue  # skip if reverse already exists
            new_graph = copy.deepcopy(graph)
            new_graph.remove_edge(parent, node)
            new_graph.add_edge(node, parent)
            if nx.is_directed_acyclic_graph(new_graph):
                # For the reversed edge, local BIC for node is with parent removed
                new_parents = set(current_parents - {parent})
                local_score = self.scorer.local_score(node, list(new_parents))
                actions.append({
                    "type": "reverse",
                    "from": parent,
                    "to": node,
                    "local_score": local_score,
                    "new_parents": list(new_parents),
                })
        return actions

        legal_ops = []
        # Generate all possible add operations
        for i, j in itertools.permutations(nodes, 2):
            if current_graph.has_edge(i, j) or i == j:
                continue
            if (i, j) in forbidden_edges:
                continue
            added = False
            try:
                if not current_graph.has_edge(i, j):
                    current_graph.add_edge(i, j)
                    added = True
                if not nx.is_directed_acyclic_graph(current_graph):
                    if added and current_graph.has_edge(i, j):
                        current_graph.remove_edge(i, j)
                    continue
                if max_indegree is not None and len(list(current_graph.predecessors(j))) > max_indegree:
                    if added and current_graph.has_edge(i, j):
                        current_graph.remove_edge(i, j)
                    continue
            except Exception:
                if added and current_graph.has_edge(i, j):
                    current_graph.remove_edge(i, j)
                continue
            op = ('+', (i, j))
            if op not in tabu_list:
                legal_ops.append(op)
            if added and current_graph.has_edge(i, j):
                current_graph.remove_edge(i, j)
        # Generate all possible remove operations
        for i, j in list(current_graph.edges()):
            if (i, j) in required_edges:
                continue
            op = ('-', (i, j))
            if op not in tabu_list:
                legal_ops.append(op)
        # Generate all possible flip operations
        for i, j in list(current_graph.edges()):
            if current_graph.has_edge(j, i):
                continue  # skip if reverse already exists
            if (j, i) in forbidden_edges or (i, j) in required_edges:
                continue
            removed = False
            added = False
            try:
                if current_graph.has_edge(i, j):
                    current_graph.remove_edge(i, j)
                    removed = True
                if not current_graph.has_edge(j, i):
                    current_graph.add_edge(j, i)
                    added = True
                if not nx.is_directed_acyclic_graph(current_graph):
                    if added and current_graph.has_edge(j, i):
                        current_graph.remove_edge(j, i)
                    if removed and not current_graph.has_edge(i, j):
                        current_graph.add_edge(i, j)
                    continue
                if max_indegree is not None and len(list(current_graph.predecessors(i))) > max_indegree:
                    if added and current_graph.has_edge(j, i):
                        current_graph.remove_edge(j, i)
                    if removed and not current_graph.has_edge(i, j):
                        current_graph.add_edge(i, j)
                    continue
            except Exception:
                if added and current_graph.has_edge(j, i):
                    current_graph.remove_edge(j, i)
                if removed and not current_graph.has_edge(i, j):
                    current_graph.add_edge(i, j)
                continue
            op = ('flip', (i, j))
            if op not in tabu_list and ('flip', (j, i)) not in tabu_list:
                legal_ops.append(op)
            if added and current_graph.has_edge(j, i):
                current_graph.remove_edge(j, i)
            if removed and not current_graph.has_edge(i, j):
                current_graph.add_edge(i, j)
        return legal_ops

        pred_adj = nx.to_numpy_array(
            current_graph, nodelist=self.nodes, weight=None, dtype=int)
        ref_adj = nx.to_numpy_array(
            self.ref_graph, nodelist=self.nodes, weight=None, dtype=int)
        _, _, _, _, current_nhd, current_shd = evaluate_generation(
            ref_adj, pred_adj, self.logger, self.nodes,
        )

        self.logger.info(
            f"Iteration {iteration} - Current Score: {current_score}, SHD={current_shd}, NHD={current_nhd}"
        )
        return current_nhd, current_shd

        # Track state
        state = {
            'graph': current_graph,
            'score': current_score,
            'dag_variables': dag_variables,
            'desc_variables': desc_variables,
            'move_count': iteration,
            'shd': current_shd,
            'nhd': current_nhd,
        }
        action = {'type': op_type, 'from': parent, 'to': child}
        reward = best_score_delta

        # Track action
        self.history.append(
            {'action': action, 'reward': reward, 'shd': current_shd, 'nhd': current_nhd})
        return state, action, reward

        """
        Applies the given operation (add, remove, flip) in-place to the graph in the state dict and updates the tabu list.
        """
        op_type, (parent, child) = best_op
        if op_type == '+':
            state['graph'].add_edge(parent, child)
            tabu_list.append(('-', (parent, child)))
        elif op_type == '-':
            if state['graph'].has_edge(parent, child):
                state['graph'].remove_edge(parent, child)
            tabu_list.append(('+', (parent, child)))
        elif op_type == 'flip':
            if state['graph'].has_edge(parent, child):
                state['graph'].remove_edge(parent, child)
            state['graph'].add_edge(child, parent)
            tabu_list.append(('flip', (parent, child)))
        else:
            raise ValueError(f"Unknown operation type: {op_type}")