import numpy as np
import pandas as pd
import logging
import copy
import networkx as nx

from collections import defaultdict, deque
from pgmpy.models import DiscreteBayesianNetwork

from rpy2.robjects.packages import importr
from rpy2.robjects import r, default_converter
from rpy2.robjects.conversion import localconverter
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects.pandas2ri as pandas2ri

import json

from llmbn.errors.generation_error import InvalidDAGError, ParseResponseError

# Ensure R has the required Bioconductor packages installed:
# In R console:
# if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager")
# BiocManager::install(c("graph","RBGL","pcalg"))

# Load Bioconductor 'graph' and 'pcalg' packages
graph_pkg = importr("graph")
pcalg     = importr("pcalg")

def df_to_cpdag_pcalg(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Convert a DAG adjacency matrix (DataFrame) to its CPDAG using R's pcalg via rpy2.

    Parameters
    ----------
    df : pd.DataFrame
        Binary adjacency matrix (0/1) of a DAG.

    Returns
    -------
    pd.DataFrame
        Adjacency matrix of the CPDAG (1: directed edge, 2: undirected edge, 0: no edge).
    """
    n = df.shape[0]
    
    # 1) Convert pandas DataFrame to R matrix and assign
    with localconverter(default_converter + pandas2ri.converter + numpy2ri.converter):
        amat = r.matrix(df.to_numpy(), nrow=n, ncol=n)
        r.assign("amat", amat)

    # 2) Build graphNEL from graphAM and compute CPDAG
    r("library(graph)")
    r("library(pcalg)")
    r("g_AM    <- new('graphAM', adjMat=amat, edgemode='directed')")
    r("g_dag   <- as(g_AM, 'graphNEL')")
    r("g_cpdag <- dag2cpdag(g_dag)")
    r("cpdag_mat <- as(g_cpdag, 'matrix')")

    # 3) Convert R matrix back to NumPy array
    with localconverter(default_converter + numpy2ri.converter):
        cpdag_np = np.array(r("cpdag_mat"))
    
    # 4) Post-process: mark bidirectional edges (1,1) as undirected (2)
    for i in range(n):
        for j in range(i + 1, n):
            if cpdag_np[i, j] == 1 and cpdag_np[j, i] == 1:
                cpdag_np[i, j] = 2
                cpdag_np[j, i] = 2

    return pd.DataFrame(cpdag_np, index=df.index, columns=df.columns)


def construct_matrix_from_bn_dict(bn_dict, dag_variables):
    if "bn" not in bn_dict:
        raise ParseResponseError("bn_dict does not contain 'bn' key")

    if "nodes" in bn_dict['bn']:
        matrix = construct_matrix_from_nodes(bn_dict, dag_variables)
    elif "edges" in bn_dict['bn']:
        matrix = construct_matrix_from_edges(bn_dict, dag_variables)
    else:
        raise ParseResponseError("bn_dict does not contain 'nodes' or 'edges' key")
    
    if not set(matrix.columns) == set(dag_variables):
        raise InvalidDAGError("matrix has different variables from dag_variables")
    if not validate_dag(matrix):
        raise InvalidDAGError("matrix is not a DAG")
    return matrix

def construct_matrix_from_edges(generation, dag_variables):
    """Construct adjacency matrix from edge list."""
    edges = generation['bn']['edges']
    matrix = pd.DataFrame(
        np.zeros((len(dag_variables), len(dag_variables))),
        columns=dag_variables,
        index=dag_variables
    )
    for edge in edges:
        if not edge['from'] in dag_variables:
            raise InvalidDAGError(f"predicted node '{edge['from']}' is not valid variable")
        if not edge['to'] in dag_variables:
            raise InvalidDAGError(f"predicted node '{edge['to']}' is not valid variable")
        matrix.at[edge['from'], edge['to']] = 1
    return matrix

def construct_matrix_from_nodes(generation, dag_variables):
    """Construct adjacency matrix from node parent relationships."""
    nodes = generation['bn']['nodes']
    matrix = pd.DataFrame(
        np.zeros((len(dag_variables), len(dag_variables))),
        columns=dag_variables,
        index=dag_variables
    )
    for node_b in nodes:
        node_b_name = node_b['node_name']
        if not node_b_name in dag_variables:
            raise InvalidDAGError(f"predicted node '{node_b_name}' is not valid variable")
        if len(node_b['parents']) > 0:
            for node_a_name in node_b['parents']:
                if not node_a_name in dag_variables:
                    raise InvalidDAGError(f"predicted node '{node_a_name}' is not valid variable")
                matrix.at[node_a_name, node_b_name] = 1
    return matrix

def validate_dag(matrix_df):
    """Check if the adjacency matrix represents a DAG (no cycles)."""
    def has_cycle(graph, node, visited, rec_stack):
        visited[node] = True
        rec_stack[node] = True
        for neighbor in range(len(graph)):
            if graph[node][neighbor] != 0:
                if not visited[neighbor]:
                    if has_cycle(graph, neighbor, visited, rec_stack):
                        return True
                elif rec_stack[neighbor]:
                    return True
        rec_stack[node] = False
        return False
    matrix = matrix_df.to_numpy().astype(int)
    num_nodes = len(matrix)
    visited = [False] * num_nodes
    rec_stack = [False] * num_nodes
    for node in range(num_nodes):
        if not visited[node]:
            if has_cycle(matrix, node, visited, rec_stack):
                raise InvalidDAGError("Graph has a cycle.")
    return True 

def topo_sort_from_ancestor_descendant(pairs):
    # Build graph and in-degree count from constraints
    graph = defaultdict(set)
    in_degree = defaultdict(int)
    nodes = set()
    for ancestor, descendant in pairs:
        if descendant not in graph[ancestor]:
            graph[ancestor].add(descendant)
            in_degree[descendant] += 1
        nodes.add(ancestor)
        nodes.add(descendant)
    for node in nodes:
        in_degree.setdefault(node, 0)
    # Kahn's algorithm
    queue = deque([node for node in nodes if in_degree[node] == 0])
    ordering = []
    while queue:
        node = queue.popleft()
        ordering.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    if len(ordering) != len(nodes):
        raise ValueError("Cycle detected! No valid topological ordering.")
    return ordering

def minimal_dag_from_ad_pairs(ad_pairs):
    # Step 1: Build the adjacency list
    graph = defaultdict(set)
    nodes = set()
    for a, d in ad_pairs:
        graph[a].add(d)
        nodes.add(a)
        nodes.add(d)
    
    # Step 2: For each edge, check if it's transitive
    def has_alternate_path(start, end, skip):
        """DFS to check if there's a path from start to end, skipping the direct edge (skip)."""
        stack = [start]
        visited = set()
        while stack:
            node = stack.pop()
            if node == end:
                return True
            for neighbor in graph[node]:
                if (node, neighbor) == skip or neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)
        return False

    # Collect all edges to check
    edges = [(a, d) for a in graph for d in graph[a]]
    # Remove transitive edges
    minimal_edges = set(edges)
    for a, d in edges:
        if has_alternate_path(a, d, (a, d)):
            minimal_edges.discard((a, d))
    
    # Build the minimal DAG adjacency list
    minimal_graph = defaultdict(set)
    for a, d in minimal_edges:
        minimal_graph[a].add(d)
    
    minimal_dag_dict = convert_to_bn_dict_format(minimal_graph)
    return minimal_dag_dict

def convert_to_bn_dict_format(minimal_graph):
    
    # Build nodes list
    nodes = []
    for name in minimal_graph.keys():
        nodes.append({
            "node_name": name,
            "parents": [parent for parent in minimal_graph.keys() if name in minimal_graph[parent]],
        })
    
    # Build edges list
    edges = []
    for parent, children in minimal_graph.items():
        for child in children:
            edges.append({
                "from": parent,
                "to": child,
            })
    
    # Compose the final structure
    dag_dict = {
        "bn": {
            "nodes": nodes,
            "edges": edges,
        }
    }
    return dag_dict


def construct_discrete_bn_from_bn_dict(
    generation_dict: dict,
    dag_variables: list,
    logger: logging.Logger,
) -> DiscreteBayesianNetwork:
    """
    Convert a generation dictionary to a discrete Bayesian network.
    
    Args:
        generation_dict: Dictionary containing the network structure
        dag_variables: List of variable names
        
    Returns:
        DiscreteBayesianNetwork object
    """
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.base import DAG
    
    
    # Create DAG from generation dict
    edges = []
    
    if "bn" in generation_dict:
        bn_dict = generation_dict['bn']
    else:
        bn_dict = generation_dict
    logger.debug(f"BN dict: %s", bn_dict)
    
    for edge in bn_dict['edges']:
        child: str = edge['to']
        parent: str = edge['from']
        edges.append((parent, child))
    
    dag = DAG(edges)
    
    # Create discrete BN
    bn = DiscreteBayesianNetwork(dag)
    
    # Ensure all nodes are in the BN
    for node in dag_variables:
        if node not in bn.nodes():
            bn.add_node(node)
    return bn


def initialize_empty_graph(dag_variables):
    """
    Initialize an empty graph with given variables.
    
    Args:
        dag_variables: List of variable names
        
    Returns:
        NetworkX DiGraph object
    """
    import networkx as nx
    G = nx.DiGraph()
    G.add_nodes_from(dag_variables)
    return G


def adjacency_df_to_bn(adjacency_df, dag_variables):
    """
    Convert adjacency DataFrame to NetworkX DiGraph.
    
    Args:
        adjacency_df: DataFrame with adjacency matrix
        dag_variables: List of variable names
        
    Returns:
        NetworkX DiGraph representing the Bayesian network structure
    """
    import networkx as nx
    G = nx.DiGraph()
    G.add_nodes_from(dag_variables)
    
    for child in dag_variables:
        for parent in dag_variables:
            if adjacency_df.loc[parent, child] == 1:
                G.add_edge(parent, child)
    
    return G


def evaluate_operation(current_graph, operation, cache, scorer, current_score, logger):
    """
    Evaluate a graph operation (add/remove edge) and return the score.
    
    Args:
        current_graph: Current graph structure (NetworkX DiGraph)
        operation: Tuple of (action, (parent, child)) where action is '+', '-', or 'flip'
        cache: Cache for storing scores
        scorer: Scorer object for computing scores
        current_score: Current score of the graph
        
    Returns:
        Score of the operation
    """
    action, (parent, child) = operation
    
    # Create a copy of the current graph
    new_graph = current_graph.copy()
    
    if action == '+':
        new_graph.add_edge(parent, child)
    elif action == '-':
        if new_graph.has_edge(parent, child):
            new_graph.remove_edge(parent, child)
    elif action == 'flip':
        # Remove existing edge and add reverse edge
        if new_graph.has_edge(parent, child):
            new_graph.remove_edge(parent, child)
        new_graph.add_edge(child, parent)
    
    # Check if the operation is valid (no cycles, etc.)
    try:
        # Check for cycles
        if not nx.is_directed_acyclic_graph(new_graph):
            logger.warning(f"Operation {operation} creates a cycle. Returning -inf.")
            return float('-inf')
        
        # Use the scorer to compute the score
        if scorer is not None:
            # Ensure that the graph only contains nodes that the scorer knows about
            graph_nodes = set(new_graph.nodes())
            if hasattr(scorer, 'state_names'):
                scorer_nodes = set(scorer.state_names.keys())
                unknown_nodes = graph_nodes - scorer_nodes
                if unknown_nodes:
                    logger.warning(f"Graph contains nodes not in scorer: {unknown_nodes}")
                    return float('-inf')
            
            try:
                score = scorer.score(new_graph)
                logger.debug(f"Operation {operation} score: {score}")
                return score
            except KeyError as e:
                logger.exception(f"KeyError in scorer: {e}")
                return float('-inf')
        else:
            logger.warning(f"Scorer is None. Returning current score: {current_score}")
            return current_score
    except:
        logger.exception(f"Error evaluating operation {operation}")
        return float('-inf')
