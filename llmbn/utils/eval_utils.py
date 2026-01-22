import numpy as np
import pandas as pd
import networkx as nx
from cdt.metrics import SHD
from llmbn.utils.graph_utils import df_to_cpdag_pcalg


def evaluate_generation(
    dag_input,
    pred_input,
    logger,
    node_order=None,
):
    """Evaluate generation metrics including precision, recall, F1, accuracy, NHD, and SHD."""
    logger.debug("Evaluating generation...")

    if isinstance(pred_input, pd.DataFrame):
        pred = pred_input.to_numpy().astype(int)
        pred_df = pred_input
    else:
        pred = pred_input
        pred_df = pd.DataFrame(pred_input, columns=node_order)
    logger.debug(f"Pred shape: {pred_df.shape}")
    
    if isinstance(dag_input, pd.DataFrame):
        dag = dag_input.to_numpy().astype(int)
        dag_df = dag_input
    else:
        dag = dag_input
        dag_df = pd.DataFrame(dag_input, columns=node_order)
    logger.debug(f"DAG shape: {dag_df.shape}")
    
    n_variables = dag.shape[0]
    logger.debug(f"Number of variables: {n_variables}")
    
    corrects = np.equal(pred, dag).astype(int)
    num_corr = np.sum(corrects) - n_variables

    tp = np.sum((pred == 1) & (dag == 1))
    pred_sum = np.sum(pred)
    dag_sum = np.sum(dag)

    precision = tp / pred_sum if pred_sum != 0 else 0
    recall = tp / dag_sum if dag_sum != 0 else 0
    logger.debug(f"Precision: {precision}, Recall: {recall}")

    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall)/(precision + recall)
    logger.debug(f"F1 Score: {f1_score}")

    accuracy = num_corr/(pred.size - n_variables)
    logger.debug(f"Accuracy: {accuracy}")

    hamming_distance = np.sum(pred != dag)
    logger.debug(f"Hamming Distance: {hamming_distance}")
    nhd = hamming_distance/(n_variables * n_variables)
    logger.debug(f"NHD: {nhd}")

    cpdag_np = df_to_cpdag_pcalg(dag_df, logger).to_numpy().astype(int)
    pred_cpdag_np = df_to_cpdag_pcalg(pred_df, logger).to_numpy().astype(int)
    logger.debug("DAG and Pred DAG converted to CPDAG in numpy array")

    shd_cdt = SHD(cpdag_np, pred_cpdag_np, False)
    logger.debug(f"SHD in CPDAG: {shd_cdt}")

    return precision, recall, f1_score, accuracy, nhd, shd_cdt


def compute_metrics(current_graph, ref_graph, dag_variables, logger):
    """
    Compute SHD and NHD between current_graph and ref_graph, log them, and return (shd, nhd).
    """
    pred_adj = nx.to_numpy_array(
        current_graph, nodelist=dag_variables, weight=None, dtype=int)
    ref_adj = nx.to_numpy_array(
        ref_graph, nodelist=dag_variables, weight=None, dtype=int)
    
    _, _, _, _, nhd, shd = evaluate_generation(
        ref_adj, pred_adj, logger, dag_variables)
    return shd, nhd


def track_action(action_type, action_data, metrics_history, logger):
    """Track action for history logging."""
    logger.debug("Action: %s, Data: %s", action_type, action_data)
    return action_type, action_data
