import pandas as pd
import os
import json
import logging
from typing import Any
from .path_utils import get_experiments_root, get_dataset_path, get_generations_path, get_experiment_path

def load_data(source, dataset, logger, config):
    """Load DAG data and variable descriptions."""
    if source == "bnlearn":
        dag, dag_variables, desc_variables = load_bnlearn_data(dataset, logger, config)
    elif source == "bnrep":
        dag, dag_variables, desc_variables = load_bnrep_data(dataset, logger, config)
    else:
        raise ValueError(f"Unknown data source: {source}")
    logger.info(
        f"[Data] Loaded {len(dag_variables)} variables."
    )
    return dag, dag_variables, desc_variables

def load_bnlearn_observation(dataset, sample, config):
    """Load observation data from BNlearn dataset."""
    file_path = get_dataset_path(
        config=config,
        source="bnlearn",
        dataset=dataset,
        file_type="samples",
    )
    df = pd.read_csv(file_path, nrows=sample)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df

def load_bnrep_observation(dataset, sample, config):
    """Load observation data from BNrep dataset."""
    file_path = get_dataset_path(
        config=config,
        source="bnrep",
        dataset=dataset,
        file_type="samples",
    )
    df = pd.read_csv(file_path, nrows=sample)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df

def load_bnlearn_data(dataset, logger, config):
    """Load BNlearn dataset DAG and variable descriptions."""
    dag_path = get_dataset_path(config, "bnlearn", dataset, "dag")
    dag = pd.read_csv(
        dag_path,
        index_col=0,
        na_filter=False
    )
    logger.info(f"[Data] DAG for dataset {dataset} loaded from {dag_path}")
    dag_variables = dag.index.to_list()
    n_variables = len(dag_variables)
    for idx in range(n_variables):
        var = dag_variables[idx]
        dag_variables[idx] = var
    desc_path = get_dataset_path(config, "bnlearn", dataset, "var")
    logger.info(f"[Data] Variable descriptions for dataset {dataset} loaded from {desc_path}")
    desc_variables = parse_var_csv_to_string_scrambled(desc_path)
    return dag, dag_variables, desc_variables

def load_bnrep_data(dataset, logger, config):
    """Load BNrep dataset DAG and variable descriptions."""
    dag_path = get_dataset_path(config, "bnrep", dataset, "dag")
    dag = pd.read_csv(
        dag_path,
        index_col=0,
        na_filter=False
    )
    dag_variables = dag.index.to_list()
    n_variables = len(dag_variables)
    for idx in range(n_variables):
        var = dag_variables[idx]
        dag_variables[idx] = var
    desc_path = get_dataset_path(config, "bnrep", dataset, "var")
    desc_variables = parse_var_csv_to_string_scrambled(desc_path)
    return dag, dag_variables, desc_variables
    
def load_observation(source, dataset, sample_size, config, logger):
    
    file_path = get_dataset_path(
        config=config,
        source=source,
        dataset=dataset,
        file_type="samples",
    )
    df = pd.read_csv(file_path, nrows=sample_size)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    logger.info(f"[Data] [{dataset}({(source)})] Observation data loaded from {file_path}")
    return df
 
def parse_var_csv_to_string(csv_path):
    """
    Parse a variable CSV file into a formatted string.
    Args:
        csv_path (str): Path to the CSV file containing variable information
    Returns:
        str: Formatted string with variable information
    """
    df = pd.read_csv(csv_path, na_filter=False)
    result = ""
    for _, row in df.iterrows():
        result += (
            f"node: {row['node']}; var_name: {row['var_name']}; "
            f"var_description: {row['var_description']}; "
            f"var_distribution: {row['var_distribution']}\n"
        )
    return result

import pandas as pd

def parse_var_csv_to_string_scrambled(csv_path):
    """
    Parse a variable CSV file into a formatted string, but in scrambled order.

    Args:
        csv_path (str): Path to the CSV file containing variable information.
        seed (int, optional): Random seed for reproducible scrambling.

    Returns:
        str: Formatted string with variable information in random order.
    """
    df = pd.read_csv(csv_path, na_filter=False)

    # Shuffle (scramble) the rows
    df = df.sample(frac=1, random_state=5).reset_index(drop=True)

    result = ""
    for _, row in df.iterrows():
        result += (
            f"node: {row['node']}; var_name: {row['var_name']}; "
            f"var_description: {row['var_description']}; "
            f"var_distribution: {row['var_distribution']}\n"
        )
    return result



def save_history(
    metrics_history,
    action_history,
    config,
    run_idx,
):
    """Save metrics and action history to JSON files."""
    experiment_path = get_experiment_path(config)
    run_name = config["data"]["experiment_data"]["experiment_name"] + f"_run_{run_idx}"
    histories_dir = os.path.join(
        experiment_path,
        config["data"]["experiment_data"]["histories"],
        run_name,
    )
    os.makedirs(histories_dir, exist_ok=True)
    
    with open(
        os.path.join(histories_dir, f"{run_name}_metrics_history.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics_history, f, indent=4, ensure_ascii=False)
    
    with open(
        os.path.join(histories_dir, f"{run_name}_action_history.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(action_history, f, indent=4, ensure_ascii=False)


def save_matrix_graph(
    matrix: pd.DataFrame,
    graph: Any,
    node_order: list,
    config,
    run_idx,
):
    """Save matrix and graph to JSON files."""
    experiment_path = get_experiment_path(config)
    run_name = config["data"]["experiment_data"]["experiment_name"] + f"_run_{run_idx}"
    histories_dir = os.path.join(
        experiment_path,
        config["data"]["experiment_data"]["histories"],
        run_name,
    )
    os.makedirs(histories_dir, exist_ok=True)
    
    # Save matrix
    matrix_file_path = os.path.join(
        histories_dir,
        f"{run_name}_matrix.json",
    )
    with open(matrix_file_path, "w", encoding="utf-8") as f:
        json.dump(matrix.tolist(), f, indent=4, ensure_ascii=False)
    
    # Save graph as adjacency matrix
    if hasattr(graph, 'edges'):
        graph_dict = {}
        for edge in graph.edges():
            if edge[0] not in graph_dict:
                graph_dict[edge[0]] = {}
            graph_dict[edge[0]][edge[1]] = 1
    else:
        graph_dict = matrix.to_dict()
    
    graph_file_path = os.path.join(
        histories_dir,
        f"{run_name}_graph.json",
    )
    with open(graph_file_path, "w", encoding="utf-8") as f:
        json.dump(graph_dict, f, indent=4, ensure_ascii=False)


def load_generations(
    dataset,
    init_generator,
    init_model,
    config,
    source_experiment=None,
):
    """Load initial generations from JSON file."""
    if init_generator is None or init_model is None:
        raise ValueError("init_generator and init_model must be provided")
    
    # Try to load from source experiment first, then fall back to current experiment
    file_path = None
    if source_experiment:
        experiments_root = get_experiments_root(config)
        experiment_path = os.path.join(experiments_root, source_experiment)
        generations_dir = os.path.join(experiment_path, "generations", "bn")
        file_path = os.path.join(
            generations_dir,
            f"{init_generator}_{init_model}_{dataset}.json",
        )
        if not os.path.exists(file_path):
            logging.info(
                "File not found in source experiment %s",
                source_experiment,
            )
    if not file_path or not os.path.exists(file_path):
        file_path = get_generations_path(
            config,
            dataset,
            init_generator,
            init_model,
        )
    
    if not os.path.exists(file_path):
        logging.warning("Generations file not found: %s", file_path)
        return [None] * 5
    
    with open(file_path, "r", encoding="utf-8") as f:
        generations = json.load(f)
    logging.info("[Generations] Loaded (%s) %d generations from %s", dataset, len(generations), file_path)
    return generations
