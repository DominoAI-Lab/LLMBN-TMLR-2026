"""
Path utility functions for PromptBN.

This module provides path management functionality based on YAML configuration,
replacing the old config.py module with a more flexible YAML-based approach.
"""

import os
from typing import Dict, Any, Optional


def resolve_path(path: str, base_dir: Optional[str] = None) -> str:
    """
    Resolve a path, handling both absolute and relative paths.
    
    Args:
        path: The path to resolve (can be absolute or relative)
        base_dir: Base directory for relative paths (defaults to current working directory)
        
    Returns:
        Resolved absolute path
    """
    if os.path.isabs(path):
        return path
    
    if base_dir is None:
        base_dir = os.getcwd()
    
    return os.path.abspath(os.path.join(base_dir, path))


def get_data_root(config: Dict[str, Any]) -> str:
    """
    Get the data root directory from configuration.
    
    Args:
        config: YAML configuration dictionary
        
    Returns:
        Resolved absolute path to data root
    """
    data_root = config['data']['input_data']['root']
    return resolve_path(data_root)


def get_experiments_root(config: Dict[str, Any]) -> str:
    """
    Get the experiments root directory from configuration.
    
    Args:
        config: YAML configuration dictionary
        
    Returns:
        Resolved absolute path to experiments root
    """
    experiments_root = config["data"]["experiment_data"]["root"]
    return resolve_path(experiments_root)


def get_experiment_path(config: Dict[str, Any]) -> str:
    """
    Get the specific experiment run directory path from configuration.
    Returns: experiments/<experiment_name>/
    """
    experiments_root = get_experiments_root(config)
    experiment_name = config['data']['experiment_data']['experiment_name']
    path = os.path.join(experiments_root, experiment_name)
    return path


def get_dataset_path(
    config: Dict[str, Any],
    source: str,
    dataset: str,
    file_type: str,
) -> str:
    """
    Get path for dataset files based on YAML configuration.
    
    Args:
        config: YAML configuration dictionary
        source: Data source ('bnlearn' or 'bnrep')
        dataset: Dataset name
        file_type: Type of file ('dag', 'var', 'samples')
        
    Returns:
        Full path to the requested file
    """
    data_root = get_data_root(config)
    
    if source == "bnlearn":
        base_dir = os.path.join(data_root, "bnlearn")
    elif source == "bnrep":
        base_dir = os.path.join(data_root, "bnrep")
    else:
        raise ValueError(f"Unknown data source: {source}")
        
    if file_type == "dag":
        return os.path.join(base_dir, dataset, f"{dataset}_dag.csv")
    elif file_type == "var":
        if source == "bnlearn":
            return os.path.join(base_dir, dataset, f"{dataset}_metadata.csv")
        else:  # bnrep
            return os.path.join(base_dir, dataset, f"{dataset}_metadata.csv")
    elif file_type == "samples":
        return os.path.join(base_dir, dataset, "samples", f"{dataset}_5000_1.csv")
    else:
        raise ValueError(f"Unknown file type: {file_type}")


def get_generations_path(
    config: Dict[str, Any],
    dataset: str,
    generator: str,
    model: str,
    source_experiment: str = None,
) -> str:
    """
    Get path for generation files based on YAML configuration.
    
    Args:
        config: YAML configuration dictionary
        dataset: Dataset name
        generator: Generator name
        model: Model name
        source_experiment: Optional source experiment name (for cross-experiment loading)
        
    Returns:
        Full path to generation file
    """
    if source_experiment:
        # Load from a different experiment directory
        experiments_root = get_experiments_root(config)
        experiment_path = os.path.join(experiments_root, source_experiment)
    else:
        experiment_path = get_experiment_path(config)
    
    generations_dir = os.path.join(experiment_path, "generations", "bn")
    os.makedirs(generations_dir, exist_ok=True)
    
    # Check if we should use run_name format (from generation workflow) or standard format
    if source_experiment:
        # For cross-experiment loading, try to find files with run_name format first
        possible_files = [
            f"generation_{generator}_{model}_{dataset}.json",  # Generation workflow format
            f"{generator}_{model}_{dataset}.json"  # Standard format
        ]
        
        for filename in possible_files:
            file_path = os.path.join(generations_dir, filename)
            if os.path.exists(file_path):
                return file_path
        
        # If no existing file found, return the standard format
        return os.path.join(generations_dir, f"{generator}_{model}_{dataset}.json")
    else:
        return os.path.join(generations_dir, f"{generator}_{model}_{dataset}.json")


def ensure_directories(config: Dict[str, Any]) -> None:
    """
    Ensure all necessary directories exist based on YAML configuration.
    Creates all output folders under the run directory.
    """
    data_root = get_data_root(config)
    experiments_root = get_experiments_root(config)
    experiment_path = get_experiment_path(config)

    directories = [
        data_root,
        experiments_root,
        experiment_path,
        os.path.join(data_root, config['data']['input_data']['source']),
        os.path.join(experiment_path, config['data']['experiment_data']['generations']),
        os.path.join(experiment_path, config['data']['experiment_data']['logs']),
        os.path.join(experiment_path, config['data']['experiment_data']['results']),
        os.path.join(experiment_path, config['data']['experiment_data']['statistics']),
    ]

    # Add histories directory if it exists in config
    if "histories" in config['data']['experiment_data']:
        directories.append(os.path.join(experiment_path, "histories"))

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
