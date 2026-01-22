import os
import json
# logging imported but not used directly in this file
import pandas as pd
# No additional typing imports needed
from .path_utils import get_experiment_path, ensure_directories

def write_result(
    config,
    dataset,
    exp_data_root,
    result_df,
    bn_generations: list[dict]=None,
    logger=None,
):
    if config['workflow'] == 'generation':
        run_name = config['generation']['generator'] + '_' + config['generation']['model'] + '_' + dataset
    elif config['workflow'] == 'refinement':
        run_name = config['refinement']['refiner'] + '_' + config['refinement']['model'] + '_' + dataset
    elif config['workflow'] == 'pipeline':
        run_name = config['generation']['generator'] + '_' + config['generation']['model'] + '_' + config['refinement']['refiner'] + '_' + config['refinement']['model'] + '_' + dataset
    else:
        raise ValueError(f"Invalid workflow type: {config['workflow']}")
    
    result_file_path = os.path.join(
        exp_data_root,
        config['data']['experiment_data']['results'],
        f"{run_name}.csv"
    )
    result_df.to_csv(result_file_path, mode='a+', index=False, header=True)
    logger.info(f"Result df written to {result_file_path}")
    
    if bn_generations is not None:
        generation_folder = os.path.join(
            exp_data_root,
            config['data']['experiment_data']['generations'],
            "bn",
        )
        os.makedirs(generation_folder, exist_ok=True)
        generation_file_path = os.path.join(
            generation_folder,
            f"{run_name}.json"
        )
        with open(generation_file_path, "w", encoding="utf-8") as f:
            json.dump(bn_generations, f, indent=4, ensure_ascii=False)
        logger.info(f"BN Generations written to {generation_file_path}")

def write_error_result(
    config,
    dataset,
    exp_data_root,
    error,
    logger,
):
    run_name = config['generation']['generator'] + '_' + config['generation']['model'] + '_' + dataset
    result_file_path = os.path.join(
        exp_data_root,
        config['data']['experiment_data']['results'],
        f"{run_name}.csv"
    )
    # generate a new dataframe with columns: dataset, run, precision, recall, f1_score, accuracy, nhd, shd, latency, total_run, validated_run, dag_ratio, exp_name, sample
    result_df = pd.DataFrame({
        'dataset': [dataset] * 5,
        'run': [0] * 5,
        'precision': [0] * 5,
        'recall': [0] * 5,
        'f1_score': [0] * 5,
        'accuracy': [0] * 5,
        'nhd': [0] * 5,
        'shd': [0] * 5,
        'latency': [0] * 5,
        'total_run': [5] * 5,
        'validated_run': [0] * 5,
        'dag_ratio': [0] * 5,
        'generator': [config['generation']['generator']] * 5 if 'generation' in config and 'generator' in config['generation'] else [None] * 5,
        'model': [config['generation']['model']] * 5 if 'generation' in config and 'model' in config['generation'] else [None] * 5,
        'sample': [config['observation']] * 5 if 'observation' in config else [None] * 5,
        'error': [error] * 5
    })
    result_df.to_csv(result_file_path, mode='a+', index=False, header=False)
    logger.info(f"ERROR result df written to {result_file_path}")
    
def init_experiment(config, logger):
    """Initialize experiment directories and return experiment run data directory."""
    # Ensure all necessary directories exist
    ensure_directories(config)
    
    # Get experiment run data directory
    exp_data_path = get_experiment_path(config)
    logger.info("Experiment run data directory: %s", exp_data_path)
    return exp_data_path
  