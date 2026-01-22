# Unified result schema for LLMBN experiment outputs
# Import these constants in both the workflow and summary scripts to ensure consistency

# Columns for generation workflow outputs
GENERATION_NUMERIC_COLS = [
    'precision', 'recall', 'f1_score', 'accuracy', 'nhd', 'shd', 'latency'
]
GENERATION_META_COLS = [
    'dataset', 'total_run', 'validated_run', 'dag_ratio', 'generator', 'model', 'sample'
]

# Columns for refinement workflow outputs
REFINEMENT_NUMERIC_COLS = [
    'init_nhd', 'init_shd', 'final_nhd', 'final_shd', 'final_score', 'move_count', 'latency'
]
REFINEMENT_META_COLS = [
    'dataset', 'refiner', 'init_generator', 'init_model', 'model', 'sample'
]

# Note:
# - 'generator' is used for generation workflow, 'refiner' for refinement.
# - 'run' is always present for grouping.
# - 'predictor' and 'algorithm' are deprecated; use 'generator' or 'refiner'.
# - If you add new columns to outputs, update this schema accordingly.
