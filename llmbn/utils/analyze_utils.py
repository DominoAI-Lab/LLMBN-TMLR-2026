import os
import glob
import pandas as pd
import numpy as np
from llmbn.utils.result_schema import (
    GENERATION_NUMERIC_COLS, GENERATION_META_COLS,
    REFINEMENT_NUMERIC_COLS, REFINEMENT_META_COLS
)

def analyze_generation(results_dir, statistics_dir, out_path):
    csv_files = glob.glob(os.path.join(results_dir, '*.csv'))
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        for col in GENERATION_META_COLS:
            if col not in df.columns:
                df[col] = None
        for col in GENERATION_NUMERIC_COLS:
            if col not in df.columns:
                df[col] = np.nan
        if 'error' not in df.columns:
            df['error'] = None
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    group_cols = [col for col in GENERATION_META_COLS]
    numeric_cols = [col for col in GENERATION_NUMERIC_COLS]
    for col in group_cols:
        all_df[col] = all_df[col].fillna('NONE').replace('', 'NONE')
    def agg_func(x):
        d = {}
        for col in numeric_cols:
            valid = pd.to_numeric(x[col], errors='coerce').dropna()
            if not valid.empty:
                mean = valid.mean()
                std = valid.std()
                d[col] = f"{mean:.3f}±{std:.3f}"
            else:
                d[col] = 'NaN'
        d['error'] = x['error'].iloc[0]
        return pd.Series(d)
    stats_df = all_df.groupby(group_cols, as_index=False).apply(agg_func, include_groups=False)
    meta = group_cols
    rest = [col for col in stats_df.columns if col not in meta]
    stats_df = stats_df[meta + rest]
    os.makedirs(statistics_dir, exist_ok=True)
    stats_df.to_csv(out_path, index=False)
    return stats_df

def analyze_refinement(results_dir, statistics_dir, out_path):
    csv_files = glob.glob(os.path.join(results_dir, '*.csv'))
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        for col in REFINEMENT_META_COLS:
            if col not in df.columns:
                df[col] = None
        for col in REFINEMENT_NUMERIC_COLS:
            if col not in df.columns:
                df[col] = np.nan
        if 'error' not in df.columns:
            df['error'] = None
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    group_cols = [col for col in REFINEMENT_META_COLS]
    numeric_cols = [col for col in REFINEMENT_NUMERIC_COLS]
    for col in group_cols:
        all_df[col] = all_df[col].fillna('NONE').replace('', 'NONE')
    def agg_func(x):
        d = {}
        for col in numeric_cols:
            valid = pd.to_numeric(x[col], errors='coerce').dropna()
            if not valid.empty:
                mean = valid.mean()
                std = valid.std()
                d[col] = f"{mean:.3f}±{std:.3f}"
            else:
                d[col] = 'NaN'
        d['error'] = x['error'].iloc[0]
        return pd.Series(d)
    stats_df = all_df.groupby(group_cols, as_index=False).apply(agg_func, include_groups=False)
    meta = group_cols
    rest = [col for col in stats_df.columns if col not in meta]
    stats_df = stats_df[meta + rest]
    os.makedirs(statistics_dir, exist_ok=True)
    stats_df.to_csv(out_path, index=False)
    return stats_df

