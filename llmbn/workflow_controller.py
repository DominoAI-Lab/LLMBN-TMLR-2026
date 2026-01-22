"""
LLMBN Workflow Controller

Orchestrates Bayesian Network structure learning workflows:
- Generation: Creates BN structures using LLM-based or traditional methods
- Refinement: Improves existing BN structures using data
- Pipeline: Combines generation and refinement in sequence

Supports both data-free (LLM) and data-dependent (statistical) approaches.
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import os

from llmbn.utils.data_utils import load_data, load_observation, load_generations
from llmbn.utils.eval_utils import evaluate_generation
from llmbn.utils.graph_utils import construct_matrix_from_bn_dict
from llmbn.errors.generation_error import MaxRetriesError, InvalidDAGError, ParseResponseError
from llmbn.errors.pipeline_error import MissingGenerationsError
from llmbn.generators.base import BaseGenerator
from llmbn.refiners.base import BaseRefiner


@dataclass
class WorkflowResult:
    """
    Container for workflow execution results with status and metrics.
    
    Attributes:
        workflow_type: Type of workflow ('generation', 'refinement', 'pipeline')
        success: Whether the workflow completed successfully
        dataset: Name of the dataset used
        generation_results: DataFrame with generation metrics (if applicable)
        refinement_results: DataFrame with refinement metrics (if applicable)
        error: Error message if workflow failed
    """
    workflow_type: str
    success: bool
    dataset: str
    generation_results: Optional[pd.DataFrame] = None
    refinement_results: Optional[pd.DataFrame] = None
    error: Optional[str] = None


class WorkflowController:
    """
    Orchestrates BN structure learning workflows (generation, refinement, pipeline).
    
    Combines LLM-based and traditional algorithms for both data-free and
    data-dependent approaches to Bayesian Network structure learning.
    
    Attributes:
        config: Configuration dictionary with workflow settings
        logger: Logger instance for output
        exp_data_dir: Experiment data directory path
        workflow_type: Type of workflow to execute
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        exp_data_dir: str,
    ) -> None:
        """
        Initialize controller with config, logger, and output directory.
        
        Args:
            config: Configuration dictionary with workflow settings
            logger: Logger instance for output
            exp_data_dir: Experiment data directory path
        """
        self.config = config
        self.logger = logger
        self.exp_data_dir = exp_data_dir
        self.workflow_type = config['workflow']
        
    def _load_data_for_dataset(
        self,
        dataset: str,
    ) -> tuple[pd.DataFrame, list, str, Optional[pd.DataFrame]]:
        """
        Load DAG, variables, descriptions, and observations for dataset.
        
        Args:
            dataset: Name of the dataset to load
            
        Returns:
            tuple: (dag, dag_variables, desc_variables, observation)
        """
        dag, dag_variables, desc_variables = load_data(
            self.config['data']['input_data']['source'],
            dataset,
            self.logger,
            self.config,
        )
        observation = None
        if self.config.get('observation', 0) > 0:
            observation = load_observation(
                self.config['data']['input_data']['source'],
                dataset,
                self.config['observation'],
                self.config,
                self.logger,
            )
            self.logger.info(f"[Data] [{dataset}] {len(observation)} observations loaded")
        return dag, dag_variables, desc_variables, observation
    
    def _load_generator(self) -> BaseGenerator:
        """
        Load appropriate generator based on configuration settings.
        
        Returns:
            Generator: Instantiated generator object
            
        Raises:
            ValueError: If unknown generator specified in config
        """
        generator_config = self.config['generation']
        generator_name = generator_config['generator']
        
        if generator_name == 'promptbn':
            from llmbn.generators.promptbn import PromptBNGenerator as Generator
            generator = Generator(
                model=generator_config['model'],
                logger=self.logger
            )
        elif generator_name == 'pgmpy_hill_climbing':
            from llmbn.generators.pgmpy_hill_climbing_generator import PgmpyHillClimbingGenerator as Generator
            generator = Generator(logger=self.logger)
        elif generator_name == 'pgmpy_mmhc':
            from llmbn.generators.pgmpy_mmhc_generator import PgmpyMMHCGenerator as Generator
            generator = Generator(logger=self.logger)
        elif generator_name == 'pgmpy_pc':
            from llmbn.generators.pgmpy_pc_generator import PgmpyPCGenerator as Generator
            generator = Generator(logger=self.logger)
        elif generator_name == 'pgmpy_ges':
            from llmbn.generators.pgmpy_ges_generator import PgmpyGESGenerator as Generator
            generator = Generator(logger=self.logger)
        else:
            raise ValueError(f"Unknown generator: {generator_name}")
        
        self.logger.info(f"[Generator] {generator_name} generator loaded")
        return generator
    
    def _load_refiner(self) -> BaseRefiner:
        """
        Load appropriate refiner based on configuration settings.
        
        Returns:
            Refiner: Instantiated refiner object
            
        Raises:
            ValueError: If unknown refiner specified in config
        """
        refinement_config = self.config['refinement']
        # Map algorithm names to their refiner classes
        # (config['refinement']['llm']['algorithm'] is kept for backward compatibility with config files)
        algorithm = refinement_config['refiner']
        if algorithm == 'reactbn':
            from llmbn.refiners.react_bn_agent import ReActBNAgent as Refiner
            refiner = Refiner(refinement_config['model'], self.logger)
        elif algorithm == 'hill_climbing':
            from llmbn.refiners.pgmpy_hill_climbing_refiner import PgmpyHillClimbingRefiner as Refiner
            refiner = Refiner(refinement_config['model'], self.logger)
        elif algorithm == 'randombn':
            from llmbn.refiners.random_bn_agent import RandomBNAgent as Refiner
            refiner = Refiner(refinement_config['model'], self.logger)
        else:
            raise ValueError(f"Unknown refiner: {algorithm}. Available refiners: 'reactbn', 'hill_climbing'")
        self.logger.info("[Refiner] %s loaded", algorithm)
        return refiner
    
    def execute_generation_workflow(
        self,
        dataset: str,
        generator: BaseGenerator,
    ) -> WorkflowResult:
        """
        Execute generation workflow for dataset using specified generator.
        
        Args:
            dataset: Name of the dataset to use
            generator: Generator instance to use
            
        Returns:
            WorkflowResult: Results container with success status and metrics
            
        Raises:
            ValueError: If dataset or generator is None
            InvalidDAGError: If generated DAG is invalid
            ParseResponseError: If LLM response cannot be parsed
            MaxRetriesError: If max retries reached
            MissingGenerationsError: If generations are missing
        """
        if dataset is None:
            raise ValueError("A dataset must be provided to execute_generation_workflow.")
        if generator is None:
            raise ValueError("A generator must be provided to execute_generation_workflow.")
        dag, dag_variables, desc_variables, observation = self._load_data_for_dataset(dataset)
        self.logger.info(f"[Workflow] Dataset {dataset} loaded successfully")
        
        try:
            self.logger.info(f"[Workflow] Starting generation workflow for dataset {dataset}")
            
            result_df = self._run_generation_experiment(
                generator=generator,
                dataset=dataset,
                desc_variables=desc_variables,
                dag_variables=dag_variables,
                observation=observation,
                dag=dag,
            )
            
            self.logger.info("------------------------------------------------")
            self.logger.info("[Workflow] Generation workflow completed successfully for dataset %s", dataset)
            self.logger.info("------------------------------------------------")
            return WorkflowResult(
                workflow_type="generation",
                success=True,
                dataset=dataset,
                generation_results=result_df
            )
            
        except (InvalidDAGError, ParseResponseError, MaxRetriesError, MissingGenerationsError) as e:
            self.logger.error("Generation workflow failed: %s", e, exc_info=True)
            return WorkflowResult(
                workflow_type="generation",
                success=False,
                dataset=dataset,
                error=str(e)
            )
    
    def execute_refinement_workflow(
        self,
        dataset: str,
        refiner: BaseRefiner,
    ) -> WorkflowResult:
        """
        Execute refinement workflow for dataset using specified refiner.
        
        Args:
            dataset: Name of the dataset to use
            refiner: Refiner instance to use
            
        Returns:
            WorkflowResult: Results container with success status and metrics
            
        Raises:
            InvalidDAGError: If refined DAG is invalid
            ParseResponseError: If LLM response cannot be parsed
            MaxRetriesError: If max retries reached
            MissingGenerationsError: If initial generations are missing
        """
        dag, dag_variables, desc_variables, observation = self._load_data_for_dataset(dataset)
        self.logger.info(f"[Data] DAG, variables, and descriptions loaded for dataset {dataset}")
        
        try:
            refinement_config = self.config['refinement']
            source_experiment = refinement_config['data'].get('source_experiment')
            generations = load_generations(
                dataset,
                init_generator=refinement_config['data']['init_generator'],
                init_model=refinement_config['data']['init_model'],
                config=self.config,
                source_experiment=source_experiment,
            )
            self.logger.info(f"[Refinement] {len(generations)} initial generations loaded")
            
            result_df = self._run_refinement_experiment(
                refiner=refiner,
                dataset=dataset,
                generations=generations,
                observation=observation,
                desc_variables=desc_variables,
                dag_variables=dag_variables,
                dag=dag,
            )
            
            self.logger.info("Refinement workflow completed successfully")
            return WorkflowResult(
                workflow_type="refinement",
                success=True,
                dataset=dataset,
                refinement_results=result_df
            )
            
        except (
            InvalidDAGError,
            ParseResponseError,
            MaxRetriesError,
            MissingGenerationsError,
        ) as e:
            self.logger.error("Refinement workflow failed: %s", e, exc_info=True)
            return WorkflowResult(
                workflow_type="refinement",
                success=False,
                dataset=dataset,
                error=str(e)
            )
    
    def execute_pipeline_workflow(
        self,
        dataset: str,
        generator: BaseGenerator,
        refiner: BaseRefiner
    ) -> WorkflowResult:
        """
        Execute generation+refinement pipeline for dataset using specified components.
        
        Args:
            dataset: Name of the dataset to use
            generator: Generator instance to use
            refiner: Refiner instance to use
            
        Returns:
            WorkflowResult: Results container with success status and metrics
            
        Raises:
            InvalidDAGError: If DAG is invalid
            ParseResponseError: If LLM response cannot be parsed
            MaxRetriesError: If max retries reached
            MissingGenerationsError: If generations are missing
        """
        self.logger.info(f"[Data] DAG, variables, and descriptions loaded for dataset {dataset}")
        
        self.logger.info("=== Starting Pipeline Workflow ===")
        
        try:
            # Step 1: Execute generation
            self.logger.info("Step 1: Executing generation workflow")
            generation_result = self.execute_generation_workflow(dataset, generator)
            
            if not generation_result.success:
                self.logger.error("Generation workflow failed, aborting pipeline")
                return WorkflowResult(
                    workflow_type="pipeline",
                    success=False,
                    dataset=dataset,
                    error=f"Generation failed: {generation_result.error}"
                )
            
            # Step 2: Execute refinement using generated structures
            self.logger.info("Step 2: Executing refinement workflow")
            refinement_result = self.execute_refinement_workflow(dataset, refiner)
            
            if not refinement_result.success:
                self.logger.error("Refinement workflow failed")
                return WorkflowResult(
                    workflow_type="pipeline",
                    success=False,
                    dataset=dataset,
                    error=f"Refinement failed: {refinement_result.error}"
                )
            
            self.logger.info("Pipeline workflow completed successfully")
            return WorkflowResult(
                workflow_type="pipeline",
                success=True,
                dataset=dataset,
                refinement_results=refinement_result.refinement_results,
            )
            
        except (InvalidDAGError, ParseResponseError, MaxRetriesError, MissingGenerationsError) as e:
            self.logger.error("Pipeline workflow failed: %s", e, exc_info=True)
            return WorkflowResult(
                workflow_type="pipeline",
                success=False,
                dataset=dataset,
                error=str(e)
            )
    
    def _run_generation_experiment(
        self,
        generator: BaseGenerator,
        dataset: str,
        desc_variables: str,
        dag_variables: list,
        observation: Optional[pd.DataFrame],
        dag: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Run BN generation experiment and collect performance metrics.
        
        Args:
            generator: Generator instance to use
            dataset: Name of the dataset
            desc_variables: Variable descriptions
            dag_variables: List of variable names
            observation: Observation data (optional)
            dag: True DAG structure for evaluation
            
        Returns:
            pd.DataFrame: Results dataframe with metrics
            
        Raises:
            InvalidDAGError: If generated DAG is invalid
            ParseResponseError: If LLM response cannot be parsed
            MaxRetriesError: If max retries reached
            MissingGenerationsError: If generations are missing
        """
        import time
        from llmbn.utils.io_utils import write_error_result, write_result
        self.logger.info(f"[Generation] Starting experiment for dataset {dataset}")
        count = 0
        count_from_last_validated = 0
        validated_count = 0
        generations = []
        results = {
            "dataset": [],
            "run": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "accuracy": [],
            "nhd": [],
            "shd": [],
            "latency": []
        }
        
        generation_config = self.config['generation']
        repeated_run = generation_config.get('repeated_run', 3)
        
        while True:
            starttime = time.time()
            count += 1
            
            try:
                prev_generations = None
                dag_validation, llm_results = generator.run(
                    desc_variables,
                    dag_variables,
                    observation=observation,
                    generations=prev_generations,
                )
            except InvalidDAGError as e:
                self.logger.exception(f"[Generation] [{dataset}] Invalid DAG: {e} in the {count}th run.")
                if count - count_from_last_validated >= 5:
                    self.logger.exception("Encountered 5 consecutive errors. Stop the experiment.")
                    write_error_result(
                        self.config,
                        dataset,
                        self.exp_data_dir,
                        "InvalidDAG",
                        self.logger,
                    )
                    raise
                continue
            except ParseResponseError:
                self.logger.exception("[Generation] [{dataset}] Unparsable Raw Generation in {count}th run", count)
                if count - count_from_last_validated >= 5:
                    self.logger.exception("Encountered 5 consecutive errors. Stop the experiment.")
                    write_error_result(
                        self.config,
                        dataset,
                        self.exp_data_dir,
                        "UnparsableResponse",
                        self.logger,
                    )
                    raise
                continue
            except MaxRetriesError:
                self.logger.exception(
                    f"Forced stop the experiment (PromptBN generator, {generation_config['llm']['model']}, {dataset}),"
                    f"reached max retries."
                )
                write_error_result(
                    self.config,
                    dataset,
                    self.exp_data_dir,
                    "MaxRetries",
                    self.logger,
                )
                raise
            except MissingGenerationsError as e:
                self.logger.exception(f"ValueError: {e}")
                write_error_result(
                    self.config,
                    dataset,
                    self.exp_data_dir,
                    "MissingGenerations",
                    self.logger,
                )
                raise

            endtime = time.time()
            if dag_validation == 1:
                validated_count += 1
                count_from_last_validated = count
                self.logger.info(f"[Generation] [{dataset}] Current Run: {count}, Validated Run: {validated_count}")
                generations.append(llm_results['Generation'])
                try:
                    self.logger.info(f"[Generation] [{dataset}] DAG {dag.shape} and Pred {llm_results['Matrix'].shape} evaluation...")
                    precision, recall, f1_score, accuracy, nhd, shd = evaluate_generation(
                        dag_input=dag,
                        pred_input=llm_results['Matrix'],
                        logger=self.logger,
                    )
                except Exception as e:
                    self.logger.exception(f"[Generation] [{dataset}] Failed to evaluate generation: {e}")
                    validated_count -= 1
                else:
                    self.logger.info(f"[Generation] [{dataset}] nhd: {nhd:.4f}; shd: {shd:.4f}; f1_score: {f1_score:.4f}; accuracy: {accuracy:.4f}")
                    results["dataset"].append(dataset)
                    results["run"].append(validated_count)
                    results["precision"].append(precision)
                    results["recall"].append(recall)
                    results["f1_score"].append(f1_score)
                    results["accuracy"].append(accuracy)
                    results["nhd"].append(nhd)
                    results["shd"].append(shd)
                    results['latency'].append(f"{int(endtime-starttime)}")
                
            if validated_count >= repeated_run:
                self.logger.info(f"[Generation] [{dataset}] Hit run requirement. Total Run Count: {count}")
                break
        
        result_df = pd.DataFrame(results)
        result_df['total_run'] = count
        result_df['validated_run'] = validated_count
        result_df['dag_ratio'] = validated_count/count
        result_df['generator'] = generator.name
        result_df['sample'] = self.config['observation']
        
        write_result(
            self.config,
            dataset,
            self.exp_data_dir,
            result_df,
            generations,
            self.logger,
        )
        return result_df
    
    def _run_refinement_experiment(
        self,
        refiner: BaseRefiner,
        dataset: str,
        generations: list,
        observation: pd.DataFrame,
        desc_variables: str,
        dag_variables: list,
        dag: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Run BN refinement experiment on initial structures and collect metrics.
        
        Args:
            refiner: Refiner instance to use
            dataset: Name of the dataset
            generations: Initial BN structures to refine
            observation: Observation data
            desc_variables: Variable descriptions
            dag_variables: List of variable names
            dag: True DAG structure for evaluation
            
        Returns:
            pd.DataFrame: Results dataframe with metrics
        """
        import numpy as np
        from llmbn.utils.data_utils import save_history, save_matrix_graph
        from llmbn.utils.io_utils import write_result
        
        exp_results = []
        
        for idx, init_generation in enumerate(generations):
            run_idx = idx + 1
            
            if self.config.get('observation', 0) == 0:
                observation = None
                self.logger.info("[Data] No observations loaded")
            else:
                observation = load_observation(
                    source=self.config['data']['input_data']['source'],
                    dataset=dataset,
                    sample_size=self.config['observation'],
                    config=self.config,
                    logger=self.logger,
                )
                self.logger.info("[Data] %d Observations loaded", len(observation))
            
            if init_generation is None:
                init_shd = np.inf
                init_nhd = np.inf
            else:
                init_generation_df = construct_matrix_from_bn_dict(
                    bn_dict=init_generation,
                    dag_variables=dag_variables,
                )
                _, _, _, _, init_nhd, init_shd = evaluate_generation(
                    dag_input=dag,
                    pred_input=init_generation_df,
                    logger=self.logger,
                )
            
            self.logger.info("run: %d, initial SHD: %.4f, NHD: %.4f", run_idx, init_shd, init_nhd)
            
            result_record = self._run_single_refinement(
                refiner=refiner,
                observation=observation,
                init_generation=init_generation,
                desc_variables=desc_variables,
                dag_variables=dag_variables,
                dag=dag,
            )
            
            self.logger.info(
                "run: %d, SHD: %.4f -> %.4f, NHD: %.4f -> %.4f",
                run_idx,
                init_shd, result_record['final_shd'], init_nhd, result_record['final_nhd']
            )
            
            exp_results.append({
                "run": run_idx,
                "init_nhd": init_nhd,
                "init_shd": init_shd,
                "final_nhd": result_record['final_nhd'],
                "final_shd": result_record['final_shd'],
                "final_score": result_record['final_score'],
                "move_count": result_record['move_count'],
                "latency": result_record['latency'],
                "try_count": result_record['try_count'],
            })

            save_history(
                metrics_history=result_record['MetricsHistory'],
                action_history=result_record['ActionHistory'],
                config=self.config,
                run_idx=run_idx,
            )
            save_matrix_graph(
                matrix=result_record['Matrix'],
                graph=result_record['Graph'],
                node_order=dag_variables,
                config=self.config,
                run_idx=run_idx,
            )

        result_df = pd.DataFrame(exp_results)
        result_df['dataset'] = dataset
        # When writing the output DataFrame for refinement, use 'refiner' column
        result_df['refiner'] = self.config['refinement']['refiner']
        result_df['init_generator'] = self.config['refinement']['data']['init_generator']
        result_df['init_model'] = self.config['refinement']['data']['init_model']
        result_df['source_experiment'] = self.config['refinement']['data']['source_experiment'] if self.workflow_type == 'refinement' else "self"
        result_df['model'] = self.config['refinement']['model']
        result_df['sample'] = self.config['observation']
        
        write_result(
            config=self.config,
            dataset=dataset,
            exp_data_root=self.exp_data_dir,
            result_df=result_df,
            bn_generations=generations,
            logger=self.logger,
        )
        
        return result_df
    
    def _run_single_refinement(
        self,
        refiner: BaseRefiner,
        observation: pd.DataFrame,
        init_generation: Optional[Dict[str, Any]],
        desc_variables: str,
        dag_variables: list,
        dag: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Run a single BN refinement iteration with error handling and metrics tracking.
        
        Args:
            refiner: Refiner instance to use
            observation: Observation data
            init_generation: Initial BN structure to refine
            desc_variables: Variable descriptions
            dag_variables: List of variable names
            dag: True DAG structure for evaluation
            
        Returns:
            Dict[str, Any]: Results dictionary with metrics and refined structure
            
        Raises:
            InvalidDAGError: If refined DAG is invalid
            MaxRetriesError: If max retries reached
        """
        import time
        # Import already available from module level
        
        starttime = time.time()
        max_retries = 3
        
        for try_count in range(max_retries):
            try:
                refiner_result = refiner.run(
                    desc_variables=desc_variables,
                    dag_variables=dag_variables,
                    dag=dag,
                    observation=observation,
                    init_generation=init_generation,
                )
                break
            except InvalidDAGError:
                if try_count >= max_retries - 1:
                    self.logger.exception("Invalid DAG: Encountered max retries. Stop the experiment.")
                    raise
            except MaxRetriesError:
                if try_count >= max_retries - 1:
                    self.logger.exception("Max retries: Forced stop the experiment.")
                    raise

        endtime = time.time()

        _, _, f1_score, accuracy, nhd, shd = evaluate_generation(
            dag_input=dag,
            pred_input=refiner_result['Matrix'],
            logger=self.logger,
        )
        self.logger.info("nhd: %.4f; shd: %.4f; f1_score: %.4f; accuracy: %.4f", nhd, shd, f1_score, accuracy)

        return {
            "try_count": try_count + 1,
            "init_nhd": refiner_result.get("init_nhd"),
            "init_shd": refiner_result.get("init_shd"),
            "final_nhd": refiner_result.get("final_nhd"),
            "final_shd": refiner_result.get("final_shd"),
            "final_score": refiner_result.get("Score"),
            "move_count": refiner_result.get("MoveCount"),
            "latency": f"{int(endtime-starttime)}",
            "Matrix": refiner_result.get("Matrix"),
            "Graph": refiner_result.get("Graph"),
            "ActionHistory": refiner_result.get("ActionHistory"),
            "MetricsHistory": refiner_result.get("MetricsHistory")
        }
    
    def execute(
        self,
        dataset: str,
        worker: Union[
            BaseGenerator, BaseRefiner, tuple[BaseGenerator, BaseRefiner],
        ],
    ) -> WorkflowResult:
        """
        Execute configured workflow (generation/refinement/pipeline) for dataset.
        
        Args:
            dataset: Name of the dataset to use
            worker: Generator, Refiner, or (Generator, Refiner) tuple based on workflow
            
        Returns:
            WorkflowResult: Results container with success status and metrics
            
        Raises:
            ValueError: If workflow type is unknown
        """
        if self.workflow_type == "generation":
            return self.execute_generation_workflow(dataset, worker)
        elif self.workflow_type == "refinement":
            return self.execute_refinement_workflow(dataset, worker)
        elif self.workflow_type == "pipeline":
            generator, refiner = worker
            return self.execute_pipeline_workflow(dataset, generator, refiner)
        else:
            raise ValueError(f"Unknown workflow type: {self.workflow_type}")

    def analyze_results(self) -> None:
        """
        Run appropriate analysis based on workflow type and save statistics.
        
        Analyzes experiment results and saves statistics to the configured output directory.
        """
        from llmbn.utils.analyze_utils import analyze_generation, analyze_refinement
        exp_root = self.config['data']['experiment_data']['experiment_name']
        results_dir = os.path.join(
            self.config['data']['experiment_data']['root'],
            exp_root,
            self.config['data']['experiment_data']['results']
        )
        statistics_dir = os.path.join(
            self.config['data']['experiment_data']['root'],
            exp_root,
            self.config['data']['experiment_data']['statistics']
        )
        out_path = os.path.join(statistics_dir, f"{exp_root}_statistics.csv")
        datasets = self.config['data']['input_data']['dataset']
        if not isinstance(datasets, list):
            datasets = [datasets]
        if self.workflow_type == 'generation':
            for dataset in datasets:
                analyze_generation(results_dir, statistics_dir, out_path)
        elif self.workflow_type == 'refinement':
            for dataset in datasets:
                analyze_refinement(results_dir, statistics_dir, out_path)
        elif self.workflow_type == 'pipeline':
            for dataset in datasets:
                analyze_generation(results_dir, statistics_dir, out_path.replace('_statistics.csv', f'_generation_statistics_{dataset}.csv'))
                analyze_refinement(results_dir, statistics_dir, out_path.replace('_statistics.csv', f'_refinement_statistics_{dataset}.csv'))
        else:
            self.logger.warning(f"[Analysis] Unknown workflow type: {self.workflow_type}")
            return None

    def run_full_experiment(self, max_workers=1) -> None:
        """
        Run experiment on all datasets with optional parallelization.
        
        Args:
            max_workers: Maximum number of parallel workers for dataset processing
            
        Raises:
            ValueError: If workflow type is unknown
        """
        self.logger.info("================================================")
        self.logger.info("Running full experiment with workflow type: %s", self.workflow_type)
        self.logger.info("================================================")
        
        datasets = self.config['data']['input_data']['dataset']
        if not isinstance(datasets, list):
            datasets = [datasets]
        max_workers = min(max_workers, len(datasets))
        # Load generator/refiner once
        if self.workflow_type == 'generation':
            worker = self._load_generator()
        elif self.workflow_type == 'refinement':
            worker = self._load_refiner()
        elif self.workflow_type == 'pipeline':
            worker = (self._load_generator(), self._load_refiner())
        else:
            raise ValueError(f"Unknown workflow type: {self.workflow_type}")
        def run_for_dataset(dataset) -> WorkflowResult:
            return self.execute(dataset, worker)
        self.logger.info("================================================")
        self.logger.info("Starting worker execution...")
        self.logger.info("================================================")
        for dataset in datasets:
            run_for_dataset(dataset)
            self.logger.info(f"[Workflow] [{dataset}] Completed workflow")
        # TODO: parallelization - R memory issue
        #     import concurrent.futures
        #     with concurrent.futures.ThreadPoolExecutor(
        #         max_workers=max_workers
        #     ) as executor:
        #         futures = [
        #             executor.submit(run_for_dataset, dataset)
        #             for dataset in datasets
        #         ]
        #         for future in concurrent.futures.as_completed(futures):
        #             future.result()  # Will raise exceptions if any
        
        self.logger.info("================================================")
        self.logger.info("Completed full experiment, starting analysis...")
        self.logger.info("================================================")
        self.analyze_results()
        self.logger.info("================================================")
        self.logger.info("Analysis completed")
        self.logger.info("================================================")