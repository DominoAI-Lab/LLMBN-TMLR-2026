# LLMBN: Bayesian Network Structure Discovery Using Large Language Models

> 🎉 **Accepted at Transactions on Machine Learning Research (TMLR), 2026!**
> Read the paper: [https://arxiv.org/pdf/2511.00574](https://arxiv.org/pdf/2511.00574)

LLMBN is an open-source framework for discovering and refining Bayesian network structures by placing large language models at the center of the learning loop. It supports both data-free structure generation from metadata and data-aware refinement with classical scores such as BIC, providing flexible and reproducible workflows for state-of-the-art structure learning across research and practical applications.

### Features

- Hybrid approach: combine LLM-based and algorithmic methods (Hill Climbing, PC, MMHC)
- Data-free generation with LLMs (OpenAI, Gemini, DeepSeek) and data-driven optimization
- Modular configuration for generation, refinement, or end-to-end workflows
- Reproducible experiments with structured outputs (generations, logs, results, statistics)
- Ready-to-use sample datasets and configurations

### Table of Contents

- [Quick Start](#quick-start)
- [Input Data: Structure and Examples](#input-data-structure-and-examples)
- [Configuration Files: Structure and Examples](#configuration-files-structure-and-examples)
- [Running Workflows](#running-workflows)
- [Troubleshooting](#troubleshooting)
- [License & Citation](#license--citation)
- [Contact](#contact)
- [Contributing](#contributing)

## 🚀 Quick Start

### Prerequisites

- Python 3.10–3.12 (Python 3.13+ is not yet supported)
- API key for one supported LLM provider:
  - [OpenAI (GPT models)](https://platform.openai.com/api-keys): Sign up and create an API key at the OpenAI platform.
  - [Gemini](https://aistudio.google.com/app/apikey): Sign in with your Google account and generate a Gemini API key.
  - [DeepSeek](https://platform.deepseek.com/api-keys): Register and create an API key on the DeepSeek platform.

> Set the appropriate API key in your `.env` file (see below). Only one provider is required.

### Installation

1. **Configure API Key**

   Set your API key for your preferred LLM provider (OpenAI, Gemini, or DeepSeek):

   ```bash
   echo "OPENAI_API_KEY=your-actual-key-here" > .env
   # or
   # echo "GEMINI_API_KEY=your-gemini-key-here" > .env
   # or
   # echo "DEEPSEEK_API_KEY=your-deepseek-key-here" > .env
   ```

2. **Install R & Required Packages**

   LLMBN requires R (≥ 4.3) with several Bioconductor packages for structure learning.

   **Install R:**

   ```bash
    # macOS
    brew install r
   
    # Ubuntu / Debian
    sudo apt-get update
    sudo apt-get install r-base
   
    # Windows
    Download from https://cran.r-project.org/
   ```

   **Install R packages (inside R console):**

   ```r
    # Install BiocManager if missing
    if (!requireNamespace("BiocManager", quietly=TRUE)) {
        install.packages("BiocManager")
    }
   
    # Install required packages
    BiocManager::install(c("graph","RBGL","pcalg"))
   ```

   **Verify R packages installation (inside R console):**

   ```r
    library(graph)
    library(RBGL)
    library(pcalg)
   ```

    If no error messages appear, the setup is correct.

    macOS ARM64 (Apple Silicon) note
    If you see errors like `libRblas.dylib` not found, reinstall R with Homebrew and OpenBLAS:

    ```bash
   brew reinstall r
   brew reinstall openblas
    ```

3. **Install LLMBN and Python Dependencies**

   Clone the repository, set up a virtual environment, and install dependencies:

   ```bash
   # Clone and enter the repo
   git clone https://github.com/DominoAI-Lab/LLMBN-TMLR-2026.git
   cd LLMBN-TMLR-2026/llmbn
   
   # (Recommended) Create a virtual environment
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install all dependencies
   bash install.sh
   ```

### Run Your First Workflow

You can get started immediately using the **sample data** and **sample configuration files** provided in this repository.

1. **Use the sample data:**

   - A sample dataset is already included under `llmbn/data/bnlearn/` (e.g., the "asia" dataset).

2. **Use the sample configuration:**

   - Sample config files are available in the `llmbn/configs/` directory (e.g., `sample_generation_only.yaml`, `sample_refinement_only.yaml`).

3. **Run a workflow:**

   - For generation:

     ```bash
     python llmbn/run.py --config llmbn/configs/sample_generation_only.yaml
     ```

   - For refinement: (after you run the generation workflow)

     ```bash
     python llmbn/run.py --config llmbn/configs/sample_refinement_only.yaml
     ```

4. **Check results:**

   - Outputs will be saved in the `experiments/` directory as specified in your config.

You're ready to explore results or customize your workflow further.

## 📂 Input Data: Structure and Examples

LLMBN supports two data sources: `bnlearn` and `bnrep`. Both use the same folder and file structure. Place your datasets under either `llmbn/data/bnlearn/{dataset_name}/` or `llmbn/data/bnrep/{dataset_name}/` as appropriate.

- **bnlearn datasets** can be downloaded from: [https://www.bnlearn.com/bnrepository/](https://www.bnlearn.com/bnrepository/)
- **bnrep datasets** can be downloaded from: [https://github.com/manueleleonelli/bnRep/tree/master/data](https://github.com/manueleleonelli/bnRep/tree/master/data)

Example folder structure (for the "asia" dataset):

```
llmbn/data/
├── bnlearn/
│   └── asia/
│       ├── asia_dag.csv         # DAG adjacency matrix
│       ├── asia_metadata.csv    # Variable metadata
│       └── samples/
│           ├── asia_5000_1.csv  # Observation samples
│           ├── asia_5000_2.csv
│           └── ...
└── bnrep/
    └── ...
```

The file naming and structure are the same for both sources.

---

**Observation Samples (CSV)**

*Each sample file contains observational data for the Bayesian network. Each row is a single observation (sample), and each column is a variable. Values are typically categorical (e.g., yes/no).* 

File: `samples/asia_5000_1.csv`

|      | asia | tub  | smoke | lung | bronc | either | xray | dysp |
| ---- | ---- | ---- | ----- | ---- | ----- | ------ | ---- | ---- |
| 1    | no   | no   | yes   | no   | no    | no     | no   | no   |
| 2    | no   | no   | no    | no   | no    | no     | no   | no   |
| 3    | no   | no   | yes   | no   | no    | no     | no   | no   |
| 4    | no   | no   | yes   | no   | no    | no     | no   | no   |
| 5    | no   | no   | yes   | no   | yes   | no     | no   | yes  |
| …    | …    | …    | …     | …    | …     | …      | …    | …    |

---

**DAG Structure (CSV)**

*The DAG CSV is an adjacency matrix representing the network structure. Rows and columns correspond to variables; a `1` in row `i`, column `j` indicates a directed edge from variable `i` to variable `j`.*

File: `asia_dag.csv`

|        | asia | tub  | smoke | lung | bronc | either | xray | dysp |
| ------ | ---- | ---- | ----- | ---- | ----- | ------ | ---- | ---- |
| asia   | 0    | 1    | 0     | 0    | 0     | 0      | 0    | 0    |
| tub    | 0    | 0    | 0     | 0    | 0     | 1      | 0    | 0    |
| smoke  | 0    | 0    | 0     | 1    | 1     | 0      | 0    | 0    |
| lung   | 0    | 0    | 0     | 0    | 0     | 1      | 0    | 0    |
| bronc  | 0    | 0    | 0     | 0    | 0     | 0      | 0    | 1    |
| either | 0    | 0    | 0     | 0    | 0     | 0      | 1    | 1    |
| xray   | 0    | 0    | 0     | 0    | 0     | 0      | 0    | 0    |
| dysp   | 0    | 0    | 0     | 0    | 0     | 0      | 0    | 0    |

---

**Variable Metadata (CSV)**

*The metadata CSV lists each variable, its name, a description, and its distribution. This provides context for interpreting the variables in the samples and DAG.*

File: `asia_metadata.csv`

| node | var_name | var_description                            | var_distribution                           |
| ---- | -------- | ------------------------------------------ | ------------------------------------------ |
| 1    | asia     | visit to Asia                              | a two-level factor with levels yes and no. |
| 2    | tub      | tuberculosis                               | a two-level factor with levels yes and no. |
| 3    | smoke    | smoking                                    | a two-level factor with levels yes and no. |
| 4    | lung     | lung cancer                                | a two-level factor with levels yes and no. |
| 5    | bronc    | bronchitis                                 | a two-level factor with levels yes and no. |
| 6    | either   | tuberculosis versus lung cancer/bronchitis | a two-level factor with levels yes and no. |
| 7    | xray     | chest X-ray                                | a two-level factor with levels yes and no. |
| 8    | dysp     | dyspnoae                                   | a two-level factor with levels yes and no. |

## 🛠️ Configuration Files: Structure and Examples

LLMBN uses YAML configuration files to define experiment workflows, data sources, and algorithm settings. Place your configuration files in the `llmbn/configs/` directory.

### What is a Configuration File?

A configuration file tells LLMBN:

- Which workflow to run (generation, refinement, or pipeline)
- Where to find your input data and where to save results
- Which generator/refiner/model to use
- How many times to repeat runs, and other experiment parameters

### Where to Put Configurations

All configuration YAMLs should be placed in the `llmbn/configs/` directory.

### Structure of a Configuration File

A typical configuration file includes the following sections:

- **workflow**: Specifies the type of workflow to run. Options are `"generation"`, `"refinement"`, or `"pipeline"`.
- **data**: Contains information about input data and experiment outputs.
  - input_data: Describes your input data setup:
    - `root`: The root directory containing all input data. See [Input Data: Structure and Examples](#input-data-structure-and-examples) for details on directory organization.
    - `dataset`: A list of dataset names to be used in the experiment.
    - `source`: The data source for the datasets (e.g., `bnlearn` or `bnrep`).
  - experiment_data: Specifies where experiment outputs are stored:
    - `root`: The root directory for all experiment outputs (intermediate and final).
    - `experiment_name`: The name of the experiment; a subfolder with this name will be created under the root.
    - `generations`, `logs`, `results`, `statistics`, etc.: Subfolders within the experiment folder for generated Bayesian networks, logs, results, and statistics, respectively.
- **observation**: The number of samples to use from the input data.
- **generation**: Defines all parameters for the generation process (e.g., generator/model selection, number of runs). This section is required for the `generation` and `pipeline` workflows. 
  - `repeated_run`: Number of times to repeat the generation process.
  - `generator`: The generator algorithm to use (e.g., `promptbn`, `pgmpy_hill_climbing`).
  - `model`: This specifies the LLM model used in LLM-driven generator. If not using LLM model, keep it an empty string `""`.
- **refinement**: Defines all parameters for the refinement process (e.g., refiner/model selection, initialization). This section is required for the `refinement` and `pipeline` workflows.
  - `data`: Specifies where to obtain the initial graphs for refinement, since refinement operates on existing structures.
    - `init_generator`: The generator used to produce the initial graphs.
    - `init_model`: The model used by the initial generator.
    - `source_experiment`: The `experiment_name` of the corresponding generation experiment whose outputs will be refined.
  - `refiner`: The refinement algorithm to use (e.g., `reactbn`).
  - `model`(optional): This specifies the LLM model used in LLM-driven refiner.

### Example: Generation Only

```yaml
workflow: "generation"
data:
  input_data:
    root: "/path/to/your/data"
    dataset: ["asia"]
    source: "bnlearn"
  experiment_data:
    root: "experiments"
    experiment_name: "generation_only"
    generations: "generations"
    logs: "logs"
    results: "results"
    statistics: "statistics"
observation: 100
generation:
  repeated_run: 3
  generator: "promptbn"
  model: "o3-mini"
```

### Example: Refinement Only

```yaml
workflow: "refinement"
data:
  input_data:
    root: "/path/to/your/data"
    dataset: ["asia"]
    source: "bnlearn"
  experiment_data:
    root: "experiments"
    experiment_name: "refinement_only"
    generations: "generations"
    histories: "histories"
    logs: "logs"
    results: "results"
    statistics: "statistics"
observation: 100
refinement:
  data:
    init_generator: "promptbn"
    init_model: "o3-mini"
    source_experiment: "generation_only"
  refiner: "reactbn"
  model: "o3-mini"
```

### Tips

- You can create new configs by copying and editing the samples in `llmbn/configs/`.
- For more details, see comments in the sample YAMLs or the documentation.

## ⚡ Running Workflows

LLMBN supports multiple workflows for Bayesian network structure learning. Choose the workflow that fits your needs, update the configuration as needed, and run with a single command.

### Workflow Options

| Workflow         | Description                                        |
| ---------------- | -------------------------------------------------- |
| Generation       | Create BN structures from scratch                  |
| Refinement       | Refine existing BN structures                      |
| Unified pipeline | Generation + refinement in one run *(Coming soon)* |

> For now, run a generation workflow and then a refinement workflow separately for a full pipeline.

### Prepare Your Data

Before running a workflow, make sure your data is organized as described in the [Input Data](#input-data-structure-and-examples) for details on file and folder organization.

### How to Run

1. Create a configuration YAML in `/path/to/your/configs/` (see above for details).

2. Run the workflow using:

   ```bash
   python llmbn/run.py --config /path/to/your/configs/<your_config_file.yaml>
   ```

3. Results and logs will be saved in the output directories specified in your config.

To use your own data, place files as described in the Input Data section, update your config YAML, and run as above.

## 🐛 Troubleshooting

- **R not found**: Install R and required packages as above
- **API key errors**: Ensure `.env` exists and contains your key (OpenAI, Gemini, or DeepSeek)
- **Data loading errors**: Check your data directory and file names
- **Import errors**: Activate your virtual environment

## 📄 License & Citation

This project is licensed under the MIT License.

If you use LLMBN in your research, please cite:

```bibtex
@article{
zhang2026bayesian,
title={Bayesian Network Structure Discovery Using Large Language Models},
author={Yinghuan Zhang and Yufei Zhang and Parisa Kordjamshidi and Zijun Cui},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2026},
url={https://openreview.net/forum?id=G4mrO8LVix},
note={}
}
```

```bibtex
@software{llmbn2026,
  title={LLMBN: Bayesian Network Structure Discovery Using Large Language Models},
  author={Zhang, Yinghuan},
  year={2026},
  url={https://github.com/DominoAI-Lab/LLMBN-TMLR-2026},
  version={1.0.0}
}
```

## 📬 Contact

For questions, support, or feedback, please:

- Open an issue on [GitHub Issues](https://github.com/DominoAI-Lab/LLMBN-TMLR-2026/issues)
- Contact the authors through the repository


## 🤝 Contributing

Contributions are welcome! To propose changes:

1. Fork the repository and create a feature branch.
2. Make focused edits with clear commit messages.
3. Ensure README examples remain accurate; add/update docs as needed.
4. Open a pull request describing the motivation and changes.

If you find a bug or have a feature request, please open an issue with steps to reproduce or a concise proposal.
