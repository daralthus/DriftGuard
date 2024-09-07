# DriftGuard 🚀

This is a sample project replicating pieces of the paper: [Are you still on track!? Catching LLM Task Drift with Activations
](https://arxiv.org/abs/2406.00799) using [Jax](https://jax.readthedocs.io/) and [Penzai](https://penzai.readthedocs.io/) on the [Gemma and Gemma 2](https://huggingface.co/docs/transformers/model_doc/gemma) models, with a simple synthetic dataset.


## Setup

Run `git lfs pull` - fetch large files in `data/`.
Install the requirements with `pip install -r requirements.txt`.
Also set your Kaggle and OpenAI API keys, get a good GPU and then you ready to hit the notebooks.

## Project structure

- `data/`: 
  - `prompts/`: Prompts generated in [notebooks/02-dataset-generation.ipynb](notebooks/02-dataset-generation.ipynb) using gpt-4o-mini. A mix of instructions, clean user inputs and poisoned user inputs containing prompt injection attempts.
  - `inference/`: Model completions and residual activations for the prompts.
  - `evals/`: Automated and manual evaluations of inference results.
  - `results/`: Final results of the trained classifiers.
- `notebooks/`: Jupyter notebooks for building up the project utilities and running experiments:
    - `01-penzai-and-activation-saving.ipynb`: A small Penzai intro (named arrays, model loading, inference, accessing activations, visualization), then defines the activation saving layer and patches the model to save intermediate activations at batch model inference.
    - `02-dataset-generation.ipynb`: Generates the synthetic datasets.
    - `03-dataset-response-and-attack-eval.ipynb`: Evaluates the model on the synthetic datasets, hitting some realizations.
    - `04-task-drift-classifier.ipynb`: Trains a classifier on residual activations.
    - `05-steering-vectors.ipynb`: PCA visualizations demonstrating linear separability and creating steering vectors to demonstrate task representation arithmetics.
- `configs/`: Configuration files for the scripts.
- `scripts/`: Scripts to process data in batches.
- `models/`: Fine tuned models.

## Running scripts

### `python scripts/save-activations.py [-h] [--config CONFIG] [--print]`:

Runs inference on a specific dataset and model to save generated text and intermediate layer activations e.g: `python scripts/save-activations.py --config configs/summarize_email-multi-gemma_2b_it.yaml --print`

```
options:
  -h, --help       Show this help message and exit
  --config CONFIG  Path to the config file.
  --print          Print the completions to the console.
```

### `python scripts/save-activations.py [-h] [--config CONFIG] [--print]`:
Run automated evaluations e.g: `python scripts/evaluate-completions.py --config configs/summarize_email-multi-gemma_2b_it.yaml --print`

```
options:
  -h, --help       Show this help message and exit
  --config CONFIG  Path to the config file.
  --print          Print the results to the console.
```