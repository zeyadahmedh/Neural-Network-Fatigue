# Neural-Network-Fatigue

A small research project exploring a **fatigue index** for neural network optimization. The goal is to measure training instability by looking at how gradients change from one optimization step to the next, and eventually use that signal inside an optimizer such as **Adam**.

## Project status

This repository is currently in an early development stage. The main pieces implemented so far are:

- a prototype implementation of the fatigue / CSI calculation in `fatigue.py`
- gradient comparison utilities based on cosine similarity
- handling for parameters that are frozen or have no gradients
- an experimental notebook, `development_notebook.ipynb`, used to prototype the idea
- a Python project setup with dependencies for PyTorch, torchvision, scikit-learn, and plotting tools

## What the project is trying to do

The current idea is to detect when optimization is becoming less stable by measuring signals such as:

- gradient conflict between the previous and current gradients
- error oscillation or related instability metrics
- parameters that should be ignored because they have zero gradients in both steps

This fatigue signal is intended to be used later in a custom optimizer step, likely by extending or overriding the `step` function in Adam.

## Current implementation

### `fatigue.py`

The main code currently contains:

- `tensor_to_array(...)` for converting PyTorch tensors to NumPy arrays
- `frozen_parameters_indices(...)` for finding parameters that have zero gradients in both steps
- `gradient_conflict(...)` for computing a distance-like measure between gradient vectors using cosine similarity
- a placeholder `error_oscillation_calculate()` function for future work

### `development_notebook.ipynb`

The notebook contains early experiments for:

- downloading and loading the MNIST dataset
- building data loaders
- testing the fatigue calculation functions interactively

## Repository layout

- `fatigue.py` — core fatigue / gradient conflict utilities
- `development_notebook.ipynb` — prototype notebook and experiments
- `pyproject.toml` — project metadata and dependencies
- `uv.lock` — locked dependency versions

## Dependencies

The project currently uses:

- `torch`
- `torchvision`
- `scikit-learn`
- `matplotlib`
- `ipython`
- `ipykernel`

## Notes

This README describes the project as it exists so far. As the fatigue metric, optimizer integration, and experiments mature, this document can be expanded with:

- mathematical details of the fatigue/CSI formula
- training results and plots
- usage examples
- instructions for running the notebook or scripts

## License

This project is licensed under the MIT License.
