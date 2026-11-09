# FYS-STK4155 - Project 2

**Feed Forward Neural Networks**

**Authors:** Giacomo Saporetti, Amanda Rae Josephine, Narne Abhinav Chowdhary

## Overview

We implement feed-forward neural networks from scratch in NumPy and use them for
(i) regression on Runge's function, compared with OLS, Ridge and Lasso on polynomial
features from Project 1 (https://github.com/GiacomoSaporetti/FYS-STK4155-Project1/tree/main), 
and (ii) multiclass classification of the MNIST handwritten digits, compared with logistic 
regression (Scikit-Learn and our own network without hidden layers) and with Scikit-Learn's `MLPClassifier`. The project covers manual backpropagation verified against automatic differentiation, gradient descent with RMSprop and Adam, L1/L2 regularisation, and hyperparameter selection by cross-validation.

## Repository contents

| File / folder                 | Description                                                                                                                                |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `Code`                        | Main directory with all code, results and plots                                                                                            |
| `doc`                         | LaTeX source of the report and corresponding PDF file                                                                                      |
| `Code/Project2.ipynb`         | Main Jupyter notebook: runs all experiments (regression on the Runge function, MNIST classification) and produces the figures and tables  |
| `Code/scratch_ffnn.py`        | Feed-forward neural network for regression, written from scratch in NumPy: sigmoid/ReLU/leaky ReLU, manual backpropagation, GD/RMSprop/Adam, L1/L2 penalties, gradient checks against `jax.grad` and finite differences |
| `Code/scratch_classifier.py`  | Feed-forward neural network for multiclass classification, written from scratch in NumPy: softmax output, cross-entropy cost, GD/RMSprop/Adam, L1/L2 penalties; `hidden_layers=()` gives logistic regression; gradient checks against `jax.grad` and finite differences |
| `Code/results`                | Cached results (CSV and joblib files) created by the notebook so that long hyperparameter scans are not recomputed. Delete the files to rerun the scans |
| `Code/figures`                | Resulting graphs                                                                                                                           |

## Requirements

- Python 3.10 or newer
- Jupyter (Notebook or JupyterLab)
- Python packages:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `joblib`
  - `jax` (only needed for the gradient check against automatic differentiation)
- The MNIST data set as a single NumPy archive, `mnist.npz` (about 11 MB and already present in the directory).

## Setup

Clone the repository:

```bash
git clone https://github.com/AmandaRaeJ16/FYS-STK4155-Project2
cd FYS-STK4155-Project2
```

Create a virtual environment (recommended) and install the dependencies:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```


## Running the code

Start Jupyter from the `Code` folder and open the notebook:

```bash
cd Code
jupyter notebook Project2.ipynb
```

Then run all cells in order (**Kernel → Restart & Run All**). The notebook is organised by the
parts of the project, so it should be run top to bottom, since later parts reuse the data and
functions defined earlier. It must be started from `Code/`, because `scratch_ffnn.py`,
`scratch_classifier.py` and `mnist.npz` are imported or loaded with relative paths.

Random seeds are fixed in the notebook, so re-running it should reproduce the same data, CV
folds and numbers.

The first run computes the hyperparameter scans and stores them in `Code/results`; later runs
load them from disk. If you change a grid, the number of epochs, or the code in
`scratch_ffnn.py` / `scratch_classifier.py`, delete the files in `Code/results` before
rerunning, otherwise the old cached results are used.

## Building the report (optional)

To compile the report from the LaTeX source:

```bash
cd doc
pdflatex main.tex
```

Run it twice if the references or table of contents are not updated. Make sure the figures
in `Code/figures` are present first.

## Notes

- The Lasso baseline can emit `ConvergenceWarning` messages, especially for very small
  `alpha`, because the polynomial features are strongly collinear. The warnings are not
  suppressed and do not stop execution.
- The full hyperparameter grid for the regularised networks (part d) takes a long time on the
  first run. The results are cached, so later runs are fast.
- The gradient check against `jax.grad` is the only part that needs `jax`.
- With the cached results the whole notebook runs in about 10-20 minutes on one CPU core.
- The end-of-run errors of the 3000-epoch run in part c (Fig. 7) can differ slightly between
  machines (rounding amplified by Adam spikes); the best-epoch values used in the report do not.
