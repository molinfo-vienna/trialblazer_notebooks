## Trialblazer Notebooks: Reproducing the results of the article

### Install the environment

If you do not have conda, you can install [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).


```
# Set the conda executable
CONDAEXEC="conda" # micromamba/miniconda/mamba/...

# Create the environment
$CONDAEXEC create -n trialblazer_article python=3.12

# Activate the environment
$CONDAEXEC activate triablazer_article

# Install the requirements
pip install -r requirements.txt

# Download the data for the base model of trialblazer
trialblazer-download

# Launch Jupyter
jupyter notebook
```

### Note on the requirements

We provide additionally the `pip freeze` and `$CONDAEXEC list` dumps.

### Reproduce the experiments

1. Download the precomputed_data_for_reproduction_with_notebooks from: https://doi.org/10.5281/zenodo.17311675
2. Unzip and place the files in trialblazer_notebooks/Data
3. Run the notebooks

Note that the trialblazer_notebook/Dataset_preprocess folder contains notebooks for dataset curation only - no raw data is provided here. All dataset resources are available in the training_set.csv in training_and_test_data folder from: https://doi.org/10.5281/zenodo.17311675

### Link to the article
Trialblazer: A chemistry-focused predictor of toxicity risks in late-stage drug development. (https://doi.org/10.1016/j.ejmech.2025.118306)
