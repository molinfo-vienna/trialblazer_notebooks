## Trialblazer Notebooks: Reproducing the results of the article

#### Description

#### Install the environment

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

#### Note on the requirements

We provide additionally the `pip freeze` and `$CONDAEXEC list` dumps.

#### Link to the article

#### Cite the article