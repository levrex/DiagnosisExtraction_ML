# DiagnosisExtraction_ML
Pipeline for building Machine Learning Classifiers tasked with extracting the RA diagnosis based on EHR data (Natural Language). 

## Installation
Prerequisite: [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environment (with jupyter notebook)

Install Jupyter Notebook:
```sh
$ conda install -c anaconda notebook
```
### Importing required modules
Before running, please install the dependencies

#### Option 1: Create environment with conda (YML)
prerequisite: conda3

```sh
$ conda env create -f ml_env.yml
$ conda activate ml_env
```

Create kernel:
```sh
$ ipython kernel install --user --name diagnosis_ra
```

Deactivate environment:
```sh
$ conda deactivate
```

#### Option 2: pip
prerequisite: pip

```sh
$ pip install -r requirements.txt
```

#### Option 3: create custom kernel with conda (Bash script)
prerequisite: conda3

```sh
$ bash build_kernel.sh
```

## How to start
Start a notebook session in the terminal 

```sh
$ notebook
```

Which will start a notebook session in the browser from which you can open the pipeline file: 
[Notebook Diagnosis](Notebook_Diagnosis_Extraction.ipynb) 

## Pipeline
![alt text](https://github.com/levrex/DiagnosisExtraction_ML/blob/master/figures/md/PipelineDiagnosisPrediction.png "Pipeline ML-Prediction RA diagnosis")
Pipeline displaying the general workflow, where the green sections are performed automatically and the blue parts require manual evaluation. A simple annotation (binary Yes or No) will suffice, thus reducing the mental load of the physician.
