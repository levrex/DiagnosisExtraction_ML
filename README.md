# DiagnosisExtraction_ML
Pipeline for building Machine Learning Classifiers tasked with extracting the RA diagnosis based on EHR data (Natural Language). 

## Installation
Prerequisite: [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environment (with jupyter notebook)

#### Option 1: pip
Before running, please install the dependencies:

```sh
$ pip install -r requirements.txt
```

#### Option 2: create custom kernel with conda
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
