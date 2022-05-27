# DiagnosisExtraction_ML
Pipeline for building Machine Learning Classifiers tasked with extracting the diagnosis based on EHR data (Natural Language / Narrative data). This repository works with Python 3.6.

Note: we used this pipeline for our study, published here: https://doi.org/10.2196/23930. We identified Rheumatoid Arthritis patients in EHR-data from two different centers to examine the universal applicability.


## Installation


#### Windows systems:
Prerequisite: Install [Anaconda](https://www.anaconda.com/distribution/) with python version 3.6+. This additionally installs the Anaconda Prompt, which you can find in the windows search bar. Use this Anaconda prompt to run the commands mentioned below.

#### Linux / Windows (dev) systems:
Prerequisite: [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environment (with jupyter notebook). Use the terminal to run the commands mentioned below.

Install Jupyter Notebook:
```sh
$ conda install -c anaconda notebook
```

### Importing required modules
Before running, please install the dependencies. 

#### Option 1: create custom kernel with conda (Bash script)
prerequisite: conda3

```sh
$ bash build_kernel.sh
```

#### Option 2: pip
prerequisite: pip

```sh
$ pip install -r requirements.txt
```

## Interactive demo
Our tool is available online as an interactive kaggle session:
[Click here for Kaggle Session](https://www.kaggle.com/code/levrex/notebook-diagnosis-extraction) 

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

## Citation
If you were to use this pipeline, please cite our paper: 

Maarseveen T, Meinderink T, Reinders M, Knitza J, Huizinga T, Kleyer A, Simon D, van den Akker E, Knevel R
Machine Learning Electronic Health Record Identification of Patients with Rheumatoid Arthritis: Algorithm Pipeline Development and Validation Study
JMIR Med Inform 2020;8(11):e23930
URL: https://medinform.jmir.org/2020/11/e23930
DOI: 10.2196/23930
PMID: 33252349

## Contact
If you experience difficulties with implementing the pipeline or if you have any other questions feel free to send me an e-mail. You can contact me on: t.d.maarseveen@lumc.nl 
