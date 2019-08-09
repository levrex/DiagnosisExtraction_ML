# Dependencies: conda

# Create a machine learning environment

conda create -n ml_env python=3.5

source activate ml_env

# If pip cant find the module try command below: 
#    pip install --upgrade pip 

pip install --upgrade pip
pip install -r requirements.txt

# link custom environment with modules to a kernel in jupyter
ipython kernel install --user --name diagnosis_ra

source deactivate