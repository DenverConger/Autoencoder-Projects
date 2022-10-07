# Autoencoder-Projects
 A series of simple projects exploring unsupervised learning


# Creating a Conda env off of the requirements file I provided.
conda create --name autoencoder-env --file requirements.txt


# Creating a conda env from scratch

#first you need to install wither anaconda or miniconda to your computer so we can make virtual enviroments to work in for dependancy issues.
conda create --name autoencoder-env python=3.9
conda activate autoencoder-env

install the libraries you need

conda list --export > requirements.txt

