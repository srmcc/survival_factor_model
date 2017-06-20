# Survival Factor Model

This package is developed to fit an integrative latent variable model based on factor analysis and the exponential Cox proportional hazards model for survival time to address the following challenges in survival time modeling: small sample sizes, high-dimensional genomic and epigenomic covariates, and heavy, informative censoring.  Furthermore, the low-dimensional latent variables of our proposed model provide a visualization of the samples that can be used to identify heterogeneity within the sample population.  We compare the performance of our model to two alternative models.  We evaluate the model based on the predictive performance on unseen samples.

# Installing the dependencies
To run this code you will need python 2.7 and the following python packages:  numpy, scipy, pandas, matplotlib, click, and sphinx (for the documentation).  
The suggested method for installing python and python packages is from source, because there is a signifcant speed differential between an optimized source installation and binary installation.
However, it is much simpler to use the anaconda package manager which can be obtained at https://www.continuum.io/downloads
Then you can simply create an "conda env" by 
```bash
conda env create -f env.yml 
#
# To activate this environment, use:
# $ source activate survival
#
# To deactivate this environment, use:
# $ source deactivate
#
```
Alternatively, you can use pip to install the dependecies
```bash
pip install -U -r requirements.txt
```
To reproduce the results in the paper, you will additionally need R, [CNTools](http://bioconductor.org/packages/release/bioc/html/CNTools.html), and [TCGA2STAT](http://www.liuzlab.org/TCGA2STAT/) to download the data.  To install the R packages, open an R environment and type
```R
## try http:// if https:// URLs are not supported
source("https://bioconductor.org/biocLite.R")
biocLite("CNTools")
install.packages("TCGA2STAT_1.0.tar.gz", repos = NULL, type = "source")
```
Alternatively, a [Nix](https://nixos.org) shell derivation has been provided which creates the required build environment. Simply run 
```bash
nix-shell shell.nix
```
in order to drop into the development shell.

# Getting Started
In order to run the code, please look at the "Getting Started" page in the documentation.

# Documentation
In order to generate the documentation, cd to docs directory and 
```bash
make html
```
in order to generate html documentation. If you have latex installed, you can generate the pdf by
```bash
make latexpdf
```
