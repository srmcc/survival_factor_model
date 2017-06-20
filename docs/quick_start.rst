.. Survival Factor Model documentation master file, created by
   sphinx-quickstart on Wed Feb 17 12:26:49 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Quick start
=================================================
If you would like to perform your own analysis on your own data, here is a quick start on what to do.

* Decide on a *project_name*.

* Make the file tree::

	mkdir xxx/survival_factor_model/project_name
	mkdir xxx/survival_factor_model/project_name/data
	mkdir xxx/survival_factor_model/project_name/analysis

* Put your data in csv files.  The rows are features and the columns are samples.  The first row should be the sample names.  The first column should be the feature names.  Move your data files into the folder::

	xxx/survival_factor_model/project_name/data/

* Create a file *data_guide.csv*. The *file_location* column contains the file names of your data files.  The *datatype* column contains a string *[normal, binom, multinom, survival]* that specifies the data type.  If the *datatype* is *[binom, multinom]*, the column *b* contains an integer.  The column *sparserange* should be *[[False, False]]*.  If the *datatype* is *survival*, then the column *etime* should have the name of the event time index in your survival data file, and the column *CEN* should be *True*, the column *Delta* should have the name of the censoring indicator index in your survival data file, and lastly the column *CENtype* should be *cindIC* for informative censoring.  A template for the file is here:: 

	xxx/survival_factor_model/config_files/templates/data_guide.csv

* Put the personalized *data_guide.csv* here:
	
	xxx/survival_factor_model/project_name/data/data_guide.csv

* Create files *specs_cv.csv* and *specs_cox_cv.csv*. In *specs_cv.csv*, the column *dzrange* should contain a list of integer latent dimensions to search over (e.g. *[2, 3]*).  The column *MULTIPRO* should contain an integer for the number of cores. In *specs_cox_cv.csv*, the column *sparserange* should contain a nested list with the sparsity parameters (e.g. *[[True, 1000], [True, 10000]]*). Templates for the files are here:: 

	xxx/survival_factor_model/config_files/templates/

* Put the personalized specification files here::

	xxx/survival_factor_model/project_name/analysis/specs_cv.csv
	xxx/survival_factor_model/project_name/analysis/specs_cox_cv.csv

* Navigate to the directory::

	cd xxx/survival_factor_model/survival/

* Open a python terminal::

	python

* Import some modules::
	
	from __future__ import division
	import pandas as pd
	import numpy as np 
	import survival_funct_v1
	import cox_funct

* Decide a seed value for your session *TOP_SEED*::

	TOP_SEED = 22343323423
	np.random.seed(TOP_SEED)
	large = 4294967295
	MASTER_SEED_CV = np.random.random_integers(0,large)
	MASTER_SEED = np.random.random_integers(0,large)	

* Decide the number of cross-validation splits *n_cv*::

	n_cv = 5

* Define your directory names::

	data_directory = ‘/xxx/survival_factor_model/project_name/data/’
	analysis_directory  = ‘/xxx/survival_factor_model/project_name/analysis/’

* Run::

	cox_funct.cross_val_cox(n_cv, data_directory, analysis_directory, False)
	survival_funct_v1.cross_val_sfa(n_cv, data_directory, analysis_directory, MASTER_SEED_CV)

* Select the best models.  Define the number of latent dimensions searched over to be *n_dz*.  Define the number of sparsity penalties searched over to be *n_cox*.  (These need to be consistent with the number of models searched in your specification csv files, of course)::

	mod_sel = survival_funct_v1.gather_plot_cv_cindex_sim(n_cv, analysis_directory)
	mod = survival_funct_v1.get_best(mod_sel.iloc[0:n_dz, :])
	mod_cox = survival_funct_v1.get_best(mod_sel.iloc[n_dz:(n_dz + n_cox), :])

* The previous code will create a pdf and a csv file of the c-indices on the validation sets from the cross validation searches in the *analysis_directory*.  The *model_selection.csv* file contains models as rows and cross-validation group as columns.  The first *n_dz* rows are the FA-ECPH-C models.  The next *n_cox* rows are the ECPH-C-L_1 models. It will also create folders and files with additional saved output from each model fit and prediction.

* Outside of the python terminal, make files *specs.csv* with the best latent dimension in the column *dzrange* (e.g. *[2]*) and *specs_cox.csv* with the best sparsity parameter in the column *sparserange* (e.g. *[[True, 1000]]*) and put them here::

	xxx/survival_factor_model/project_name/analysis/specs.csv
	xxx/survival_factor_model/project_name/analysis/specs_cox.csv

* Back in the python terminal, run the final analysis::

	cox_funct.final_fit_cox(data_directory, analysis_directory, False)
	survival_funct_v1.final_fit_sfa(data_directory, analysis_directory, MASTER_SEED)

* The previous code will create folders and files with saved output from each model fit and prediction.  The c-index of the final prediction for the test set is the last element of the list in these files::

	xxx/survival_factor_model/project_name/analysis/final_fit_results/model_selection_output_final.txt
	xxx/survival_factor_model/project_name/analysis/final_fit_results_cox/model_selection_output_coxfinal.txt

* Figures for the FA-ECPH-C models can be found here::

	xxx/survival_factor_model/project_+ disease_type +/analysis/cv_5_results/cv_run0/model_0_0/learn/expected_values/
	xxx/survival_factor_model/project_+ disease_type +/analysis/cv_5_results/cv_run0/model_0_0/val/
	xxx/survival_factor_model/project_+ disease_type +/analysis/final_fit_results/model_0_0/learn/expected_values/
	xxx/survival_factor_model/project_+ disease_type +/analysis/final_fit_results/model_0_0/val/

* Just a note regarding the pipeline.  Both python functions survival_funct_v1.cross_val_sfa, cox_funct.cross_val_cox will make the learn/val/test set random splits of your samples only if the splits do not already exist in the analysis folder.  This impacts the reproducibility, since the following code with and without making splits will give different results::

	survival_funct_v1.cross_val_sfa(n_cv, data_directory, analysis_directory, MASTER_SEED_CV) 