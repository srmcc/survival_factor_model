.. Survival Factor Model documentation master file, created by
   sphinx-quickstart on Wed Feb 17 12:26:49 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Reproducing the paper
=================================================
If you would like to reproduce the analysis in *A latent variable model for survival time prediction with censoring and diverse covariates*, please do the following.


* Navigate to the directory.::

	cd xxx/survival_factor_model/survival/

* A single command will download the data (with the exception of the *gold_standard* files, discussed below), pre-process the data, create and place the necessary specification files to run FA-ECPH-C and ECPH-C-L_1, perform 5-fold cross validation, and perform the final analysis.  It will take some time to run.  Enter something like:: 

	python reproduce_pipeline.py --root_directory /xxx/survival_factor_model/ --disease_type LGG --gold_standard True

* The *root_directory* is your directory /xxx/survival_factor_model/.
* The *disease_type* is one of [LGG, GBM, LUAD, LUSC, SIM1, SIM2, SIM3, SIM4].  LGG and GBM need to be run before SIM1-4.
* The *gold_standard* is *True* or *False*.  Enter *True* if the *disease_type* is [LGG, GBM, LUAD, LUSC] and you would like to be able to reproduce the gold standard predictions, otherwise enter *False*. For *True* to work, you first need to do the following:

	* LGG: download `Supplementary Appendix 2 <http://www.nejm.org/doi/suppl/10.1056/NEJMoa1402121/suppl_file/nejmoa1402121_appendix_2.xlsx>`_.  Save the Clinical.Output sheet as a csv file at::

	 xxx/survival_factor_model/config_files/external_files/tcga_lgg_nejmoa1402121_appendix_293.csv, sha256sum 5214e4a3a5c8f417fdb64fa4a504c30d7144506ac70ff44e2584c6c93e3af3a3

	* GBM: download `Supplementary Appendix 7 <http://www.sciencedirect.com/science/MiamiMultiMediaURL/1-s2.0-S0092867413012087/1-s2.0-S0092867413012087-mmc7.xlsx/272196/html/S0092867413012087/24011dc159bc85db34f2e32ceb911ef0/mmc7.xlsx>`_.  Save the Clinical Data sheet (after deleting the first two rows) as a csv file at::

	 xxx/survival_factor_model/config_files/external_files/tcga_gbm_mm7_544.csv, sha256sum 213d4b9208771910b6a420f351646b8e42559378daf1c47ef1308237c8911769

	* LUAD: download `Supplementary Table 2 <http://www.nature.com/nature/journal/v511/n7511/extref/nature13385-s2.xlsx>`_.  Save the S_Table 7-Clinical&Molecular_Summary sheet (after deleting the first 4 rows) as a csv file at::

	 xxx/survival_factor_model/config_files/external_files/tcga_luad_nature13385-s2.csv, sha256sum 23f28cf7c15d5e38759db275d62c65108e0d2463b3f741bc1f2b8fb02609bc1e

	* LUSC: download the `Supplemental Files <http://www.nature.com/nature/journal/v489/n7417/extref/nature11404-s2.zip>`_.  Unzip and save data.file.S7.5.clinical.and.genomic.data.table.xls (after deleting the first 3 rows and adding the column header `Expression Subtype’ over the last column)  at::

	 xxx/survival_factor_model/config_files/external_files/tcga_lusc_data.file.S7.5.clinical.and.genomic.data.table.csv, sha256sum c3af680743e5882c751c0740988b10110d0d8bef7722dc0d2057ea57d6f86f51


The previous code will create folders and files with saved output from each model fit and prediction for cross-validation and the final models.  The cross-validation and final fit c-indices can be found here::

	xxx/survival_factor_model/project_+ disease_type +/analysis/model_selection.csv
	xxx/survival_factor_model/project_+ disease_type +/analysis/final_fit_results/model_selection_output_final.txt
	xxx/survival_factor_model/project_+ disease_type +/analysis/final_fit_results_cox/model_selection_output_coxfinal.txt
	xxx/survival_factor_model/project_+ disease_type +/analysis/final_fit_results_cox_gs/model_selection_output_coxfinal.txt

The *model_selection.csv* file contains models as rows and cross-validation group as columns.  The first 4 rows are the FA-ECPH-C models.  The next 3 rows are the ECPH-C-L_1 models.  If *gold_standard* is *True*, then the last row is the gold standard ECPH-C model.  The txt files contain the c-indices of the final prediction on the test set.  It is the last element of the list.

Figures for the FA-ECPH-C models can be found here::

	xxx/survival_factor_model/project_+ disease_type +/analysis/cv_5_results/cv_run0/model_0_0/learn/expected_values/
	xxx/survival_factor_model/project_+ disease_type +/analysis/cv_5_results/cv_run0/model_0_0/val/
	xxx/survival_factor_model/project_+ disease_type +/analysis/final_fit_results/model_0_0/learn/expected_values/
	xxx/survival_factor_model/project_+ disease_type +/analysis/final_fit_results/model_0_0/val/

Once you have run all eight disease types, you can enter::

	python make_table2.py --root_directory /xxx/survival_factor_model/ --gold_standard True

This will create a .txt file with the results in Table 2 at::
	
	/xxx/survival_factor_model/table2.txt



