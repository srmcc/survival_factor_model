from __future__ import division
import survival_funct_v1
import cox_funct
import clean_data
from time import time
import os
import numpy as np
import plot_funct
import pandas as pd
import re


def pipeline(root_directory, disease_type, gold_standard, plot_only=False):
	# gold_standard = True
	# gold_standard = False

	# # ["LGG", "GBM", "LUAD", "LUSC", "SIM1", "SIM2", "SIM3", "SIM4"]

	# disease_type = "GBM"
	# disease_type = "LGG"
	# disease_type = "LUAD"
	# disease_type = "LUSC"
	# disease_type = "SIM1"
	# disease_type = "SIM3"
	# disease_type = "SIM4"
	# disease_type = "SIM2"

	# root_directory = '/home/smccurdy/scratch/survival/survival_factor_model/'
	project_name= "project_" + disease_type
	data_directory, analysis_directory = survival_funct_v1.new_project(root_directory, project_name)
	large = 4294967295

	if disease_type == "LGG":
		TOP_SEED= 1234233330#9893
		gs_file_name = 'tcga_lgg_nejmoa1402121_appendix_293.csv'
	if disease_type == "GBM":
		TOP_SEED= 3453462234#232
		gs_file_name = 'tcga_gbm_mm7_544.csv'
	if disease_type == "LUAD":
		TOP_SEED= 9823423
		gs_file_name = 'tcga_luad_nature13385-s2.csv'
	if disease_type == "LUSC":
		TOP_SEED= 723423123#4234
		gs_file_name = 'tcga_lusc_data.file.S7.5.clinical.and.genomic.data.table.csv'
	if disease_type == "SIM1":
		TOP_SEED= 884810362
		param_analysis_directory = root_directory + 'project_LGG/analysis/'
		ntest=69
		ntrain=279-69
		sim_frac=0
		sim_type = 1

	if disease_type == "SIM2":
		TOP_SEED=2941402902
		param_analysis_directory = root_directory + 'project_LGG/analysis/'
		ntest=69
		ntrain=279-69
		#sim_frac=.9
		#sim_type = 2
		sim_frac=0
		sim_type = 1

	if disease_type == "SIM3":
		TOP_SEED=352522181
		param_analysis_directory = root_directory + 'project_GBM/analysis/'
		ntest=17
		ntrain=71-17
		sim_frac=0
		sim_type = 1

	if disease_type == "SIM4":
		TOP_SEED=569377263
		param_analysis_directory = root_directory + 'project_GBM/analysis/'
		ntest=69
		ntrain=279-69
		sim_frac=0
		sim_type = 1

	if not plot_only:
		assert TOP_SEED < large
		np.random.seed(TOP_SEED)
		if disease_type not in ["LGG", "GBM", "LUAD", "LUSC"]:
			MASTER_SEED_DG = np.random.random_integers(0,large)

		MASTER_SEED_CV = np.random.random_integers(0,large)
		MASTER_SEED = np.random.random_integers(0,large)


		rscript_path = os.getenv('RSCRIPT_PATH', '/usr/bin/Rscript')
		if disease_type in ["LGG", "GBM", "LUAD", "LUSC"]:
			if os.listdir(data_directory+ 'not_in_use/') == []:
				os.system(rscript_path + ' download_data.R '+ disease_type + ' ' + root_directory + project_name+ '/data/not_in_use/')

		#must move in gold standard file to /data/not_in_use/
		if gold_standard==True:
			if os.path.exists( '../config_files/external_files/'+ gs_file_name):
				os.system('cp ../config_files/external_files/'+ gs_file_name + ' ' + data_directory + 'not_in_use/')
			else:
				print('You need ../config_files/external_files/'+ gs_file_name + ' to run gold_standard = True.  quitting.')
				return

		if len(os.listdir(data_directory)) == 1:
			if disease_type in ["LGG", "GBM", "LUAD", "LUSC"]:
				clean_data.clean_data(disease_type, data_directory, gold_standard)
			else:
				if os.path.exists(param_analysis_directory):
					dz, paramX, paramt, CEN=survival_funct_v1.load_params(param_analysis_directory)
					survival_funct_v1.gen_data_to_csv(ntrain, ntest, dz, paramX, paramt, CEN, root_directory + project_name + '/', sim_frac, sim_type, MASTER_SEED_DG)
				else:
					print('directory' + param_analysis_directory + 'does not exist.  need to run LGG or GBM first. quitting')
					return

		os.system('cp -r ../config_files/'+ disease_type+ '/* '+ analysis_directory)
		## put 5 spec files (or 4 if no gs), and sample splits into analysis directory.
		n_cv=5
		print('hello')

		cox_funct.cross_val_pca_cox(n_cv, data_directory, analysis_directory)

		cox_funct.cross_val_cox(n_cv, data_directory, analysis_directory, False)

		if disease_type in ["LGG", "GBM", "LUAD", "LUSC"] and gold_standard == True:
			cox_funct.cross_val_cox(n_cv, data_directory, analysis_directory, gold_standard)


		t = time()
		survival_funct_v1.cross_val_sfa(n_cv, data_directory, analysis_directory, MASTER_SEED_CV)
		print(time() - t) / 60, 'min'

		if disease_type in ["LGG", "GBM", "LUAD", "LUSC"] and gold_standard == True:
			mod_sel = survival_funct_v1.gather_plot_cv_cindex(n_cv, analysis_directory, disease_type)
		else:
			mod_sel = survival_funct_v1.gather_plot_cv_cindex_sim(n_cv, analysis_directory, disease_type)


		mod=survival_funct_v1.get_best(mod_sel.iloc[0:4, :])
		print('best sfa mod', mod)
		specs = pd.read_csv(analysis_directory + 'specs_cv.csv', delimiter=',', index_col=0)
		dzrange_string = specs.loc[0, 'dzrange']
		dzrange = eval(dzrange_string.replace(';', ','))
		specs.loc[0, 'dzrange'] = '[' + str(dzrange[mod])+ ']'
		specs.to_csv(analysis_directory + 'specs.csv', delimiter=',', index_col=0)
		
		mod_cox = survival_funct_v1.get_best(mod_sel.iloc[4:7, :])
		print('best cox mod', mod_cox)
		specs = pd.read_csv(analysis_directory + 'specs_cox_cv.csv', delimiter=',', index_col=0)
		sparserange_string = specs.loc[0, 'sparserange']
		sparserange = eval(sparserange_string.replace(';', ','))
		specs.loc[0, 'sparserange'] = '[' + str(sparserange[mod_cox])+ ']'
		specs.to_csv(analysis_directory + 'specs_cox.csv', delimiter=',', index_col=0)

		mod_pca_cox = survival_funct_v1.get_best(mod_sel.iloc[7:10, :])
		print('best pca cox mod', mod_pca_cox)
		specs = pd.read_csv(analysis_directory + 'specs_cv.csv', delimiter=',', index_col=0) #same dimensions as fa
		dzrange_string = specs.loc[0, 'dzrange']
		dzrange = eval(dzrange_string.replace(';', ','))
		specs.loc[0, 'dzrange'] = '[' + str(dzrange[mod_pca_cox])+ ']'
		specs.to_csv(analysis_directory + 'specs_pca_cox.csv', delimiter=',', index_col=0)


		means=np.mean(mod_sel, axis=1)
		print(means)
		stdvs= np.std(mod_sel, axis=1)
		print(stdvs)

		for model in range(3):
			survival_funct_v1.getsparse_cox(n_cv, analysis_directory, model)


		## first need to make sure specs are correct.
		## final fits:
		cox_funct.final_fit_pca_cox(data_directory, analysis_directory)

		cox_funct.final_fit_cox(data_directory, analysis_directory, False)
		if disease_type in ["LGG", "GBM", "LUAD", "LUSC"] and gold_standard == True:
			cox_funct.final_fit_cox(data_directory, analysis_directory, gold_standard)

		t = time()
		survival_funct_v1.final_fit_sfa(data_directory, analysis_directory, MASTER_SEED)
		print(time() - t) / 60, 'min'

	if gold_standard:
		data_guide_gs=pd.read_csv(data_directory + 'data_guide_gs.csv', sep=',')
		survivalloc =data_directory+ data_guide_gs.loc[:, 'file_location'].values[-1]
		i=0
		n_cv=5
		plot_funct.prettyplot(n_cv, i, survivalloc, analysis_directory, disease_type)
		for filename in data_guide_gs.loc[:, 'file_location']:
			gs= pd.read_csv(data_directory +filename, sep=',', index_col=0)
			if re.search(r'age', filename) and not re.search(r'stage', filename):
				plot_funct.prettyplot2(n_cv, i, survivalloc, gs.loc['yearstobirth', :].astype('float64'), 'age', analysis_directory, disease_type)
				if disease_type =='LGG':
					plot_funct.prettyplot3(n_cv, i, survivalloc, plot_funct.make_categorical(gs.loc['neoplasm_histologic_grade',:].replace([1, 2, 3], ['G1', 'G2', 'G3'])), 'grade', analysis_directory, disease_type)
					plot_funct.prettyplot3(n_cv, i, survivalloc, plot_funct.make_categorical(gs.loc['cqcf_method_of_sample_procuremen', :].replace([3, 2, 1], ['Gross Total Resection', 'Subtotal Resection', 'Biopsy Only'])), 'resection', analysis_directory, disease_type)
			if re.search(r'sub', filename):
				plot_funct.prettyplot3(n_cv, i, survivalloc, gs, 'subtype', analysis_directory, disease_type)
				if disease_type =='LGG':
					plot_funct.prettyplot3_final(survivalloc, gs, 'subtype', analysis_directory, disease_type)
			if re.search(r'MGMT', filename):
				plot_funct.prettyplot3(n_cv, i, survivalloc, plot_funct.bin_to_cat(gs.loc['METHYLATED', :], 'METHYLATED' , 'UNMETHYLATED'), 'MGMT', analysis_directory, disease_type)
				plot_funct.prettyplot3(n_cv, i, survivalloc, plot_funct.bin_to_cat(gs.loc['WT', :], 'WT' , 'R132H') , 'IDH1', analysis_directory, disease_type)
			if re.search(r'stage', filename):
				if disease_type=="LUAD":
					plot_funct.prettyplot3(n_cv, i, survivalloc, gs, 'nstage', analysis_directory, disease_type)
				if disease_type=="LUSC":
					plot_funct.prettyplot3(n_cv, i, survivalloc, gs, 'tstage', analysis_directory, disease_type)
			if re.search(r'smoking', filename):
				plot_funct.prettyplot3(n_cv, i, survivalloc, gs, 'smoking', analysis_directory, disease_type)
			if re.search(r'gender', filename):
				plot_funct.prettyplot3(n_cv, i, survivalloc, plot_funct.bin_to_cat(gs.loc['female', :], 'female', 'male'), 'gender', analysis_directory, disease_type)

	if disease_type in ["SIM3", "SIM4"]:
		ngroup=3
		trueZ= pd.read_csv(data_directory+'not_in_use/Ztrue.csv' , sep=",", index_col=0)
		if os.path.exists(data_directory+'not_in_use/groups.csv'):
			groups=pd.read_csv(data_directory+'not_in_use/groups.csv', sep=',', index_col = 0)
		else:
			groups=survival_funct_v1.assign_group(trueZ, ngroup)
			groups.to_csv(data_directory+'not_in_use/groups.csv',  delimiter=',')
		plot_funct.prettyplot3_final_sim(plot_funct.make_categorical(groups.iloc[0, :]), 'subtype', root_directory, project_name, disease_type)

	if disease_type in ["LGG", "GBM", "LUAD", "LUSC"] and gold_standard == True:
		mod_sel = survival_funct_v1.gather_plot_cv_cindex(n_cv, analysis_directory, disease_type)
	else:
		n_cv=5
		mod_sel = survival_funct_v1.gather_plot_cv_cindex_sim(n_cv, analysis_directory, disease_type)