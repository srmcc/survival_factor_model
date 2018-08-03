from __future__ import division
import survival_funct_v1
import numpy as np
import pandas as pd
import click

#root_directory = '/home/smccurdy/scratch/survival/survival_factor_model/'

@click.command()
@click.option('--root_directory', default = '/home/smccurdy/scratch/survival/survival_factor_model/', prompt ='The root directory of the project', help='This is your directory /xx/xx/survival_factor_model/')
@click.option('--gold_standard', default= True, prompt='Do gold standard analysis?',
              help="True if [LGG, GBM, LUAD, LUSC] and if external files are downloaded, otherwise False")

def make_table2(root_directory, gold_standard):
	if gold_standard == 'True':
		gold_standard = True
	elif gold_standard =='False':
		gold_standard = False
	row_names = []
	for disease_type in ["LGG", "GBM", "LUAD", "LUSC"]:
		if gold_standard==True:
			row_names = row_names + 4* [disease_type]
		else:
			row_names = row_names + 3* [disease_type]
	for disease_type in ["SIM1", "SIM2", "SIM3", "SIM4"]:
		row_names = row_names + 3* [disease_type]
	if gold_standard== True:
		Table2 = pd.DataFrame(np.zeros((28, 5)), columns =["Disease", "Model no.", "Test Set c-index", "CV c-index mean", "CV c-index standard deviation"], index = row_names) 
	else:
		Table2 = pd.DataFrame(np.zeros((16, 5)), columns =["Disease", "Model no.", "Test Set c-index", "CV c-index mean", "CV c-index standard deviation"], index = row_names) 
	
	for disease_type in ["LGG", "GBM", "LUAD", "LUSC", "SIM1", "SIM2", "SIM3", "SIM4"]:
		mod_sel =pd.read_csv(root_directory + 'project_' +disease_type +'/analysis/model_selection.csv', index_col=[0])
		mod=survival_funct_v1.get_best(mod_sel.iloc[0:4, :])
		mod_cox = survival_funct_v1.get_best(mod_sel.iloc[4:7, :])
		mod_pca_cox = survival_funct_v1.get_best(mod_sel.iloc[7:10, :])
		means=np.mean(mod_sel, axis=1)
		means= np.round(means, 2)
		stdvs= np.std(mod_sel, axis=1)
		stdvs= np.round(stdvs, 2)
		if disease_type in ["LGG", "GBM", "LUAD", "LUSC"] and gold_standard == True:
			Table2.loc[disease_type, "Disease"] = [mod, 4+ mod_cox, 7+ mod_pca_cox, 11]
			Table2.loc[disease_type, "CV c-index mean"] = [means[mod], means[4+ mod_cox], means[7+ mod_pca_cox], means[11]]
			Table2.loc[disease_type, "CV c-index standard deviation"] = [stdvs[mod], stdvs[4+ mod_cox], stdvs[7+ mod_pca_cox], stdvs[11]]
			with open(root_directory + 'project_' +disease_type +'/analysis/final_fit_results/model_selection_output_final.txt', 'r') as f:
				with open(root_directory + 'project_' +disease_type +'/analysis/final_fit_results_cox/model_selection_output_coxfinal.txt', 'r') as g:
					with open(root_directory + 'project_' +disease_type +'/analysis/final_fit_results_pca_cox/model_selection_output_pca_coxfinal.txt', 'r') as h:
						with open(root_directory + 'project_' +disease_type +'/analysis/final_fit_results_cox_gs/model_selection_output_coxfinal.txt', 'r') as i:
							Table2.loc[disease_type, "Test Set c-index"] = [np.round(eval(f.readline())[-1][0], 2), np.round(eval(g.readline())[-1][0], 2), np.round(eval(h.readline())[-1][0], 2), np.round(eval(i.readline())[-1][0], 2)]
		else:
			Table2.loc[disease_type, "Disease"] = [mod, 4+ mod_cox, 7+ mod_pca_cox]
			Table2.loc[disease_type, "CV c-index mean"] = [means[mod], means[4+ mod_cox], means[7+ mod_pca_cox]]
			Table2.loc[disease_type, "CV c-index standard deviation"] = [stdvs[mod], stdvs[4+ mod_cox], stdvs[7+ mod_pca_cox]]
			with open(root_directory + 'project_' +disease_type +'/analysis/final_fit_results/model_selection_output_final.txt', 'r') as f:
				with open(root_directory + 'project_' +disease_type +'/analysis/final_fit_results_cox/model_selection_output_coxfinal.txt', 'r') as g:
					with open(root_directory + 'project_' +disease_type +'/analysis/final_fit_results_pca_cox/model_selection_output_pca_coxfinal.txt', 'r') as h:
						Table2.loc[disease_type, "Test Set c-index"] = [np.round(eval(f.readline())[-1][0], 2), np.round(eval(g.readline())[-1][0], 2), np.round(eval(h.readline())[-1][0], 2)]
		
	Table2.to_csv(root_directory + "table2.txt", sep = "&", line_terminator = "\\\n")

if __name__ == '__main__':
    make_table2()