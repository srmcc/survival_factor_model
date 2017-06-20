import os
import pandas as pd
import numpy as np
import scipy.stats
import glob
import re
import copy

def make_categorical(row):
	## row should be (samples, )
	row = row.replace(np.nan, 'missing')
	items = list(np.unique(row))
	empty = np.zeros((len(items), row.shape[0]))
	for s, samples in enumerate(row.index):
		i = items.index(row.loc[samples])  
		empty[i, s]=1
	cat = pd.DataFrame(empty, index = items, columns = row.index)
	if 'missing' in list(cat.index):
		cat.loc[:, (cat.loc['missing', :]==1)]= np.nan
		cat= cat.drop('missing')
	return(cat)

def clean_data(disease_type, data_directory, gold_standard):

	if disease_type=="LGG":
		data_types=["RNASeq2","miRNASeq", "Mutation","CNV_SNP", "CNA_SNP", "Methylation", "Clinical", "barcodes"]
		methyl_type="450K"
		gs_file_path = data_directory + 'not_in_use/tcga_lgg_nejmoa1402121_appendix_293.csv'
	if disease_type=="GBM":
		data_types=["RNASeq2", "Mutation","CNV_SNP", "CNA_SNP", "Methylation"]
		methyl_type="27K"
		gs_file_path = data_directory + 'not_in_use/tcga_gbm_mm7_544.csv'
	if disease_type=="LUAD":
		data_types=["RNASeq2","miRNASeq", "Mutation","CNV_SNP", "CNA_SNP"]
		gs_file_path = data_directory + 'not_in_use/tcga_luad_nature13385-s2.csv'
	if disease_type=="LUSC":
		data_types=["RNASeq2", "Mutation","CNV_SNP", "CNA_SNP", "Methylation"]
		methyl_type="27K"
		gs_file_path = data_directory + 'not_in_use/tcga_lusc_data.file.S7.5.clinical.and.genomic.data.table.csv'


	filenames =sorted(glob.glob(data_directory + 'not_in_use/*.csv'))
	if gs_file_path in filenames:
		filenames.remove(gs_file_path)
	for f in filenames:
		if re.search(r'plate', f):
			todrop = f
		else:
			todrop = False
	if todrop in filenames:
		filenames.remove(todrop)

	data_types=["RNASeq2","miRNASeq", "Mutation","CNV_SNP", "CNA_SNP", "Methylation", "Clinical"]
	data_names=[]
	for f in filenames:
		for item in data_types:
			if re.search(r'%s' %item, f):
				if item=='RNASeq2':
					if re.search(r'barcodes', f):
						data_names.append("RNASeq2_barcodes")
					else:
						data_names.append("RNASeq2")
				else:
					data_names.append(item)


	data=[]
	for f in filenames:
		dat = pd.read_csv(f, sep=',', index_col=0)
		newindex=[]
		for item in dat.index:
			if isinstance(item, str):
				newindex.append(item.replace('d1.', '').replace('d2.', ''))
			else:
				newindex.append(item)
		dat.index=newindex
		f_clin = re.search(r'Clinical', f)
		if f_clin:
			clinical=dat
			clinical.loc["daystodeath", :]=clinical.loc["daystodeath", :].astype("float")
			clinical.loc["daystolastfollowup", :]=clinical.loc["daystolastfollowup", :].astype("float")
			clinical.loc["vitalstatus", :]=clinical.loc["vitalstatus", :].astype("float")
			clinical.loc["eventtime", clinical.loc["vitalstatus", :]==1] =clinical.loc["daystodeath", clinical.loc["vitalstatus", :]==1]
			clinical.loc["eventtime", clinical.loc["vitalstatus", :]==0] =clinical.loc["daystolastfollowup", clinical.loc["vitalstatus", :]==0]
			clinical.loc["eventtime", :]= clinical.loc["eventtime", :].astype("float") 
			survival= clinical.loc[["eventtime", "vitalstatus"], :]
		f_bar = re.search(r'barcodes', f)
		if f_bar:
			barcodes=dat.T
			for item in barcodes.columns:
				barcodes.loc['sname', item]= barcodes.loc['x', item][0:12]
			for item in barcodes.columns:
				barcodes.loc['plate', item]= barcodes.loc['x', item][21:25]
			barcodes.columns=barcodes.loc['sname', :]
			barcodes=barcodes.loc[:, clinical.columns]
			barcodes=barcodes.drop('sname')
			barcodes.index=['fullbarcode', 'plate']
		data.append(dat)

	data.append(survival)
	data_names.append("Survival")
	data.append(barcodes)
	data_names.append("RNASeq2_barcodes_plate")

	missing_cutoff= 0.1
	cutoff= 0.3
	good_item=[]
	##how many datatypes will you be fitting in the model?
	nfiles= len(data)
	##do not change this line.
	col_names= ['file_location','datatype', 'b','sparserange','etime', 'CEN', 'Delta', 'CENtype']
	data_guide = pd.DataFrame(np.empty((nfiles, len(col_names)))*np.nan, columns=col_names)
	for i, item in enumerate(data):
		if data_names[i]=='RNASeq2_barcodes':
			continue
		print(data_names[i])
		print(item.shape)
		item=item.loc[:, np.isfinite(survival.iloc[0,:].astype(float))]
		print(item.shape)
		if data_names[i] == "Survival":
			data_guide.loc[i, 'file_location'] = 'tcga_' +disease_type+'_' +data_names[i]+ '_'+str(num_samp)+'.csv'
			data_guide.loc[i, 'datatype'] = 'survival'
			#this is the index name of the event time row
			data_guide.loc[i, 'etime'] = 'eventtime'
			#this is the index name of the event indicator row
			data_guide.loc[i, 'Delta'] = 'vitalstatus'
			#options for 'CEN' (censoring) are True (there is censoring in the data) or False (there is no censoring)
			data_guide.loc[i, 'CEN']= True
			#options for 'CENtype' are 'cind' (uninformative censoring) or 'cindIC' (informative censoring)
			data_guide.loc[i, 'CENtype'] = 'cindIC'
			item.to_csv(data_directory + 'tcga_' +disease_type+'_' +data_names[i]+ '_'+str(num_samp)+'.csv', sep=',')
			good_item.append(item)
		elif data_names[i] in ["Clinical","RNASeq2_barcodes_plate"]:
			item.to_csv(data_directory + 'not_in_use/tcga_' +disease_type+'_' +data_names[i]+ '_'+str(num_samp)+'.csv', sep=',')
			good_item.append(item)
			if data_names[i] == "Clinical":
				clinical = item
		else:
			num_samp=item.shape[1]
			print('num_samp', num_samp)
			if data_names[i] =="Mutation":
				data_guide.loc[i, 'datatype'] = 'binom'
				data_guide.loc[i, 'b'] = 1
			else:
				data_guide.loc[i, 'datatype'] = 'normal'		
			data_guide.loc[i, 'sparserange'] ='[[False, False]]'
			if data_names[i] =="Mutation" and disease_type in ["LUSC", "LUAD"]:
				cutoffp=0.1
			elif data_names[i] == "Methylation" and disease_type == "LGG":
				cutoffp=0.03
			else:
				cutoffp=cutoff
			print('cutoff', cutoffp)
			item= item[np.sum(np.isnan(item), axis=1) < missing_cutoff* num_samp]
			print(item.shape)
			item_var=np.nanvar(item, axis=1)
			item=item[scipy.stats.rankdata(item_var)> (1-cutoffp) *len(item_var)] 
			print(item.shape)
			item.to_csv(data_directory + 'tcga_' +disease_type+'_' +data_names[i]+ '_'+str(num_samp)+'_' +str(item.shape[0])+ '.csv', sep=',')
			data_guide.loc[i, 'file_location']= 'tcga_' +disease_type+'_' +data_names[i]+ '_'+str(num_samp)+'_' +str(item.shape[0])+ '.csv'
			good_item.append(item)
	data_guide.dropna(axis=0, how='all', inplace=True)
	data_guide.index = list(range(data_guide.shape[0]))
	data_guide.to_csv(data_directory + 'data_guide.csv')

	if gold_standard:
		if disease_type=="LGG":
			if not os.path.isfile(gs_file_path):
				print('gold standard file does not exist, exiting')
				#return
			gs = pd.read_csv(gs_file_path, sep=',', index_col=0)
			gs = gs.T
			gs = gs.replace('G1', 1)
			gs = gs.replace('G2', 2)
			gs = gs.replace('G3', 3)
			gs = gs.replace('Gross Total Resection', 3)
			gs = gs.replace('Subtotal Resection', 2)
			gs = gs.replace('Biopsy Only', 1)
			gs = gs.loc[:, clinical.columns]
			subtype = make_categorical(gs.loc['IDH/1p19q Subtype', :])
			subtype.to_csv(data_directory + 'tcga_LGG_subtype_' +str(num_samp)+ '.csv', sep= ',')
			age=pd.DataFrame(clinical.loc['yearstobirth'].reshape((1, num_samp)), index = ['yearstobirth'], columns= clinical.columns)
			age_grade_resection = pd.concat([age, gs.loc[['neoplasm_histologic_grade','cqcf_method_of_sample_procuremen'], clinical.columns]])
			age_grade_resection.to_csv(data_directory + 'tcga_LGG_age_grade_resection_'+str(num_samp)+ '.csv', sep= ',')
			nfiles = 2
			data_guide_gs = pd.DataFrame(np.empty((nfiles, len(col_names)))*np.nan, columns=col_names)
			data_guide_gs.loc[0, 'file_location'] = 'tcga_LGG_age_grade_resection_'+str(num_samp)+ '.csv'
			data_guide_gs.loc[0, 'datatype'] = 'normal'
			data_guide_gs.loc[1, 'file_location'] = 'tcga_LGG_subtype_' +str(num_samp)+ '.csv'
			data_guide_gs.loc[1, 'datatype'] = 'multinom'
			data_guide_gs.loc[1, 'b'] = 1
			data_guide_gs.loc[0:1, 'sparserange'] ='[[False, False]]'
			data_guide_gs=pd.concat([data_guide_gs,data_guide.loc[data_guide.loc[ :,'datatype']=='survival', :]])
			data_guide_gs.index = list(range(data_guide_gs.shape[0]))
			data_guide_gs.to_csv(data_directory + 'data_guide_gs.csv')
		if disease_type=="GBM":
			if not os.path.isfile(gs_file_path):
				print('gold standard file does not exist, exiting')
				#return
			gs = pd.read_csv(gs_file_path, sep=',', index_col=0)
			gs = gs.T
			gs = gs.loc[:, clinical.columns]
			subtype_exp = make_categorical(gs.loc['Expression\rSubclass', :]) 
			subtype_meth = make_categorical(gs.loc['MGMT Status', :])
			subtype_idh = make_categorical(gs.loc['IDH1\r status', :])
			subtype_exp.to_csv(data_directory + 'tcga_GBM_expression_subclass_' +str(num_samp)+'.csv', sep=',')
			subtype_mi = pd.concat([subtype_meth, subtype_idh])
			subtype_mi.loc[['METHYLATED', 'WT'],:].to_csv(data_directory + 'tcga_GBM_MGMT_IDH1_'+str(num_samp)+ '.csv', sep= ',')
			age=pd.DataFrame(clinical.loc['yearstobirth'].reshape((1, num_samp)), index = ['yearstobirth'], columns= clinical.columns)
			age.to_csv(data_directory + 'tcga_GBM_age_'+str(num_samp)+ '.csv', sep= ',')
			nfiles = 3
			data_guide_gs = pd.DataFrame(np.empty((nfiles, len(col_names)))*np.nan, columns=col_names)
			data_guide_gs.loc[0, 'file_location'] = 'tcga_GBM_age_'+str(num_samp)+ '.csv'
			data_guide_gs.loc[0, 'datatype'] = 'normal'
			data_guide_gs.loc[1, 'file_location'] = 'tcga_GBM_expression_subclass_' +str(num_samp)+'.csv'
			data_guide_gs.loc[1, 'datatype'] = 'multinom'
			data_guide_gs.loc[1, 'b'] = 1
			data_guide_gs.loc[2, 'file_location'] = 'tcga_GBM_MGMT_IDH1_'+str(num_samp)+ '.csv'
			data_guide_gs.loc[2, 'datatype'] = 'binom'
			data_guide_gs.loc[2, 'b'] = 1
			data_guide_gs.loc[0:2, 'sparserange'] ='[[False, False]]'
			data_guide_gs=pd.concat([data_guide_gs,data_guide.loc[data_guide.loc[ :,'datatype']=='survival', :]])
			data_guide_gs.index = list(range(data_guide_gs.shape[0]))
			data_guide_gs.to_csv(data_directory + 'data_guide_gs.csv')
		if disease_type=="LUAD":
			if not os.path.isfile(gs_file_path):
				print('gold standard file does not exist, exiting')
				#return
			gs = pd.read_csv(gs_file_path, sep=',', index_col=0)
			gs = gs.T
			gs = gs.loc[:, clinical.columns]
			subtype= make_categorical(gs.loc['expression_subtype', :])
			subtype.to_csv(data_directory + 'tcga_LUAD_subtype_' +str(num_samp)+'.csv', sep=',')
			nstage=make_categorical(clinical.loc['pathologyNstage', :])
			nstage.to_csv(data_directory + 'tcga_LUAD_nstage_'+str(num_samp)+'.csv', sep=',')
			nfiles = 2
			data_guide_gs = pd.DataFrame(np.empty((nfiles, len(col_names)))*np.nan, columns=col_names)
			data_guide_gs.loc[0, 'file_location'] = 'tcga_LUAD_subtype_' +str(num_samp)+'.csv'
			data_guide_gs.loc[0, 'datatype'] = 'multinom'
			data_guide_gs.loc[0, 'b'] = 1
			data_guide_gs.loc[1, 'file_location'] = 'tcga_LUAD_nstage_'+str(num_samp)+'.csv'
			data_guide_gs.loc[1, 'datatype'] = 'multinom'
			data_guide_gs.loc[1, 'b'] = 1
			data_guide_gs.loc[0:1, 'sparserange'] ='[[False, False]]'
			data_guide_gs=pd.concat([data_guide_gs,data_guide.loc[data_guide.loc[ :,'datatype']=='survival', :]])
			data_guide_gs.index = list(range(data_guide_gs.shape[0]))
			data_guide_gs.to_csv(data_directory + 'data_guide_gs.csv')
		if disease_type=="LUSC":
			if not os.path.isfile(gs_file_path):
				print('gold standard file does not exist, exiting')
				#return
			gs= pd.read_csv(gs_file_path, sep=',', index_col=0)
			gs=gs.T
			gs=gs.loc[:, clinical.columns]
			smoking= make_categorical(gs.loc['Smoking Status', :])
			smoking.to_csv(data_directory + 'tcga_LUSC_smoking_'+str(num_samp)+ '.csv', sep= ',')
			age=pd.DataFrame(clinical.loc['yearstobirth'].reshape((1, num_samp)), index = ['yearstobirth'], columns= clinical.columns)
			age.to_csv(data_directory + 'tcga_LUSC_age_'+str(num_samp)+ '.csv', sep= ',')
			gender=make_categorical(clinical.loc['gender', :])
			gender.to_csv(data_directory + 'tcga_LUSC_gender_'+str(num_samp)+ '.csv', sep= ',')
			Tstage=make_categorical(clinical.loc['pathologyTstage', :])
			Tstage.to_csv(data_directory + 'tcga_LUSC_tstage_'+str(num_samp)+ '.csv', sep=',')
			nfiles = 4
			data_guide_gs = pd.DataFrame(np.empty((nfiles, len(col_names)))*np.nan, columns=col_names)
			data_guide_gs.loc[0, 'file_location'] = 'tcga_LUSC_age_'+str(num_samp)+ '.csv'
			data_guide_gs.loc[0, 'datatype'] = 'normal'
			data_guide_gs.loc[1, 'file_location'] =  'tcga_LUSC_gender_'+str(num_samp)+ '.csv'
			data_guide_gs.loc[1, 'datatype'] = 'binom'
			data_guide_gs.loc[1, 'b'] = 1
			data_guide_gs.loc[2, 'file_location'] = 'tcga_LUSC_tstage_'+str(num_samp)+ '.csv'
			data_guide_gs.loc[2, 'datatype'] = 'multinom'
			data_guide_gs.loc[2, 'b'] = 1
			data_guide_gs.loc[3, 'file_location'] = 'tcga_LUSC_smoking_'+str(num_samp)+ '.csv'
			data_guide_gs.loc[3, 'datatype'] = 'multinom'
			data_guide_gs.loc[3, 'b'] = 1
			data_guide_gs.loc[0:3, 'sparserange'] ='[[False, False]]'
			data_guide_gs=pd.concat([data_guide_gs,data_guide.loc[data_guide.loc[ :,'datatype']=='survival', :]])
			data_guide_gs.index = list(range(data_guide_gs.shape[0]))
			data_guide_gs.to_csv(data_directory + 'data_guide_gs.csv')
