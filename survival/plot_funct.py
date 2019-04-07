from __future__ import division
from time import time
import sys
import os
import numpy as np
import scipy
import pandas as pd
##plotting commands
import matplotlib
#workaround for x - windows
matplotlib.use('Agg')
#from ggplot import *
import pylab as pl
import matplotlib.pyplot as plt

if matplotlib.__version__[0] != '1':
    matplotlib.style.use('classic')

def bin_to_cat(vector, oldname, newname):
	newvector= pd.concat([vector, np.ones(vector.shape) - vector], axis=1)
	newvector.columns= [oldname, newname]
	newvector= newvector.transpose()
	return newvector 

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

def prettyplot(n_cv, i, survivalloc, analysis_directory, disease_type):
	##potential bug later on--> i used twice.
	cv_testsamples= pd.read_csv(analysis_directory + 'cv_'+str(n_cv)+ '_samples_splits/cv_sample_split_' +str(i)+'.csv', sep=',', index_col=0)
	testsamples= pd.read_csv(analysis_directory + 'test_samples.csv', sep=',', index_col=0)
	zloc = 'cv_'+str(n_cv)+ '_results/cv_run'+str(i)+'/model_0_0/learn/expected_values/EZ_given_x.csv'
	zloc_v = 'cv_'+str(n_cv)+ '_results/cv_run'+str(i)+'/model_0_0/val/EZ_given_x.csv'
	survival = pd.read_csv(survivalloc, sep=",", index_col=0)
	eps= np.min(survival.ix[0, survival.iloc[0, :]!=0])
	survival.ix[0, survival.iloc[0, :]==0]= eps/10  #replacing zero survival times with 1/10 the smallest time.
	survival_v = survival.loc[:, list(cv_testsamples.values[:,0])] 
	survival= survival.drop(list(cv_testsamples.values[:,0]), axis=1)
	survival= survival.drop(list(testsamples.values[:,0]), axis=1)
	EZ = pd.read_csv(analysis_directory+ zloc, sep=",", index_col=0)
	EZ.columns=survival.columns 
	t= np.log(survival.iloc[0, :])
	delta= survival.iloc[1, :]
	x = EZ.iloc[0, :]
	y = EZ.iloc[1, :]
	fig, ax = plt.subplots()
	#cm = plt.cm.get_cmap('Greys')
	cm = plt.cm.get_cmap('winter')
	markers = ['o', 'v']
	vminn= np.min(t)
	print("miminum value", vminn)
	vmaxx=np.max(t) 
	for k, m in enumerate(markers):
		if m== 'o':
			j= (delta==1)
		elif m=='v':
			j= (delta==0)
		#j= (m=='o')*(delta==1) +(m=='v')*(delta==0)
		im=ax.scatter(x[j], y[j], c=t[j], cmap= cm, marker=m, s=100, vmin= vminn, vmax = vmaxx, edgecolors='black')
	cbar=fig.colorbar(im, ax=ax)
	cbar.ax.tick_params(labelsize=24)
	ax.tick_params(axis='both', labelsize=24)
	ax.set_xlabel(r'$E[z_1| x]$', fontsize=24)
	ax.set_ylabel(r'$E[z_2| x]$', fontsize=24)
	#labels = [item.get_text() for item in ax.get_xticklabels()]
	labels = ax.get_xticklabels()
	print(labels)
	ax.set_xticklabels(labels, rotation='vertical')
	#ax.arrow(0, 0, paramt[0][1][0][0][0], paramt[0][1][0][0][1], head_width=.3, head_length=.3, fc='k', ec='k')
	# ax.set_title('Volume and percent change')
	# ax.legend()
	fig.tight_layout()
	# plt.savefig(analysis_directory+'cv_'+str(n_cv)+ '_results/cv_run'+str(i)+'/model_0_0/learn/expected_values/'+disease_type+ '_EZ_given_x_grey_eventtime.pdf',
	#             bbox_inches='tight')
	plt.savefig(analysis_directory+'cv_'+str(n_cv)+ '_results/cv_run'+str(i)+'/model_0_0/learn/expected_values/'+disease_type+ '_EZ_given_x_color.pdf', bbox_inches='tight')
	# plt.savefig(analysis_directory+'cv_'+str(n_cv)+ '_results/cv_run'+str(i)+'/model_0_0/learn/expected_values/'+disease_type+ '_EZ_given_x_color_eventtime.pdf', bbox_inches='tight')
	#plt.savefig(analysis_directory+'cv_'+str(n_cv)+ '_results/cv_run'+str(i)+'/model_0_0/learn/expected_values/'+disease_type+ '_EZ_given_x_grey_eventtime.eps', format='eps', dpi=1200, bbox_inches='tight')
	plt.savefig(analysis_directory+'cv_'+str(n_cv)+ '_results/cv_run'+str(i)+'/model_0_0/learn/expected_values/'+disease_type+ '_EZ_given_x_color.eps', format='eps', dpi=1200, bbox_inches='tight')
	ylim= ax.get_ylim()
	xlim= ax.get_xlim()
	## now coding on the validation set. scale is set by learning set, axis are set by learning set. 
	EZ_v = pd.read_csv(analysis_directory+ zloc_v, sep=",", index_col=0)
	EZ_v.columns=survival_v.columns
	t=np.log( survival_v.iloc[0, :])
	delta= survival_v.iloc[1, :]
	x = EZ_v.iloc[0, :]
	y = EZ_v.iloc[1, :]
	figv, axv = plt.subplots()
	##remove this, makes it use different axes
	vminn= np.min(t)
	vmaxx=np.max(t) 
	print(vmaxx, vminn)
	for k, m in enumerate(markers):
		if m== 'o':
			j= (delta==1)
		elif m=='v':
			j= (delta==0)
		#j= (m=='o')*(delta==1) +(m=='v')*(delta==0)
		im=axv.scatter(x[j], y[j], c=t[j], cmap= cm, marker=m, s=100, vmin= vminn, vmax = vmaxx, edgecolors='black')
	cbarv=figv.colorbar(im, ax=axv)
	cbarv.ax.tick_params(labelsize=24)
	axv.tick_params(axis='both', labelsize=24)
	axv.set_xlabel(r'$E[z_1| x]$', fontsize=24)
	axv.set_ylabel(r'$E[z_2| x]$', fontsize=24)
	labels = [item.get_text() for item in axv.get_xticklabels()]
	axv.set_xticklabels(labels, rotation='vertical')
	#ax.arrow(0, 0, paramt[0][1][0][0][0], paramt[0][1][0][0][1], head_width=.3, head_length=.3, fc='k', ec='k')
	# ax.set_title('Volume and percent change')
	# ax.legend()
	axv.set_ylim(ylim)
	axv.set_xlim(xlim)
	figv.tight_layout()
	plt.savefig(analysis_directory+'cv_'+str(n_cv)+ '_results/cv_run'+str(i)+'/model_0_0/val/'+disease_type+ '_EZ_given_x_val_color.pdf', bbox_inches='tight')
	#plt.savefig(analysis_directory+'cv_'+str(n_cv)+ '_results/cv_run'+str(i)+'/model_0_0/val/'+ disease_type+ '_EZ_given_x_grey_eventtime.pdf', bbox_inches='tight')
	plt.savefig(analysis_directory+'cv_'+str(n_cv)+ '_results/cv_run'+str(i)+'/model_0_0/val/'+ disease_type+ '_EZ_given_x_val_color.eps', format='eps', dpi = 1200, bbox_inches='tight')


def prettyplot2(n_cv, i, survivalloc, colorvector, colorname, analysis_directory, disease_type):
	cv_testsamples= pd.read_csv(analysis_directory + 'cv_'+str(n_cv)+ '_samples_splits/cv_sample_split_' +str(i)+'.csv', sep=',', index_col=0)
	testsamples= pd.read_csv(analysis_directory + 'test_samples.csv', sep=',', index_col=0)
	zloc = 'cv_'+str(n_cv)+ '_results/cv_run'+str(i)+'/model_0_0/learn/expected_values/EZ_given_x.csv'
	survival = pd.read_csv(survivalloc, sep=",", index_col=0)
	survival= survival.drop(list(cv_testsamples.values[:,0]), axis=1)
	survival= survival.drop(list(testsamples.values[:,0]), axis=1)
	colorvector = colorvector.drop(list(cv_testsamples.values[:,0]))
	colorvector = colorvector.drop(list(testsamples.values[:,0])) 
	EZ = pd.read_csv(analysis_directory+ zloc, sep=",", index_col=0)
	EZ.columns=survival.columns 
	t= survival.iloc[0, :]
	delta= survival.iloc[1, :]
	x = EZ.iloc[0, :]
	y = EZ.iloc[1, :]
	fig, ax = plt.subplots()
	cm = plt.cm.get_cmap('winter')
	#cm = plt.cm.get_cmap('Greys')
	markers = ['o', 'v']
	vminn= np.min(colorvector)
	vmaxx=np.max(colorvector) 
	for k, m in enumerate(markers):
		if m== 'o':
			j= (delta==1)
		elif m=='v':
			j= (delta==0)
		im=ax.scatter(x[j], y[j], c=colorvector[j], cmap=cm, marker=m, s=100, vmin= vminn, vmax = vmaxx, edgecolors='black')
	cbar= fig.colorbar(im, ax=ax)
	cbar.ax.tick_params(labelsize=24)
	ax.tick_params(axis='both', labelsize=24)
	ax.set_xlabel(r'$E[z_1| x]$', fontsize=24)
	ax.set_xlabel(r'$E[z_2| x]$', fontsize=24)
	labels = [item.get_text() for item in ax.get_xticklabels()]
	ax.set_xticklabels(labels, rotation='vertical')
	#ax.arrow(0, 0, paramt[0][1][0][0][0], paramt[0][1][0][0][1], head_width=.3, head_length=.3, fc='k', ec='k')
	# ax.set_title('Volume and percent change')
	#ax.legend()
	fig.tight_layout()
	plt.savefig(analysis_directory+'cv_'+str(n_cv)+ '_results/cv_run'+str(i)+'/model_0_0/learn/expected_values/'+disease_type+'_EZ_given_x_' + colorname + '_color.pdf', bbox_inches='tight')
	plt.savefig(analysis_directory+'cv_'+str(n_cv)+ '_results/cv_run'+str(i)+'/model_0_0/learn/expected_values/'+disease_type+'_EZ_given_x_' + colorname + '_color.eps', format='eps', dpi=1200, bbox_inches='tight')

def prettyplot3(n_cv, i, survivalloc, colorcat, colorname, analysis_directory, disease_type):
	cv_testsamples= pd.read_csv(analysis_directory + 'cv_'+str(n_cv)+ '_samples_splits/cv_sample_split_' +str(i)+'.csv', sep=',', index_col=0)
	testsamples= pd.read_csv(analysis_directory + 'test_samples.csv', sep=',', index_col=0)
	zloc = 'cv_'+str(n_cv)+ '_results/cv_run'+str(i)+'/model_0_0/learn/expected_values/EZ_given_x.csv'
	survival = pd.read_csv(survivalloc, sep=",", index_col=0)
	survival= survival.drop(list(cv_testsamples.values[:,0]), axis=1)
	survival= survival.drop(list(testsamples.values[:,0]), axis=1)
	colorcat = colorcat.drop(list(cv_testsamples.values[:,0]), axis=1)
	colorcat = colorcat.drop(list(testsamples.values[:,0]), axis=1) 
	if np.sum(np.sum(np.isnan(colorcat)))!=0:
		colorcat.loc['Missing']=0
		colorcat.loc['Missing', (np.sum(np.isnan(colorcat))!=0)]=1
	EZ = pd.read_csv(analysis_directory+ zloc, sep=",", index_col=0)
	EZ.columns=survival.columns 
	fig, ax = plt.subplots()
	#cm = plt.cm.get_cmap('rainbow')
	#cm = plt.cm.get_cmap('Greys')
	cm = plt.cm.get_cmap('winter')
	vminn= np.min(range(colorcat.shape[0]))-1
	vmaxx= np.max(range(colorcat.shape[0]))
	colorlist=matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vminn, vmax=vmaxx, clip=False), cmap=cm).to_rgba(range(colorcat.shape[0]))
	markers = ['o', 'v']
	leg=[]
	names=[]
	for l in range(colorcat.shape[0]):
		x = EZ.ix[0, colorcat.loc[:, (colorcat.iloc[l, :]==1)].columns]
		y = EZ.ix[1, colorcat.loc[:, (colorcat.iloc[l, :]==1)].columns]
		delta= survival.ix[1, colorcat.loc[:, (colorcat.iloc[l, :]==1)].columns]
		for k, m in enumerate(markers):
			if m== 'o':
				j= (delta==1)
			elif m=='v':
				j= (delta==0)
			#j= (m=='o')*(delta==1) +(m=='v')*(delta==0)
			#j= (m=='o')&(delta==1) | (m=='v')&(delta==0)
			if k==0:
				im=ax.scatter(x[j], y[j], color=colorlist[l], edgecolors='black', marker=m, s=100)
				leg.append(im)
				names.append(colorcat.index[l]) 
				print(k, l, colorlist[l], x[j].shape)	
			else:
				im=ax.scatter(x[j], y[j], color=colorlist[l], edgecolors='black', marker=m, s=100)
				# leg.append(im)
			# 	names.append(colorcat.index[l]) 
			# 	print(k, l, colorlist[l], x[j].shape)
	#fig.colorbar(im, ax=ax)
	ax.legend(leg, names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=24)
	ax.tick_params(axis='both', labelsize=24)
	ax.set_xlabel(r'$E[z_1| x]$', fontsize=24)
	ax.set_ylabel(r'$E[z_2| x]$', fontsize=24)
	labels = [item.get_text() for item in ax.get_xticklabels()]
	ax.set_xticklabels(labels, rotation='vertical')
	#ax.arrow(0, 0, paramt[0][1][0][0][0], paramt[0][1][0][0][1], head_width=.3, head_length=.3, fc='k', ec='k')
	# ax.set_title('Volume and percent change')
	fig.tight_layout()
	# plt.savefig(analysis_directory+'cv_'+str(n_cv)+ '_results/cv_run'+str(i)+'/model_0_0/learn/expected_values/'+disease_type+'_EZ_given_x_' + colorname + '.pdf', bbox_inches='tight')
	# plt.savefig(analysis_directory+'cv_'+str(n_cv)+ '_results/cv_run'+str(i)+'/model_0_0/learn/expected_values/'+disease_type+'_EZ_given_x_' + colorname + '.eps', format='eps', dpi=1200, bbox_inches='tight')
	plt.savefig(analysis_directory+'cv_'+str(n_cv)+ '_results/cv_run'+str(i)+'/model_0_0/learn/expected_values/'+disease_type+'_EZ_given_x_' + colorname + '_color.pdf', bbox_inches='tight')
	plt.savefig(analysis_directory+'cv_'+str(n_cv)+ '_results/cv_run'+str(i)+'/model_0_0/learn/expected_values/'+disease_type+'_EZ_given_x_' + colorname + '_color.eps', format='eps', dpi=1200, bbox_inches='tight')


def prettyplot3_pca(n_cv, i, survivalloc, colorcat, colorname, analysis_directory, disease_type):
	cv_testsamples= pd.read_csv(analysis_directory + 'cv_'+str(n_cv)+ '_samples_splits/cv_sample_split_' +str(i)+'.csv', sep=',', index_col=0)
	testsamples= pd.read_csv(analysis_directory + 'test_samples.csv', sep=',', index_col=0)
	zloc = 'cv_'+str(n_cv)+ '_results_pca_cox/cv_run'+str(i)+'/model_pca_cox_0/learn_pca_cox/lamdaVxT_dz.csv'
	survival = pd.read_csv(survivalloc, sep=",", index_col=0)
	survival= survival.drop(list(cv_testsamples.values[:,0]), axis=1)
	survival= survival.drop(list(testsamples.values[:,0]), axis=1)
	colorcat = colorcat.drop(list(cv_testsamples.values[:,0]), axis=1)
	colorcat = colorcat.drop(list(testsamples.values[:,0]), axis=1) 
	if np.sum(np.sum(np.isnan(colorcat)))!=0:
		colorcat.loc['Missing']=0
		colorcat.loc['Missing', (np.sum(np.isnan(colorcat))!=0)]=1
	EZ = pd.read_csv(analysis_directory+ zloc, sep=",", index_col=0)
	EZ.columns=survival.columns 
	fig, ax = plt.subplots()
	#cm = plt.cm.get_cmap('rainbow')
	cm = plt.cm.get_cmap('Greys')
	#cm = plt.cm.get_cmap('winter')
	vminn= np.min(range(colorcat.shape[0]))-1
	vmaxx= np.max(range(colorcat.shape[0]))
	colorlist=matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vminn, vmax=vmaxx, clip=False), cmap=cm).to_rgba(range(colorcat.shape[0]))
	markers = ['o', 'v']
	leg=[]
	names=[]
	for l in range(colorcat.shape[0]):
		x = EZ.ix[0, colorcat.loc[:, (colorcat.iloc[l, :]==1)].columns]
		y = EZ.ix[1, colorcat.loc[:, (colorcat.iloc[l, :]==1)].columns]
		delta= survival.ix[1, colorcat.loc[:, (colorcat.iloc[l, :]==1)].columns]
		for k, m in enumerate(markers):
			if m== 'o':
				j= (delta==1)
			elif m=='v':
				j= (delta==0)
			#j= (m=='o')*(delta==1) +(m=='v')*(delta==0)
			#j= (m=='o')&(delta==1) | (m=='v')&(delta==0)
			if k==0:
				im=ax.scatter(x[j], y[j], color=colorlist[l], edgecolors='black', marker=m, s=100)
				leg.append(im)
				names.append(colorcat.index[l]) 
				print(k, l, colorlist[l], x[j].shape)	
			else:
				im=ax.scatter(x[j], y[j], color=colorlist[l], edgecolors='black', marker=m, s=100)
				# leg.append(im)
			# 	names.append(colorcat.index[l]) 
			# 	print(k, l, colorlist[l], x[j].shape)
	#fig.colorbar(im, ax=ax)
	ax.legend(leg, names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=24)
	ax.tick_params(axis='both', labelsize=24)
	ax.set_xlabel(r'$U_1$', fontsize=24)
	ax.set_ylabel(r'$U_2$', fontsize=24)
	labels = [item.get_text() for item in ax.get_xticklabels()]
	ax.set_xticklabels(labels, rotation='vertical')
	#ax.arrow(0, 0, paramt[0][1][0][0][0], paramt[0][1][0][0][1], head_width=.3, head_length=.3, fc='k', ec='k')
	# ax.set_title('Volume and percent change')
	fig.tight_layout()
	# plt.savefig(analysis_directory+'cv_'+str(n_cv)+ '_results/cv_run'+str(i)+'/model_0_0/learn/expected_values/'+disease_type+'_EZ_given_x_' + colorname + '.pdf', bbox_inches='tight')
	# plt.savefig(analysis_directory+'cv_'+str(n_cv)+ '_results/cv_run'+str(i)+'/model_0_0/learn/expected_values/'+disease_type+'_EZ_given_x_' + colorname + '.eps', format='eps', dpi=1200, bbox_inches='tight')
	plt.savefig(analysis_directory+'cv_'+str(n_cv)+ '_results_pca_cox/cv_run'+str(i)+'/model_pca_cox_0/learn_pca_cox/'+disease_type+'_U_given_x_' + colorname + '_color.pdf', bbox_inches='tight')
	plt.savefig(analysis_directory+'cv_'+str(n_cv)+ '_results_pca_cox/cv_run'+str(i)+'/model_pca_cox_0/learn_pca_cox/'+disease_type+'_U_given_x_' + colorname + '_color.eps', format='eps', dpi=1200, bbox_inches='tight')



def prettyplot_final(survivalloc, analysis_directory, disease_type):
	##potential bug later on--> i used twice.
	#cv_testsamples= pd.read_csv(analysis_directory + 'cv_'+str(n_cv)+ '_samples_splits/cv_sample_split_' +str(i)+'.csv', sep=',', index_col=0)
	testsamples= pd.read_csv(analysis_directory + 'test_samples.csv', sep=',', index_col=0)
	zloc = 'final_fit_results/model_0_0/learn/expected_values/EZ_given_x.csv'
	zloc_v = 'final_fit_results/model_0_0/val/EZ_given_x.csv'
	survival = pd.read_csv(survivalloc, sep=",", index_col=0)
	survival_v = survival.loc[:, list(testsamples.values[:,0])] 
	survival= survival.drop(list(testsamples.values[:,0]), axis=1)
	EZ = pd.read_csv(analysis_directory+ zloc, sep=",", index_col=0)
	EZ.columns=survival.columns 
	t= survival.iloc[0, :]
	delta= survival.iloc[1, :]
	for z1 in range(EZ.shape[0]):
		for z2 in range(z1 +1):
			if z2< z1:
				x = EZ.iloc[z2, :]
				y = EZ.iloc[z1, :]
				fig, ax = plt.subplots()
				#cm = plt.cm.get_cmap('Greys')
				cm = plt.cm.get_cmap('winter')
				markers = ['o', 'v']
				vminn= np.min(t)
				vmaxx=np.max(t) 
				for k, m in enumerate(markers):
					if m== 'o':
						j= (delta==1)
					elif m=='v':
						j= (delta==0)
					#j= (m=='o')*(delta==1) +(m=='v')*(delta==0)
					im=ax.scatter(x[j], y[j], c=t[j], cmap= cm, marker=m, s=100, vmin= vminn, vmax = vmaxx, edgecolors= 'black')
				cbar=fig.colorbar(im, ax=ax)
				cbar.ax.tick_params(labelsize=24)
				ax.tick_params(axis='both', labelsize=24)
				ax.set_xlabel(r'$E[z_%d| x]$' % (z2 +1), fontsize=24)
				ax.set_ylabel(r'$E[z_%d| x]$' %(z1 +1), fontsize=24)
				labels = [item.get_text() for item in ax.get_xticklabels()]
				ax.set_xticklabels(labels, rotation='vertical')
				#ax.arrow(0, 0, paramt[0][1][0][0][0], paramt[0][1][0][0][1], head_width=.3, head_length=.3, fc='k', ec='k')
				# ax.set_title('Volume and percent change')
				# ax.legend()
				fig.tight_layout()
				plt.savefig(analysis_directory+'final_fit_results/model_0_0/learn/expected_values/'+disease_type +'_EZ_given_x_'+str(z1)+'_'+ str(z2)+ '_color.pdf',
				            bbox_inches='tight')
				plt.savefig(analysis_directory+'final_fit_results/model_0_0/learn/expected_values/'+disease_type +'_EZ_given_x_'+str(z1)+'_'+ str(z2)+ '_color.eps', format='eps', dpi=1200,
				            bbox_inches='tight')
				ylim= ax.get_ylim()
				xlim= ax.get_xlim()
	## now coding on the validation set. scale is set by learning set, axis are set by learning set. 
	EZ_v = pd.read_csv(analysis_directory+ zloc_v, sep=",", index_col=0)
	EZ_v.columns=survival_v.columns
	t= survival_v.iloc[0, :]
	delta= survival_v.iloc[1, :]
	for z1 in range(EZ.shape[0]):
		for z2 in range(z1 +1):
			if z2<z1:
				x = EZ_v.iloc[z2, :]
				y = EZ_v.iloc[z1, :]
				figv, axv = plt.subplots()
				for k, m in enumerate(markers):
					if m== 'o':
						j= (delta==1)
					elif m=='v':
						j= (delta==0)
					#j= (m=='o')*(delta==1) +(m=='v')*(delta==0)
					im_v=axv.scatter(x[j], y[j], c=t[j], cmap= cm, marker=m, s=100, vmin= vminn, vmax = vmaxx, edgecolors ='black')
				cbarv=figv.colorbar(im_v, ax=axv)
				cbarv.ax.tick_params(labelsize=24)
				axv.tick_params(axis='both', labelsize=24)
				axv.set_xlabel(r'$E[z_%d| x]$' %(z2 +1), fontsize=24)
				axv.set_ylabel(r'$E[z_%d| x]$' %(z1 +1), fontsize=24)
				labels = [item.get_text() for item in axv.get_xticklabels()]
				axv.set_xticklabels(labels, rotation='vertical')
				#ax.arrow(0, 0, paramt[0][1][0][0][0], paramt[0][1][0][0][1], head_width=.3, head_length=.3, fc='k', ec='k')
				# ax.set_title('Volume and percent change')
				# ax.legend()
				axv.set_ylim(ylim)
				axv.set_xlim(xlim)
				figv.tight_layout()
				plt.savefig(analysis_directory+'final_fit_results/model_0_0/val/'+ disease_type+ '_EZ_given_x_'+str(z1)+'_'+ str(z2)+ '_color.pdf',
				            bbox_inches='tight')
				plt.savefig(analysis_directory+'final_fit_results/model_0_0/val/'+ disease_type+ '_EZ_given_x_'+str(z1)+'_'+ str(z2)+ '_color.eps', format='eps', dpi=1200,
				            bbox_inches='tight')


def prettyplot3_final(survivalloc, colorcat, colorname, analysis_directory, disease_type):
	testsamples= pd.read_csv(analysis_directory + 'test_samples.csv', sep=',', index_col=0)
	zloc = 'final_fit_results/model_0_0/learn/expected_values/EZ_given_x.csv'
	zloc_v = 'final_fit_results/model_0_0/val/EZ_given_x.csv'
	survival = pd.read_csv(survivalloc, sep=",", index_col=0)
	survival_v = survival.loc[:, list(testsamples.values[:,0])] 
	survival= survival.drop(list(testsamples.values[:,0]), axis=1)
	#t= survival.iloc[0, :]
	#delta= survival.iloc[1, :]
	if np.sum(np.sum(np.isnan(colorcat)))!=0:
		colorcat.loc['Missing']=0
		colorcat.loc['Missing', (np.sum(np.isnan(colorcat))!=0)]=1
	colorcat_v=colorcat.loc[:, list(testsamples.values[:,0])]
	colorcat = colorcat.drop(list(testsamples.values[:,0]), axis=1) 
	EZ = pd.read_csv(analysis_directory+ zloc, sep=",", index_col=0)
	EZ.columns=survival.columns 
	#cm = plt.cm.get_cmap('rainbow')
	#cm = plt.cm.get_cmap('Greys')
	cm = plt.cm.get_cmap('winter')
	vminn= np.min(range(colorcat.shape[0]))-1
	vmaxx= np.max(range(colorcat.shape[0]))
	colorlist=matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vminn, vmax=vmaxx, clip=False), cmap=cm).to_rgba(range(colorcat.shape[0]))
	markers = ['o', 'v']
	for z1 in range(EZ.shape[0]):
		for z2 in range(z1 +1):
			if z2<z1:
				fig, ax = plt.subplots()
				leg=[]
				names=[]
				for l in range(colorcat.shape[0]):
					x = EZ.ix[z2, colorcat.loc[:, (colorcat.iloc[l, :]==1)].columns]
					y = EZ.ix[z1, colorcat.loc[:, (colorcat.iloc[l, :]==1)].columns]
					delta= survival.ix[1, colorcat.loc[:, (colorcat.iloc[l, :]==1)].columns]
					for k, m in enumerate(markers):
						if m== 'o':
							j= (delta==1)
						elif m=='v':
							j= (delta==0)
						#j= (m=='o')*(delta==1) +(m=='v')*(delta==0)
						#j= (m=='o')&(delta==1) | (m=='v')&(delta==0)
						if k==0:
							im=ax.scatter(x[j], y[j], color=colorlist[l], edgecolors='black', marker=m, s=100)
							leg.append(im)
							names.append(colorcat.index[l]) 
							print(k, l, colorlist[l], x[j].shape, m)	
						else:
							im=ax.scatter(x[j], y[j], color=colorlist[l], edgecolors='black', marker=m, s=100)
							# leg.append(im)
						# 	names.append(colorcat.index[l]) 
						# 	print(k, l, colorlist[l], x[j].shape)
				#fig.colorbar(im, ax=ax)
				ax.legend(leg, names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=24)
				ax.tick_params(axis='both', labelsize=24)
				ax.set_xlabel(r'$E[z_%d| x]$'%(z2 +1), fontsize=24)
				ax.set_ylabel(r'$E[z_%d| x]$'%(z1 +1), fontsize=24)
				labels = [item.get_text() for item in ax.get_xticklabels()]
				ax.set_xticklabels(labels, rotation='vertical')
				#ax.arrow(0, 0, paramt[0][1][0][0][0], paramt[0][1][0][0][1], head_width=.3, head_length=.3, fc='k', ec='k')
				# ax.set_title('Volume and percent change')
				fig.tight_layout()
				plt.savefig(analysis_directory+'final_fit_results/model_0_0/learn/expected_values/'+disease_type+'_EZ_given_x_'+str(z1)+'_'+ str(z2) +'_'+ colorname + '_color.pdf', bbox_inches='tight')
				plt.savefig(analysis_directory+'final_fit_results/model_0_0/learn/expected_values/'+disease_type+'_EZ_given_x_'+str(z1)+'_'+ str(z2) +'_'+ colorname + '_color.eps', format='eps', dpi=1200, bbox_inches='tight')
	# figLegend = pl.figure()
	# pl.figlegend(*ax.get_legend_handles_labels(), loc = 'upper left')
	# figLegend.savefig(analysis_directory+'final_fit_results/model_0_0/learn/expected_values/'+disease_type+'_'+ colorname + '_EZ_given_x_legend.pdf')
	EZ_v = pd.read_csv(analysis_directory+ zloc_v, sep=",", index_col=0)
	EZ_v.columns=survival_v.columns
	#t= survival_v.iloc[0, :]
	#delta= survival_v.iloc[1, :]
	markers = ['o', 'v']
	for z1 in range(EZ.shape[0]):
		for z2 in range(z1 +1):
			if z2<z1:
				fig_v, ax_v = plt.subplots()
				leg_v=[]
				names_v=[]
				for l in range(colorcat_v.shape[0]):
					x = EZ_v.ix[z2, colorcat_v.loc[:, (colorcat_v.iloc[l, :]==1)].columns]
					y = EZ_v.ix[z1, colorcat_v.loc[:, (colorcat_v.iloc[l, :]==1)].columns]
					delta= survival_v.ix[1, colorcat_v.loc[:, (colorcat_v.iloc[l, :]==1)].columns]
					for k, m in enumerate(markers):
						if m== 'o':
							j= (delta==1)
						elif m=='v':
							j= (delta==0)
						#j= (m=='o')*(delta==1) +(m=='v')*(delta==0)
						#j= (m=='o')&(delta==1) | (m=='v')&(delta==0)
						if k==0:
							im_v=ax_v.scatter(x[j], y[j], color=colorlist[l], edgecolors='black', marker=m, s=100)
							leg_v.append(im_v)
							names_v.append(colorcat_v.index[l]) 
							print(k, l, colorlist[l], x[j].shape, m)	
						else:
							im_v=ax_v.scatter(x[j], y[j], color=colorlist[l], edgecolors='black', marker=m, s=100)
							# leg.append(im)
						# 	names.append(colorcat.index[l]) 
						# 	print(k, l, colorlist[l], x[j].shape)
				#fig.colorbar(im, ax=ax)
				ax_v.legend(leg_v, names_v, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=24)
				ax_v.tick_params(axis='both', labelsize=24)
				ax_v.set_xlabel(r'$E[z_%d| x]$'%(z2 +1), fontsize=24)
				ax_v.set_ylabel(r'$E[z_%d| x]$'%(z1 +1), fontsize=24)
				labels = [item.get_text() for item in ax_v.get_xticklabels()]
				ax_v.set_xticklabels(labels, rotation='vertical')
				#ax.arrow(0, 0, paramt[0][1][0][0][0], paramt[0][1][0][0][1], head_width=.3, head_length=.3, fc='k', ec='k')
				# ax.set_title('Volume and percent change')
				fig_v.tight_layout()
				plt.savefig(analysis_directory+'final_fit_results/model_0_0/val/'+disease_type+'_EZ_given_x_'+str(z1)+'_'+ str(z2) +'_'+ colorname + '_val_color.pdf', bbox_inches='tight')
				plt.savefig(analysis_directory+'final_fit_results/model_0_0/val/'+disease_type+'_EZ_given_x_'+str(z1)+'_'+ str(z2) +'_'+ colorname + '_val_color.eps', format='eps', dpi=1200, bbox_inches='tight')


def prettyplot3_final_sim(colorcat, colorname, root_directory, project_name, disease_type):
	data_directory= root_directory + project_name +'/data/'
	analysis_directory= root_directory+ project_name +'/analysis/'
	survivalloc=  root_directory + project_name +'/data/survival.csv'
	testsamples= pd.read_csv(analysis_directory + 'test_samples.csv', sep=',', index_col=0)
	zloc = 'final_fit_results/model_0_0/learn/expected_values/EZ_given_x.csv'
	zloc_v = 'final_fit_results/model_0_0/val/EZ_given_x.csv'
	survival = pd.read_csv(survivalloc, sep=",", index_col=0)
	trueZ= pd.read_csv(data_directory+'not_in_use/Ztrue.csv' , sep=",", index_col=0)
	#trueZ= pd.read_csv(data_directory+'data_no_cutoff/Ztrue.csv' , sep=",", index_col=0)
	if np.sum(np.sum(np.isnan(colorcat)))!=0:
		colorcat.loc['Missing']=0
		colorcat.loc['Missing', (np.sum(np.isnan(colorcat))!=0)]=1
	#cm = plt.cm.get_cmap('Greys')
	cm = plt.cm.get_cmap('winter')
	vminn= np.min(range(colorcat.shape[0]))-1
	vmaxx= np.max(range(colorcat.shape[0]))
	colorlist=matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vminn, vmax=vmaxx, clip=False), cmap=cm).to_rgba(range(colorcat.shape[0]))
	markers = ['o', 'v']
	for z1 in range(trueZ.shape[0]):
		for z2 in range(z1 +1):
			if z2<z1:
				fig_v, ax_v = plt.subplots()
				leg_v=[]
				names_v=[]
				for l in range(colorcat.shape[0]):
					x = trueZ.ix[z2, colorcat.loc[:, (colorcat.iloc[l, :]==1)].columns]
					y = trueZ.ix[z1, colorcat.loc[:, (colorcat.iloc[l, :]==1)].columns]
					delta= survival.ix[1, colorcat.loc[:, (colorcat.iloc[l, :]==1)].columns]
					for k, m in enumerate(markers):
						if m== 'o':
							j= (delta==1)
						elif m=='v':
							j= (delta==0)
						#j= (m=='o')*(delta==1) +(m=='v')*(delta==0)
						#j= (m=='o')&(delta==1) | (m=='v')&(delta==0)
						if k==0:
							im_v=ax_v.scatter(x[j], y[j], color=colorlist[l], edgecolors='black', marker=m, s=100)
							leg_v.append(im_v)
							names_v.append(colorcat.index[l]) 
							print(k, l, colorlist[l], x[j].shape, m)	
						else:
							im_v=ax_v.scatter(x[j], y[j], color=colorlist[l], edgecolors='black', marker=m, s=100)
							# leg.append(im)
						# 	names.append(colorcat.index[l]) 
						# 	print(k, l, colorlist[l], x[j].shape)
				#fig.colorbar(im, ax=ax)
				ax_v.legend(leg_v, names_v, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=24)
				ax_v.tick_params(axis='both', labelsize=24)
				ax_v.set_xlabel(r'$ z_%d$'%(z2 +1), fontsize=24)
				ax_v.set_ylabel(r'$ z_%d$'%(z1 +1), fontsize=24)
				labels = [item.get_text() for item in ax_v.get_xticklabels()]
				ax_v.set_xticklabels(labels, rotation='vertical')
				#ax.arrow(0, 0, paramt[0][1][0][0][0], paramt[0][1][0][0][1], head_width=.3, head_length=.3, fc='k', ec='k')
				# ax.set_title('Volume and percent change')
				fig_v.tight_layout()
				plt.savefig(analysis_directory+'final_fit_results/model_0_0/'+disease_type+ '_trueZ'+str(z1)+'_'+ str(z2) +'_'+ colorname + '_all_color.pdf', bbox_inches='tight')
				plt.savefig(analysis_directory+'final_fit_results/model_0_0/'+disease_type+ '_trueZ'+str(z1)+'_'+ str(z2) +'_'+ colorname + '_all_color.eps', format='eps', dpi=1200, bbox_inches='tight')
	del ax_v, fig_v, im_v, leg_v, names_v
	survival_v = survival.loc[:, list(testsamples.values[:,0])] 
	survival= survival.drop(list(testsamples.values[:,0]), axis=1)
	#t= survival.iloc[0, :]
	#delta= survival.iloc[1, :]
	colorcat_v=colorcat.loc[:, list(testsamples.values[:,0])]
	colorcat = colorcat.drop(list(testsamples.values[:,0]), axis=1) 
	EZ = pd.read_csv(analysis_directory+ zloc, sep=",", index_col=0)
	EZ.columns=survival.columns 
	EZ_v = pd.read_csv(analysis_directory+ zloc_v, sep=",", index_col=0)
	EZ_v.columns=survival_v.columns
	# EZ_all= pd.concat([EZ, EZ_v], axis=1)
	# EZ_all= pd.DataFrame(EZ_all, columns= list(EZ.columns)+list(EZ_v.columns))
	# EZ_all=EZ_all.loc[:, trueZ.columns]
	# print EZ_all.shape
	# forQ = scipy.linalg.svd(trueZ.dot(EZ_all.T))
	# Q= forQ[2].T.dot(forQ[0])
	# EZ_all = Q.T.dot(EZ_all)
	# EZ_all= pd.DataFrame(EZ_all, columns= trueZ.columns)
	# EZ= EZ_all.ix[:, EZ.columns]
	# EZ_v= EZ_all.ix[:, EZ_v.columns]
	#cm = plt.cm.get_cmap('rainbow')
	#cm = plt.cm.get_cmap('Greys')
	cm = plt.cm.get_cmap('winter')
	vminn= np.min(range(colorcat.shape[0]))-1
	vmaxx= np.max(range(colorcat.shape[0]))
	colorlist=matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vminn, vmax=vmaxx, clip=False), cmap=cm).to_rgba(range(colorcat.shape[0]))
	markers = ['o', 'v']
	for z1 in range(EZ.shape[0]):
		for z2 in range(z1 +1):
			if z2<z1:
				fig, ax = plt.subplots()
				leg=[]
				names=[]
				for l in range(colorcat.shape[0]):
					x = EZ.ix[z2, colorcat.loc[:, (colorcat.iloc[l, :]==1)].columns]
					y = EZ.ix[z1, colorcat.loc[:, (colorcat.iloc[l, :]==1)].columns]
					delta= survival.ix[1, colorcat.loc[:, (colorcat.iloc[l, :]==1)].columns]
					for k, m in enumerate(markers):
						if m== 'o':
							j= (delta==1)
						elif m=='v':
							j= (delta==0)
						#j= (m=='o')*(delta==1) +(m=='v')*(delta==0)
						#j= (m=='o')&(delta==1) | (m=='v')&(delta==0)
						if k==0:
							im=ax.scatter(x[j], y[j], color=colorlist[l], edgecolors='black', marker=m, s=100)
							leg.append(im)
							names.append(colorcat.index[l]) 
							print(k, l, colorlist[l], x[j].shape, m)	
						else:
							im=ax.scatter(x[j], y[j], color=colorlist[l], edgecolors='black', marker=m, s=100)
							# leg.append(im)
						# 	names.append(colorcat.index[l]) 
						# 	print(k, l, colorlist[l], x[j].shape)
				#fig.colorbar(im, ax=ax)
				ax.legend(leg, names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=24)
				# ax.set_xlabel(r'$Q E[z_%d| x]$'%(z2 +1), fontsize=20)
				# ax.set_ylabel(r'$Q E[z_%d| x]$'%(z1 +1), fontsize=20)
				ax.set_xlabel(r'$E[z_%d| x]$'%(z2 +1), fontsize=24)
				ax.set_ylabel(r'$E[z_%d| x]$'%(z1 +1), fontsize=24)
				labels = [item.get_text() for item in ax.get_xticklabels()]
				ax.set_xticklabels(labels, rotation='vertical')
				#ax.arrow(0, 0, paramt[0][1][0][0][0], paramt[0][1][0][0][1], head_width=.3, head_length=.3, fc='k', ec='k')
				# ax.set_title('Volume and percent change')
				fig.tight_layout()
				plt.savefig(analysis_directory+'final_fit_results/model_0_0/learn/expected_values/'+disease_type+'_EZ_given_x_'+str(z1)+'_'+ str(z2) +'_'+ colorname + '_color.pdf', bbox_inches='tight')
				plt.savefig(analysis_directory+'final_fit_results/model_0_0/learn/expected_values/'+disease_type+'_EZ_given_x_'+str(z1)+'_'+ str(z2) +'_'+ colorname + '_color.eps', format='eps', dpi=1200, bbox_inches='tight')
	EZ_v = pd.read_csv(analysis_directory+ zloc_v, sep=",", index_col=0)
	EZ_v.columns=survival_v.columns
	#t= survival_v.iloc[0, :]
	#delta= survival_v.iloc[1, :]
	markers = ['o', 'v']
	for z1 in range(EZ.shape[0]):
		for z2 in range(z1 +1):
			if z2<z1:
				fig_v, ax_v = plt.subplots()
				leg_v=[]
				names_v=[]
				for l in range(colorcat_v.shape[0]):
					x = EZ_v.ix[z2, colorcat_v.loc[:, (colorcat_v.iloc[l, :]==1)].columns]
					y = EZ_v.ix[z1, colorcat_v.loc[:, (colorcat_v.iloc[l, :]==1)].columns]
					delta= survival_v.ix[1, colorcat_v.loc[:, (colorcat_v.iloc[l, :]==1)].columns]
					for k, m in enumerate(markers):
						if m== 'o':
							j= (delta==1)
						elif m=='v':
							j= (delta==0)
						#j= (m=='o')*(delta==1) +(m=='v')*(delta==0)
						#j= (m=='o')&(delta==1) | (m=='v')&(delta==0)
						if k==0:
							im_v=ax_v.scatter(x[j], y[j], color=colorlist[l], edgecolors='black', marker=m, s=100)
							leg_v.append(im_v)
							names_v.append(colorcat_v.index[l]) 
							print(k, l, colorlist[l], x[j].shape, m)	
						else:
							im_v=ax_v.scatter(x[j], y[j], color=colorlist[l], edgecolors='black', marker=m, s=100)
							# leg.append(im)
						# 	names.append(colorcat.index[l]) 
						# 	print(k, l, colorlist[l], x[j].shape)
				#fig.colorbar(im, ax=ax)
				ax_v.legend(leg_v, names_v, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=24)
				ax_v.tick_params(axis='both', labelsize=24)
				ax_v.set_xlabel(r'$E[z_%d| x]$'%(z2 +1), fontsize=24)
				ax_v.set_ylabel(r'$E[z_%d| x]$'%(z1 +1), fontsize=24)
				labels = [item.get_text() for item in ax_v.get_xticklabels()]
				ax_v.set_xticklabels(labels, rotation='vertical')
				# ax_v.set_xlabel(r'$Q E[z_%d| x]$'%(z2 +1), fontsize=20)
				# ax_v.set_ylabel(r'$ Q E[z_%d| x]$'%(z1 +1), fontsize=20)
				#ax.arrow(0, 0, paramt[0][1][0][0][0], paramt[0][1][0][0][1], head_width=.3, head_length=.3, fc='k', ec='k')
				# ax.set_title('Volume and percent change')
				fig_v.tight_layout()
				plt.savefig(analysis_directory+'final_fit_results/model_0_0/val/'+disease_type+'_EZ_given_x_'+str(z1)+'_'+ str(z2) +'_'+ colorname + '_val_color.pdf', bbox_inches='tight')
				plt.savefig(analysis_directory+'final_fit_results/model_0_0/val/'+disease_type+'_EZ_given_x_'+str(z1)+'_'+ str(z2) +'_'+ colorname + '_val_color.eps', format='eps', dpi=1200, bbox_inches='tight')

