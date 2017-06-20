"""
Survival Factor Model, Copyright (C) 2017,  Shannon McCurdy and Regents of the University of California.

Contact:  Shannon McCurdy, smccurdy@berkeley.edu.

    This file is part of Survival Factor Model.

    Survival Factor Model is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Survival Factor Model is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Survival Factor Model program.  If not, see <http://www.gnu.org/licenses/>.
"""


from __future__ import division
import os
import numpy as np
from sklearn import linear_model
import scipy
import scipy.stats
from time import time
import pickle
import cindex_code
import survival_funct_v1
import pandas as pd
import traceback
import sys


def cox(X, tE, Delta, niter, sparse, plotloc):
    ##initialize lamda and wt
    dx, nsamp = X.shape
    X = np.vstack([np.ones((1, nsamp)), X])
    Lam0 = np.sum(Delta) / np.sum(tE)
    wt0 = np.zeros((1, dx))
    w0 = np.vstack([np.log(Lam0), wt0.T])
    for i in range(niter):
        eta = X.T.dot(w0)
        # print(i, eta)
        y = np.sqrt(tE.T) * np.exp(eta / 2) * (
            eta + (Delta / tE).T * np.exp(-eta) - np.ones(eta.shape))
        # print("cox iter and y", i, y)
        x_tilde = (np.diag((np.sqrt(tE.T) * np.exp(eta / 2))[:, 0])).dot(X.T)
        ## doing some checks
        print("cox checks", y.shape, np.sum(np.isfinite(y)),  x_tilde.shape, np.sum(np.isfinite(x_tilde)))
        if np.sum(np.isfinite(y))< y.shape[0]:
            print("problem iteration", i)
            print("eta", np.sum(np.isfinite(eta)),np.sum(np.isfinite((Delta / tE).T * np.exp(-eta))), np.sum(np.isfinite(np.exp(eta / 2) )) )
            print(" big eta", np.sum(eta/ 2 > 709))
            # print("te", tE)
            # print("delta", Delta)
        if not sparse[0]:
            Lfit = linear_model.LinearRegression(fit_intercept=False)
            Lfit.fit(x_tilde, y)
            wnew = Lfit.coef_
            #print(wnew -
            #      (scipy.linalg.inv(x_tilde.T.dot(x_tilde)).dot(x_tilde.T.dot(y))).T)
            ux, lamx, vtx = scipy.linalg.svd(x_tilde, full_matrices=False)
            wnew_alt = (vtx.T.dot(np.diag(1 / lamx)).dot(ux.T).dot(y)).T
            #print('no inv', wnew-wnew_alt)
            ux, lamx, vtx = scipy.linalg.svd(X, full_matrices=False)
            wnew_alt2 = np.log(Delta /
                               tE).dot(vtx.T).dot(np.diag(1 / lamx)).dot(ux.T)
            print('exact?', wnew - wnew_alt2)
            wnew = wnew.reshape((dx + 1, 1))
        else:
            LLfit = linear_model.LassoLars(sparse[1],
                                           fit_intercept=False,
                                           eps=10e-9)
            LLfit.fit(x_tilde, y)
            wnew = LLfit.coef_
            wnew = wnew.reshape((dx + 1, 1))
            print('objective', 1 / (2 * nsamp) * (y - x_tilde.dot(wnew)).T.dot(
                y - x_tilde.dot(wnew)) - sparse[1] * np.sum(np.abs(wnew)))
        #print('wnew')
        print('converged?', np.sum(w0 - wnew))
        w0 = wnew
    Lamp = np.exp(wnew[0, :])
    wtp = wnew[1:, :].T
    print("cox shapes", X[1:, :].shape, wtp.shape)
    ##making a log file and saving parameters
    if not os.path.exists(plotloc + 'expected_values/'):
        os.makedirs(plotloc + 'expected_values/')
    texp = expsurvtime_cox(X[1:, :], wtp, Lamp, plotloc + 'expected_values/')
    if not os.path.exists(plotloc + 'learned_parameters/'):
        os.makedirs(plotloc + 'learned_parameters/')
    with open(plotloc + 'learned_parameters/' + 'cox_log.txt', 'w') as g:
        g.write('time of run' + ',' + str(time()) + '\n')
        g.write('plotloc' + ',' + str(plotloc) + '\n')
        g.write('number of iterations' + ',' + str(niter) + '\n')
        g.write('sparse' + ',' + str(sparse) + '\n')
        g.write('Xshape' + ',' + str(X.shape) + '\n')
    # with open(plotloc + 'learned_parameters/' + 'wt.py') as f:
    #     pickle.dump(wtp, f)
    pd.DataFrame(wtp).to_csv(plotloc + 'learned_parameters/' + 'wt.csv')
    # with open(plotloc + 'learned_parameters/' + 'Lam.py') as f:
    #     pickle.dump(Lamp, f)
    pd.DataFrame(Lamp).to_csv(plotloc + 'learned_parameters/' + 'Lam.csv')
    return wtp, Lamp


##change name to expsurvtime_cox()
def expsurvtime_cox(X, wt, Lam, plotloc):
    texp = 1 / Lam * np.exp(-wt.dot(X))
    # f = open(plotloc + 'Et_given_x.py', 'w')
    # pickle.dump(texp, f)
    # f.close()
    pd.DataFrame(texp).to_csv(plotloc + 'Et_given_x.csv')
    return texp


## expected c - index for fitting the simple cox model on data simulated with
## the FA+ survival model
def expected_cindex_cox(paramX, paramt, dztrue, nsamp, ntrain, niter, sparse, CEN, plotloc):
    expected_c_test = []
    train_data = []
    test_data = []
    for i in range(0, ntrain):
        train_data.append(survival_funct_v1.gen_data(nsamp, dztrue, paramX, paramt, CEN))
        if CEN == True:
            test_data.append(survival_funct_v1.gen_data(nsamp, dztrue, paramX, paramt[0],
                                                                 CEN=False))
        elif CEN == False:
            test_data.append(survival_funct_v1.gen_data(nsamp, dztrue, paramX, paramt,
                                                                 CEN=False))
        ##or could generate test data with censoring
    for i in range(0, ntrain):
        (dataparamX, dataparamt, Z_hidden) = train_data[i]
        X, tE, Delta = get_compiled_x_and_t(dataparamX, dataparamt, CEN)
        if not os.path.exists(plotloc + 'learn_cox_' + str(i) + '/'):
                os.makedirs(plotloc + 'learn_cox_' + str(i) + '/')
        wtp, Lamp = cox(X, tE, Delta, niter, sparse,
                        plotloc + 'learn_cox_' + str(i) + '/')
        for j in range(0, ntrain):
            (dataparamX_test, dataparamt_test, Z_hidden_test) = test_data[j]
            X_test, tE_test, Delta_test = get_compiled_x_and_t(dataparamX_test,
                                                 dataparamt_test,
                                                 CEN=False)
            if not os.path.exists(plotloc + 'val_cox_' + str(i) + '_' + str(j) + '/'):
                os.makedirs(plotloc + 'val_cox_' + str(i) + '_' + str(j) + '/')
            est_test = expsurvtime_cox(
                X_test, wtp, Lamp,
                plotloc + 'val_cox_' + str(i) + '_' + str(j) + '/')
            estdelta_test = np.ones((1, nsamp))
            ## because no censoring on test data.
            ## also no ties.
            c_test = cindex_code.cindex(tE_test, Delta_test, est_test, estdelta_test)
            expected_c_test.append(c_test)
            print('expected c test', expected_c_test, c_test, i, j)
    return expected_c_test

def expected_cindex_2step(paramX, paramt, dztrue, dztrain, nsamp, ntrain, niter, convgap, mh_params, step, CEN, plotloc):
    expected_c_test = []
    train_data = []
    test_data = []
    for i in range(0, ntrain):
        train_data.append(survival_funct_v1.gen_data(nsamp, dztrue, paramX, paramt, CEN))
        if CEN == True:
            test_data.append(survival_funct_v1.gen_data(nsamp, dztrue, paramX, paramt[0],
                                                                 CEN=False))
        elif CEN == False:
            test_data.append(survival_funct_v1.gen_data(nsamp, dztrue, paramX, paramt,
                                                                 CEN=False))
        ##or could generate test data with censoring
    for i in range(0, ntrain):
        (dataparamX, dataparamt, Z_hidden) = train_data[i]
        if not os.path.exists(plotloc + 'learn_' + str(i) + '/'):
            os.makedirs(plotloc + 'learn_2step_' + str(i) + '/')
        ## need to train data with EM_all no time data to get EZ
        INIT = True
        train = survival_funct_v1.EM_all(dataparamX, False, dztrain, niter, convgap,
                       mh_params, step, False, INIT, 
                       plotloc + 'learn_2step_' + str(i) + '/')
        if len(train) != 6:
            print('stopping now')
            return train
        (dataparamXp, dataparamtp, Xmeans, q, Ldata, EZ) = train
        tE, Delta = get_compiled_t(dataparamt, CEN)
        sparse = [False]
        wtp, Lamp = cox(EZ, tE, Delta, niter, sparse,
                        plotloc + 'learn_2step_' + str(i) + '/')
        for j in range(0, ntrain):
            (dataparamX_test, dataparamt_test, Z_hidden_test) = test_data[j]
            ## need to get EZ_test given EM_all no time data
            dataparamXq, dataparamtq = survival_funct_v1.dataswitch(
                dataparamX_test, dataparamt_test, False, dataparamXp,
                dataparamtp, Xmeans, CEN)
            ## EZ_no_surv()
            EZ_test, EZZtn_test = survival_funct_v1.E_step_no_surv(dataparamXq, dztrain)
            tE_test, Delta_test = get_compiled_t(dataparamt_test,
                                                 CEN=False)
            if not os.path.exists(plotloc + 'val_2step_' + str(i) + '_' + str(j) + '/'):
                os.makedirs(plotloc + 'val_2step_' + str(i) + '_' + str(j) + '/')
            est_test = expsurvtime_cox(
                EZ_test, wtp, Lamp,
                plotloc + 'val_2step_' + str(i) + '_' + str(j) + '/')
            estdelta_test = np.ones((1, nsamp))
            ## because no censoring on test data.
            ## also no ties.
            c_test = cindex_code.cindex(tE_test, Delta_test, est_test, estdelta_test)
            expected_c_test.append(c_test)
            print('expected c test', expected_c_test, c_test, i, j)
    return expected_c_test




##making an alternative model which is purely cox to compare to.
def get_compiled_x_and_t(dataparamX, dataparamt, CEN):
    for j, datapara in enumerate(dataparamX):
        [X, (datatype, theta, sparse)] = datapara
        if datatype == 'multinom':
            ## add an if statement to remove one component of multinomial data.
            ## to get the right number of dof
            print("multinom")
            X = X[:-1, :]
        if j == 0:
            X_comp = X
        else:
            X_comp = np.vstack([X_comp, X])
    if CEN == False:
        [tE, (datatype, thetat, sparse)] = dataparamt
        Delta = np.ones(tE.shape)
    elif CEN == True:
        [[tE, (datatypet, thetat, sparset)], [Delta, (datatypec, thetac,
                                                      sparsec)]] = dataparamt
    return X_comp, tE, Delta

def get_compiled_x(dataparamX):
    for j, datapara in enumerate(dataparamX):
        [X, (datatype, theta, sparse)] = datapara
        if datatype == 'multinom':
            ## add an if statement to remove one component of multinomial data.
            ## to get the right number of dof
            X = X[:-1, :]
        if j == 0:
            X_comp = X
        else:
            X_comp = np.vstack([X_comp, X])
    return X_comp

def get_compiled_t(dataparamt, CEN):
    if CEN == False:
        [tE, (datatype, thetat, sparse)] = dataparamt
        Delta = np.ones(tE.shape)
    elif CEN == True:
        [[tE, (datatypet, thetat, sparset)], [Delta, (datatypec, thetac,
                                                      sparsec)]] = dataparamt
    return tE, Delta



#     # #cindex_code cindex_test, though this is structured slightly differently.
#     # #def cindex_alt_Cox
def learn_predict_cindex_cox(dataparamX_test, dataparamt_test, CEN_test, dataparamX,
                   dataparamt, CEN, niter, sparse, plotloc):
    """

    this code takes in test data (dataparamX_test) and training data (dataparamX)
    learns the maximum likelihood parameters
    (the parameters in dataparamX0), predicts the expected survival time using the
    maximum likelhood parameters, and calculates the c-index on the test data and test
    predictions
    """
    X, tE, Delta = get_compiled_x_and_t(dataparamX, dataparamt, CEN)
    ##learn
    if not os.path.exists(plotloc + 'learn_cox/'):
        os.makedirs(plotloc + 'learn_cox/')
    wtp, Lamp = cox(X, tE, Delta, niter, sparse, plotloc+ 'learn_cox/')
    print(Lamp, np.sum(wtp == 0))
    for i, item in enumerate(wtp[0, :]):
        if item != 0: print(i)
    X_test, tE_test, Delta_test = get_compiled_x_and_t(
        dataparamX_test, dataparamt_test, CEN_test)
    #predict
    if not os.path.exists(plotloc + 'val_cox/'):
        os.makedirs(plotloc + 'val_cox/')
    print("shapes", X.shape, X_test.shape, wtp.shape)
    est_test = expsurvtime_cox(X_test, wtp, Lamp, plotloc + 'val_cox/')
    #estdelta_test = np.ones(est_test.shape)
    ## because no censoring on prediction.
    ## cindex on test set
    c_test = cindex_code.cindex_ties(tE_test, Delta_test, est_test)[0]
    return c_test


def model_selection_fixed_data_cox(k, dataparamX_test, dataparamt_test, CEN_test, dataparamX,
                   dataparamt, CEN, niter, sparserange, plotloc): 
    f = open(plotloc + 'model_selection_output_cox' +str(k) +'.txt', 'w')
    for nmod, sparse in enumerate(sparserange):
        if not os.path.exists(plotloc + 'model_cox_' + str(nmod) + '/'):
                os.makedirs(plotloc + 'model_cox_' + str(nmod) + '/')
        else:
            print('exiting model selection because directory already exists')
            return
        cindex = learn_predict_cindex_cox(dataparamX_test, dataparamt_test, CEN_test, dataparamX,
                       dataparamt, CEN, niter, sparse, plotloc + 'model_cox_' + str(nmod) + '/')
        f.write(str(nmod) + ',' + str(sparse) + ', ' + str(cindex) + '\n')
    f.close()
    return


def nan_to_mean(data):
    data_nonan= []
    for X in data:
        print('checking nans before', np.sum(np.sum(np.isnan(X))))
        Xmean = np.nanmean(X, 1)
        inds = np.where(np.isnan(X))
        X.iloc[inds] = np.take(Xmean, inds[0])
        print('checking nans after', np.sum(np.sum(np.isnan(X))))
        data_nonan.append(X)
    return data_nonan


def cross_val_cox(n_cv, data_directory, analysis_directory, gold_standard):
    data_guide, data, sample_name_ref = survival_funct_v1.load_data(data_directory, gold_standard)
    data = nan_to_mean(data)
    cv_splits, test_samples = survival_funct_v1.make_splits(n_cv, analysis_directory, sample_name_ref)
    test_samples = list(test_samples.values[:, 0])
    file_path = analysis_directory + 'cv_' + str(n_cv) + '_results_cox' + gold_standard*'_gs' + '/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    else:
        print('directory already exists. exiting so that results are not overwritten')
        return
    specs_file_path = analysis_directory + 'specs_cox' + gold_standard*'_gs' + (not gold_standard) *'_cv' + '.csv'
    if not os.path.isfile(specs_file_path):
        print('spec file does not exist, exiting')
        return
    else:
        specs = pd.read_csv(specs_file_path, delimiter=',', index_col=0)
        sparserange_string = specs.loc[0, 'sparserange']
        sparserange = eval(sparserange_string.replace(';', ','))
    for i in range(n_cv):
        os.makedirs(file_path + 'cv_run' + str(i) + '/')
        dataparamX_test, dataparamt_test, CEN_test, dataparamX, dataparamt, CEN = survival_funct_v1.make_dataparamXt(
            n_cv, cv_splits, test_samples, i, data_guide, data,
            file_path + 'cv_run' + str(i) + '/')
        model_selection_fixed_data_cox(i, dataparamX_test, dataparamt_test, CEN_test, dataparamX,
                   dataparamt, CEN, specs.loc[0, 'niter'], sparserange, file_path + 'cv_run' + str(i) + '/')
    return

### this code fits the two step modelling procedure: first FA, then Cox with EZ as covariates.
def cross_val_2step(n_cv, data_directory, analysis_directory):
    """
    only differences are filenames
    simplified sparserange call because you wouldn't run this without time 
    and the fact that model_selection_2step calls learn_predict_cindex_2step, which has the guts of the difference
    i still have it load mh_params even though it doesn't use them.
    """
    data_guide, data, sample_name_ref = survival_funct_v1.load_data(data_directory, False)
    data = nan_to_mean(data)
    cv_splits, test_samples = survival_funct_v1.make_splits(n_cv, analysis_directory, sample_name_ref)
    test_samples= list(test_samples.values[:, 0])
    sparserange_string = data_guide.loc[0:(len(data) -2), 'sparserange']
    sparserange = []
    for k in range(sparserange_string.shape[0]):
        sparserange.append(eval(sparserange_string[k].replace(';', ',')))
    if not os.path.exists(analysis_directory + 'cv_'+ str(n_cv) + '_results_2step/'):
        os.makedirs(analysis_directory + 'cv_'+ str(n_cv) + '_results_2step/')
    else:
        print('directory already exists. exiting so that results are not overwritten')
        return
    if not os.path.isfile(analysis_directory + 'specs_cv.csv'):
        print('spec file does not exist, exiting')
        return
    else:
        specs = pd.read_csv(analysis_directory + 'specs_cv.csv', delimiter=',', index_col=0)
        dzrange_string = specs.loc[0, 'dzrange']
        dzrange = eval(dzrange_string.replace(';', ','))
        # niter = specs['niter']
        # convgap = specs['convgap']
        # batch_number = specs['batch_number']
        # nrep_batch = specs['nrep_batch']
        # burn = specs['nrep_burn']
        # MULTIPRO = specs['MULTIPRO']
        # PRE_SEED = specs['PRE_SEED']
        # mh_params = batch_number, nrep_batch, burn, MULTIPRO, PRE_SEED
        mh_params = specs.loc[0, 'batch_number'], specs.loc[0, 'nrep_batch'], specs.loc[0, 'burn'], specs.loc[0, 'MULTIPRO'], specs.loc[0, 'PRE_SEED']
        # step = specs['step']
    for i in range(n_cv):
        os.makedirs(analysis_directory + 'cv_' + str(n_cv) + '_results_2step/cv_run' + str(i) + '/')
        dataparamX_test, dataparamt_test, CEN_test, dataparamX, dataparamt, CEN = survival_funct_v1.make_dataparamXt(
            n_cv, cv_splits, test_samples, i, data_guide, data,
            analysis_directory + 'cv_' + str(n_cv) + '_results_2step/cv_run' + str(i) + '/')
        INIT = True 
        model_selection_fixed_data_2step(i, dataparamX_test, dataparamt_test, CEN_test,
                    dataparamX, dataparamt, dzrange, sparserange,
                    specs.loc[0, 'niter'], specs.loc[0, 'convgap'], mh_params, specs.loc[0, 'step'], CEN, INIT,
                    analysis_directory + 'cv_' + str(n_cv) + '_results_2step/cv_run' + str(i) + '/')
    return

def model_selection_fixed_data_2step(k, dataparamX_test, dataparamt_test, CEN_test,
                    dataparamX, dataparamt, dzrange, sparserange,
                    niter, convgap, mh_params, step, CEN, INIT, plotloc): 
    """
    only differences are file output names and that calls learn_predict_cindex_2step
    main difference is in learn_predict_cindex_2step
    """
    f = open(plotloc + 'model_selection_output_2step' +str(k) +'.txt', 'w')
    ### picking which parameters to use.
    #nmodel will contain the cumulative product of alphabets (total dimension) up to n
    nmodel = len(sparserange[0])
    for datatype in range(len(sparserange)):
        if nmodel != len(sparserange[datatype]):
            print('exiting model selection because datatypes have different number of sparsity parameters')
            return
    for nmod in range(nmodel):
        ##now i need code to feed in the sparsepick argurments into dataparamX, dataparamt.
        dataparamXs = []
        sparserun = []
        for j, datapara in enumerate(dataparamX):
            [X, (datatype, theta, sparse)] = datapara
            dataparamXs.append([X, (datatype, theta, sparserange[j][nmod])])
            sparserun.append(sparserange[j][nmod])
        print('sparserun', sparserun)
        #print(paramX)
        #print(paramXs)
        for m, dz in enumerate(dzrange):
            print('nmod, m', nmod, m)
            if not os.path.exists(plotloc + 'model_' + str(nmod) + '_' + str(m) + '/'):
                os.makedirs(plotloc + 'model_' + str(nmod) + '_' + str(m) + '/')
            else:
                print('exiting model selection because directory already exists')
                return
            cindex = learn_predict_cindex_2step(
                dataparamX_test, dataparamt_test, CEN_test, dataparamXs, dataparamt,
                dz, niter, convgap, mh_params, step, CEN, INIT,
                plotloc + 'model_' + str(nmod) + '_' + str(m) + '/')
            print('dz, sparserun, cindex', dz, sparserun, cindex)
            f.write(str(dz) + ', ' + str(sparserun) + ', ' + str(cindex) + '\n')
            #answer[m, nmod, :] = [(dz, sparserun), cindex]
    f.close()
    return

def learn_predict_cindex_2step(dataparamX_test, dataparamt_test, CEN_test, dataparamX,
                   dataparamt, dz, niter, convgap, mh_params, step, CEN, INIT, plotloc):
    """

    this code takes in test data (dataparamX_test) and training data (dataparamX)
    learns the maximum likelihood parameters
    (the parameters in dataparamX0), predicts the expected survival time using the
    maximum likelhood parameters, and calculates the c-index on the test data and test
    predictions
    """
    tE, Delta = get_compiled_t(dataparamt, CEN)
    ##learn
    if not os.path.exists(plotloc + 'learn_2step/'):
        os.makedirs(plotloc + 'learn_2step/')
        os.makedirs(plotloc + 'learn_2step/fa/')
    try:
        INIT=True
        train = survival_funct_v1.EM_all(dataparamX, False, dz, niter, convgap, mh_params, step, False,
                  INIT, plotloc+ 'learn_2step/fa/')
        if len(train) == 6:
            print('here1')
            (dataparamXp, dataparamtp, Xmeans, q, Ldata, EZ) = train
        else:
            print('stopping now')
            return train
        sparse = [False]
        wtp, Lamp = cox(EZ, tE, Delta, niter, sparse, plotloc+ 'learn_2step/')
        print(Lamp, np.sum(wtp == 0))
        for i, item in enumerate(wtp[0, :]):
            if item != 0: print(i)
        #predict
        dataparamXq, dataparamtq = survival_funct_v1.dataswitch(
            dataparamX_test, dataparamt_test, CEN_test, dataparamXp,
            dataparamtp, Xmeans, CEN)
        EZ_test, EZZtn_test = survival_funct_v1.E_step_no_surv(dataparamXq, dz)
        tE_test, Delta_test = get_compiled_t(dataparamt_test, CEN_test)
        if not os.path.exists(plotloc + 'val_2step/'):
            os.makedirs(plotloc + 'val_2step/')
        pd.DataFrame(EZ_test).to_csv(plotloc + 'val_2step/EZ_given_x.csv')
        est_test = expsurvtime_cox(EZ_test, wtp, Lamp, plotloc + 'val_2step/')
        estdelta_test = np.ones(est_test.shape)
        ## because no censoring on prediction.
        ## cindex on test set
        print('now calculating cindex')
        ## this takes too long with 300 samples.
        c_test = cindex_code.cindex_ties(tE_test, Delta_test, est_test)[0]
        #c_test = cindex_code.cindex(tE_test, Delta_test, est_test, estdelta_test)
    except AssertionError:
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb)
        # Fixed format
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]
        with open(plotloc + 'assertion_error.txt', 'w') as f:
            f.write('An error occurred on line {} in statement {}'.format(line, text))
        c_test = np.nan
    return c_test


def final_fit_cox(data_directory, analysis_directory, gold_standard):
    data_guide, data, sample_name_ref = survival_funct_v1.load_data(data_directory, gold_standard)
    data = nan_to_mean(data)
    if os.path.isfile(analysis_directory + 'test_samples.csv'):
        print('test samples are set aside')
        test_samples = pd.read_csv(analysis_directory + 'test_samples.csv', delimiter=',', index_col = 0)
    else:
        print('test samples file missing, exiting.')
        return
    test_samples= list(test_samples.values[:, 0])
    file_path = analysis_directory + 'final_fit_results_cox' + gold_standard*'_gs' + '/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    else:
        print('directory already exists. exiting so that results are not overwritten')
        return
    specs_file_path = analysis_directory + 'specs_cox'+ gold_standard*'_gs' + '.csv'
    if not os.path.isfile(specs_file_path):
        print('spec file does not exist, exiting')
        return
    else:
        specs = pd.read_csv(specs_file_path, delimiter=',', index_col=0)
        sparserange_string = specs.loc[0, 'sparserange']
        sparserange = eval(sparserange_string.replace(';', ','))
    dataparamX_test, dataparamt_test, CEN_test, dataparamX, dataparamt, CEN = survival_funct_v1.make_dataparamXt_final(
        test_samples, data_guide, data,
        file_path)
    model_selection_fixed_data_cox('final', dataparamX_test, dataparamt_test, CEN_test, dataparamX,
               dataparamt, CEN, specs.loc[0, 'niter'], sparserange, file_path)
    return


