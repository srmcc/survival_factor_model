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
from time import time
import sys
import os
import numpy as np
import scipy
import scipy.linalg.blas
import pandas as pd
import multiprocessing
import random
import sklearn
from sklearn import linear_model
import pickle
import scipy.stats
##plotting commands
import matplotlib
#workaround for x - windows
matplotlib.use('Agg')
#from ggplot import *
import pylab as pl
import glob
import re
import matplotlib.pyplot as plt
import cindex_code
import traceback
import warnings
import copy
#import pdb

warnings.resetwarnings()
warnings.filterwarnings("error", "Early stopping the lars path, as the residues are small and")
#warnings.filterwarnings("error", "DeprecationWarning: using a non-integer number instead of an integer will result in an error in the future")

if matplotlib.__version__[0] != '1':
    matplotlib.style.use('classic')

#X = dx by nsamp
#tE = 1 by nsamp
#Delta = 1 by n samp
#Wx = dx by dz
#wt = 1 by dz
#Z = dz by nsamp
#Lam = scalar
#alpha = 1 by nsamp
##sparse = [True, sparsity threshold, sparsedensity (for createparam)]
#paramX = [(datatypeX, thetaX, sparseX), (),()]
## dataparam = [[X, (datatypeX, thetaX, sparseX)], [],[]]
#datatypes for paramX
#normal
#binom
#multinom
#etime
#cind
#cindIC


def paramcount(nfeatures, nnormalfeatures, dz, nsamp):
    """
    this does a parameter count for non - sparse models.  if delta is included
    in nfeatures, it only does informative censoring.  need to think about
    multinomial model and counting properly - - i think it's just
    nfeature_multi = dim_multi - 1.
    """
    nW = nfeatures * dz
    nunident = dz * (dz - 1) / 2
    ##insert rotation matrix
    nmu = nfeatures
    nPsi = nnormalfeatures
    nmodel = nW - nunident + nmu + nPsi
    if nsamp > nfeatures:
        print("nsamp more than nfeatures")
        pcount = (nmu + nfeatures * (nfeatures + 1) / 2) - nmodel
    else:
        pcount = (nmu + nsamp * (2 * nfeatures - nsamp + 1) / 2) - nmodel
    print('hope this is positive: dof of data - dof of model',
          nfeatures * nsamp - nmodel)
    return pcount, nmodel


def softmax(x):
    dx, nsamp = x.shape
    e_x = np.exp(x)
    sm = (e_x / np.sum(e_x, axis=0))
    ##summing over features dx
    return sm


def lamxi(xi):
    if xi.shape == ():
        xi = np.array([xi])
    lamx = (scipy.special.expit(xi) - 1 / 2) / (2 * xi)
    lamx[xi == 0] = 1 / 8
    lamx[xi > 709] = (1 / (4 * xi) )[xi>709]
    return lamx


#create parameters
def createparam(dx, dz, nsamp, datatype, b, sparse):
    if datatype == 'normal' or datatype == 'binom' or datatype == 'multinom':
        if not sparse[0]:
            Wx = np.random.multivariate_normal(
                np.zeros(dz), np.diag(np.ones(dz)), dx)
        else:
            Wx = scipy.sparse.rand(dx, dz, sparse[2])
            Wx = Wx.toarray() + np.vstack((np.diag(np.random.normal(
                [0], [1], dz)), np.zeros((dx - dz, dz))))
    if datatype == 'normal':
        Psix = np.random.gamma([1], [1], dx)[:, None]
        Mux = np.zeros(dx)
        theta = (Wx, Mux, Psix)
    elif datatype == 'binom':
        Mux = np.zeros(dx)
        xi = np.ones((dx, nsamp))
        theta = (Wx, Mux, xi, b)
    elif datatype == 'multinom':
        Wx[dx - 1, :] = 0
        ##fix for multinom
        Mux = np.zeros(dx)
        Mux[-1] = 0
        ##fix for multinom
        xi = np.ones((dx, nsamp))
        # xi[-1:, :] = 0
        # ##fix for multinom
        alpha = np.ones((1, nsamp))
        theta = (Wx, Mux, xi, alpha, b)
    elif datatype == 'etime' or datatype == 'cindIC':
        wt = np.random.normal([0], [1], dz)
        wt = np.reshape(wt, (1, dz))
        Lam = np.random.gamma(shape=[1], scale=[1])
        theta = (wt, Lam)
    elif datatype == 'cind':
        Lamc = np.random.gamma(shape=[1], scale=[1])
        theta = (Lamc)
    else:
        print('unknown datatype')
    return (datatype, theta, sparse)


def emptyparam(dx, dz, nsamp, datatype, b, sparse):
    if datatype == 'normal' or datatype == 'binom' or datatype == 'multinom':
        Wx = np.zeros((dx, dz))
    if datatype == 'normal':
        Psix = np.zeros(dx)
        Mux = np.zeros(dx)
        theta = (Wx, Mux, Psix)
    elif datatype == 'binom':
        Mux = np.zeros(dx)
        xi = np.ones((dx, nsamp))
        theta = (Wx, Mux, xi, b)
    elif datatype == 'multinom':
        Wx[dx - 1, :] = 0
        ##fix for multinom
        Mux = np.zeros(dx)
        Mux[-1] = 0
        ##fix for multinom
        xi = np.ones((dx, nsamp))
        # xi[-1:, :] = 0
        # ##fix for multinom
        alpha = np.ones((1, nsamp))
        theta = (Wx, Mux, xi, alpha, b)
    elif datatype == 'etime' or datatype == 'cindIC':
        wt = np.zeros((1, dz))
        Lam = 0
        theta = (wt, Lam)
    elif datatype == 'cind':
        Lamc = 0
        theta = (Lamc)
    else:
        print('unknown datatype')
    return (datatype, theta, sparse)

#simulating from the model.
def gen_data(nsamp, dz, paramX, paramt=False, CEN=False):
    Zmean = np.zeros(dz)
    Zcov = np.diag(np.ones(dz))
    Z = (np.random.multivariate_normal(Zmean, Zcov, nsamp)).T
    dataparamX = []
    for i, para in enumerate(paramX):
        datatype, theta, sparse = para
        if datatype == 'normal':
            Wx, Mux, Psix = theta
            # X = Wx.dot(Z) + np.random.multivariate_normal(
            #     Mux, np.diag(Psix[:, 0]), nsamp).T
            # # faster way to simulate multivariate normal
            X = (Wx.dot(Z) + np.array([Mux, ] * nsamp).T
                 + np.sqrt(Psix) * (np.random.normal(size=(Wx.shape[0], nsamp))))
            dataparamX.append([X, para])
        elif datatype == 'binom':
            Wx, Mux, xi, b = theta
            Muxn = np.array([Mux, ] * nsamp).T
            X = np.random.binomial(b, scipy.special.expit(Muxn + Wx.dot(Z)))
            dataparamX.append([X, para])
        elif datatype == 'multinom':
            Wx, Mux, xi, alpha, b = theta
            assert np.sum(Wx[-1:, :]) == 0
            Muxn = np.array([Mux, ] * nsamp).T
            assert np.sum(Muxn[-1:, :]) == 0
            # assert np.sum(xi[-1:, :]) == 0
            dx = Wx.shape[0]
            p = softmax(Muxn + Wx.dot(Z))
            X = np.empty((dx, nsamp))
            for n in range(0, nsamp):
                X[:, n] = np.random.multinomial(b, p[:, n])
            dataparamX.append([X, para])
        else:
            print('unknown datatype')
            return
    if not paramt:
        dataparamt = False
    else:
        if not CEN:
            datatype, theta, sparse = paramt
            wt, Lam = theta
            t = np.random.exponential(1 / (Lam * np.exp(wt.dot(Z))))
            dataparamt = [t, paramt]
        elif CEN:
            [(datatypet, thetat, sparset),
             (datatypec, thetac, sparsec)] = paramt
            wt, Lam = thetat
            t = np.random.exponential(1 / (Lam * np.exp(wt.dot(Z))))
            if datatypec == 'cind':
                Lamc = thetac
                c = np.random.exponential(1 / (Lamc), nsamp)
                c = np.reshape(c, (1, nsamp))
            elif datatypec == 'cindIC':
                wtc, Lamc = thetac
                c = np.random.exponential(1 / (Lamc * np.exp(wtc.dot(Z))))
            tE = np.min(np.array([t, c]), axis=0)
            Delta = 1 * (tE == t)
            dataparamt = [[tE, (datatypet, thetat, sparset)],
                          [Delta, (datatypec, thetac, sparsec)]]
    return (dataparamX, dataparamt, Z)


#simulating reps of Z
def sim_Z(dz, nsamp, nrep):
    Zmean = np.zeros(dz)
    Zcov = np.diag(np.ones(dz))
    Zrep = np.empty((nrep, dz, nsamp))
    ZZtnrep = np.empty((nrep, nsamp, dz, dz))
    for i in range(0, nrep):
        Zrep[i, :, :] = (np.random.multivariate_normal(Zmean, Zcov, nsamp)).T
        ZZtnrep[i, :, :, :] = np.array([np.outer(Zrep[i, :, dn], Zrep[
            i, :, dn]) for dn in range(0, nsamp)])
    return (Zrep, ZZtnrep)


def sim_Z_Cox_functions(Zrep, wt):
    nrep, dz, nsamp = Zrep.shape
    expwtZrep = np.empty((nrep, nsamp))
    ZexpwtZrep = np.empty((nrep, dz, nsamp))
    ZZtexpwtZrep = np.empty((nrep, nsamp, dz, dz))
    for i in range(0, nrep):
        expwtZrep[i, :] = np.exp(wt.dot(Zrep[i, :, :]))
        ZexpwtZrep[i, :, :] = np.multiply(Zrep[i, :, :],
                                          np.exp(wt.dot(Zrep[i, :, :])))
        ZZtexpwtZrep[i, :, :, :] = np.array([np.outer(ZexpwtZrep[
            i, :, dn], Zrep[i, :, dn]) for dn in range(0, nsamp)])
    return (expwtZrep, ZexpwtZrep, ZZtexpwtZrep)


##using samples from Z|X metropolis - hastings to calculate E - step
def E_step_MH(Zrep):
    if len(Zrep.shape) == 3:
        nrep, dz, nsamp = Zrep.shape
        EZ = np.average(Zrep, axis=0)
        #EZZt = np.average([Zrep[x, :, :].dot(Zrep[x, :, :].T)
        #for x in range(0, nrep)], axis=0)
        EZZtn = np.array([np.average(
            [np.outer(Zrep[x, :, n], Zrep[x, :, n]) for x in range(0, nrep)],
            axis=0) for n in range(0, nsamp)])
    elif len(Zrep.shape) == 2:
        ##this use case is not an expectation, it simply calculates different
        ## functions of Z for use in lnprobdatagivenZ
        dz, nsamp = Zrep.shape
        EZ = Zrep
        #EZZt = Zrep.dot.(Zrep.T)
        EZZtn = np.array([np.outer(Zrep[:, dn], Zrep[:, dn])
                          for dn in range(0, nsamp)])
    #return(EZ, EZZt, EZZtn)
    return (EZ, EZZtn)


def E_step_MH_Cox(Zrep, wt):
    if len(Zrep.shape) == 3:
        nrep, dz, nsamp = Zrep.shape
        EexpwtZ = np.average(
            [np.exp(wt.dot(Zrep[x, :, :])) for x in range(0, nrep)],
            axis=0)
        ZexpwtZ = [np.multiply(Zrep[x, :, :], np.exp(wt.dot(Zrep[x, :, :])))
                   for x in range(0, nrep)]
        EZexpwtZ = np.average(ZexpwtZ, axis=0)
        #np.multiply(Zrep[x, :,:],np.exp(wt.dot(Zrep[x, :,:])))[:,0] -
        #Zrep[x, :, 0] * np.exp(wt.dot(Zrep[x, :,:]))[0, 0]
        EZZtexpwtZ = np.average(
            np.array([[np.outer(ZexpwtZ[x][:, y], Zrep[x, :, y])
                       for x in range(0, nrep)] for y in range(0, nsamp)]),
            axis=1)
    elif len(Zrep.shape) == 2:
        ##this use case is not an expectation, it simply calculates different
        #functions of Z for use in lnprobdatagivenZ
        dz, nsamp = Zrep.shape
        EexpwtZ = np.exp(wt.dot(Zrep))
        EZexpwtZ = np.multiply(Zrep, np.exp(wt.dot(Zrep)))
        #np.multiply(Zrep[x, :,:],np.exp(wt.dot(Zrep[x, :,:])))[:,0] -
        #Zrep[x, :, 0] * np.exp(wt.dot(Zrep[x, :,:]))[0, 0]
        EZZtexpwtZ = np.array([np.outer(EZexpwtZ[:, dn], Zrep[:, dn])
                               for dn in range(0, nsamp)])
    return (EexpwtZ, EZexpwtZ, EZZtexpwtZ)


##exact (with variational approx) calculation for no - survival E - step factor
## analysis with different datatypes.
def E_step_no_surv(dataparamX, dz):
    nsamp = dataparamX[0][0].shape[1]
    Cninv = np.empty((nsamp, dz, dz))
    for dn in range(0, nsamp):
        Cninv[dn, :, :] = np.diag(np.ones(dz))
    EZ = np.empty((dz, nsamp))
    EZZtn = np.empty((nsamp, dz, dz))
    temp = np.zeros((dz, nsamp))
    for k, datapara in enumerate(dataparamX):
        [X, (datatype, theta, sparse)] = datapara
        dx = X.shape[0]
        if datatype == 'normal':
            Wx0, Mux0, Psix0 = theta
            Psix0_inv = 1 / Psix0
            Mux0n = np.array([Mux0, ] * nsamp).T
            temp = temp + Wx0.T.dot(Psix0_inv * (X - Mux0n))
            for dn in range(0, nsamp):
                Cninv[dn, :, :] = (
                    Cninv[dn, :, :] + Wx0.T.dot(Psix0_inv * Wx0))
        elif datatype == 'binom':
            Wx0, Mux0, xi0, b = theta
            Mux0n = np.array([Mux0, ] * nsamp).T
            lamxi_xi0 = lamxi(xi0)
            temp = (temp + Wx0.T.dot(X - b / 2 * np.ones(X.shape)) - 2 * b *
                    Wx0.T.dot((lamxi_xi0 * Mux0n)))
            ## or the same, Mux - 2 * b * Wx0.T.dot((np.array([Mux0, ]).T * lamxi_xi0
            for dn in range(0, nsamp):
                for dz1 in range(0, dz):
                    for dz2 in range(0, dz):
                        Cninv[dn, dz1, dz2] = (
                            Cninv[dn, dz1, dz2] + 2 * b *
                            np.sum([lamxi_xi0[i, dn] * Wx0[i, dz1] * Wx0[
                                i, dz2] for i in range(0, dx)]))
        elif datatype == 'multinom':
            Wx0, Mux0, xi0, alpha0, b = theta
            Mux0n = np.array([Mux0, ] * nsamp).T
            lamxi_xi0 = lamxi(xi0)
            temp = (temp + Wx0.T.dot(X - b / 2 * np.ones(X.shape)) + Wx0.T.dot(
                2 * b * lamxi_xi0 * alpha0) - 2 * b *
                    Wx0.T.dot((lamxi_xi0 * Mux0n)))
            for dn in range(0, nsamp):
                for dz1 in range(0, dz):
                    for dz2 in range(0, dz):
                        Cninv[dn, dz1, dz2] = (
                            Cninv[dn, dz1, dz2] + 2 * b *
                            np.sum([lamxi_xi0[i, dn] * Wx0[i, dz1] * Wx0[
                                i, dz2] for i in range(0, dx)]))
        else:
            print('unknown datatype')
            return
    for dn in range(0, nsamp):
        Cn = scipy.linalg.inv(Cninv[dn, :, :])
        EZ[:, dn] = Cn.dot(temp[:, dn])
        EZZtn[dn, :, :] = Cn + np.outer(EZ[:, dn], EZ[:, dn])
    return (EZ, EZZtn)


## Q's are lnP(datatype|Z), and built to take in either Z or E[f(Z)]'s
def Q_Cox(wt, Lam, tE, Delta, EZ, EexpwtZ, datatype):
    #print('q_cox datatype', datatype)
    if datatype == 'cind':
        q_cox = np.sum(Delta) * np.log(Lam) - Lam * np.sum(tE)
    elif datatype == 'cindIC' or datatype == 'etime':
        q_cox = (np.sum(Delta) * np.log(Lam) + wt.dot(EZ).dot(Delta.T) - Lam *
                 EexpwtZ.dot(tE.T))
    return q_cox


def Q_Norm(Wx, Mux, Psix, X, EZ, EZZtn, sparse):
    nsamp = X.shape[1]
    dx, dz = Wx.shape
    EZZt = np.sum(EZZtn, 0)
    Psix_inv = 1 / Psix
    ##because Psix is diagonal.
    #Psix_inv = scipy.linalg.inv(Psix)
    Ginv = Wx.T.dot(Psix_inv * Wx)
    ##normal part of Cinv
    #print('Ginv in Q', Ginv)
    Muxn = np.array([Mux, ] * nsamp).T
    Psix_invdotMuxn = Psix_inv * Muxn
    WxdotEZ = Wx.dot(EZ)
    ##np.log(np.prod(np.diag(Psix))) instead of np.linalg.slogdet(Psix)[1]
    #(because Psix is diagonal) and actually, need to use:
    # - nsamp/2 * np.sum(np.log(np.diag(Psix)))
    #because otherwise product becomes 0 and it becomes infinite
    q_FA = (
        -nsamp * dx / 2 * np.log(2 * np.pi) - nsamp / 2 * np.sum(np.log(Psix))
        - 1 / 2 * np.sum(np.diag(X.T.dot(Psix_inv * X))) +
        np.sum(np.diag(X.T.dot(Psix_inv * WxdotEZ))) - 1 / 2 *
        np.sum(np.diag(EZZt.dot(Ginv))))
    #print('at q_norm', q_FA)
    q_FA = (q_FA - 1 / 2 * np.sum(np.diag(Muxn.T.dot(Psix_invdotMuxn))) -
            np.sum(np.diag(Muxn.T.dot(Psix_inv * WxdotEZ))) +
            np.sum(np.diag(X.T.dot(Psix_invdotMuxn))))
    #print('at q_norm', q_FA)
    if sparse[0]:
        q_FA = q_FA - sparse[1] * np.sum(np.abs(Wx))
    return q_FA


def Q_BIN(Wx, Mux, xi, alpha, b, X, EZ, EZZtn, sparse, datatype):
    nsamp = X.shape[1]
    dx, dz = Wx.shape
    temp = 0
    lamxi_xi = lamxi(xi)
    Muxn = np.array([Mux, ] * nsamp).T
    WxdotEZ = Wx.dot(EZ)
    prefactor = 0
    if datatype == 'multinom':
        ##adding in the alpha_n pieces.
        prefactor = (prefactor - b * (1 - dx / 2) * np.sum(alpha) - b *
                     np.sum(lamxi_xi.dot((alpha**2).T)) + 2 * b *
                     np.diag(lamxi_xi.T.dot(WxdotEZ)).dot(alpha.T))
        prefactor = prefactor + 2 * b * np.sum(lamxi_xi * Muxn * alpha)
        # print('alpha part', prefactor)
    #this part is now calcuated in the tensor expression below.
    # for dn in range(0, nsamp):
    #     for i in range(0, dx):
    #         temp = (temp - b * lamxi_xi[i, dn]
    #                 * Wx[i, :].dot(EZZtn[dn, :, :]).dot(Wx[i, :].T)
    temp = (-b * np.sum(lamxi_xi * np.einsum('i...i->i...',
                                             np.tensordot(
                                                 np.tensordot(Wx,
                                                              EZZtn,
                                                              axes=([1], [1])),
                                                 Wx,
                                                 axes=([2], [1])))))
    q_BIN = (prefactor +
             np.sum(np.diag((X - b / 2 * np.ones(X.shape)).T.dot(WxdotEZ))) + b
             * np.sum(np.log(scipy.special.expit(xi))) - b / 2 * np.sum(xi) +
             temp + np.sum(b * lamxi_xi * (xi**2)))
    q_BIN = (q_BIN + np.sum(X * Muxn) - b / 2 * np.sum(Muxn) - b * 2 * np.sum(
        WxdotEZ * Muxn * lamxi_xi) - b / 2 * np.sum(lamxi_xi * Muxn**2))
    if sparse[0]:
        q_BIN = q_BIN - sparse[1] * np.sum(np.abs(Wx))
    return q_BIN


def Q_Latent(EZZtn):
    nsamp, dz, dz = EZZtn.shape
    q_latent = (
        -nsamp * dz / 2 * np.log(2 * np.pi) - 1 / 2 *
        np.sum([np.sum(np.diag(EZZtn[dn, :, :])) for dn in range(nsamp)]))
    return q_latent


##simulating ln(p(X|Z)) for use in loss function - ln(p(x)).
##for Q put in EZ, EZZtn, EexpwtZ, EexptwtcZ
def lnprobdatagivenZ(dataparamX, dataparamt, Z, ZZtn, expwtZ, expwtcZ, CEN, VERBOSE=False):
    ##if dataparamt == False, then make expwtZ, expwtcZ == False.
    lnprobdatagivenZ = 0
    for i, datapara in enumerate(dataparamX):
        [X, (datatype, theta, sparse)] = datapara
        nsamp = X.shape[1]
        if datatype == 'normal':
            Wx, Mux, Psix = theta
            dx, dz = Wx.shape
            lnprobdatagivenZ = (
                lnprobdatagivenZ + Q_Norm(Wx, Mux, Psix, X, Z, ZZtn, sparse))
            if VERBOSE: print('qnorm', Q_Norm(Wx, Mux, Psix, X, Z, ZZtn, sparse))
        elif datatype == 'binom':
            Wx, Mux, xi, b = theta
            lnprobdatagivenZ = (
                lnprobdatagivenZ + Q_BIN(Wx, Mux, xi, np.zeros((1, nsamp)), b,
                                         X, Z, ZZtn, sparse, datatype))
            if VERBOSE: print('qbin', Q_BIN(Wx, Mux, xi, np.zeros((1,nsamp)), b, X, Z, ZZtn, sparse, datatype))
        elif datatype == 'multinom':
            Wx, Mux, xi, alpha, b = theta
            lnprobdatagivenZ = (lnprobdatagivenZ + Q_BIN(
                Wx, Mux, xi, alpha, b, X, Z, ZZtn, sparse, datatype))
            if VERBOSE: print('qmultinom', Q_BIN(
                Wx, Mux, xi, alpha, b, X, Z, ZZtn, sparse, datatype))
        else:
            print('unknown datatype')
            return
    #if dataparamt == False:
    #print('lnprobdatagivenZ: no time leg')
    #if dataparamt != False:
    if not isinstance(dataparamt, bool):
        if not CEN:
            [tE, (datatypet, thetat, sparset)] = dataparamt
            wt, Lam = thetat
            # print('lnprobdatagivenZ', datatypet)
            lnprobdatagivenZ = (lnprobdatagivenZ + Q_Cox(
                wt, Lam, tE, np.ones((1, nsamp)), Z, expwtZ, datatypet))
            if VERBOSE: print('qcox', Q_Cox(wt, Lam, tE, np.ones((1, nsamp)), Z, expwtZ, datatypet))
        elif CEN:
            [[tE, (datatypet, thetat, sparset)],
             [Delta, (datatypec, thetac, sparsec)]] = dataparamt
            wt, Lam = thetat
            Deltac = np.ones(Delta.shape) - Delta
            lnprobdatagivenZ = (lnprobdatagivenZ + Q_Cox(wt, Lam, tE, Delta, Z,
                                                         expwtZ, datatypet))
            if VERBOSE: print('qcox', Q_Cox(wt, Lam, tE, Delta, Z, expwtZ, datatypet))
            if datatypec == 'cind':
                Lamc = thetac
                lnprobdatagivenZ = (lnprobdatagivenZ + Q_Cox(
                    np.zeros((1, dz)), Lamc, tE, Deltac, Z, False, datatypec))
                if VERBOSE: print('qcox cind', Q_Cox(np.zeros((1, dz)), Lamc, tE, Deltac, Z, False, datatypec))
            elif datatypec == 'cindIC':
                wtc, Lamc = thetac
                lnprobdatagivenZ = (lnprobdatagivenZ + Q_Cox(
                    wtc, Lamc, tE, Deltac, Z, expwtcZ, datatypec))
                if VERBOSE: print('qcox cindIC', Q_Cox(wtc, Lamc, tE, Deltac, Z, expwtcZ, datatypec))
    assert np.isinf(lnprobdatagivenZ) == False
    return lnprobdatagivenZ


##exact (with variational approx) calculation for no - survival loss function
#- ln(p(x)) with different datatypes.
def loss_function_no_surv(dataparamX, EZ, EZZtn):
    nsamp = dataparamX[0][0].shape[1]
    Z = np.zeros(EZ.shape)
    ZZtn = np.zeros(EZZtn.shape)
    lnprobdata = lnprobdatagivenZ(dataparamX, False, Z, ZZtn, False, False,
                                  False)
    ##this calcuates the bits that are independent of z in p(x|z)
    for dn in range(0, nsamp):
        Cn = EZZtn[dn, :, :] - np.outer(EZ[:, dn], EZ[:, dn])
        ##Cn is the covariance matrix, EZ[:, dn] is the mean.
        lnprobdata = (lnprobdata + 1 / 2 *
                      EZ[:, dn].T.dot(scipy.linalg.inv(Cn)).dot(EZ[:, dn]) -
                      np.linalg.slogdet(Cn)[1] / 2)
        ## no numerical part because that canceled out in the integration.
    return -lnprobdata


##ln(p(X, Z)) for metropolis - hastings.  this function also calculates Q(model) when Z arguments are expectations of functions of Z.
def jointlogprob(dataparamX, dataparamt, Z, ZZtn, expwtZ, expwtcZ, CEN):
    jntlnprob = lnprobdatagivenZ(dataparamX, dataparamt, Z, ZZtn, expwtZ,
                                 expwtcZ, CEN) + Q_Latent(ZZtn)
    return jntlnprob


def pick_out_ith_individual(i, dataparamX, dataparamt, CEN):
    ##pick out the ith individual
    dataparamXi = []
    for k, datapara in enumerate(dataparamX):
        [X, (datatype, theta, sparse)] = datapara
        Xi = X[:, i]
        Xi = np.reshape(Xi, (X.shape[0], 1))
        if datatype == 'binom':
            Wx, Mux, xi, b = theta
            theta = (Wx, Mux, np.reshape(xi[:, i], (X.shape[0], 1)), b)
        elif datatype == 'multinom':
            Wx, Mux, xi, alpha, b = theta
            theta = (Wx, Mux, np.reshape(xi[:, i], (X.shape[0], 1)),
                     np.reshape(alpha[:, i], (1, 1)), b)
        dataparamXi.append([Xi, (datatype, theta, sparse)])
    #if dataparamt == False:
    if isinstance(dataparamt, bool):
        dataparamti = False
    else:
        if not CEN:
            [tE, (datatype, theta, sparse)] = dataparamt
            tEi = np.reshape(tE[:, i], (1, 1))
            dataparamti = [tEi, (datatype, theta, sparse)]
        elif CEN:
            [[tE, (datatypet, thetat, sparset)],
             [Delta, (datatypec, thetac, sparsec)]] = dataparamt
            tEi = np.reshape(tE[:, i], (1, 1))
            Deltai = np.reshape(Delta[:, i], (1, 1))
            dataparamti = [[tEi, (datatypet, thetat, sparset)],
                           [Deltai, (datatypec, thetac, sparsec)]]
    return (dataparamXi, dataparamti)


def calculate_time_functions_for_MH(dataparamt, Zti, CEN):
    #calculating functions of Z for jointlogprob
    Zti, ZZti = E_step_MH(Zti)
    #if dataparamt == False:
    if isinstance(dataparamt, bool):
        expwtZi, expwtcZi = False, False
    #print('MH_sim_Z_given_data_p: no time leg')
    else:
        if not CEN:
            [tE, (datatype, theta, sparse)] = dataparamt
            wt, Lam = theta
            expwtZi, ZexpwtZi, ZZtexpwtZi = E_step_MH_Cox(Zti, wt)
            expwtcZi = False
        elif CEN:
            [[tE, (datatypet, thetat, sparset)],
             [Delta, (datatypec, thetac, sparsec)]] = dataparamt
            wt, Lam = thetat
            if datatypec == 'cind':
                expwtZi, ZexpwtZi, ZZtexpwtZi = E_step_MH_Cox(Zti, wt)
                expwtcZi = False
            elif datatypec == 'cindIC':
                wtc, Lamc = thetac
                expwtZi, ZexpwtZi, ZZtexpwtZi = E_step_MH_Cox(Zti, wt)
                expwtcZi, ZexpwtcZi, ZZtexpwtcZi = E_step_MH_Cox(Zti, wtc)
    return (Zti, ZZti, expwtZi, expwtcZi)


##simulating Z|X for metropolis - hastings.
def MH_sim_Z_given_data_p((i, dz, dataparamX, dataparamt, nrep, CEN, burn,
                           SEED, sigma2)):
    #print(SEED)
    np.random.seed(SEED)
    (dataparamXi, dataparamti) = pick_out_ith_individual(i, dataparamX,
                                                         dataparamt, CEN)
    #Zcov = np.diag(np.ones(dz))
    Zrep = np.empty((nrep + burn, dz))
    #counter = 0
    #initialize
    #sigma2 = 1
    Zti, EZZtn = E_step_no_surv(dataparamXi, dz)
    #Zti = np.random.multivariate_normal(np.zeros(dz), sigma2 * Zcov)
    Zti = np.reshape(Zti, (dz, 1))
    Zcov = EZZtn[0] - np.outer(Zti, Zti)
    ##this is simply to get the random calls to align with the missing data, to enable checking.
    #Zti= so_can_compare(np.zeros(dz), dz, dataparamXi)
    (Zti, ZZti, expwtZi, expwtcZi) = calculate_time_functions_for_MH(
        dataparamti, Zti, CEN)
    q = jointlogprob(dataparamXi, dataparamti, Zti, ZZti, expwtZi, expwtcZi,
                     CEN)
    c = 0
    #acceptance = np.empty((nrep + burn, 1))
    #done = False
    #simulate
    #while c<(nrep + burn) and done == False:
    while c < (nrep + burn):
        Ztpi = np.random.multivariate_normal(Zti[:, 0], sigma2 * Zcov)
        Ztpi = np.reshape(Ztpi, (dz, 1))
        ##this is simply to get the random calls to align with the missing data, to enable checking.
        #Ztpi= so_can_compare(Zti[:, 0], dz, dataparamXi)
        (Ztpi, ZZtnpi, expwtZpi, expwtcZpi) = calculate_time_functions_for_MH(
            dataparamti, Ztpi, CEN)
        #calculating cutoff
        qp = jointlogprob(dataparamXi, dataparamti, Ztpi, ZZtnpi, expwtZpi,
                          expwtcZpi, CEN)
        #cutoff = np.exp(jointlogprob(dataparamXi, dataparamti, Ztpi, ZZtnpi, expwtZpi, expwtcZpi, CEN) - jointlogprob(dataparamXi, dataparamti, Zti, ZZti, expwtZi, expwtcZi, CEN))
        cutoff = np.exp(qp - q)
        # print(c, cutoff)
        # print('qp, q', qp, q, cutoff == np.exp(jointlogprob(dataparamXi, dataparamti, Ztpi, ZZtnpi, expwtZpi, expwtcZpi, CEN) - jointlogprob(dataparamXi, dataparamti, Zti, ZZti, expwtZi, expwtcZi, CEN)))
        assert np.isnan(cutoff) == False
        if cutoff >= 1:
            Zrep[c, :] = np.copy(Ztpi[:, 0])
            Zti = np.copy(Ztpi)
            # ZZti = np.copy(ZZtnpi)
            # expwtZi = np.copy(expwtZpi)
            # expwtcZi = np.copy(expwtcZpi)
            q = np.copy(qp)
        #    acceptance[c, :] = 1
        else:
            if np.random.binomial(1, cutoff) == 1:
                Zrep[c, :] = np.copy(Ztpi[:, 0])
                Zti = np.copy(Ztpi)
                # ZZti = np.copy(ZZtnpi)
                # expwtZi = np.copy(expwtZpi)
                # expwtcZi = np.copy(expwtcZpi)
                q = np.copy(qp)
        #        acceptance[c, :] = 1
            else:
                Zrep[c, :] = np.copy(Zti[:, 0])
        #        acceptance[c, :] = 0
        #        if c == burn:
        #            arate = np.sum(acceptance[0:burn, :])/burn
        #            #print(arate)
        #            counter = counter + 1 * (arate>0.1 and arate<.6)
        #            #print(counter)
        c = c + 1
    return Zrep[burn:(burn + nrep), :]


def MH_sim_Z_given_data_p_error_wrapper((i, dz, dataparamX, dataparamt, nrep, CEN, burn,
                           SEED, sigma2)):
    try:
        Zrep= MH_sim_Z_given_data_p((i, dz, dataparamX, dataparamt, nrep, CEN, burn,
                           SEED, sigma2))
    except AssertionError:
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb)
        # Fixed format
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]
        with open('assertion_error_sample_'+str(i)+'_'+str(time())+'.txt', 'w') as f:
            f.write('An error occurred on line {} in statement {}'.format(line, text))
        Zrep=np.empty((nrep, dz))
        Zrep=np.nan
    return Zrep


## simulating log p(x) using metropolis hastings
def mh_sim_log_marginal_data((i, dz, dataparamX, dataparamt, Zrep, CEN, SEED)):
    ##Zrep needs to be the samples from mh_sim_Z_given_data_p
    np.random.seed(SEED)
    (dataparamXi, dataparamti) = pick_out_ith_individual(i, dataparamX,
                                                         dataparamt, CEN)
    ##making the first sample the reference point.
    zstar = Zrep[0, :, i, None]
    (Zti, ZZti, expwtZi, expwtcZi) = calculate_time_functions_for_MH(
        dataparamti, zstar, CEN)
    qstar = jointlogprob(dataparamXi, dataparamti, Zti, ZZti, expwtZi,
                         expwtcZi, CEN)
    #print('estimate of log p(zstar, x)', qstar, i)
    ##initializing arrays
    nsamp_sim = Zrep.shape[0] - 1
    qrep = np.empty(nsamp_sim)
    gen_den_zrep = np.empty(nsamp_sim)
    q_gen_den_samp = np.empty(nsamp_sim)
    cutoff_zrep = np.empty(nsamp_sim)
    cutoff_gen_den_samp = np.empty(nsamp_sim)
    ##generating samples from the candidate generating density
    sigma2 = 1
    Zcov = np.diag(np.ones(dz))
    gen_den_samp = np.random.multivariate_normal(zstar[:, 0], sigma2 * Zcov,
                                                 nsamp_sim)
    for k in range(nsamp_sim):
        ##calculating the q function for points sampled from z|x.  (the zeroth point is the reference point)
        Zti = Zrep[k + 1, :, i, None]
        (Zti, ZZti, expwtZi, expwtcZi) = calculate_time_functions_for_MH(
            dataparamti, Zti, CEN)
        qrep[k] = jointlogprob(dataparamXi, dataparamti, Zti, ZZti, expwtZi,
                               expwtcZi, CEN)
        #the candidate generating probability density for the points sampled from z|x and the reference point zstar
        gen_den_zrep[k] = 1 / (np.power(
            np.sqrt(2 * np.pi), dz)) * np.exp(-np.sum((zstar - Zti)**2) / 2)
        ##calculating the q function for points sampled from the candidate generating density
        Zti = gen_den_samp[k, :, None]
        (Zti, ZZti, expwtZi, expwtcZi) = calculate_time_functions_for_MH(
            dataparamti, Zti, CEN)
        q_gen_den_samp[k] = jointlogprob(dataparamXi, dataparamti, Zti, ZZti,
                                         expwtZi, expwtcZi, CEN)
        cutoff_zrep[k] = np.min([1, np.exp(qstar - qrep[k])])
        cutoff_gen_den_samp[k] = np.min([1, np.exp(q_gen_den_samp[k] - qstar)])
    ## estimate of p(zstar|x)
    p_zstargivendata = np.sum(cutoff_zrep *
                              gen_den_zrep) / np.sum(cutoff_gen_den_samp)
    #print('estimate of p(zstar|x)', p_zstargivendata, np.log(p_zstargivendata), i)
    log_marginal_data = qstar - np.log(p_zstargivendata)
    return log_marginal_data


def pick_scale(dz, dataparamX0, dataparamt0, nrep_batch, CEN, burn, plotloc):
    sigma2 = [6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, .5, 0.25, 0.1]
    SEED1=1398413422#3234
    SEED2= 909879879#38453
    dn=0
    acoll=[]
    rcoll=[]
    neffcoll=[]
    for item in sigma2:
        draws1= MH_sim_Z_given_data_p((dn, dz, dataparamX0, dataparamt0,
                                           nrep_batch, CEN, burn, SEED1, item))
        arate1=len(np.unique(draws1[:, 0]))/ nrep_batch
        draws2= MH_sim_Z_given_data_p((dn, dz, dataparamX0, dataparamt0,
                                           nrep_batch, CEN, burn, SEED2, item))
        arate2= len(np.unique(draws2[:, 0]))/ nrep_batch
        acoll.append(np.abs((arate1 + arate2)/2 - .234))
        psi= np.vstack((np.diag(draws1.dot(draws1.T)), np.diag(draws2.dot(draws2.T)))).T
        rhat, neff=converg_monitor(psi)
        rcoll.append(rhat)
        neffcoll.append(neff)
        if np.average((arate1, arate2)) >= 0.134 and np.average((arate1, arate2)) <= 0.334 and rhat <= 1.2 and neff>=10:
            with open(plotloc+ 'MH_scale.txt', 'w') as f:
                f.write("scale is " + str(item) + ", acceptance is " + str(np.average((arate1, arate2))) + ", rhat is " + str(rhat) + ", neff is " + str(neff) )
            return(item)
    best=acoll.index(np.min(acoll))
    with open(plotloc+ 'MH_scale.txt', 'w') as f:
        f.write("unable to find good scale, try larger burn and nrep_batch.  used scale " + str(sigma2[best]) + ", acceptance diff from 0.234 " + str(acoll[best]) + ", rhat " + str(rcoll[best]) + ", neff " + str(neffcoll[best]))
    return(sigma2[best])


def converg_monitor(psi):
    """
    from gelman and rubin's book
    m sequences of length n (after discarding first half)
    psi is a scalar quantity and psi_ij i is in n, j is in m.
    """
    n, m = psi.shape
    psidotj = 1/n * np.sum(psi, axis=0)
    psidotdot= 1/m *np.sum(psidotj)
    B= n/(m-1)*np.sum((psidotj -psidotdot)**2)
    Sjsq =  1/(n-1)*np.sum((psi -psidotj)**2, axis=0)
    W = 1/m *np.sum(Sjsq)
    varplus = (n-1)/n * W + 1/n * B
    Rhat = np.sqrt(varplus/W)
    neff= m * n* varplus/B
    neff = np.min((n*m, neff))
    return(Rhat, neff)


def M_step_Norm(Wx0, Psix0, X, EZ, EZZtn, sparse):
    nsamp = X.shape[1]
    dx, dz = Wx0.shape
    EZZt = np.sum(EZZtn, axis=0)
    if not sparse[0]:
        Wxp = X.dot(EZ.T).dot(scipy.linalg.inv(EZZt))
        #Psixp = 1/nsamp * (X.dot(X.T) - Wxp.dot(EZZt).dot(Wxp.T))
        Psixp = 1 / nsamp * np.diag(X.dot(X.T) - Wxp.dot(EZZt).dot(
            Wxp.T))[:, None]
        #print(Psixp)
        #Psixp = 1/nsamp * np.diag(X.dot(X.T) - Wxp.dot(EZ.dot(X.T)))
        #print(Psixp)
        #Psixp = np.diag(Psixp)
    else:
        Wxp = np.empty((dx, dz))
        ######### test for when alpha is 0
        #alt = np.empty((dx, dz))
        ######### end test
        LEZZt = np.linalg.cholesky(EZZt)
        print('checking svd of cholesky', scipy.linalg.svd(LEZZt)[1])
        Y = X.dot(EZ.T).dot(scipy.linalg.inv(LEZZt.T))
        for i in range(0, dx):
            #print('cutoff', Psix0[i, i] * sparse[1]/dz)
            LLfit = linear_model.LassoLars(Psix0[i, 0] * sparse[1] / dz,
                                           fit_intercept=False)
            ##otherwise intercept is included in code and not penalized.
            LLfit.fit(LEZZt.T, Y[i, :])
            Wxp[i, :] = LLfit.coef_
            print('Wxp all zero?', np.sum(Wxp[i, :]) == 0)
            #print(1/(2 * dz) * (Y[i, :] - Wxp[i, :].dot(LEZZt)).dot(Y[i, :] - Wxp[i, :].dot(LEZZt)))
            ######### test for when alpha is 0
            # Lfit = linear_model.LinearRegression(fit_intercept=False)
            # Lfit.fit(LEZZt.T,Y[i, :])
            # alt[i, :] = Lfit.coef_
            # print('number close', np.sum(np.isclose(Wxp[i, :], alt[i, :])))
            # assert np.allclose(Wxp[i, :], alt[i, :])
            ######### end test
            #print(1/(2 * dz) * (Y[i, :] - alt[i, :].dot(LEZZt)).dot(Y[i, :] - alt[i, :].dot(LEZZt)) + Psix0[i, i] * sparse[1]/ dz * np.sum(np.abs(alt[i, :])))
            #print(1/(2 * dz) * (Y[i, :] - alt[i, :].dot(LEZZt)).dot(Y[i, :] - alt[i, :].dot(LEZZt)))
            #print(Y[i, :] - LEZZt.T.dot(alt[i, :]))
        print('M_step_Norm, sparse: should all be positive, abs(full) - abs(sparse)', np.abs(
            X.dot(EZ.T).dot(scipy.linalg.inv(EZZt))) - np.abs(Wxp))
        ######### test for when alpha is 0
        # assert np.allclose(X.dot(EZ.T).dot(scipy.linalg.inv(EZZt)), alt)
        ######### end test
        #print(np.diag(Psix0))
        Psixp = 1 / nsamp * np.diag(X.dot(X.T) - Wxp.dot(EZZt).dot(
            Wxp.T))[:, None]
    assert Psixp.shape == (dx, 1)
    small = 0.00005
    print("checks on w, psi", np.sum(Psixp > small), np.linalg.matrix_rank(
        Wxp,
        tol=None))
    assert np.sum(Psixp > small) == dx
    #assert np.linalg.matrix_rank(Wxp, tol=None) == np.min([dx, dz])
    #Psixp = np.diag(Psixp)
    return (Wxp, Psixp)


def M_step_BIN(Wx0, Mux0, xi0, alpha0, b, X, EZ, EZZtn, sparse, datatype):
    ##EZZtn[n, j, k]
    nsamp = X.shape[1]
    dx, dz = Wx0.shape
    Wxp = np.empty((dx, dz))
    xip = np.empty((dx, nsamp))
    temp = np.tensordot(Wx0, EZZtn, axes=([1], [1]))
    ##[dx = i, nsamp, dz = i, k]
    Mux0n = np.array([Mux0, ] * nsamp).T
    print('M_step_BIN:  isnan temp', np.sum(np.isnan(temp)),
          np.sum(np.isnan(EZZtn)))
    for i in range(0, dx):
        xip[i, :] = np.sqrt(
            np.tensordot(Wx0[i, :],
                         temp[i, :, :],
                         axes=([0], [1])) + 2 * Mux0[i] * Wx0[i, :].dot(EZ) +
            Mux0n[i, :]**2 + (datatype == 'multinom') * (-2 * Wx0[i, :].dot(
                EZ) * alpha0 + alpha0**2 - 2 * alpha0 * Mux0[i]))
        #print('less than zero?',
        #    np.sum((np.tensordot(Wx0[i, :], temp[i, :, :], axes=([0], [1]))
        #    + (datatype == 'multinom') * ( - 2 * Wx0[i, :].dot(EZ) * alpha0
        #    + alpha0 ** 2))<0), 'equal zero?',
        #   np.sum((np.tensordot(Wx0[i, :], temp[i, :, :], axes=([0], [1]))
        #   + (datatype == 'multinom') * ( - 2 * Wx0[i, :].dot(EZ) * alpha0
        #    + alpha0 ** 2)) == 0))
        print('Wx0 all zero?', np.sum(Wx0[i, :]) == 0)
        print('isnan?', np.sum(np.isnan(lamxi(xip[i, :]))),
              np.sum(np.isnan(xip[i, :])))
    # if datatype == 'multinom':
    #     xip[-1:, :] = 0
    #         ##fix for multinom
    lamxi_xip = lamxi(xip)
    if datatype == 'multinom':
        alphap = (
            np.diag(lamxi_xip.T.dot(Wx0).dot(EZ)) + lamxi_xip.T.dot(Mux0) -
            (1 - dx / 2) / 2) / np.sum(lamxi_xip,
                                       axis=0)
        alphap = np.reshape(alphap, (1, nsamp))
        print('alpha', alphap[0][0:5])
    else:
        alphap = alpha0
    if not sparse[0]:
        for i in range(0, dx):
            L = np.linalg.cholesky(2 * b * np.tensordot(lamxi_xip[i, :],
                                                        EZZtn,
                                                        axes=([0], [0])))
            #print('checking svd of cholesky', scipy.linalg.svd(L)[1])
            Y = ((X[i, :] - b / 2 * np.ones(X[i, :].shape) - 2 * b * Mux0[i] *
                  lamxi_xip[i, :] + (datatype == 'multinom') * 2 * b *
                  (alphap *
                   lamxi_xip[i, :])).dot(EZ.T).dot(scipy.linalg.inv(L.T)))
            Lfit = linear_model.LinearRegression(fit_intercept=False)
            Lfit.fit(L.T, Y.T)
            Wxp[i, :] = Lfit.coef_
    else:
        for i in range(0, dx):
            L = np.linalg.cholesky(2 * b * np.tensordot(lamxi_xip[i, :],
                                                        EZZtn,
                                                        axes=([0], [0])))
            #print('checking svd of cholesky', scipy.linalg.svd(L)[1])
            Y = ((X[i, :] - b / 2 * np.ones(X[i, :].shape) - 2 * b * Mux0[i] *
                  lamxi_xip[i, :] + (datatype == 'multinom') * 2 * b *
                  (alphap *
                   lamxi_xip[i, :])).dot(EZ.T).dot(scipy.linalg.inv(L.T)))
            LLfit = linear_model.LassoLars(sparse[1] / dz, fit_intercept=False)
            ##otherwise intercept is included in code and not penalized.
            LLfit.fit(L.T, Y.T)
            Wxp[i, :] = LLfit.coef_
    Muxp = ((np.sum(X,
                    axis=1) - b / 2 * nsamp - 2 * b *
             np.diag(lamxi_xip.dot(EZ.T).dot(Wxp.T)) + 2 * b *
             lamxi_xip.dot(alphap.T)[:, 0]) / (2 * b * np.sum(lamxi_xip,
                                                              axis=1)))
    if datatype == 'multinom':
        assert np.sum(Wx0[-1:, :]) == 0
        assert np.sum(Mux0n[-1:, :]) == 0
        # assert np.sum(xi0[-1:, :]) == 0
        Wxp[-1:, :] = 0
        ##fix for multinom
        Muxp[-1] = 0
        ##fix for multinom
        #assert np.linalg.matrix_rank(Wxp, tol=None) == np.min([dx-1, dz])
    else:
        print(np.linalg.matrix_rank(Wxp, tol=None))
        # assert np.linalg.matrix_rank(Wxp, tol=None) == np.min([dx, dz])
    # print np.linalg.matrix_rank(Wxp, tol=None)
    return (Wxp, Muxp, xip, alphap)


##newton - raphson m - step update for exponential time data
def M_step_NR(wt0, Lam0, tE, Delta, EZ, EexpwtZ, EZexpwtZ, EZZtexpwtZ, step):
    dz, nsamp = EZ.shape
    A = EexpwtZ.dot(tE.T)
    A = np.reshape(A, (1, 1))
    B = EZexpwtZ.dot(tE.T)
    B = np.reshape(B, (1, dz))
    D = np.sum(
        np.array([tE[:, dn] * EZZtexpwtZ[dn, :, :] for dn in range(0, nsamp)]),
        axis=0)
    alle = np.bmat([[A, B], [B.T, D]])
    assert scipy.linalg.svd(alle)[1][-1:] > .000001
    all_inv = scipy.linalg.inv(alle)
    Ap = all_inv[0, 0, None, None]
    Bp = all_inv[0, 1:, None].T
    DCAB = all_inv[1:, 1:]
    print('M_step_NR: svd of [[Ap, Bp], [Bp.T, DCAB]] ',
          scipy.linalg.svd(np.bmat([[Ap, Bp], [Bp.T, DCAB]]))[1])
    Mp = np.sum(Delta) - Lam0 * A
    Mp = np.reshape(Mp, (1, 1))
    Np = EZ.dot(Delta.T).T - Lam0 * B
    #print("Mp", Mp)
    #print("Np", Np)
    beta0 = np.log(Lam0)
    betap = beta0 + step * (Ap * Mp + Bp.dot(Np.T)) / Lam0
    Lamp = np.exp(betap)
    #print("betap", betap)
    wtp = wt0 + step * (Bp.T * Mp + DCAB.dot(Np.T)).T / Lam0
    #print("wtp", wtp, wtp.shape)
    return (wtp, Lamp)


def pPCAsoln(X, dz):
    dx, nsamp = X.shape
    Ux, lamx, VxT = scipy.linalg.svd(X, full_matrices=False)
    if dx > dz:
        SigpPCA = sum(lamx[dz:]**2) / (nsamp * (dx - dz))
        WxpPCA = Ux[:, 0:dz].dot(np.diag(np.sqrt(lamx[0:dz]**2 - SigpPCA * np.ones(
            dz))))
        #PsixpPCA = np.diag(SigpPCA * np.ones(dx))
        PsixpPCA = (SigpPCA * np.ones(dx))[:, None]
    if dx <= dz:
        SigpPCA = lamx[-1]**2 / nsamp
        WxpPCA = np.ones((dx, dz))
        PsixpPCA = (SigpPCA * np.ones(dx))[:, None]
        print('dx<= dz, not ppca initialization')
    return WxpPCA, PsixpPCA


def EM_initialize(dataparamX, dataparamt, dz, CEN):
    nsamp = dataparamX[0][0].shape[1]
    Xmeans = []
    dataparamX0 = []
    dataparamt0 = []
    for i, datapara in enumerate(dataparamX):
        [X, (datatype, theta, sparse)] = datapara
        dx = X.shape[0]
        if datatype == 'normal':
            Xmean = np.mean(X, 1)
            Xmeans.append(Xmean)
            # reporting out means for normal data.
            Xc = X - np.array([Xmean, ] * nsamp).T
            #initialize.  centering normal data and giving it mean zero.
            (Wx0, Psix0) = pPCAsoln(Xc, dz)
            Mux0 = np.zeros(Xmean.shape)
            theta0 = (Wx0, Mux0, Psix0)
            datapara0 = [Xc, (datatype, theta0, sparse)]
            dataparamX0.append(datapara0)
        elif datatype == 'binom':
            Xmean = np.mean(X, 1)
            Xmeans.append(Xmean)
            Xc = X - np.array([Xmean, ] * nsamp).T
            #initialize
            (Wx0, Psix0) = pPCAsoln(Xc, dz)
            xi0 = np.ones((dx, nsamp))
            b = theta[3]
            theta0 = (Wx0, Xmean, xi0, b)
            datapara0 = [X, (datatype, theta0, sparse)]
            dataparamX0.append(datapara0)
        elif datatype == 'multinom':
            Xmean = np.mean(X, 1)
            Xmeans.append(Xmean)
            Xc = X - np.array([Xmean, ] * nsamp).T
            #initialize
            (Wx0, Psix0) = pPCAsoln(Xc, dz)
            Wx0[-1:, :] = 0
            ##fix for multinom
            Xmean[-1] = 0
            ##fix for multinom
            xi0 = np.ones((dx, nsamp))
            # xi0[-1:, :] = 0
            # ##fix for multinom
            alpha0 = np.ones((1, nsamp))
            b = theta[4]
            theta0 = (Wx0, Xmean, xi0, alpha0, b)
            datapara0 = [X, (datatype, theta0, sparse)]
            dataparamX0.append(datapara0)
        else:
            print('unknown datatype')
            return
    #if dataparamt == False:
    if isinstance(dataparamt, bool):
        print('initialization: no time leg')
        dataparamt0 = dataparamt
    else:
        if not CEN:
            [tE, (datatype, theta, sparse)] = dataparamt
            Lam0 = nsamp / np.sum(tE)
            wt0 = np.zeros((1, dz))
            #wt0 = np.random.normal([0], [1], dz)
            #wt0 = np.reshape(wt0, (1, dz))
            theta0 = (wt0, Lam0)
            dataparamt0 = [tE, (datatype, theta0, sparse)]
        elif CEN:
            [[tE, (datatypet, thetat, sparset)],
             [Delta, (datatypec, thetac, sparsec)]] = dataparamt
            Lam0 = np.sum(Delta) / np.sum(tE)
            wt0 = np.zeros((1, dz))
            #wt0 = np.random.normal([0], [1], dz)
            #wt0 = np.reshape(wt0, (1, dz))
            thetat0 = wt0, Lam0
            Deltac = np.ones(Delta.shape) - Delta
            if datatypec == 'cind':
                Lamc0 = np.sum(Deltac) / np.sum(tE)
                thetac0 = Lamc0
            elif datatypec == 'cindIC':
                Lamc0 = np.sum(Deltac) / np.sum(tE)
                wtc0 = np.zeros((1, dz))
                #wtc0 = np.random.normal([0], [1], dz)
                #wtc0 = np.reshape(wtc0, (1, dz))
                thetac0 = wtc0, Lamc0
            dataparamt0 = [[tE, (datatypet, thetat0, sparset)],
                           [Delta, (datatypec, thetac0, sparsec)]]
    return (dataparamX0, dataparamt0, Xmeans)


def check_convergence(dataparamXp, dataparamtp, dataparamX0, dataparamt0, CEN):
    maxdiff = 0
    for i, datapara in enumerate(dataparamX0):
        [X, (datatype, theta0, sparse)] = datapara
        #theta0 = datapara[1][1]
        #[X, (datatype, thetap, sparse)] = dataparamXp[i]
        thetap = dataparamXp[i][1][1]
        if datatype == 'normal':
            Wx0, Mux0, Psix0 = theta0
            Wxp, Muxp, Psixp = thetap
            maxdiff = np.max([np.max(np.abs(Wxp - Wx0)), maxdiff])
        elif datatype == 'binom':
            Wx0, Mux0, xi0, b = theta0
            Wxp, Muxp, xip, b = thetap
            maxdiff = np.max([np.max(np.abs(Wxp - Wx0)), maxdiff])
        elif datatype == 'multinom':
            Wx0, Mux0, xi0, alpha0, b = theta0
            Wxp, Muxp, xip, alphap, b = thetap
            maxdiff = np.max([np.max(np.abs(Wxp - Wx0)), maxdiff])
        else:
            print('unknown datatype')
            return
    #if dataparamt0 == 0:
    if isinstance(dataparamt0, bool):
        print('check_convergence: no time leg')
    else:
        if not CEN:
            [tE, (datatype, thetat0, sparse)] = dataparamt0
            #thetat0 = dataparamt0[1][1]
            thetatp = dataparamtp[1][1]
            wt0, Lam0 = thetat0
            wtp, Lamp = thetatp
            maxdiff = np.max([np.max(np.abs(wtp - wt0)), maxdiff])
        elif CEN:
            [[tE, (datatypet, thetat0, sparset)],
             [Delta, (datatypec, thetac0, sparsec)]] = dataparamt0
            #thetat0 = dataparamt0[0][1][1]
            thetatp = dataparamtp[0][1][1]
            #thetac0 = dataparamt0[1][1][1]
            thetacp = dataparamtp[1][1][1]
            wt0, Lam0 = thetat0
            wtp, Lamp = thetatp
            if datatypec == 'cind':
                #Lamcp = thetac0
                maxdiff = np.max([np.max(np.abs(wtp - wt0)), maxdiff])
            elif datatypec == 'cindIC':
                wtc0, Lamc0 = thetac0
                wtcp, Lamcp = thetacp
                maxdiff = np.max([np.max(np.abs(wtp - wt0)), np.max(np.abs(
                    wtcp - wtc0)), maxdiff])
    return maxdiff


def EM_all(dataparamX, dataparamt, dz, niter, convgap, mh_params, step, CEN,
           INIT, plotloc):
    #initialize
    if INIT:
        (dataparamX0, dataparamt0, Xmeans) = EM_initialize(dataparamX,
                                                           dataparamt, dz, CEN)
    else:
        dataparamX0, dataparamt0 = dataparamX, dataparamt
    del dataparamX, dataparamt
    nsamp = dataparamX0[0][0].shape[1]
    q = np.empty((niter, 2))
    Ldata = np.empty((niter, 2))
    for i in range(0, niter):
        #E - step
        #if dataparamt0 == 0:
        if isinstance(dataparamt0, bool):
            print('E - step: no time leg, estep is analytical ' +
                  '(under variational approximation)')
            EZ, EZZtn = E_step_no_surv(dataparamX0, dz)
            EexpwtZ, EexpwtcZ = False, False
            ## recording parameters
            qq = jointlogprob(dataparamX0, dataparamt0, EZ, EZZtn, EexpwtZ,
                              EexpwtcZ, CEN)
            q[i, :] = np.array([qq, i])
            Ldata[i, :] = np.array([loss_function_no_surv(
                dataparamX0, EZ, EZZtn), i])
            sigma2=False
        #print('loss function diff', loss_function_no_surv(dataparamX0,
        #EZ, EZZtn) - loss_function_no_surv_alt(dataparamX0))
        #print('here')
        else:
            ##creating samples for empirical expectations.
            #Zrep = MH_sim_Z_given_data((dz, dataparamX0, dataparamt0,
            #nrep, CEN, burn = 1000))
            batch_number, nrep_batch, burn, MULTIPRO, PRE_SEED = mh_params
            nrep = batch_number * nrep_batch
            Zrep = np.empty((nrep, dz, nsamp))
            if not os.path.exists(plotloc + 'mh_seeds/'):
                os.makedirs(plotloc + 'mh_seeds/')
            if not PRE_SEED:
                if sys.platform == "linux" or sys.platform == "linux2":
                    large = 4294967295
                    #large = 100000000000
                elif sys.platform == "darwin":
                    large = 4294967295
                SEED = np.random.randint(
                    0,
                    large,
                    size=(nsamp * batch_number))
                assert np.unique(SEED).shape == SEED.shape
                f = open(plotloc + 'mh_seeds/'+ 'seed_' + str(i) + '.py', 'w')
                pickle.dump(SEED, f)
                f.close()
            else:
                f = open(plotloc +  'mh_seeds/'+ 'seed_' + str(i) + '.py', 'r')
                SEED = pickle.load(f)
                f.close()
            if i==0:
                sigma2= pick_scale(dz, dataparamX0, dataparamt0, nrep_batch, CEN, burn, plotloc + 'mh_seeds/')
                #sigma2=1
            # for dn in range(nsamp):
            #     Zrepj = MH_sim_Z_given_data_p_error_wrapper((dn, dz, dataparamX0, dataparamt0,
            #                                  nrep_batch, CEN, burn, SEED[dn * batch_number], sigma2))
            #     print('no problems', dn)
            # return
            if MULTIPRO==1:
                for dn in range(nsamp):
                    for j in range(batch_number):
                        Zrep[(nrep_batch * j):(nrep_batch * j + nrep_batch), :, dn] = MH_sim_Z_given_data_p_error_wrapper((dn, dz, dataparamX0, dataparamt0,
                                                     nrep_batch, CEN, burn, SEED[dn * batch_number + j], sigma2))
            elif MULTIPRO>1:
                numThreads = MULTIPRO
                pool = multiprocessing.Pool(numThreads)
                # results = pool.map_async(tester, [(x) for x in range(nsamp)])
                # pool.close()
                # pool.join()
                # results = results.get()
                # print(results)
                # return
                results = pool.map_async(MH_sim_Z_given_data_p_error_wrapper,
                                         [(dn, dz, dataparamX0, dataparamt0,
                                           nrep_batch, CEN, burn,
                                           SEED[dn * batch_number + j], sigma2)
                                          for dn in range(nsamp)                
                                          for j in range(batch_number)])
                # set the pool to work
                pool.close()
                # party's over, kids
                pool.join()
                # wait for all tasks to finish
                results = results.get()
                for dn in range(nsamp):
                    for j in range(batch_number):
                        Zrep[(nrep_batch * j):(nrep_batch * j + nrep_batch), :,
                             dn] = (results[dn * batch_number + j])
            assert np.isnan(np.sum(Zrep)) == False 
            ## simulating marginal likelihood of data (up to a constant)
            ##need save the seed to make reproducable.
            # if sys.platform == "linux" or sys.platform == "linux2":
            #     large = 4294967295
            #     #large = 100000000000
            # elif sys.platform == "darwin":
            #     large = 4294967295
            # SEED2 = np.random.randint(0, large, size=nsamp)
            # log_marg_data = np.empty(nsamp)
            # for dn in range(nsamp):
            #     log_marg_data[dn] = mh_sim_log_marginal_data((
            #         dn, dz, dataparamX0, dataparamt0, Zrep, CEN, SEED2[dn]))
            ##calculating emperical expectations
            EZ, EZZtn = E_step_MH(Zrep)
            if not CEN:
                [tE, (datatype, thetat0, sparse)] = dataparamt0
                wt0, Lam0 = thetat0
                (EexpwtZ, EZexpwtZ, EZZtexpwtZ) = E_step_MH_Cox(Zrep, wt0)
                EexpwtcZ = False
            elif CEN:
                [[tE, (datatypet, thetat0, sparset)],
                 [Delta, (datatypec, thetac0, sparsec)]] = dataparamt0
                wt0, Lam0 = thetat0
                (EexpwtZ, EZexpwtZ, EZZtexpwtZ) = E_step_MH_Cox(Zrep, wt0)
                if datatypec == 'cind':
                    Lamc0 = thetac0
                    EexpwtcZ = False
                elif datatypec == 'cindIC':
                    wtc0, Lamc0 = thetac0
                    (EexpwtcZ, EZexpwtcZ, EZZtexpwtcZ) = E_step_MH_Cox(Zrep,
                                                                       wtc0)
                    ## recording parameters
            qq = jointlogprob(dataparamX0, dataparamt0, EZ, EZZtn, EexpwtZ,
                              EexpwtcZ, CEN)
            q[i, :] = np.array([qq, i])
            # Ldata[i, :] = np.array([np.sum(log_marg_data), i])
            #M - step
        dataparamXp = []
        for k, datapara in enumerate(dataparamX0):
            [X, (datatype, theta0, sparse)] = datapara
            if datatype == 'normal':
                Wx0, Mux0, Psix0 = theta0
                (Wxp, Psixp) = M_step_Norm(Wx0, Psix0, X, EZ, EZZtn, sparse)
                dataparamXp.append([X, (datatype, (Wxp, Mux0, Psixp), sparse)])
                ## no update for mean
            elif datatype == 'binom':
                Wx0, Mux0, xi0, b = theta0
                alpha0 = np.zeros((1, nsamp))
                (Wxp, Muxp, xip, alphap) = M_step_BIN(
                    Wx0, Mux0, xi0, alpha0, b, X, EZ, EZZtn, sparse, datatype)
                ##no update for alpha0
                dataparamXp.append([X, (datatype, (Wxp, Muxp, xip, b), sparse)
                                    ])
            elif datatype == 'multinom':
                Wx0, Mux0, xi0, alpha0, b = theta0
                (Wxp, Muxp, xip, alphap) = M_step_BIN(
                    Wx0, Mux0, xi0, alpha0, b, X, EZ, EZZtn, sparse, datatype)
                dataparamXp.append([X, (datatype, (Wxp, Muxp, xip, alphap, b),
                                        sparse)])
            else:
                print('unknown datatype')
        #if dataparamt0 == 0:
        if isinstance(dataparamt0, bool):
            print('Mstep: no time leg')
            print('in m step', isinstance(dataparamt0, bool))
            dataparamtp = copy.deepcopy(dataparamt0)
            print('in m step', isinstance(dataparamtp, bool))
            EexpwtpZ, EexpwtcpZ, EexpwtZ, EexpwtcZ = False, False, False, False
        else:
            if not CEN:
                [tE, (datatype, thetat0, sparse)] = dataparamt0
                #wt0, Lam0 = thetat0
                ##do update
                wtp, Lamp = M_step_NR(wt0, Lam0, tE, np.ones((1, nsamp)), EZ,
                                      EexpwtZ, EZexpwtZ, EZZtexpwtZ, step)
                thetatp = (wtp, Lamp)
                dataparamtp = [tE, (datatype, thetatp, sparse)]
                (EexpwtpZ, EZexpwtpZ, EZZtexpwtpZ) = E_step_MH_Cox(Zrep, wtp)
                EexpwtcpZ = False
            elif CEN:
                #[[tE, (datatypet, thetat0, sparset)],
                # [Delta, (datatypec, thetac0, sparsec)]] = dataparamt0
                #wt0, Lam0 = thetat0
                #print(wt0, Lam0, tE, Delta)
                (wtp, Lamp) = M_step_NR(wt0, Lam0, tE, Delta, EZ, EexpwtZ,
                                        EZexpwtZ, EZZtexpwtZ, step)
                thetatp = (wtp, Lamp)
                (EexpwtpZ, EZexpwtpZ, EZZtexpwtpZ) = E_step_MH_Cox(Zrep, wtp)
                Deltac = np.ones(Delta.shape) - Delta
                if datatypec == 'cind':
                    #Lamcp = thetac0
                    ## there are no updates past initialization for this case.
                    dataparamtp = [[tE, (datatypet, thetatp, sparset)],
                                   [Delta, (datatypec, thetac0, sparsec)]]
                    EexpwtcpZ = False
                elif datatypec == 'cindIC':
                    #wtc0, Lamc0 = thetac0
                    (wtcp, Lamcp) = M_step_NR(wtc0, Lamc0, tE, Deltac, EZ,
                                              EexpwtcZ, EZexpwtcZ, EZZtexpwtcZ,
                                              step)
                    thetacp = (wtcp, Lamcp)
                    dataparamtp = [[tE, (datatypet, thetatp, sparset)],
                                   [Delta, (datatypec, thetacp, sparsec)]]
                    (EexpwtcpZ, EZexpwtcpZ, EZZtexpwtcpZ) = E_step_MH_Cox(Zrep,
                                                                          wtcp)
        ### checking that Q increases
        qqp = jointlogprob(dataparamXp, dataparamtp, EZ, EZZtn, EexpwtpZ,
                           EexpwtcpZ, CEN)
        if (qqp < qq):
            print("didn't maximize! trying smaller step sizes Newton -" +
                  " Raphson updates on the time arms.")
            #print("Psis", scipy.linalg.svd(Psixp)[1], scipy.linalg.svd(Psix0)[1])
            #print(" - Lam * np.exp(wt.dot(Z)).dot(tE.T)",
            #  - Lamp * EexpwtZ.dot(tE.T), - Lam0 * EexpwtZ.dot(tE.T))
            for j in range(10, 110, 10):
                print(j)
                if isinstance(dataparamt0, bool):
                #if dataparamt0 == 0:
                    print('Mstep: no time leg')
                    dataparamtp = dataparamt0
                    print('in m step', isinstance(dataparamtp, bool))
                else:
                    stepj = step / j
                    if not CEN:
                        [tE, (datatype, thetat0, sparse)] = dataparamt0
                        #wt0, Lam0 = thetat0
                        wtp, Lamp = M_step_NR(wt0, Lam0, tE,
                                              np.ones((1, nsamp)), EZ, EexpwtZ,
                                              EZexpwtZ, EZZtexpwtZ, stepj)
                        thetatp = (wtp, Lamp)
                        dataparamtp = [tE, (datatype, thetatp, sparse)]
                        (EexpwtpZ, EZexpwtpZ, EZZtexpwtpZ) = E_step_MH_Cox(
                            Zrep, wtp)
                        EexpwtcpZ = False
                    elif CEN:
                        [[tE, (datatypet, thetat0, sparset)],
                         [Delta, (datatypec, thetac0, sparsec)]] = dataparamt0
                        print('sumdelta', np.sum(Delta))
                        #wt0, Lam0 = thetat0
                        (wtp, Lamp) = M_step_NR(wt0, Lam0, tE, Delta, EZ,
                                                EexpwtZ, EZexpwtZ, EZZtexpwtZ,
                                                stepj)
                        thetatp = (wtp, Lamp)
                        (EexpwtpZ, EZexpwtpZ, EZZtexpwtpZ) = E_step_MH_Cox(
                            Zrep, wtp)
                        #Deltac = np.ones(Delta.shape) - Delta
                        if datatypec == 'cind':
                            #Lamcp = thetac0
                            ## there are no updates past initialization for this case.
                            dataparamtp = [[tE, (datatypet, thetatp, sparset)],
                                           [Delta, (datatypec, thetac0,
                                                    sparsec)]]
                            EexpwtcpZ = False
                        elif datatypec == 'cindIC':
                            #wtc0, Lamc0 = thetac0
                            (wtcp, Lamcp) = M_step_NR(wtc0, Lamc0, tE, Deltac,
                                                      EZ, EexpwtcZ, EZexpwtcZ,
                                                      EZZtexpwtcZ, stepj)
                            thetacp = (wtcp, Lamcp)
                            dataparamtp = [[tE, (datatypet, thetatp, sparset)],
                                           [Delta, (datatypec, thetacp,
                                                    sparsec)]]
                            (EexpwtcpZ, EZexpwtcpZ,
                             EZZtexpwtcpZ) = E_step_MH_Cox(Zrep, wtcp)
                qqp = jointlogprob(dataparamXp, dataparamtp, EZ, EZZtn,
                                   EexpwtpZ, EexpwtcpZ, CEN)
                if (qqp > qq):
                    print('yay maximized')
                    break
                else:
                    continue
            if (qqp < qq):
                print('did not manage to maximize')
                return (dataparamXp, dataparamtp, dataparamX0, dataparamt0, EZ,
                        EZZtn, EexpwtZ, EexpwtcZ, EexpwtpZ, EexpwtcpZ, q, Ldata, i)
        if np.isnan(qqp):
            return (dataparamXp, dataparamtp, dataparamX0, dataparamt0, EZ,
                    EZZtn, EexpwtZ, EexpwtcZ, q, Ldata, i)
        maxdiff = check_convergence(dataparamXp, dataparamtp, dataparamX0,
                                    dataparamt0, CEN)
        print(maxdiff)
        if maxdiff < convgap:
            print('resassigning and ending, abs(diff) less than convgap',
                  convgap, 'iter', i)
            dataparamX0 = np.copy(dataparamXp)
            dataparamt0 = np.copy(dataparamtp)
            q = q[0:i + 1]
            Ldata = Ldata[0:i + 1]
            break
        else:
            dataparamX0 = np.copy(dataparamXp)
            if isinstance(dataparamtp, bool):
                dataparamt0 = copy.deepcopy(dataparamtp)
            else:
                dataparamt0 = np.copy(dataparamtp)
    ##one more e-step to save E[z|x, t]
    ##
    if not os.path.exists(plotloc + 'expected_values/'):
        os.makedirs(plotloc + 'expected_values/')
    EZ_last = e_step(dataparamX0, dataparamt0, dz, nsamp, mh_params, CEN, plotloc + 'expected_values/', sigma2)
    #expsurvtime() saves E[z|x], and E[t|x]
    if isinstance(dataparamt0, bool):
    #if dataparamt0 == 0:
        print('no time prediction')
    else:
        (texp, var) = expsurvtime(dataparamX0, dataparamt0, plotloc + 'expected_values/')
    if not os.path.exists(plotloc + 'learned_parameters/'):
        os.makedirs(plotloc + 'learned_parameters/')
    save_params(dataparamX0, dataparamt0, Xmeans, dz, niter, mh_params, Ldata, CEN, plotloc + 'learned_parameters/')
    return (dataparamX0, dataparamt0, Xmeans, q, Ldata, EZ_last)


## could do more encapsulation!
def e_step(dataparamX0, dataparamt0, dz, nsamp, mh_params, CEN, plotloc, sigma2):
    """
    this performs the e-step and saves E[Z|x] for no time and E[Z|x, t, Delta] with time.
    """
    #if dataparamt0 == 0:
    if isinstance(dataparamt0, bool):
        print('E - step: no time leg, estep is analytical ' +
              '(under variational approximation)')
        EZ, EZZtn = E_step_no_surv(dataparamX0, dz)
        # f = open(plotloc + 'EZ_given_x.py', 'w')
        # pickle.dump(EZ, f)
        # f.close()
        pd.DataFrame(EZ).to_csv(plotloc + 'EZ_given_x.csv')
    else:
        ##creating samples for empirical expectations.
        #Zrep = MH_sim_Z_given_data((dz, dataparamX0, dataparamt0,
        #nrep, CEN, burn = 1000))
        batch_number, nrep_batch, burn, MULTIPRO, PRE_SEED = mh_params
        nrep = batch_number * nrep_batch
        Zrep = np.empty((nrep, dz, nsamp))
        if not PRE_SEED:
            if sys.platform == "linux" or sys.platform == "linux2":
                large = 4294967295
                #large = 100000000000
            elif sys.platform == "darwin":
                large = 4294967295
            SEED = np.random.randint(
                0,
                large,
                size=(nsamp * batch_number))
            assert np.unique(SEED).shape == SEED.shape
            f = open(plotloc + 'seed_EZ_given_x_t_Delta.py', 'w')
            pickle.dump(SEED, f)
            f.close()
        else:
            f = open(plotloc + 'seed_EZ_given_x_t_Delta.py', 'r')
            SEED = pickle.load(f)
            f.close()
        #sigma2=1
        if MULTIPRO == 1:
            for dn in range(nsamp):
                for j in range(batch_number):
                    Zrep[(nrep_batch * j):(nrep_batch * j + nrep_batch), :, dn] = MH_sim_Z_given_data_p_error_wrapper((dn, dz, dataparamX0, dataparamt0,
                                                     nrep_batch, CEN, burn, SEED[dn * batch_number + j], sigma2))
        elif MULTIPRO> 1:
            numThreads = MULTIPRO
            pool = multiprocessing.Pool(numThreads)
            results = pool.map_async(MH_sim_Z_given_data_p_error_wrapper,
                                     [(dn, dz, dataparamX0, dataparamt0,
                                       nrep_batch, CEN, burn,
                                       SEED[dn * batch_number + j], sigma2)
                                      for dn in range(nsamp)
                                      for j in range(batch_number)])
            # set the pool to work
            pool.close()
            # party's over, kids
            pool.join()
            # wait for all tasks to finish
            results = results.get()
            for dn in range(nsamp):
                for j in range(batch_number):
                    Zrep[(nrep_batch * j):(nrep_batch * j + nrep_batch), :,
                         dn] = (results[dn * batch_number + j])
        assert np.isnan(np.sum(Zrep)) == False
        ##calculating emperical expectations
        EZ, EZZtn = E_step_MH(Zrep)
        # f = open(plotloc + 'EZ_given_x_t_Delta.py', 'w')
        # pickle.dump(EZ, f)
        # f.close()
        pd.DataFrame(EZ).to_csv(plotloc + 'EZ_given_x_t_Delta.csv')
    return(EZ)


def save_params(dataparamX0, dataparamt0, Xmeans, dz, niter, mh_params, Ldata, CEN, plotloc):
    ##consider instead http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.savetxt.html#numpy.savetxt
    ## numpy.savetxt ?
    with open(plotloc + 'EM_log.txt', 'w') as g:
        g.write('time of run' + ',' + str(time()) + '\n')
        g.write('plotloc' + ',' + str(plotloc) + '\n')
        g.write('dz' + ',' + str(dz) + '\n')
        g.write('number of iterations' + ',' + str(niter) + '\n')
        g.write('metropolis hastings parameters' + ',' + str(mh_params) + '\n')
        g.write('likelihood of data' + ',' + str(Ldata) + '\n' )
        for k, datapara in enumerate(dataparamX0):
            [X, (datatype, theta, sparse)] = datapara
            g.write(str(k) + ',' + str(X.shape) + ',' + datatype + ',' + str(sparse) + '\n')
            # with open(plotloc + 'sparse_' + str(k) + '.py', 'w') as f:
            #     pickle.dump(sparse, f)
            pd.DataFrame(sparse).to_csv(plotloc + 'sparse_' + str(k) + '.csv')
            if datatype == 'normal':
                Wx, Mux, Psix = theta
                if sum(Mux) == 0:
                    Mux = Xmeans[k]
                theta = (Wx, Mux, Psix)
                vnames = ['Wx', 'Mux', 'Psix']
                for j, item in enumerate(theta):
                    # with open(plotloc + vnames[j] + '_' + str(k) + '.py', 'w') as f:
                    #     pickle.dump(item, f)
                    pd.DataFrame(item).to_csv(plotloc + vnames[j] + '_' + str(k) + '.csv')
                g.write(str(k) + ',' + 'num Wx elements =0, ' + str(np.sum(theta[0] == 0)) + '\n')
            elif datatype == 'binom':
                vnames = ['Wx', 'Mux', 'xi', 'b']
                for j, item in enumerate(theta):
                    # with open(plotloc + vnames[j] + '_' + str(k) + '.py', 'w') as f:
                    #     pickle.dump(item, f)
                    if type(item) != np.ndarray:
                        item = [item]
                    pd.DataFrame(item).to_csv(plotloc + vnames[j] + '_' + str(k) + '.csv')
                g.write(str(k) + ',' + 'num Wx elements =0, ' + str(np.sum(theta[0] == 0)) + '\n')   
            elif datatype == 'multinom':
                vnames = ['Wx', 'Mux', 'xi', 'alpha', 'b']
                for j, item in enumerate(theta):
                    # with open(plotloc + vnames[j] + '_' + str(k) + '.py', 'w') as f:
                    #     pickle.dump(item, f)
                    if type(item) != np.ndarray:
                        item = [item]
                    pd.DataFrame(item).to_csv(plotloc + vnames[j] + '_' + str(k) + '.csv')
                g.write(str(k) + ',' + 'num Wx elements =0, ' + str(np.sum(theta[0] == 0)) + '\n')
            else:
                print('unknown datatype')
        if isinstance(dataparamt0, bool):
        #if dataparamt0 == 0:
            g.write('no time data' + '\n')
        else:
            if not CEN:
                g.write('time data, no censoring' + '\n')
                [tE, (datatype, thetat, sparse)] = dataparamt0
                vnames = ['wt', 'Lam']
                for j, item in enumerate(thetat):
                    # with open(plotloc + vnames[j] + '.py', 'w') as f:
                    #     pickle.dump(item, f)
                    pd.DataFrame(item).to_csv(plotloc + vnames[j] + '.csv') 
            elif CEN:
                [[tE, (datatypet, thetat, sparset)],
                 [Delta, (datatypec, thetac, sparsec)]] = dataparamt0
                vnames = ['wt', 'Lam']
                for j, item in enumerate(thetat):
                    # with open(plotloc + vnames[j] + '.py', 'w') as f:
                    #     pickle.dump(item, f)
                    pd.DataFrame(item).to_csv(plotloc + vnames[j] + '.csv') 
                if datatypec == 'cind':
                    g.write('time data, with non-informative censoring' + '\n')
                    vnames = ['Lamc']
                    for j, item in enumerate([thetac]):
                        # with open(plotloc + vnames[j] + '.py', 'w') as f:
                        #     pickle.dump(item, f)
                        pd.DataFrame([item]).to_csv(plotloc + vnames[j] + '.csv')
                elif datatypec == 'cindIC':
                    g.write('time data, with informative censoring' + '\n')
                    vnames = ['wtc', 'Lamc']
                    for j, item in enumerate(thetac):
                        # with open(plotloc + vnames[j] + '.py', 'w') as f:
                        #     pickle.dump(item, f)
                        pd.DataFrame(item).to_csv(plotloc + vnames[j] + '.csv')


##expected error
def dataswitch(dataparamX_test, dataparamt_test, CEN_test, dataparamX0,
               dataparamt0, Xmeans, CEN0):
    ##combines X and t, Delta from (dataparamX_test, data paramt_test)
    #with parameters from (dataparamX0, dataparamt0).
    dataparamXp_new = []
    for i, datapara in enumerate(dataparamX0):
        X = dataparamX_test[i][0]
        dx, nsamptest = X.shape
        if datapara[1][0] == 'normal':
            paramX = (datapara[1][0], (datapara[1][1][0], Xmeans[i],
                                       datapara[1][1][2]), datapara[1][2])
            ## being careful about means!
        if datapara[1][0] == 'binom':
            ###taking the average learned xi for all n in the test set
            paramX = (datapara[1][0],
                      (datapara[1][1][0], datapara[1][1][1],
                       np.array([np.mean(datapara[1][1][2],
                                         axis=1), ] * nsamptest).T,
                       datapara[1][1][3]), datapara[1][2])
        if datapara[1][0] == 'multinom':
            ###taking the average learned xi_i for all n in the test set.
            #same for alpha_n
            paramX = (datapara[1][0],
                      (datapara[1][1][0], datapara[1][1][1], np.array(
                          [np.mean(datapara[1][1][2],
                                   axis=1), ] * nsamptest).T,
                       np.mean(datapara[1][1][3]) * np.ones((1, nsamptest)),
                       datapara[1][1][4]), datapara[1][2])
        dataparamXp_new.append([X, paramX])
    if isinstance(dataparamt0, bool):
    #if dataparamt0 == 0:
        print('dataswitch: no time leg')
        dataparamtp_new = False
    else:
        if CEN0 == CEN_test:
            if not CEN0:
                paramt = dataparamt0[1]
                tE = dataparamt_test[0]
                dataparamtp_new = [tE, paramt]
            elif CEN0:
                paramtt = dataparamt0[0][1]
                paramtc = dataparamt0[1][1]
                tE = dataparamt_test[0][0]
                Delta = dataparamt_test[1][0]
                dataparamtp_new = [[tE, paramtt], [Delta, paramtc]]
        elif CEN0 != CEN_test:
            if not CEN_test:
                paramt = dataparamt0[0][1]
                tE = dataparamt_test[0]
                dataparamtp_new = [tE, paramt]
            elif CEN_test:
                paramtt = dataparamt0[1]
                paramtc = (False,
                           'this parameter was not present in dataparamt0')
                tE = dataparamt_test[0][0]
                Delta = dataparamt_test[1][0]
                dataparamtp_new = [[tE, paramtt], [Delta, paramtc]]
    return (dataparamXp_new, dataparamtp_new)


##expected survival time given x and parameters
def expsurvtime(dataparamX, dataparamt, PLOT):
    """
    returns expected survival time if dataparamt!=0.  otherwise returns likelihood of data.  sorry about the hack.
    """
    nsamp = dataparamX[0][0].shape[1]
    dz = dataparamX[0][1][1][0].shape[1]
    #if dataparamt != 0:
    if not isinstance(dataparamt, bool):
        ## wt, Lam = dataparamt[1][1] is correct in simulations because
        ## there is no censoring in the simulated test data
        #wt, Lam = dataparamt[1][1]
        if len(dataparamt[0]) == 2:
            tE = dataparamt[0][0]
            Delta = dataparamt[1][0]
            wt, Lam = dataparamt[0][1][1]
            print(wt, Lam)
        elif len(dataparamt[0]) == 1:
            tE = dataparamt[0]
            Delta = np.ones(tE.shape)
            wt, Lam = dataparamt[1][1]
            print(wt, Lam)
        texp = np.empty((1, nsamp))
        EZ, EZZtn = E_step_no_surv(dataparamX, dz)
        ## this calcuates the covariance matrix for Z and (mean of Z + wt) for
        ## when Z is integrated out of p(x, z, t) = p(t| z)p(x|z) p(z)  with
        ## the X_test data and the Xp parameters.
        for dn in range(0, nsamp):
            Cn = EZZtn[dn, :, :] - np.outer(EZ[:, dn], EZ[:, dn])
            ##Cn is the covariance matrix, EZ[:, dn] is the mean.
            ## remember EZ has Cn in it.
            #print(Cn)
            texp[:, dn] = 1 / Lam * np.exp(1 / 2 * wt.dot(Cn).dot(wt.T) -
                                           EZ[:, dn].T.dot(wt.T))
            ## no numerical part because that canceled out in the integration.
        var = False
        #think about variance later.
        #print(EZ.T)
        if PLOT != False:
            if len(PLOT) == 2:
                (plotloc, group) = PLOT
                x = EZ[0, :]
                y = EZ[1, :]
                fig, ax = plt.subplots()
                ax.scatter(x, y, c=group, s=250, edgecolors='black')
                ax.set_xlabel(r'$E[z_1| x]$', fontsize=20)
                ax.set_ylabel(r'$E[z_2| x]$', fontsize=20)
                # ax.set_title('Volume and percent change')
                #ax.legend()
                fig.tight_layout()
                plt.savefig(plotloc + 'EZ_given_x.pdf',
                            bbox_inches='tight')
                # f = open(plotloc + 'EZ_given_x.py', 'w')
                # pickle.dump(EZ, f)
                # f.close()
                pd.DataFrame(EZ).to_csv(plotloc + 'EZ_given_x.csv')
                # f = open(
                #     plotloc + 'Et_given_x.py',
                #     'w')
                # pickle.dump(texp, f)
                # f.close()
                pd.DataFrame(texp).to_csv(plotloc + 'Et_given_x.csv')
            else:
                plotloc = PLOT
                x = EZ[0, :]
                y = EZ[1, :]
                fig, ax = plt.subplots()
                cm = plt.cm.get_cmap('Greys')
                markers = ['o', 'v']
                for k, m in enumerate(markers):
                    i= (m=='o')*(Delta==1) +(m=='v')*(Delta==0)
                    i=i[0, :]
                    im=ax.scatter(x[i], y[i], c=tE[0, i], cmap=cm, marker=m, s=100, vmin= np.min(tE), vmax = np.max(tE), edgecolors='black')
                    if k == 0:
                        fig.colorbar(im, ax=ax)
                ax.set_xlabel(r'$E[z_1| x]$', fontsize=20)
                ax.set_ylabel(r'$E[z_2| x]$', fontsize=20)
                fig.tight_layout()
                plt.savefig(plotloc + 'EZ_given_x.pdf',
                            bbox_inches='tight')
                pd.DataFrame(EZ).to_csv(plotloc + 'EZ_given_x.csv')
                # f = open(
                #     plotloc + 'Et_given_x.py',
                #     'w')
                # pickle.dump(texp, f)
                # f.close()
                pd.DataFrame(texp).to_csv(plotloc + 'Et_given_x.csv') 
    else:
        ##this bit is for just FA.  insteaod of returning a texp, it returns the loss function of the data.
        EZ, EZZtn = E_step_no_surv(dataparamX, dz)
        ## this calcuates the covariance matrix for Z and (mean of Z + wt) for
        ## when Z is integrated out of p(x, z, t) = p(t| z)p(x|z) p(z)  with
        ## the X_test data and the Xp parameters.
        ## calculates p(x)
        loss_function = loss_function_no_surv(dataparamX, EZ, EZZtn)
        texp=loss_function
        var=False
        if PLOT!=False:
            plotloc = PLOT
            x = EZ[0, :]
            y = EZ[1, :]
            fig, ax = plt.subplots()
            im=ax.scatter(x, y, c='gray', s=100, edgecolors='black')
            ax.set_xlabel(r'$E[z_1| x]$', fontsize=20)
            ax.set_ylabel(r'$E[z_2| x]$', fontsize=20)
            fig.tight_layout()
            plt.savefig(plotloc + 'EZ_given_x.pdf',
                        bbox_inches='tight')
            pd.DataFrame(EZ).to_csv(plotloc + 'EZ_given_x.csv')
    return (texp, var)


##simulating ln(p(t|Z)) for use in f(t)
def lnprobtgivenZ(tE, paramt, Z):
    wt, Lam = paramt[1]
    lnprobdatagivenZ_t = (
        np.log(Lam) + wt.dot(Z) - Lam * np.exp(wt.dot(Z)) * tE)
    return lnprobdatagivenZ_t


##simulating ln f(t)
def sim_lnprobt(tE, paramt, Zrep):
    nrep = Zrep.shape[0]
    #av = np.average([lnprobtgivenZ(tE, paramt, Zrep[x, :, :]) for x in range(0, nrep)])
    #print(av)
    #sim_probdata = np.average(np.exp([lnprobtgivenZ(tE, paramt, Zrep[x, :, :]) - av for x in range(0, nrep)]))
    sim_probdata = np.average(np.exp([lnprobtgivenZ(tE, paramt, Zrep[x, :, :])
                                      for x in range(0, nrep)]))
    print(sim_probdata)
    #sim_lnprobdata = av + np.log(sim_probdata)
    sim_lnprobdata = np.log(sim_probdata)
    return sim_lnprobdata


###simulating f(t)
def sim_probt(paramt, dz, nrep, tmax, scale):
    t = range(0, tmax)
    t = np.array([x / scale for x in t])
    Zrep_p, discard = sim_Z(dz, 1, nrep)
    simpt = [np.exp(sim_lnprobt(tE, paramt, Zrep_p)) for tE in t]
    return (simpt, t)


def assign_group(Z, ngroup):
    dz, nsamp = Z.shape
    group = np.empty((1, nsamp))
    if dz != 2:
        print("don't know how to assign group!")
        return
    else:
        angle = 360 / ngroup
        for dn in range(nsamp):
            angle_p = np.arctan2(Z.iloc[1, dn], Z.iloc[0, dn]) * 180 / np.pi
            if angle_p < 0:
                angle_p = 360 + angle_p
            for nangle in range(ngroup):
                if nangle * angle < angle_p and angle_p <= (
                        nangle + 1) * angle:
                    group[:, dn] = nangle
    group=pd.DataFrame(group, columns=Z.columns)
    return group



##changed so you put in dataparamX already having learned parameters.
## cindex_icluster
def predict_cindex(dataparamX_test, dataparamt_test, CEN_test, dataparamX0, dataparamt0,
                    Xmeans, CEN0, plotloc):
    """
    this code takes in test data (dataparamX_test) and maximum likelihood parameters
    (the parameters in dataparamX0), predicts the expected survival time using the
    maximum likelhood parameters, and calculates the c-index on the test data and test
    predictions
    """
    dataparamXp, dataparamtp = dataswitch(dataparamX_test, dataparamt_test,
                                          CEN_test, dataparamX0, dataparamt0,
                                          Xmeans, CEN0)
    #switched in the test data into dataparamXp, dataparamtp, the learned parameters
    print('train', dataparamt0)
    print('test', dataparamt_test)
    print('should be test t with training param', dataparamtp)
    ##predicting test - set survival time from learned parameters
    (est_test, var_test) = expsurvtime(dataparamXp, dataparamtp, plotloc)
                                                ##plotting of E[z|x] enabled
    #if dataparamt0 != 0:
    if not isinstance(dataparamt0, bool):
        estdelta_test = np.ones(est_test.shape)
        ##no censoring on prediction using parameters from training data.
        ## changed to handle censoring on test data.
        if CEN_test == False:
            tE_test = dataparamt_test[0]
            Delta_test = np.ones(tE_test.shape)
        if CEN_test == True:
            tE_test = dataparamt_test[0][0]
            Delta_test = dataparamt_test[1][0]
        print(tE_test, Delta_test, est_test, estdelta_test)
        #c_test = cindex_code.cindex(tE_test, Delta_test, est_test, estdelta_test)
        c_test = cindex_code.cindex_ties(tE_test, Delta_test, est_test)[0]
    else:
        ##this is the case for no survival data.  instead of cindex, returns loss function of data
        c_test= est_test
    return c_test

### model_selection_cindex_icluster
def learn_predict_cindex(
        dataparamX_test, dataparamt_test, CEN_test, dataparamX, dataparamt, dz,
        niter, convgap, mh_params, step, CEN, INIT, plotloc):
    """

    this code takes in test data (dataparamX_test) and training data (dataparamX)
    learns the maximum likelihood parameters
    (the parameters in dataparamX0), predicts the expected survival time using the
    maximum likelhood parameters, and calculates the c-index on the test data and test
    predictions
    """
    ##learn parameters from training data
    if not os.path.exists(plotloc + 'learn/'):
        os.makedirs(plotloc + 'learn/')
    try:
        train = EM_all(dataparamX, dataparamt, dz, niter, convgap, mh_params, step,
                       CEN, INIT, plotloc + 'learn/')
        if len(train) == 6:
            print('here1')
            (dataparamX0, dataparamt0, Xmeans, q, Ldata, EZ) = train
            print('test', dataparamt_test)
            print('learned', dataparamt0)
        else:
            print('stopping now')
            return train
        print('here2')
        if not os.path.exists(plotloc + 'val/'):
            os.makedirs(plotloc + 'val/')
        c_test = predict_cindex(dataparamX_test, dataparamt_test, CEN_test, dataparamX0,
                                dataparamt0, Xmeans, CEN, plotloc + 'val/')
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


def model_selection_fixed_data(k, dataparamX_test, dataparamt_test, CEN_test,
                             dataparamX, dataparamt, dzrange, sparserange,
                    niter, convgap, mh_params, step, CEN, INIT, plotloc):
    f = open(plotloc + 'model_selection_output_' +str(k) +'.txt', 'w')
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
            cindex = learn_predict_cindex(
                dataparamX_test, dataparamt_test, CEN_test, dataparamXs, dataparamt,
                dz, niter, convgap, mh_params, step, CEN, INIT,
                plotloc + 'model_' + str(nmod) + '_' + str(m) + '/')
            print('dz, sparserun, cindex', dz, sparserun, cindex)
            f.write(str(dz) + ', ' + str(sparserun) + ', ' + str(cindex) + '\n')
            #answer[m, nmod, :] = [(dz, sparserun), cindex]
    f.close()


def load_data(data_directory, gold_standard):
    filenames = glob.glob(data_directory + '*.csv')
    data = []
    data_order = []
    c = 0
    for f in filenames:
        f_short = re.search(r'/[\w\.-]+.csv', f)
        if f_short:
            print f_short.group()[1:]
            f_short = f_short.group()[1:]
        else:
            print('something went wrong with regular expressions')
            return
        if f_short in ['data_guide.csv' , 'data_guide_gs.csv']:
            if gold_standard and f_short==  'data_guide_gs.csv':
                data_guide = pd.read_csv(f, sep=',')
            if not gold_standard and f_short== 'data_guide.csv':
                data_guide = pd.read_csv(f, sep=',')
        else:
            data_file = pd.read_csv(f, sep=',', index_col=0)
            print('data_file name and shape', f_short, data_file.shape)
            if c == 0:
                c = 1
                sample_name_ref = data_file.columns
            else:
                ## this is a check to make sure that sample names are in the correct order.
                if not (sample_name_ref == data_file.columns).all():
                    print('sample lists in ' + data_directory + ' do not match, ending load data')
                    return
            print('checking nans', np.sum(np.sum(np.isnan(data_file))))
            data.append(data_file)
            data_order.append(f_short)
    ## putting the data in the order of the dataguide.  keep survival data last.
    data_ordered = []
    for i in range(data_guide.shape[0]):
        #print("index", i, data_order.index(data_guide.loc[i, "file_location"]))
        data_ordered.append(data[data_order.index(data_guide.loc[i, "file_location"])])
    return(data_guide, data_ordered, sample_name_ref)


def make_splits(n_cv, analysis_directory, sample_name_ref):
    cv_splits = []
    if os.path.exists(analysis_directory + 'cv_' + str(n_cv) + '_samples_splits/'):
        print('cross validation sample split already exists for n_cv=', n_cv)
        filenames = sorted(glob.glob(
            analysis_directory + 'cv_'+ str(n_cv) + '_samples_splits/' + 'cv_sample_split_*.csv'))
        for f in filenames:
            cv_splits.append(pd.read_csv(f, delimiter=',', index_col = 0))
        assert len(cv_splits) == n_cv
        if os.path.isfile(analysis_directory + 'test_samples.csv'):
            print('test samples are already set aside')
            test_samples = pd.read_csv(analysis_directory + 'test_samples.csv', delimiter=',', index_col = 0)
        else:
            print('test samples file missing, reproducing from sample_name_ref and loaded cv_splits')
            for dn in range(n_cv):
                test_samples = sample_name_ref.drop(list(cv_splits[i].values[:, 0]))
    else:
        print('need to split samples into cross validation groups')
        os.makedirs(analysis_directory + 'cv_'+ str(n_cv) + '_samples_splits/')
        if os.path.isfile(analysis_directory + 'test_samples.csv'):
            print('test samples are already set aside')
            test_samples = pd.read_csv(analysis_directory + 'test_samples.csv', delimiter=',', index_col = 0)
            sample_name_trainval = sample_name_ref.drop(list(test_samples.values[:, 0]))
            nsamp_trainval = len(sample_name_trainval)
            print('size of test set, size of training + validation set',
                  len(test_samples), nsamp_trainval)
            sample_name_trainval = pd.DataFrame(random.sample(sample_name_trainval, nsamp_trainval))
        else:
            print('need to set aside test samples')
            total_samples = len(sample_name_ref)
            ## setting aside 25% of the data as a test set
            nsamp_test = total_samples // 4
            randomorder = pd.DataFrame(random.sample(sample_name_ref, total_samples))
            test_samples = randomorder[0:nsamp_test]
            test_samples.to_csv(analysis_directory + 'test_samples.csv', delimiter=',')
            sample_name_trainval = randomorder[nsamp_test:]
            nsamp_trainval = total_samples - nsamp_test
        ## now splitting remainder samples into cv_groups
        nsamp_cv = nsamp_trainval // n_cv
        nsamp_remainder = nsamp_trainval - nsamp_cv * n_cv
        for dn in range(n_cv):
            if dn < nsamp_remainder:
                cv_split = sample_name_trainval[dn * (nsamp_cv + 1): (dn + 1)*(nsamp_cv + 1)]
            elif dn >= nsamp_remainder:
                cv_split = (
                    sample_name_trainval[nsamp_remainder * (nsamp_cv + 1)
                    + (dn - nsamp_remainder) * (nsamp_cv):
                    nsamp_remainder * (nsamp_cv + 1) + (dn - nsamp_remainder + 1) * (nsamp_cv)])
            cv_split.to_csv(analysis_directory + 'cv_' + str(n_cv) + '_samples_splits/'
                           + 'cv_sample_split_' + str(dn) + '.csv', delimiter=',')
            cv_splits.append(cv_split)
        cv_set = set()
        for i in range(n_cv):
            cv_set = cv_set | set(cv_splits[i].values[:, 0])
        assert len(cv_set) == nsamp_trainval
    return(cv_splits, test_samples)

def make_dataparamXt(n_cv, cv_splits, test_samples, i, data_guide, data, analysis_directory):
    dataparamX = []
    dataparamt = []
    dataparamX_test = []
    dataparamt_test = []
    cv_test_samples = list(cv_splits[i].values[:, 0])
    for j in range(len(data)):
        datatype = data_guide.loc[j, 'datatype']
        ## this dz and sparse don't matter because they will be set instead in model_selection_fixed data
        dz = 2
        sparse = [False]
        if datatype != 'survival':
            b = data_guide.loc[j, 'b']
            cv_test_data = data[j].loc[:, cv_test_samples]
            print('test data shape, i , j', cv_test_data.shape, i, j)
            cv_train_data = data[j].drop(cv_test_samples, axis=1)
            print('dropped cv test samples from train data', cv_train_data.shape)
            cv_train_data = cv_train_data .drop(test_samples, axis=1)
            print('dropped uber test samples from train data', cv_train_data.shape)
            print('shape of dropped duplicates', cv_train_data.drop_duplicates().shape)
            print('number of features that are zero for all samples on training data',
                  np.sum(np.sum(cv_train_data, axis=1) == 0))
            if np.sum(np.sum(cv_train_data, axis=1) == 0) != 0:
                all_zero_features = cv_train_data[np.sum(cv_train_data, axis=1) == 0].index
                pd.DataFrame(all_zero_features).to_csv(analysis_directory + 'all_zero_features_data_' + str(j) + '.csv')
                cv_test_data = cv_test_data.drop(all_zero_features)
                print('dropped zero features test data', cv_test_data.shape)
                cv_train_data = cv_train_data.drop(all_zero_features)
                print('dropped zero features train data', cv_train_data.shape)
            dataparamX_test.append([np.array(cv_test_data), emptyparam(cv_test_data.shape[0], dz, cv_test_data.shape[1], datatype, b, sparse)])
            dataparamX.append([np.array(cv_train_data), emptyparam(cv_train_data.shape[0], dz, cv_train_data.shape[1], datatype, b, sparse)])
        else:
            CEN = data_guide.loc[j, 'CEN']
            if np.sum(data[j].loc[data_guide.loc[j, 'etime'], :]==0)!= 0:
                print("some samples have zero event time", np.sum(data[j].loc[data_guide.loc[j, 'etime'], :]==0))
                eps = np.min(data[j].loc[data_guide.loc[j, 'etime'], :][data[j].loc[data_guide.loc[j, 'etime'], :] != 0])/10
                print("replacing those zeros with 1/10 the next smallest event time")
                data[j].loc[data_guide.loc[j, 'etime'], :][data[j].loc[data_guide.loc[j, 'etime'], :] == 0] = eps
            if CEN == False:
                CEN_test = False
            cv_test_t = data[j].loc[data_guide.loc[j, 'etime'], cv_test_samples]
            cv_test_t = np.array(cv_test_t)
            cv_test_t = np.reshape(cv_test_t, (1, len(cv_test_t)))
            print(cv_test_t.shape)
            cv_train_t = data[j].loc[data_guide.loc[j, 'etime']].drop(cv_test_samples)
            cv_train_t = cv_train_t.drop(test_samples)
            cv_train_t = np.array(cv_train_t)
            cv_train_t = np.reshape(cv_train_t, (1, len(cv_train_t)))
            if CEN:
                datatypec = data_guide.loc[j, 'CENtype']
                cv_test_Delta = data[j].loc[data_guide.loc[j, 'Delta'], cv_test_samples]
                cv_test_Delta = np.array(cv_test_Delta)
                cv_test_Delta = np.reshape(cv_test_Delta, (1, len(cv_test_Delta)))
                if np.sum(cv_test_Delta) == len(cv_test_samples):
                    CEN_test = False
                else:
                    CEN_test = True
                cv_train_Delta = data[j].loc[data_guide.loc[j, 'Delta']].drop(cv_test_samples)
                cv_train_Delta = cv_train_Delta.drop(test_samples)
                cv_train_Delta = np.array(cv_train_Delta)
                cv_train_Delta = np.reshape(cv_train_Delta, (1, len(cv_train_Delta)))
                if np.sum(cv_train_Delta) == cv_train_Delta.shape[1]:
                    ## since there's no censoring on the training data subset, resetting this variable.
                    CEN = False
            if not CEN:
                dataparamt = [cv_train_t, emptyparam(cv_train_t.shape[0], dz, cv_train_t.shape[1], 'etime', np.nan, sparse)]
            else:
                dataparamt.append([cv_train_t, emptyparam(cv_train_t.shape[0], dz, cv_train_t.shape[1], 'etime', np.nan, sparse)])
                dataparamt.append([cv_train_Delta, emptyparam(cv_train_Delta.shape[0], dz, cv_train_Delta.shape[1], datatypec, np.nan, sparse)])
            if not CEN_test:
                dataparamt_test = [cv_test_t, emptyparam(cv_test_t.shape[0], dz, cv_test_t.shape[1], 'etime', np.nan, sparse)]
            else:
                dataparamt_test.append([cv_test_t, emptyparam(cv_test_t.shape[0], dz, cv_test_t.shape[1], 'etime', np.nan, sparse)])
                dataparamt_test.append([cv_test_Delta, emptyparam(cv_test_Delta.shape[0], dz, cv_test_Delta.shape[1], datatypec, np.nan, sparse)])
    if dataparamt ==[]:
        dataparamt = False
        CEN = False
        dataparamt_test = False
        CEN_test = False
    return(dataparamX_test, dataparamt_test, CEN_test, dataparamX, dataparamt, CEN)


def make_dataparamXt_final(test_samples, data_guide, data, analysis_directory):
    dataparamX = []
    dataparamt = []
    dataparamX_test = []
    dataparamt_test = []
    for j in range(len(data)):
        datatype = data_guide.loc[j, 'datatype']
        ## this dz and sparse don't matter because they will be set instead in model_selection_fixed data
        dz = 2
        sparse = [False]
        if datatype != 'survival':
            b = data_guide.loc[j, 'b']
            test_data = data[j].loc[:, test_samples]
            print('test data shape, j', test_data.shape, j)
            train_data = data[j].drop(test_samples, axis=1)
            print('dropped  test samples from train data', train_data.shape)
            print('shape of dropped duplicates', train_data.drop_duplicates().shape)
            print('number of features that are zero for all samples on training data',
                  np.sum(np.sum(train_data, axis=1) == 0))
            if np.sum(np.sum(train_data, axis=1) == 0) != 0:
                all_zero_features = train_data[np.sum(train_data, axis=1) == 0].index
                pd.DataFrame(all_zero_features).to_csv(analysis_directory + 'all_zero_features_data_' + str(j) + '.csv')
                test_data = test_data.drop(all_zero_features)
                print('dropped zero features test data', test_data.shape)
                train_data = train_data.drop(all_zero_features)
                print('dropped zero features train data', train_data.shape)
            dataparamX_test.append([np.array(test_data), emptyparam(test_data.shape[0], dz, test_data.shape[1], datatype, b, sparse)])
            dataparamX.append([np.array(train_data), emptyparam(train_data.shape[0], dz, train_data.shape[1], datatype, b, sparse)])
        else:
            CEN = data_guide.loc[j, 'CEN']
            if np.sum(data[j].loc[data_guide.loc[j, 'etime'], :]==0)!= 0:
                print("some samples have zero event time", np.sum(data[j].loc[data_guide.loc[j, 'etime'], :]==0))
                eps = np.min(data[j].loc[data_guide.loc[j, 'etime'], :][data[j].loc[data_guide.loc[j, 'etime'], :] != 0])/10
                print("replacing those zeros with 1/10 the next smallest event time")
                data[j].loc[data_guide.loc[j, 'etime'], :][data[j].loc[data_guide.loc[j, 'etime'], :] == 0] = eps
            if CEN == False:
                CEN_test = False
            test_t = data[j].loc[data_guide.loc[j, 'etime'], test_samples]
            test_t = np.array(test_t)
            test_t = np.reshape(test_t, (1, len(test_t)))
            print(test_t.shape)
            train_t = data[j].loc[data_guide.loc[j, 'etime']].drop(test_samples)
            train_t = np.array(train_t)
            train_t = np.reshape(train_t, (1, len(train_t)))
            if CEN:
                datatypec = data_guide.loc[j, 'CENtype']
                test_Delta = data[j].loc[data_guide.loc[j, 'Delta'], test_samples]
                test_Delta = np.array(test_Delta)
                test_Delta = np.reshape(test_Delta, (1, len(test_Delta)))
                if np.sum(test_Delta) == len(test_samples):
                    CEN_test = False
                else:
                    CEN_test = True
                train_Delta = data[j].loc[data_guide.loc[j, 'Delta']].drop(test_samples)
                train_Delta = np.array(train_Delta)
                train_Delta = np.reshape(train_Delta, (1, len(train_Delta)))
                if np.sum(train_Delta) == train_Delta.shape[1]:
                    ## since there's no censoring on the training data subset, resetting this variable.
                    CEN = False
            if not CEN:
                dataparamt = [train_t, emptyparam(train_t.shape[0], dz, train_t.shape[1], 'etime', np.nan, sparse)]
            else:
                dataparamt.append([train_t, emptyparam(train_t.shape[0], dz, train_t.shape[1], 'etime', np.nan, sparse)])
                dataparamt.append([train_Delta, emptyparam(train_Delta.shape[0], dz, train_Delta.shape[1], datatypec, np.nan, sparse)])
            if not CEN_test:
                dataparamt_test = [test_t, emptyparam(test_t.shape[0], dz, test_t.shape[1], 'etime', np.nan, sparse)]
            else:
                dataparamt_test.append([test_t, emptyparam(test_t.shape[0], dz, test_t.shape[1], 'etime', np.nan, sparse)])
                dataparamt_test.append([test_Delta, emptyparam(test_Delta.shape[0], dz, test_Delta.shape[1], datatypec, np.nan, sparse)])
    if dataparamt ==[]:
        dataparamt = False
        CEN = False
        dataparamt_test = False
        CEN_test = False
    return(dataparamX_test, dataparamt_test, CEN_test, dataparamX, dataparamt, CEN)


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

def cross_val_sfa(n_cv, data_directory, analysis_directory, MASTER_SEED):
    if MASTER_SEED!=False:
        np.random.seed(MASTER_SEED)
        f = open(analysis_directory + 'cross_val_sfa_seed.pkl', 'w')
        pickle.dump(MASTER_SEED, f)
    else:
        if not os.path.exists(analysis_directory + 'cross_val_sfa_seed.pkl'):
            print('no seed, exiting!')
            return
        else:
            f = open(analysis_directory + 'cross_val_sfa_seed.pkl', 'r')
            MASTER_SEED=pickle.load(f)
            np.random.seed(MASTER_SEED)
    data_guide, data, sample_name_ref = load_data(data_directory, False)
    data = nan_to_mean(data)
    cv_splits, test_samples = make_splits(n_cv, analysis_directory, sample_name_ref)
    test_samples= list(test_samples.values[:, 0])
    if np.sum(data_guide.loc[:, 'datatype']=='survival'):
        sparserange_string = data_guide.loc[0:(len(data) -2), 'sparserange']
    else:
        ##no survival needs longer sparserange.
        sparserange_string = data_guide.loc[0:(len(data) -1), 'sparserange']
    sparserange = []
    for k in range(sparserange_string.shape[0]):
        sparserange.append(eval(sparserange_string[k].replace(';', ',')))
    if not os.path.exists(analysis_directory + 'cv_'+ str(n_cv) + '_results/'):
        os.makedirs(analysis_directory + 'cv_'+ str(n_cv) + '_results/')
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
        os.makedirs(analysis_directory + 'cv_' + str(n_cv) + '_results/cv_run' + str(i) + '/')
        dataparamX_test, dataparamt_test, CEN_test, dataparamX, dataparamt, CEN = make_dataparamXt(n_cv, cv_splits, test_samples, i, data_guide, data, analysis_directory + 'cv_' + str(n_cv) + '_results/cv_run' + str(i) + '/')
        INIT = True  ## for now, don't give specific intializations for parameters
        model_selection_fixed_data(i, dataparamX_test, dataparamt_test, CEN_test,
                                   dataparamX, dataparamt, dzrange, sparserange,
                                   specs.loc[0, 'niter'], specs.loc[0, 'convgap'], mh_params, specs.loc[0, 'step'], CEN, INIT,
                                   analysis_directory + 'cv_' + str(n_cv) + '_results/cv_run' + str(i) + '/')
    return


def gather_plot_cv_cindex(n_cv, analysis_directory, disease_type):
    for i in range(n_cv):
        file_1 = analysis_directory + 'cv_' + str(n_cv) + '_results/cv_run' + str(i) + '/model_selection_output_'+str(i) + '.txt'
        file_2 = analysis_directory + 'cv_' + str(n_cv) + '_results_cox/cv_run' + str(i) + '/model_selection_output_cox'+str(i) + '.txt'
        file_3 = analysis_directory + 'cv_' + str(n_cv) + '_results_cox_gs/cv_run' + str(i) + '/model_selection_output_cox'+str(i) + '.txt'
        # file_4 = analysis_directory + 'cv_' + str(n_cv) + '_results_2step/cv_run' + str(i) + '/model_selection_output_2step'+str(i) + '.txt'
        f_1 = pd.read_csv(file_1, sep=',', header=None)
        f_2 = pd.read_csv(file_2, sep=',', header=None)
        f_3 = pd.read_csv(file_3, sep=',', header=None)
        # f_4 = pd.read_csv(file_4, sep=',', header=None)
        if i == 0:
            # mod_sel = pd.concat([f_1.iloc[:, f_1.shape[1]-1], f_2.iloc[:, f_2.shape[1]-1], f_3.iloc[:, f_3.shape[1]-1], f_4.iloc[:, f_4.shape[1]-1]])
            mod_sel = pd.concat([f_1.iloc[:, f_1.shape[1]-1], f_2.iloc[:, f_2.shape[1]-1], f_3.iloc[:, f_3.shape[1]-1]])
            print(mod_sel.shape)
        else:
            # mod_sel = pd.concat([mod_sel, pd.concat([f_1.iloc[:, f_1.shape[1]-1], f_2.iloc[:, f_2.shape[1]-1], f_3.iloc[:, f_3.shape[1]-1], f_4.iloc[:, f_4.shape[1]-1]])], axis=1)
            mod_sel = pd.concat([mod_sel, pd.concat([f_1.iloc[:, f_1.shape[1]-1], f_2.iloc[:, f_2.shape[1]-1], f_3.iloc[:, f_3.shape[1]-1]])], axis=1)
            print mod_sel.shape
    mod_sel.index = range(mod_sel.shape[0])
    mod_sel.columns = range(mod_sel.shape[1])
    for i in range(mod_sel.shape[0]):
        for j in range(mod_sel.shape[1]):
            item = mod_sel.iloc[i, j]
            if item == ' nan':
                item=np.nan
            else:
                item = eval(item[2:-1])
            mod_sel.iloc[i, j] = item
    print mod_sel
    fig, ax = plt.subplots()
    means=np.mean(mod_sel, axis=1)
    for item in mod_sel.index:
        im=ax.scatter((item+1)*np.ones(mod_sel.loc[item, :].shape), mod_sel.loc[item, :], color='gray', marker='o', s=100)
        imm=ax.scatter(item+1, means.loc[item], color='black', marker='_', s=100)  
    a= ax.legend([im, imm], ['c-index', 'mean c-index'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.set_ylabel(str(n_cv) + '-fold cross-validation c-index')
    ax.set_xlabel('model type')
    pl.xticks(range(1, mod_sel.shape[0]+1), mod_sel.index)
    if disease_type in ['LGG', 'LUAD']:
        txt = matplotlib.offsetbox.TextArea("Model Type:\n0 - FA-EPH-C "+r'$d_z =2$'+ "\n1 - FA-EPH-C "+r'$d_z =3$'+ " \n2 - FA-EPH-C "+r'$d_z =4$'+ " \n3 - FA-EPH-C "+r'$d_z =5$'+ "\n4 - EPH-C-L"+r'$_1 \gamma = 1e3$'+ "\n5 - EPH-C-L"+r'$_1 \gamma=1e4$'+ "\n6 - EPH-C-L"+r'$_1 \gamma =1e5$'+ "\n7 - EPH-C Gold Standard")
        box= a._legend_box
        box.get_children().append(txt)
        box.set_figure(box.figure)
    if disease_type =='LUSC':
        txt = matplotlib.offsetbox.TextArea("Model Type:\n0 - FA-EPH-C "+r'$d_z =2$'+ "\n1 - FA-EPH-C "+r'$d_z =5$'+ " \n2 - FA-EPH-C "+r'$d_z =10$'+ " \n3 - FA-EPH-C "+r'$d_z =15$'+ "\n4 - EPH-C-L"+r'$_1 \gamma = 1e3$'+ "\n5 - EPH-C-L"+r'$_1 \gamma=1e4$'+ "\n6 - EPH-C-L"+r'$_1 \gamma =1e5$'+ "\n7 - EPH-C Gold Standard")
        box= a._legend_box
        box.get_children().append(txt)
        box.set_figure(box.figure)
    if disease_type =='GBM':
        txt = matplotlib.offsetbox.TextArea("Model Type:\n0 - FA-EPH-C "+r'$d_z =2$'+ "\n1 - FA-EPH-C "+r'$d_z =3$'+ "\n2 - FA-EPH-C "+r'$d_z =4$'+ "\n3 - FA-EPH-C "+r'$d_z =5$' +"\n4 - EPH-C-L"+r'$_1 \gamma = 5e3$'+ "\n5 - EPH-C-L"+r'$_1 \gamma=1e4$'+ "\n6 - EPH-C-L"+r'$_1 \gamma =1e5$'+ "\n7 - EPH-C Gold Standard")
        box= a._legend_box
        box.get_children().append(txt)
        box.set_figure(box.figure)
    fig.tight_layout()
    plt.savefig(analysis_directory + 'model_selection_plot.pdf', bbox_inches='tight')
    mod_sel.to_csv(analysis_directory + 'model_selection.csv')
    return mod_sel

def gather_plot_cv_cindex_sim(n_cv, analysis_directory, disease_type):
    for i in range(n_cv):
        file_1 = analysis_directory + 'cv_' + str(n_cv) + '_results/cv_run' + str(i) + '/model_selection_output_'+str(i) + '.txt'
        file_2 = analysis_directory + 'cv_' + str(n_cv) + '_results_cox/cv_run' + str(i) + '/model_selection_output_cox'+str(i) + '.txt'
        # file_4 = analysis_directory + 'cv_' + str(n_cv) + '_results_2step/cv_run' + str(i) + '/model_selection_output_2step'+str(i) + '.txt'
        f_1 = pd.read_csv(file_1, sep=',', header=None)
        f_2 = pd.read_csv(file_2, sep=',', header=None)
        # f_4 = pd.read_csv(file_4, sep=',', header=None)
        if i == 0:
            # mod_sel = pd.concat([f_1.iloc[:, f_1.shape[1]-1], f_2.iloc[:, f_2.shape[1]-1], f_3.iloc[:, f_3.shape[1]-1], f_4.iloc[:, f_4.shape[1]-1]])
            mod_sel = pd.concat([f_1.iloc[:, f_1.shape[1]-1], f_2.iloc[:, f_2.shape[1]-1]])
            print(mod_sel.shape)
        else:
            # mod_sel = pd.concat([mod_sel, pd.concat([f_1.iloc[:, f_1.shape[1]-1], f_2.iloc[:, f_2.shape[1]-1], f_3.iloc[:, f_3.shape[1]-1], f_4.iloc[:, f_4.shape[1]-1]])], axis=1)
            mod_sel = pd.concat([mod_sel, pd.concat([f_1.iloc[:, f_1.shape[1]-1], f_2.iloc[:, f_2.shape[1]-1]])], axis=1)
            print mod_sel.shape
    mod_sel.index = range(mod_sel.shape[0])
    mod_sel.columns = range(mod_sel.shape[1])
    for i in range(mod_sel.shape[0]):
        for j in range(mod_sel.shape[1]):
            item = mod_sel.iloc[i, j]
            if item == ' nan':
                item=np.nan
            else:
                item = eval(item[2:-1])
            mod_sel.iloc[i, j] = item
    print mod_sel
    fig, ax = plt.subplots()
    means=np.mean(mod_sel, axis=1)
    for item in mod_sel.index:
        im=ax.scatter((item+1)*np.ones(mod_sel.loc[item, :].shape), mod_sel.loc[item, :], color='gray', marker='o', s=100)
        imm=ax.scatter(item+1, means.loc[item], color='black', marker='_', s=100)  
    a= ax.legend([im, imm], ['c-index', 'mean c-index'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.set_ylabel(str(n_cv) + '-fold cross-validation c-index')
    ax.set_xlabel('model type')
    pl.xticks(range(1, mod_sel.shape[0]+1), mod_sel.index)
    if disease_type in ['SIM1', 'SIM2', 'SIM3', 'SIM4']:
        txt = matplotlib.offsetbox.TextArea("Model Type:\n0 - FA-EPH-C "+r'$d_z =2$'+ "\n1 - FA-EPH-C "+r'$d_z =3$'+ "\n2 - FA-EPH-C "+r'$d_z =4$'+ "\n3 - FA-EPH-C "+r'$d_z =5$' + "\n4 - EPH-C-L"+r'$_1 \gamma = 5e4$'+ "\n5 - EPH-C-L"+r'$_1 \gamma=1e5$'+ "\n6 - EPH-C-L"+r'$_1 \gamma =1e6$')
        box= a._legend_box
        box.get_children().append(txt)
        box.set_figure(box.figure)
    fig.tight_layout()
    plt.savefig(analysis_directory + 'model_selection_plot.pdf', bbox_inches='tight')
    mod_sel.to_csv(analysis_directory + 'model_selection.csv')
    return mod_sel

def gather_plot_cv_cindex_2step(n_cv, analysis_directory):
    for i in range(n_cv):
        file_1 = analysis_directory + 'cv_' + str(n_cv) + '_results_2step/cv_run' + str(i) + '/model_selection_output_2step'+str(i) + '.txt'
        # file_4 = analysis_directory + 'cv_' + str(n_cv) + '_results_2step/cv_run' + str(i) + '/model_selection_output_2step'+str(i) + '.txt'
        f_1 = pd.read_csv(file_1, sep=',', header=None)
        # f_4 = pd.read_csv(file_4, sep=',', header=None)
        if i == 0:
            # mod_sel = pd.concat([f_1.iloc[:, f_1.shape[1]-1], f_2.iloc[:, f_2.shape[1]-1], f_3.iloc[:, f_3.shape[1]-1], f_4.iloc[:, f_4.shape[1]-1]])
            mod_sel = f_1.iloc[:, f_1.shape[1]-1]
            print(mod_sel.shape)
        else:
            # mod_sel = pd.concat([mod_sel, pd.concat([f_1.iloc[:, f_1.shape[1]-1], f_2.iloc[:, f_2.shape[1]-1], f_3.iloc[:, f_3.shape[1]-1], f_4.iloc[:, f_4.shape[1]-1]])], axis=1)
            mod_sel = pd.concat([mod_sel, f_1.iloc[:, f_1.shape[1]-1]], axis=1)
            print mod_sel.shape
    mod_sel.index = range(mod_sel.shape[0])
    mod_sel.columns = range(mod_sel.shape[1])
    for i in range(mod_sel.shape[0]):
        for j in range(mod_sel.shape[1]):
            item = mod_sel.iloc[i, j]
            if item == ' nan':
                item=np.nan
            else:
                item = eval(item[2:-1])
            mod_sel.iloc[i, j] = item
    print mod_sel
    mod_sel.to_csv(analysis_directory + 'model_selection_2step.csv')
    return mod_sel


def final_fit_sfa(data_directory, analysis_directory, MASTER_SEED):
    if MASTER_SEED!=False:
        np.random.seed(MASTER_SEED)
        f = open(analysis_directory + 'final_fit_sfa_seed.pkl', 'w')
        pickle.dump(MASTER_SEED, f)
    else:
        if not os.path.exists(analysis_directory + 'final_fit_sfa_seed.pkl'):
            print('no seed, exiting!')
            return
        else:
            f = open(analysis_directory + 'final_fit_sfa_seed.pkl', 'r')
            MASTER_SEED=pickle.load(f)
            np.random.seed(MASTER_SEED)
    data_guide, data, sample_name_ref = load_data(data_directory, False)
    data = nan_to_mean(data)
    if os.path.isfile(analysis_directory + 'test_samples.csv'):
        print('test samples are set aside')
        test_samples = pd.read_csv(analysis_directory + 'test_samples.csv', delimiter=',', index_col = 0)
    else:
        print('test samples file missing, exiting.')
        return
    test_samples= list(test_samples.values[:, 0])
    sparserange_string = data_guide.loc[0:(len(data) -2), 'sparserange']
    sparserange = []
    for k in range(sparserange_string.shape[0]):
        sparserange.append(eval(sparserange_string[k].replace(';', ',')))
    if not os.path.exists(analysis_directory + 'final_fit_results/'):
        os.makedirs(analysis_directory + 'final_fit_results/')
    else:
        print('directory already exists. exiting so that results are not overwritten')
        return
    if not os.path.isfile(analysis_directory + 'specs.csv'):
        print('spec file does not exist, exiting')
        return
    else:
        specs = pd.read_csv(analysis_directory + 'specs.csv', delimiter=',', index_col=0)
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
    dataparamX_test, dataparamt_test, CEN_test, dataparamX, dataparamt, CEN = make_dataparamXt_final(test_samples, data_guide, data, analysis_directory + 'final_fit_results/')
    INIT = True  ## for now, don't give specific intializations for parameters
    model_selection_fixed_data('final', dataparamX_test, dataparamt_test, CEN_test,
                               dataparamX, dataparamt, dzrange, sparserange,
                               specs.loc[0, 'niter'], specs.loc[0, 'convgap'], mh_params, specs.loc[0, 'step'], CEN, INIT,
                               analysis_directory + 'final_fit_results/')
    return


###sparsity tools
def getsparse_cox(n_cv, analysis_directory, model):
    for i in range(n_cv):
        file_1 = analysis_directory + 'cv_' + str(n_cv) + '_results_cox/cv_run' + str(i) + '/model_cox_' + str(model) +'/learn_cox/learned_parameters/wt.csv'
        wt = pd.read_csv(file_1, sep=',', index_col=0)
        if i == 0:
            gamma = [np.sum(wt!=0, axis=1)]
        else:
            gamma.append(np.sum(wt!=0, axis=1))
    return(np.mean(gamma))


def load_params(analysis_directory):
    plotloc = analysis_directory + 'final_fit_results/model_0_0/learn/learned_parameters/'
    paramX = []
    num= len(glob.glob(plotloc+ 'Mux*.csv'))
    for k in range(num):
        filenames = glob.glob(plotloc+ '*'+ str(k)+'.csv')
        sparse= pd.read_csv(plotloc+ 'sparse_'+ str(k)+'.csv' , sep=',', index_col=0).values[:,0]
        theta=[]
        if k==0:
           dzfile = glob.glob(plotloc+ 'Wx_' +str(k)+'.csv')
           dz = pd.read_csv(dzfile[0], sep=',', index_col=0).shape[1]
        if len(filenames)==4:
            datatype = 'normal'
            vnames = ['Wx', 'Mux', 'Psix']
            for j, item in enumerate(vnames):
                theta.append(pd.read_csv(plotloc + vnames[j] + '_' + str(k) + '.csv',  sep=',', index_col=0).values)
            paramX.append((datatype, theta, sparse))
        elif len(filenames)==5:
            datatype = 'binom'
            vnames = ['Wx', 'Mux', 'xi', 'b']
            for j, item in enumerate(vnames):
                if item != 'b':
                    theta.append(pd.read_csv(plotloc + vnames[j] + '_' + str(k) + '.csv',  sep=',', index_col=0).values)
                else:
                    theta.append(pd.read_csv(plotloc + vnames[j] + '_' + str(k) + '.csv',  sep=',', index_col=0).values[0][0])
            paramX.append((datatype, theta, sparse))
        elif len(filenames)==6:
            datatype = 'multinom'
            vnames = ['Wx', 'Mux', 'xi', 'alpha', 'b']
            for j, item in enumerate(vnames):
                if item != 'b':
                    theta.append(pd.read_csv(plotloc + vnames[j] + '_' + str(k) + '.csv',  sep=',', index_col=0).values)
                else:
                    theta.append(pd.read_csv(plotloc + vnames[j] + '_' + str(k) + '.csv',  sep=',', index_col=0).values[0][0])
            paramX.append((datatype, theta, sparse))
    filenames = glob.glob(plotloc + 'wt*.csv')
    if len(filenames) ==2:
        print('here')
        wt= pd.read_csv(plotloc +'wt.csv',  sep=',', index_col=0).values
        wtc= pd.read_csv(plotloc +'wtc.csv',  sep=',', index_col=0).values
        Lam = pd.read_csv(plotloc +'Lam.csv',  sep=',', index_col=0).values
        Lamc = pd.read_csv(plotloc +'Lamc.csv',  sep=',', index_col=0).values
        paramt= [('etime', (wt, Lam), [False, False]), ('cindIC', (wtc, Lamc), [False, False])]
        CEN=True
    elif len(filenames)==1:
        wt= pd.read_csv(plotloc +'wt.csv',  sep=',', index_col=0).values
        Lam = pd.read_csv(plotloc +'Lam.csv',  sep=',', index_col=0).values
        filenames2 = glob.glob(plotloc+ 'Lam*.csv')
        if len(filenames2)==1:
            paramt= [('etime', (wt, Lam), [False, False])]
            CEN=False
        elif len(filenames2)==2:
            Lamc = pd.read_csv(plotloc +'Lamc.csv',  sep=',', index_col=0).values
            paramt= [('etime', (wt, Lam), [False, False]), ('cind', (Lamc), [False, False])]
            CEN=True
    else:
        paramt=False
        CEN=False
    return(dz, paramX, paramt, CEN)

def gen_data_to_csv(ntrain, ntest, dz, paramX, paramt, CEN, plotloc, frac, sim_type, SEED):
    if SEED!=False:
        np.random.seed(SEED)
        f = open(plotloc + 'data/not_in_use/data_generating_seed.pkl', 'w')
        pickle.dump(SEED, f)
    else:
        if not os.path.exists(plotloc + 'data/not_in_use/data_generating_seed.pkl'):
            print('no seed, exiting!')
            return
        else:
            f = open(plotloc + 'data/not_in_use/data_generating_seed.pkl', 'r')
            SEED=pickle.load(f)
            np.random.seed(SEED)
            print('set seed from file')
            #print(cmp(state, np.random.RandomState().get_state()))
    nsamp=ntrain + ntest
    Zmean = np.zeros(dz)
    Zcov = np.diag(np.ones(dz))
    Z = (np.random.multivariate_normal(Zmean, Zcov, nsamp)).T
    Z=pd.DataFrame(Z, columns=['sim_'+str(x) for x in range(nsamp)])
    print(type(Z.columns[0]))
    Z.to_csv(plotloc + 'data/not_in_use/Ztrue.csv')
    pd.DataFrame(Z.columns[ntrain:nsamp]).to_csv(plotloc +'analysis/test_samples.csv', sep=',')
    dataparamX = []
    nfiles= len(paramX) + 1 
    ##do not change this line.
    col_names= ['file_location','datatype', 'b','sparserange','etime', 'CEN', 'Delta', 'CENtype']
    data_guide = pd.DataFrame(np.empty((nfiles, len(col_names)))*np.nan, columns=col_names)
    for i, para in enumerate(paramX):
        datatype, theta, sparse = para
        if datatype == 'normal':
            Wx, Mux, Psix = theta
            if sim_type==1 or sim_type==2:
                zero_index= np.random.choice(Wx.shape[0], np.int(frac*Wx.shape[0]), replace=False)
                pd.DataFrame(zero_index).to_csv(plotloc+ 'data/not_in_use/Wx_zero_indices_' +str(i) + '.csv')
                if sim_type==1: 
                    Wx[zero_index, :] = 0
                    print(np.sum(np.sum(Wx==0)))
                else:
                    Wx[zero_index, -2:] = 0
            # X = Wx.dot(Z) + np.random.multivariate_normal(
            #     Mux, np.diag(Psix[:, 0]), nsamp).T
            # # faster way to simulate multivariate normal
            X = (Wx.dot(Z) + np.array([Mux, ] * nsamp).T
                 + np.sqrt(Psix) * (np.random.normal(size=(Wx.shape[0], nsamp))))
        elif datatype == 'binom':
            Wx, Mux, xi, b = theta
            data_guide.loc[i, 'b'] = b
            if sim_type==1 or sim_type==2:
                zero_index= np.random.choice(Wx.shape[0], np.int(frac*Wx.shape[0]), replace=False)
                pd.DataFrame(zero_index).to_csv(plotloc+ 'data/not_in_use/Wx_zero_indices_' +str(i) + '.csv')
                if sim_type==1: 
                    Wx[zero_index, :] = 0
                    print(np.sum(np.sum(Wx==0)))
                else:
                    Wx[zero_index, -2:] = 0
            Muxn = np.array([Mux, ] * nsamp).T
            X = np.random.binomial(b, scipy.special.expit(Muxn + Wx.dot(Z)))
        elif datatype == 'multinom':
            Wx, Mux, xi, alpha, b = theta
            data_guide.loc[i, 'b'] = b
            if sim_type==1 or sim_type==2:
                zero_index= np.random.choice(Wx.shape[0], np.int(frac*Wx.shape[0]), replace=False)
                pd.DataFrame(zero_index).to_csv(plotloc+ 'data/not_in_use/Wx_zero_indices_' +str(i) + '.csv')
                if sim_type==1: 
                    Wx[zero_index, :] = 0
                    print(np.sum(np.sum(Wx==0)))
                else:
                    Wx[zero_index, -2:] = 0
            assert np.sum(Wx[-1:, :]) == 0
            Muxn = np.array([Mux, ] * nsamp).T
            assert np.sum(Muxn[-1:, :]) == 0
            # assert np.sum(xi[-1:, :]) == 0
            dx = Wx.shape[0]
            p = softmax(Muxn + Wx.dot(Z))
            X = np.empty((dx, nsamp))
            for n in range(0, nsamp):
                X[:, n] = np.random.multinomial(b, p[:, n])
        else:
            print('unknown datatype')
            return
        print(X.shape)
        X=pd.DataFrame(X[0, :, :], columns=['sim_'+str(x) for x in range(nsamp)])
        X.to_csv(plotloc+'data/X_'+str(i)+'.csv', sep=',')
        data_guide.loc[i, 'file_location']= 'X_'+str(i)+'.csv'  
        data_guide.loc[i, 'datatype'] = datatype 
        data_guide.loc[i, 'sparserange'] ='[[False, False]]'
        if sim_type==2:
            X.drop(zero_index).to_csv(plotloc+'data/X_inform_'+str(i)+'.csv', sep=',')
    if not paramt:
        dataparamt = False
        data_guide.dropna(axis=0, how='all', inplace=True)  
    else:
        if not CEN:
            datatype, theta, sparse = paramt
            wt, Lam = theta
            if sim_type==2 or sim_type==3:
                wt[:-2]=0
            t = np.random.exponential(1 / (Lam * np.exp(wt.dot(Z))))
            t=pd.DataFrame(t, columns=['sim_'+str(x) for x in range(nsamp)], index='eventtime')
            t.to_csv(plotloc+'data/t.csv', sep=',')
            data_guide.loc[i+1, 'file_location'] = 't.csv'
            data_guide.loc[i+1, 'datatype'] = 'survival'
            data_guide.loc[i+1, 'etime'] = 'eventtime'
            #this is the index name of the event indicator row
            data_guide.loc[i+1, 'Delta'] = False
            data_guide.loc[i+1, 'CEN'] = CEN
            data_guide.loc[i+1, 'CENtype'] = CEN
        elif CEN:
            [(datatypet, thetat, sparset),
             (datatypec, thetac, sparsec)] = paramt
            wt, Lam = thetat
            if sim_type==2 or sim_type==3:
                wt[:-2]=0
            t = np.random.exponential(1 / (Lam * np.exp(wt.dot(Z))))
            if datatypec == 'cind':
                Lamc = thetac
                c = np.random.exponential(1 / (Lamc), nsamp)
                c = np.reshape(c, (1, nsamp))
            elif datatypec == 'cindIC':
                wtc, Lamc = thetac
                if sim_type==2 or sim_type==3:
                    wtc[:-2]=0
                c = np.random.exponential(1 / (Lamc * np.exp(wtc.dot(Z))))
            tE = np.min(np.array([t, c]), axis=0)
            Delta = 1 * (tE == t)
            dataparamt = [[tE, (datatypet, thetat, sparset)],
                          [Delta, (datatypec, thetac, sparsec)]]
            survtrain=pd.DataFrame(np.vstack([tE[:, 0:ntrain],Delta[:, 0:ntrain] ]) )
            survtest=pd.DataFrame(np.vstack([t[:, ntrain:nsamp], np.ones(ntest).reshape(1, ntest) ]) )
            surv=pd.concat([survtrain, survtest], axis=1)
            print(surv.shape)
            surv.index=['tE', 'delta']
            surv.columns=['sim_'+str(x) for x in range(nsamp)]
            #surv.ix['tE', surv.ix['tE'] < 0.1]=0
            #surv.ix['tE', surv.ix['tE'] > 1e5]=1e5
            surv.to_csv(plotloc+'data/survival.csv', sep=',')
            data_guide.loc[i+1, 'file_location'] = 'survival.csv'
            data_guide.loc[i+1, 'datatype'] = 'survival' 
            data_guide.loc[i+1, 'etime'] = 'tE'
            #this is the index name of the event indicator row
            data_guide.loc[i+1, 'Delta'] = 'delta'
            data_guide.loc[i+1, 'CEN']= CEN
            data_guide.loc[i+1, 'CENtype'] = datatypec
    data_guide.to_csv(plotloc + 'data/data_guide.csv') 

def get_best(mod_sel):
    means= np.mean(mod_sel, axis=1)
    best=list(means).index(np.max(means))
    print ('best', best)
    stds= np.std(mod_sel, axis=1)
    top= means + stds
    bottom = means -stds
    better=[]
    for i, item in enumerate(means):
        if top.iloc[i] < top.iloc[best] and bottom.iloc[i]> bottom.iloc[best]:
            print('ones better', i)
            better.append(i)
    if better ==[]:
        return best
    elif len(better)==1:
        return better[0]
    else:
        means2 = means.iloc[better]
        best2=list(means).index(np.max(means2))
        print('second try at best', best2)
        better2=[]
        for i, item in enumerate(means):
            if top.iloc[i] < top.iloc[best2] and bottom.iloc[i]> bottom.iloc[best2]:
                better2.append(i)
                print('second try ones better', i)
        if better2 ==[]:
            return best2
        elif len(better2)==1:
            return better2[0]
        else:
            print('need to make code recursive')

def new_project(directory, project_name):
    if not os.path.exists(directory + project_name + '/'):
        os.makedirs(directory + project_name+ '/')
        data_directory= directory + project_name+ '/data/'
        analysis_directory= directory + project_name+ '/analysis/'
        os.makedirs(data_directory)
        os.makedirs(analysis_directory)
        os.makedirs(data_directory + 'not_in_use/')
        # not working because of conda
        # if project_name in ["LGG" ,"GBM", "LUAD", "LUSC"]:
        #   os.system('Rscript download_data.R '+ project_name + ' ' + directory + project_name+ '/data/not_in_use/')
    else:
        data_directory= directory + project_name+ '/data/'
        analysis_directory= directory + project_name+ '/analysis/'
    return(data_directory, analysis_directory)





