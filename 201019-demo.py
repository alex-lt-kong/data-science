# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 22:15:06 2020

@author: mamsds
"""

from typing import Dict
from typing import List
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.stats.outliers_influence import variance_inflation_factor

import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import scipy as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings

import importlib
asgmt1 = importlib.import_module('200920_asgmt1_normality-and-other-tests')
ch2 = importlib.import_module('200922_ch2_linear-regression-and-assumptions')


def box_cox_transformer(lambda1: float, x1: np.ndarray, y1: np.ndarray):

    if lambda1 != 0:
        BCy = (y1 ** lambda1 - 1) / lambda1
    else:
        BCy = np.log(y1)

    BCfmodel = ch2.ols_model_helper(patsy_formula = 'BCy ~ x1',
                     data = {'BCy': BCy, 'x1': x1},
                     with_intercept = True,
                     verbosity = 0,
                     significance_level = 0.05)

    #BCfmodel = sm.OLS(BCy, x1).fit()
    # This statement is used by Dr. Yu. My understanding is that both statements should be fine.

    SSE_lambda = ((BCfmodel.predict() - BCy) ** 2).sum()

    loglf = (lambda1 - 1) * ((np.log(y1)).sum()) - len(y1) * (np.log(SSE_lambda / len(y1))) / 2

    return { 'lambda': lambda1, 'loglf': loglf }

def assumptions_12346_checker(model):

    extresid = model.get_influence().resid_studentized_external
    pred = model.predict()

    print('Externally Studentized Residual Plots:')
    plt.scatter(pred, extresid, s = 1)
    plt.xlabel('ŷ (the predicted dependent variable)')
    plt.ylabel('ri (the externally studentized residual(i.e. y[i] - ŷ[i]))')
    plt.axhline(y=2, color='g', linestyle='-.', linewidth = 0.5)
    plt.axhline(y=0, color='r', linestyle='-.', linewidth = 0.5)
    plt.axhline(y=-2, color='g', linestyle='-.', linewidth = 0.5)
    plt.show()
    print('Interpretation of the plot:\nIf:')
    print('(1) The plot has no pattern (for possible patterns and their meanings, refer to pages 28 - 30 of lec5_ch2-variable-selection-and-model-diagnostics.pdf);')
    print('(2) Points are around 0; and')
    print('(3) Most of the points are inside the band |r[i]| <= 2')
    print('We say that Assumptions A1 - A4 are likely to be valid.')


    print('\nNormality Check:')
    sm.qqplot(data=extresid, dist=sp.stats.distributions.norm, line='45')
    plt.show()
    print('If many points do not fall near the red line then it is likely that assumption 4 is invalid.')

    asgmt1.shapiro_wilk_test(significance_level = 0.01, dataset = extresid)
    print('\n')
   # z = (extresid - np.mean(extresid))/np.std(extresid)
    asgmt1.kolmogorov_smirnov_test(significance_level = 0.01, dataset = extresid, standardized = False)

    asgmt1.anderson_darling_test(dataset = extresid)

def lec5_box_cox_transformation_remedy_non_normality_non_constant_variance():

    parent_dir = pathlib.Path(__file__).parent.absolute()
    df = pd.read_csv(os.path.join(parent_dir, 'hprice.txt'), sep='\t')

    y = df['Y']
    X = pd.concat([pd.DataFrame(df['X1']), pd.DataFrame(df['X2']), pd.DataFrame(df['X3']), pd.DataFrame(df['X4'])], axis = 1)

  #  X = sm.add_constant(X)
  #  print(X)
    BC = pd.DataFrame(columns = ['lambda', 'loglf'])
    lambda1 = np.arange(-2, 2, 0.01)
    # this range should be related to the [-2, +2] mentioned in the interpretation of assumptions_12346_checker()
    lambda1[200] = 0
    for i in range(1, len(lambda1)):
        BC.loc[i] = box_cox_transformer(lambda1[i], X, y)

    plt.plot(BC['lambda'], BC['loglf'])
    plt.show()
    print(BC['lambda'][BC['loglf'] == BC['loglf'].max()])

    mylambda = 0.11
    # this 0.11 comes from the above statement
    BCy = (y ** mylambda - 1) / mylambda

    dataBC = pd.DataFrame({'x1': df['X1'], 'x2': df['X2'], 'x3' : df['X3'], 'x4': df['X4'], 'BCy': BCy})

    model = ch2.ols_model_helper(patsy_formula = 'BCy ~ x1 + x2 + x3 + x4',
                     data = dataBC,
                     with_intercept = True,
                     verbosity = 0,
                     significance_level = 0.05)
    assumptions_12346_checker(model)
    print('Refer to pages 48 and 49 of lec5-6_ch2...')

def basic_test():


    from scipy.stats import shapiro, kstest

    print('Refer to page 229 of lec2-3_ch1_introduction_with_mjd')

    da1 = pd.read_csv('sp500_with-headers.txt', sep = "\s+")
    rt = np.diff(np.log(da1["close"]))
   # rt = np.append(np.nan, rt)
    print(shapiro(rt))
    print('')
    #  asgmt1.shapiro_wilk_test(significance_level = 0.01, dataset = rt)
 #   z = (rt - np.mean(rt)) / np.std(rt)

 #   print(kstest(z, cdf='norm'))
  #  print('')
    asgmt1.kolmogorov_smirnov_test(significance_level = 0.01, dataset = rt, standardized = False)
 #   asgmt1.kolmogorov_smirnov_test(significance_level = 0.01, dataset = z, standardized = True)

    print('\n')
    asgmt1.anderson_darling_test(dataset = rt)

def main():

    basic_test()
 #   return
    lec5_box_cox_transformation_remedy_non_normality_non_constant_variance()

if __name__ == '__main__':

    main()