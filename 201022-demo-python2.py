# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 22:15:06 2020

@author: mamsds
"""


#from statsmodels.sandbox.regression.predstd import wls_prediction_std
#from statsmodels.stats.outliers_influence import variance_inflation_factor

#import itertools
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
import scipy as sp
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
#import warnings

#import importlib
#asgmt1 = importlib.import_module('200920_asgmt1_normality-and-other-tests')
#ch2 = importlib.import_module('200922_ch2_linear-regression-and-assumptions')




def basic_test():


    from scipy.stats import shapiro, kstest

   # print('Refer to page 229 of lec2-3_ch1_introduction_with_mjd')

    da1 = pd.read_csv('/home/mamsds/Documents/d/hkust/courses/mfit5003_data-analysis/source-code/sp500_with-headers.txt', sep = "\s+")
    rt = np.diff(np.log(da1["close"]))
   # rt = np.append(np.nan, rt)

    #  asgmt1.shapiro_wilk_test(significance_level = 0.01, dataset = rt)
    z = (rt - np.mean(rt)) / np.std(rt)

    print(kstest(z, cdf='norm'))
  #  print('')
 #   asgmt1.kolmogorov_smirnov_test(significance_level = 0.01, dataset = rt, standardized = False)
 #   asgmt1.kolmogorov_smirnov_test(significance_level = 0.01, dataset = z, standardized = True)

 #   print('\n')
  #  asgmt1.anderson_darling_test(dataset = rt)

def main():

    print("Python version")
    print (sys.version)
    basic_test()
 #   return
 #   lec5_box_cox_transformation_remedy_non_normality_non_constant_variance()

if __name__ == '__main__':

    main()