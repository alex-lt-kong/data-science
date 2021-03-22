#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 22:08:22 2020

@author: mamsds
"""

from typing import List
from statsmodels.sandbox.regression.predstd import wls_prediction_std

import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import scipy as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf

import importlib
ch2 = importlib.import_module('linear-regression-and-assumptions')

def model_evaluator(dataset: np.ndarray,
                    column_names_of_explanatory_aka_independent_variables: List[str],
                    column_name_of_response_aka_dependent_variable: str):

    x_names = column_names_of_explanatory_aka_independent_variables
    y_name = column_name_of_response_aka_dependent_variable
    y = dataset[y_name]

    model = sm.OLS(y, sm.add_constant(dataset[list(x_names)]))
    # add_constant: intercept included
    regr = model.fit()

    return {'model': regr,
            'variables': list(x_names),
            'aic': regr.aic,
            'bic': regr.bic,
            'r2': regr.rsquared,
            'adj_r2': regr.rsquared_adj}

def forward_selector(dataset: np.ndarray,
                     column_names_of_USED_explanatory_aka_independent_variables: List[str],
                     column_names_of_ALL_explanatory_aka_independent_variables: List[str],
                     evaluation_criterion: str,
                     column_name_of_response_aka_dependent_variable: str):

  #  print(f'USED: {column_names_of_USED_explanatory_aka_independent_variables}')
  #  print(f'ALL: {column_names_of_ALL_explanatory_aka_independent_variables}')
    criterion = evaluation_criterion.lower()
    assert (criterion == 'aic' or criterion == 'bic' or criterion == 'r2' or criterion == 'adj_r2')

    used_x_names = column_names_of_USED_explanatory_aka_independent_variables
    x_names = column_names_of_ALL_explanatory_aka_independent_variables
    y_name = column_name_of_response_aka_dependent_variable

    unused_x_names = [new_x for new_x in x_names if new_x not in used_x_names]
    results = []

    for new_x in unused_x_names:
        results.append(model_evaluator(dataset = dataset,
                    column_names_of_explanatory_aka_independent_variables = used_x_names + [new_x],
                    column_name_of_response_aka_dependent_variable = y_name))
   #     print(results[len(results) - 1]['variables'])

    models = pd.DataFrame(results)

    if criterion == 'aic' or criterion == 'bic':
        model_each = models.loc[models[criterion].idxmin()]
    else:
        model_each = models.loc[models[criterion].idxmax()]
    return model_each

def get_best_model_with_forward_selection(evaluation_criterion: str,
        dataset: np.ndarray,
        column_names_of_explanatory_aka_independent_variables: List[str],
        column_name_of_response_aka_dependent_variable: str):

    criterion = evaluation_criterion.lower()
    assert (criterion == 'aic' or criterion == 'bic' or criterion == 'r2' or criterion == 'adj_r2')
    print(f'=== Picking model by [{criterion}] with FORWARD sequential selection ===')

    y_name = column_name_of_response_aka_dependent_variable
    x_names = column_names_of_explanatory_aka_independent_variables

    best_models = pd.DataFrame(columns = ['model', 'aic', 'bic','r2', 'adj_r2', 'variables'])
  #  best_models = pd.DataFrame(columns = [criterion, 'variables'])
    exp = []

    for i in range(1, len(x_names) + 1):
        best_models.loc[i] = forward_selector(dataset = dataset,
                  column_names_of_USED_explanatory_aka_independent_variables = exp,
                  column_names_of_ALL_explanatory_aka_independent_variables = x_names,
                  evaluation_criterion = criterion,
                  column_name_of_response_aka_dependent_variable = y_name)
        exp = best_models.loc[i]['variables']
    if criterion == 'aic' or criterion == 'bic':
        best_models = best_models.sort_values(by = criterion, ascending = True)
    else:
        best_models = best_models.sort_values(by = criterion, ascending = False)

    print(best_models[[criterion, 'variables']])
    return best_models


def get_best_model_with_backward_elimination(
        dataset: np.ndarray,
        evaluation_criterion: str,
        column_names_of_explanatory_aka_independent_variables: List[str],
        column_name_of_response_aka_dependent_variable: str):

    criterion = evaluation_criterion.lower()
    assert (criterion == 'aic' or criterion == 'bic' or criterion == 'r2' or criterion == 'adj_r2')

    print(f'=== Picking model by [{criterion}] with BACKWARD sequential elimination ===')
    x_names = column_names_of_explanatory_aka_independent_variables
    y_name = column_name_of_response_aka_dependent_variable

    best_models = pd.DataFrame(columns = ['model', 'aic', 'bic','r2', 'adj_r2', 'variables'])

    best_models.loc[len(x_names)] = model_evaluator(dataset = dataset,
                    column_names_of_explanatory_aka_independent_variables = x_names,
                    column_name_of_response_aka_dependent_variable = y_name)

    while len(x_names) > 1:
        best_models.loc[len(x_names) - 1] = backward_eliminator(
                    dataset = dataset, evaluation_criterion = criterion,
                    column_names_of_explanatory_aka_independent_variables = x_names,
                    column_name_of_response_aka_dependent_variable = y_name)
        x_names = best_models.loc[len(x_names) - 1]['variables']

    if criterion == 'aic' or criterion == 'bic':
        best_models = best_models.sort_values(by = criterion, ascending = True)
    else:
        best_models = best_models.sort_values(by = criterion, ascending = False)
    print(best_models[[criterion, 'variables']])
    return best_models

def backward_eliminator(dataset: np.ndarray, evaluation_criterion: str,
                    column_names_of_explanatory_aka_independent_variables: List[str],
                    column_name_of_response_aka_dependent_variable: str):

    criterion = evaluation_criterion
    assert (criterion == 'aic' or criterion == 'bic' or criterion == 'r2' or criterion == 'adj_r2')
    x_names = column_names_of_explanatory_aka_independent_variables
    results = []
    for combo in itertools.combinations(x_names, len(x_names) - 1):
        results.append(model_evaluator(dataset = dataset,
                    column_names_of_explanatory_aka_independent_variables = combo,
                    column_name_of_response_aka_dependent_variable = column_name_of_response_aka_dependent_variable))
        models = pd.DataFrame(results)
        if criterion == 'bic' or criterion == 'aic':
            model_each = models.loc[models[criterion].idxmin()]
        else:
            model_each = models.loc[models[criterion].idxmax()]
    return model_each

def get_best_model_with_enumeration(evaluation_criterion: str,
                      dataset: np.ndarray,
                      column_names_of_explanatory_aka_independent_variables: List[str],
                      column_name_of_response_aka_dependent_variable: str):
    # This is called "Best Subset Selection approach" by Dr. Yu.

    criterion = evaluation_criterion.lower()
    assert (criterion == 'aic' or criterion == 'bic' or criterion == 'r2' or criterion == 'adj_r2')

    print(f'=== Picking model by [{criterion}] with ENUMERATION ===')
    x_names = column_names_of_explanatory_aka_independent_variables
    y_name = column_name_of_response_aka_dependent_variable


    best_models = [None] * (len(x_names) + 1)

    for i in range(1, len(x_names) + 1):
        results = []
        for x_names_combination in itertools.combinations(x_names, i):
            results.append(model_evaluator(dataset = dataset,
                    column_names_of_explanatory_aka_independent_variables = x_names_combination,
                    column_name_of_response_aka_dependent_variable = y_name))

        if criterion == 'bic' or criterion == 'aic':
            # AIC and BIC can be considered as measures of error.
            # Therefore, the smaller the better.
            temp_min = 2333333
            for j in range(len(results)):
                if results[j][criterion] < temp_min:
                    best_models[i] = results[j]
                    temp_min = results[j][criterion]
        else:
            temp_max = -2333333
            for j in range(len(results)):
                if results[j][criterion] > temp_max:
                    best_models[i] = results[j]
                    temp_max = results[j][criterion]
    best_models.pop(0)

    best_models = pd.DataFrame(best_models)

    if criterion == 'aic' or criterion == 'bic':
        best_models = best_models.sort_values(by = criterion, ascending = True)
    else:
        best_models = best_models.sort_values(by = criterion, ascending = False)
    best_models = best_models.reset_index(drop=True)
    print(best_models[[criterion, 'variables']])
    return best_models

def question0():

    parent_dir = pathlib.Path(__file__).parent.absolute()
    df0 = pd.read_csv(os.path.join(parent_dir, 'weeklyinterest.txt'), sep='\s+', header = None)
    df = df0[df0.iloc[:, 6] > 0]

    aaa_diff = np.diff(df.iloc[:, 9])
    tbond10_diff = np.diff(df.iloc[:, 5])
    tbond30_diff = np.diff(df.iloc[:, 6])

    fedfund_diff = np.diff(df.iloc[:, 3])
    prime_diff = np.diff(df.iloc[:, 8])

    y = aaa_diff

    df = pd.concat([pd.DataFrame(y),
                    pd.DataFrame(tbond10_diff),
                   pd.DataFrame(tbond30_diff),
                   pd.DataFrame(fedfund_diff),
                   pd.DataFrame(prime_diff)], axis = 1)
    df.columns = ['aaa_dif', 'tbond10_dif', 'tbond30_dif', 'fedfund_dif', 'prime_dif']

    #print(df)
    get_best_model_with_enumeration(evaluation_criterion = 'AIC',
      dataset = df,
      column_names_of_explanatory_aka_independent_variables = ['tbond10_dif', 'tbond30_dif', 'fedfund_dif', 'prime_dif'],
      column_name_of_response_aka_dependent_variable = 'aaa_dif')

    print('')
    get_best_model_with_backward_elimination(
        evaluation_criterion = 'adj_r2',
        dataset = df,
        column_names_of_explanatory_aka_independent_variables = ['tbond10_dif', 'tbond30_dif', 'fedfund_dif', 'prime_dif'],
        column_name_of_response_aka_dependent_variable = 'aaa_dif')
    print('')

    get_best_model_with_backward_elimination(
        evaluation_criterion = 'aic',
        dataset = df,
        column_names_of_explanatory_aka_independent_variables = ['tbond10_dif', 'tbond30_dif', 'fedfund_dif', 'prime_dif'],
        column_name_of_response_aka_dependent_variable = 'aaa_dif')

    print('')
    get_best_model_with_forward_selection(evaluation_criterion = 'AIC',
    dataset = df,
    column_names_of_explanatory_aka_independent_variables = ['tbond10_dif', 'tbond30_dif', 'fedfund_dif', 'prime_dif'],
    column_name_of_response_aka_dependent_variable = 'aaa_dif')

def question1():

    parent_dir = pathlib.Path(__file__).parent.absolute()
    df = pd.read_csv(os.path.join(parent_dir, 'hw3_q1.txt'), sep='\s+')

    print('')
    model = get_best_model_with_enumeration(evaluation_criterion = 'adj_r2',
              dataset = df,
              column_names_of_explanatory_aka_independent_variables = ['sales', 'AE', 'cost'],
              column_name_of_response_aka_dependent_variable = 'EPS').iloc[0]
    print('')
    get_best_model_with_enumeration(evaluation_criterion = 'AIC',
              dataset = df,
              column_names_of_explanatory_aka_independent_variables = ['sales', 'AE', 'cost'],
              column_name_of_response_aka_dependent_variable = 'EPS')
    print('')
    get_best_model_with_enumeration(evaluation_criterion = 'bic',
              dataset = df,
              column_names_of_explanatory_aka_independent_variables = ['sales', 'AE', 'cost'],
              column_name_of_response_aka_dependent_variable = 'EPS')



    print('')
    get_best_model_with_backward_elimination(
        evaluation_criterion = 'BIC',
        dataset = df,
        column_names_of_explanatory_aka_independent_variables = ['sales', 'AE', 'cost'],
        column_name_of_response_aka_dependent_variable = 'EPS')

    print('')
    get_best_model_with_forward_selection(evaluation_criterion = 'adj_r2',
        dataset = df,
        column_names_of_explanatory_aka_independent_variables = ['sales', 'AE', 'cost'],
        column_name_of_response_aka_dependent_variable = 'EPS')

    print(f'The ultimately chosen model: {model}')
    model = model['model']

    print('')
    ch2.predict_y_with_model(model = model,
                             intercept_included = True,
                             independent_variables = [[175, 240], [20, 25]],
                             iv_names = ['AE', 'cost'],
                             significance_level = 0.03)

def question2():

    parent_dir = pathlib.Path(__file__).parent.absolute()
    df = pd.read_csv(os.path.join(parent_dir, 'hw3_q2.txt'), sep='\s+')

    print(df.head())
    df['cbrt_carat'] = df['carat'] ** (1 / 3)
    print(df.head())
  #  return
    ivs = ['colorscore', 'clarityscore', 'cutscore', 'cbrt_carat']
    dv = 'price'


    model = ch2.ols_model_helper(patsy_formula = 'price ~ colorscore + clarityscore + cutscore + cbrt_carat',
                     data = df,
                     with_intercept = True,
                     verbosity = 2,
                     significance_level = 0.05)
    ch2.assumptions_12346_checker(model)

    y = df[dv]
    X = df[ivs]

    BC = pd.DataFrame(columns = ['lambda', 'loglf'])
    lambda1 = np.arange(-2.5, 2.5, 0.01)
    # this range should be related to the [-2, +2] mentioned in the interpretation of assumptions_12346_checker()
   # lambda1[200] = 0
    print(lambda1[249])
    lambda1[250] = 0
    print(lambda1[251])
    # How about this???


    for i in range(1, len(lambda1)):
        BC.loc[i] = ch2.box_cox_transformer(lambda1[i], X, y)

    plt.plot(BC['lambda'], BC['loglf'])
    plt.show()
    print(BC['lambda'][BC['loglf'] == BC['loglf'].max()])

    mylambda = 0.1
    # this 0.1 comes from the above print() statement
    BCy = (y ** mylambda - 1) / mylambda

    df['price_transformed'] = BCy

    model = ch2.ols_model_helper(patsy_formula = 'price_transformed ~ colorscore + clarityscore + cutscore + cbrt_carat',
                     data = df,
                     with_intercept = True,
                     verbosity = 3,
                     significance_level = 0.03)
    ch2.assumptions_12346_checker(model)

    # ivs = ['colorscore', 'clarityscore', 'cutscore', 'cbrt_carat']
    ch2.predict_y_with_model(model = model,
                             intercept_included = True,
                             independent_variables = [[4, 2], [6, 4], [4, 3], [0.58**(1/3), 1.03**(1/3)]],
                             iv_names = ivs,
                             significance_level = 0.03)


    lowers = [10.973969, 12.533602]
    uppers = [12.379540, 13.950204]

    real_lowers = [0, 0]
    real_uppers = [0, 0]

    real_lowers[0] = min(box_cox_transform_inverse(lowers[0], mylambda),
                           box_cox_transform_inverse(uppers[0], mylambda))
    real_uppers[0] = max(box_cox_transform_inverse(lowers[0], mylambda),
                           box_cox_transform_inverse(uppers[0], mylambda))
    real_lowers[1] = min(box_cox_transform_inverse(lowers[1], mylambda),
                           box_cox_transform_inverse(uppers[1], mylambda))
    real_uppers[1] = max(box_cox_transform_inverse(lowers[1], mylambda),
                           box_cox_transform_inverse(uppers[1], mylambda))

    print(real_lowers)
    print(real_uppers)

def box_cox_transform_inverse(yt, lambda1=0):
    # http://www.css.cornell.edu/faculty/dgr2/_static/files/R_html/Transformations.html#3_transformation_and_back-transformation

    assert lambda1 != 0 # Not implemented
    return math.exp(np.log(1 + lambda1 * yt) / lambda1)


def main():

    pd.set_option('display.max_columns', None)
 #   pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 270)

    print('\n\n ===== Question 0 =====\n\n')
    question0()

    print('\n\n ===== Question 1 =====\n\n')
    question1()

    print('\n\n ===== Question 2 =====\n\n')
    question2()


if __name__ == '__main__':
    main()