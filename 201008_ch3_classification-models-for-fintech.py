#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 20:48:25 2020

@author: Alex
"""

from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_splitble way to f
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import random


def plot_decision_regions(X, y, classifier, test_idx = None, resolution = 0.02):

    markers = ('x', 'o', '^', 'v')
    colors = ('red', 'green', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1 # feature 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1 # feature 2

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha = 0.2, cmap = cmap)
    plt.xlim(xx1.min(), xx2.max())
    plt.ylim(xx2.min(), xx2.max())

    #plot class samples
    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=X[y == c1, 0], y=X[y == c1, 1], alpha = 0.8, c = cmap(idx), marker = 'o', label = c1, s = 50)

    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], alpha = 0.8, linewidth = 1, c = 'k', marker = 'x', s = 55, label = 'testing data')

def model_assessment_accuracy_precision_recall_f1(classifier, test_y: np.ndarray, test_x: np.ndarray, verbosity: int):

    predict_y = classifier.predict(test_x)
    cf_mat = confusion_matrix(y_true = test_y, y_pred = predict_y)
    if verbosity >= 1:
        print(cf_mat)

    if verbosity >= 2:
        fig, ax = plt.subplots(figsize = (2.5, 2.5))
        ax.matshow(cf_mat, cmap = plt.cm.Blues, alpha = 0.5)
        for i in range(cf_mat.shape[0]):
            for j in range(cf_mat.shape[1]):
                ax.text(x = j, y = i, s = cf_mat[i, j], va = 'center', ha = 'center')
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual (True) Class')
        plt.show()
    if verbosity >= 3:
        print("""Interpretation of the confusion matrix:
                             |      Predict Class      |
                             |  Positive  |  Negative  |
-------------------------------------------------------|
  Actual (True) |  Positive  |     TP     |     FN     |
  Class         |  Negative  |     FP     |     TN     |
          """)

    if verbosity >= 2:
        print(classification_report(test_y, predict_y))
    if verbosity >= 3:
        print('Interpretation of the classification report:')
        print('Precision: TP / (TP + FP) or TN / (FN + TN)')
        print('Recall: TP / (TP + FN) or TN / (FP + TN)')
        print('F1: 2 / (1 / Precision + 1 / Recall)')
        print('Usually, Precision, recall, f1-score and accuracy are enough.')

def model_assessment_roc_curve(classifier, test_y: np.ndarray, test_x: np.ndarray):

    false_positive_rate, true_positive_rate, thresholds = roc_curve(test_y, classifier.predict_proba(test_x)[:, 1])

    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.title('Receiver Operating Characteristic Curve')
    plt.plot(false_positive_rate, true_positive_rate, 'b', marker = 'o', label = 'AUC = {}'.format(roc_auc))
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('True Positive rate (i.e. Recall)')
    plt.xlabel('False Positive rate')
    plt.show()

def lec6_house_price_classifier_one_variable():

    parent_dir = pathlib.Path(__file__).parent.absolute()
    df = pd.read_csv(os.path.join(parent_dir, 'house_price.txt'), sep=',')

    df.loc[:, ('y1')] = (df['price'] > df['price'].median()).astype(int)
    df = df.drop(['index', 'price', 'sq_price'], axis = 1)

    df_low = df.loc[df['y1'] == 0][['area', 'bathrooms', 'y1']]
    df_high = df.loc[df['y1'] == 1][['area', 'bathrooms', 'y1']]

    train_x, test_x, train_y, test_y = train_test_split(df[['area']], df['y1'], test_size = 0.25, random_state = 33)
    # random_state:  Pass an int for reproducible output across multiple function calls.

    ss = StandardScaler()
    train_x_std = ss.fit_transform(train_x)
    test_x_std = ss.transform(test_x)
    # These two statements are used to make sure that all data are in the same scale.

    lr = LogisticRegression(solver = 'liblinear').fit(train_x_std, train_y)

    x_range = np.linspace(df['area'].min(), df['area'].max(), 1000)
    # np.linspace: Return evenly spaced numbers over a specified interval.
    x_range  = x_range.reshape(-1, 1)
    x_range_std = ss.transform(x_range)

    y_prob = lr.predict_proba(x_range_std)
    # Interpretation of y_prob:
    # For each row, there will be two floating-point numbers [0.23780654318010663, 0.7621934568198934]
    # 0.23780654318010663 -> The probability of 0
    # 0.7621934568198934 -> The probability of 1
    plt.plot(x_range, y_prob[:, 1], 'b-', label = 'High')
    plt.plot(x_range, y_prob[:, 0], 'r--', label = 'low')

    plt.scatter(df_low['area'], df_low['y1'], marker = 'o', s = 50, c = 'red', label = 'Low')
    plt.scatter(df_high['area'], df_high['y1'], marker = 'x', s = 50, c = 'blue', label = 'High')

    plt.xlabel('area')
    plt.ylabel('')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()


    model_assessment_accuracy_precision_recall_f1(classifier = lr, test_y = test_y, test_x = test_x_std, verbosity = 1)
    model_assessment_roc_curve(classifier = lr, test_y = test_y, test_x = test_x_std)

def lec6_house_price_classifier_two_variables():

    parent_dir = pathlib.Path(__file__).parent.absolute()
    df = pd.read_csv(os.path.join(parent_dir, 'house_price.txt'), sep=',')

    df.loc[:, ('y1')] = (df['price'] > df['price'].median()).astype(int)
    df = df.drop(['index', 'price', 'sq_price'], axis = 1)

    df_low = df.loc[df['y1'] == 0][['area', 'bathrooms', 'y1']]
    df_high = df.loc[df['y1'] == 1][['area', 'bathrooms', 'y1']]

    train_x, test_x, train_y, test_y = train_test_split(df[['area', 'bathrooms']], df['y1'], test_size = 0.25, random_state = 33)
    # random_state:  Pass an int for reproducible output across multiple function calls.

    ss = StandardScaler()
    train_x_std = ss.fit_transform(train_x)
    test_x_std = ss.transform(test_x)
    # These two statements are used to make sure that all data are in the same scale.

    rnd = random.randint(0, 2) # The following three statements are equivalent.
    if rnd == 0:
        lr = LogisticRegression(solver = 'liblinear').fit(train_x_std, train_y)
        # default solver is lbfgs and is used for multi-class classification.
    elif rnd == 1:
        lr = LogisticRegression(solver = 'liblinear', fit_intercept = True).fit(train_x_std, train_y)
    else:
        lr = LogisticRegression(solver = 'liblinear', penalty = 'l2', C = 1).fit(train_x_std, train_y)
        # Regularization basically adds the penalty as model complexity increases.
        # Difference between L1 and L2 regularization: L1 regularization tries to estimate the median of the data while the L2 regularization tries to estimate the mean of the data to avoid overfitting.

    plt.scatter(df_low['area'], df_low['bathrooms'], marker = 'o', s = 50, c = 'red', label = 'Low')
    plt.scatter(df_high['area'], df_high['bathrooms'], marker = 'x', s = 50, c = 'blue', label = 'High')

    plt.xlabel('area')
    plt.ylabel('bathrooms')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

    x_combined_std = np.vstack((train_x_std, test_x_std))
    y_combined = np.hstack((train_y, test_y))

    plot_decision_regions(X = x_combined_std, y = y_combined, classifier = lr, test_idx = range(35, 47))

    plt.xlabel('area [standardized]')
    plt.ylabel('bathrooms [standardized]')
    plt.legend(loc = 'upper left')
    plt.show()

    model_assessment_accuracy_precision_recall_f1(classifier = lr, test_y = test_y, test_x = test_x_std, verbosity = 3)
    model_assessment_roc_curve(classifier = lr, test_y = test_y, test_x = test_x_std)

def lec6_support_vector_machine():

    parent_dir = pathlib.Path(__file__).parent.absolute()
    df = pd.read_csv(os.path.join(parent_dir, 'house_price.txt'), sep=',')

    df.loc[:, ('y1')] = (df['price'] > df['price'].median()).astype(int)
    df = df.drop(['index', 'price', 'sq_price'], axis = 1)

    df_low = df.loc[df['y1'] == 0][['area', 'bathrooms', 'y1']]
    df_high = df.loc[df['y1'] == 1][['area', 'bathrooms', 'y1']]

    train_x, test_x, train_y, test_y = train_test_split(df[['area', 'bathrooms']], df['y1'], test_size = 0.25, random_state = 33)

    ss = StandardScaler()
    train_x_std = ss.fit_transform(train_x)
    test_x_std = ss.transform(test_x)

    rnd = random.randint(0, 2) # The following three statements are equivalent.
    if rnd == 0:
        lsvc = LinearSVC().fit(train_x_std, train_y)
    elif rnd == 1:
        lsvc = LinearSVC(C = 1).fit(train_x_std, train_y)
    else:
        lsvc = LinearSVC(penalty = 'l2', C = 1).fit(train_x_std, train_y)


    x_combined_std = np.vstack((train_x_std, test_x_std))
    y_combined = np.hstack((train_y, test_y))

    plot_decision_regions(X = x_combined_std, y = y_combined, classifier = lsvc, test_idx = range(35, 47))

    plt.xlabel('area [standardized]')
    plt.ylabel('bathrooms [standardized]')
    plt.legend(loc = 'upper left')
    plt.show()

    model_assessment_accuracy_precision_recall_f1(classifier = lsvc, test_y = test_y, test_x = test_x_std, verbosity = 3)

def main():

    print('\n\n===== Lec6: House Price Classifier with One Variable =====')
    lec6_house_price_classifier_one_variable()

    print('\n\n===== Lec6: House Price Classifier with Two Variables =====')
    lec6_house_price_classifier_two_variables()

    print('\n\n===== Lec6: Support Vector Machine with Two Variables =====')
    lec6_support_vector_machine()

if __name__ == '__main__':

    main()