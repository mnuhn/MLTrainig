#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf

import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)
import sys

#tf.logging.set_verbosity(tf.logging.INFO)

# Define linear regression routine
def linreg(dataset, splitpercentage, FEATURES, LABEL):

    # option to only use fraction of shuffled entries 
    fracuse = 1.0
    dataset = dataset.sample(frac=fracuse)
    num_rows = dataset.shape[0]
    
     # split full_set into training set and test set
    split_testtrain = int(splitpercentage*num_rows)
    training_set = full_set[:split_testtrain].reset_index(drop=True)
    test_set = full_set[split_testtrain:].reset_index(drop=True)

    # Feature cols
    feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in FEATURES]

    # Build the Estimator for linear regression
    model = tf.estimator.LinearRegressor(feature_columns=feature_cols)
    # Build a DNNRegressor, with 2x"nodes"-unit hidden layers, with the feature columns
    # defined above as input.
    #nodes = 200
    #model = tf.estimator.DNNRegressor(
    #    hidden_units=[nodes, nodes], feature_columns=feature_cols)

    # Train the model.
    # By default, the Estimators log output every 100 steps.
    #model.train(input_fn=lambda: input_fn(training_set, FEATURES, LABEL), steps=10000)
    model.train(input_fn=lambda: input_fn(training_set, FEATURES, LABEL), steps=100)
    
    return [0,1,0]


# Define input data sets for train and test data
def input_fn(data_set, features, label):
    feature_cols = {k: tf.constant(data_set[k].values) for k in features}
    labels = tf.constant(data_set[label].values)
    return feature_cols, labels

# Define input data sets for prediction data
def input_fn_pred(data_set, features):
    feature_cols = {k: tf.constant(data_set[k].values) for k in features}
    return feature_cols


def main(unused_argv):

    # Load datasets
    athenafile = "/home/rummelm/local_files/resid_Athena_2.csv"
    full_set = pd.read_csv(athenafile, skiprows=1)
    num_resid = full_set.shape[1]-1

    # Read out feature columns from csv file
    COLUMNS = []
    for i in range(num_resid):
        COLUMNS.append("Resid"+str(i+1))
    FEATURES = COLUMNS
    COLUMNS.append("gval")
    FEATURES = FEATURES[:-1]
    LABEL = "gval"

    # Define full set
    full_set = pd.read_csv(athenafile, skiprows=1, names=COLUMNS)
    # option to only use fraction of shuffled entries 
    fracuse = 1.0
    full_set = full_set.sample(frac=fracuse)
    num_rows = full_set.shape[0]
    #print(full_set)
    #sys.exit(0)

    # split full_set into training set and test set
    split_testtrain = int(0.75*num_rows)
    training_set = full_set[:split_testtrain].reset_index(drop=True)
    test_set = full_set[split_testtrain:].reset_index(drop=True)

    # Feature cols
    feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in FEATURES]

    # Build the Estimator for linear regression
    #model = tf.estimator.LinearRegressor(feature_columns=feature_cols)
    # Build a DNNRegressor, with 2x"nodes"-unit hidden layers, with the feature columns
    # defined above as input.
    nodes = 200
    model = tf.estimator.DNNRegressor(
        hidden_units=[nodes, nodes], feature_columns=feature_cols)

    # Train the model.
    # By default, the Estimators log output every 100 steps.
    model.train(input_fn=lambda: input_fn(training_set, FEATURES, LABEL), steps=10000)
    #model.train(input_fn=lambda: input_fn(training_set, FEATURES, LABEL), steps=100)
    
    # Evaluate how the model performs on data it has not yet seen.
    eval_result = model.evaluate(input_fn=lambda: input_fn(test_set, FEATURES, LABEL), steps=1)
    #sys.exit(0)

    # test predictions more detailed: save predicted values for g
    # all actual values of the label g in the test set
    gval_testvalues = test_set.drop(COLUMNS[:-1], axis=1)
    # test set without labels
    test_set_nolabel = test_set.drop(columns='gval')
    # length of test set
    ntest = len(test_set_nolabel)
    # predictions for g from the above
    gval_testpredictions = model.predict(input_fn=lambda: input_fn_pred(test_set_nolabel, FEATURES))
    gval_testpredictionsislice = list(itertools.islice(gval_testpredictions, ntest))
    # store in list
    gval_testpredlist = []
    for entry in gval_testpredictionsislice:
        gval_testpredlist.append(entry["predictions"][0])

    # Find what values of g are being used
    gvals = []
    for i in range(len(gval_testvalues)):
        gval = gval_testvalues.loc[i][0]
        if not gval in gvals:
            gvals.append(gval)
    gvals = np.sort(gvals)

    # calculate loss by hand
    losshand = 0.
    for i in range(ntest):
        losshand += (gval_testvalues.loc[i][0]-gval_testpredlist[i])**2
    print("loss calculated by hand: "+str(losshand))

    # Produce histograms and percentiles for predicted g values for each actual value of g
    percentiles = []
    for g in gvals:
        gpreds = []
        for i in range(ntest):
            if gval_testvalues.loc[i][0] == g:
                gpreds.append(gval_testpredlist[i])
        # make histograms
        plt.hist(gpreds, bins='auto')
        plt.title("g = "+str(g)+"e-13")
        plt.savefig("histo_Athena_"+str(g)+".pdf", bbox_inches='tight')
        plt.clf()
        # calculate percentiles
        if g == 0.:
            gperc = np.percentile(gpreds,95)
        else:
            gperc = np.percentile(gpreds,5)
        percentiles.append(gperc)
        print([g,gperc])

    # Plot g vs 5% of DNN predicted values for g
    fig, ax1 = plt.subplots()
    #plt.rc('text', usetex=True)
    ax1.loglog(gvals[1:], percentiles[1:], 'bo', [0.5,1500])
    ax1.set_xlim(0.05,150.)
    #ax1.set_ylim(0.1,200)
    ax1.axhline(y=percentiles[0], color='red', linestyle='--')
    ax1.set_xlabel(r'$g\, [10^{-13}\, {\rm GeV}^{-1}]$', fontsize=16)
    ax1.set_ylabel(r'$G_{\rm DNN}(\mathcal{F}_{i,{\rm sim}}\,|\,g)$', fontsize=18)
    ax1.annotate(r'$95\% \, {\rm of}\, G_{\rm DNN}(\mathcal{F}_{i,{\rm sim}}\,|\,g=0)$', xy=(0.06, 1.1*percentiles[0]), fontsize=12, color='red')
    ax2 = plt.axes([0, 0, 1, 1])
    ip = InsetPosition(ax1, [0.1,0.45,0.45,0.5])
    ax2.set_axes_locator(ip)
    mark_inset(ax1, ax2, loc1=2, loc2=1, fc="none", ec='0.5')
    ax2.loglog(gvals[1:], percentiles[1:], 'bo', [0.5,1500])
    ax2.axhline(y=percentiles[0], color='red', linestyle='--')
    ax2.set_xlim(2.5,6.5)
    ax2.set_ylim(0.3,5.3)
    fig.savefig("Athena_05percentile.pdf", bbox_inches='tight')


    """fplot = plt.figure()
    plt.rc('text', usetex=True)
    plt.xlim((0.05,150.))
    #plt.ylim((900,110000))
    plt.loglog(gvals[1:], percentiles[1:], 'bo', [0.5,1500])
    plt.axhline(y=percentiles[0], color='red', linestyle='--')
    plt.xlabel(r'$g\, [10^{-13}\, {\rm GeV}^{-1}]$', fontsize=18)
    plt.ylabel(r'$5\%\, {\rm of DNN Predictions for }\, g$', fontsize=18)
    plt.annotate(r'$95\% \, {\rm of DNN predictions for} \, g=0$', xy=(0.1, 1.1*percentiles[0]), fontsize=14, color='red')
    fplot.savefig("Athena_05percentile.pdf", bbox_inches='tight')"""

"""if __name__ == "__main__":
    tf.app.run()"""
