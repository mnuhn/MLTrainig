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
    training_set = dataset[:split_testtrain].reset_index(drop=True)
    test_set = dataset[split_testtrain:].reset_index(drop=True)

    # Feature cols
    feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in FEATURES]

    # Build the Estimator for linear regression
    #model = tf.estimator.LinearRegressor(feature_columns=feature_cols)
    # Build a DNNRegressor, with 2x"nodes"-unit hidden layers, with the feature columns
    # defined above as input.
    nodes = 2
    model = tf.estimator.DNNRegressor(
        hidden_units=[nodes, nodes], feature_columns=feature_cols)

    # Train the model.
    # By default, the Estimators log output every 100 steps.
    #model.train(input_fn=lambda: input_fn(training_set, FEATURES, LABEL), steps=10000)
    model.train(input_fn=lambda: input_fn(training_set, FEATURES, LABEL), steps=100)
    
    #return [0,1,0]


# Define input data sets for train and test data
def input_fn(data_set, features, label):
    feature_cols = {k: tf.constant(data_set[k].values) for k in features}
    labels = tf.constant(data_set[label].values)
    return feature_cols, labels

# Define input data sets for prediction data
def input_fn_pred(data_set, features):
    feature_cols = {k: tf.constant(data_set[k].values) for k in features}
    return feature_cols


