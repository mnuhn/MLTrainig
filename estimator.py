from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)
import sys

#tf.logging.set_verbosity(tf.logging.INFO)

# Define linear regression routine: use architecture that is easily switched to
# DNN (same layout that has been used for axion stuff). Not very suited though
# to extracting standard info like bias and slope
def linreg_rough(dataset, splitpercentage, FEATURES, LABEL):

    # Option to only use fraction of shuffled entries.
    fracuse = 1.0
    dataset = dataset.sample(frac=fracuse)
    num_rows = dataset.shape[0]
    
    # Split full_set into training set and test set.
    split_testtrain = int(splitpercentage*num_rows)
    training_set = dataset[:split_testtrain].reset_index(drop=True)
    test_set = dataset[split_testtrain:].reset_index(drop=True)

    # Feature columns:
    feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in FEATURES]
    
    # Build the estimator for linear regression.
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

    # Evaluate how the model performs on data it has not yet seen.
    eval_result = model.evaluate(input_fn=lambda: input_fn(test_set, FEATURES, LABEL), steps=1)

    # Get variable names and values.
    varnames = model.get_variable_names()

    return eval_result

# More pedestrian linear regression that gives more detailed information.
def linreg_detailed(dataset, splitpercentage, FEATURE, LABEL):

    # Parameters:
    rng = np.random
    learning_rate = 0.01
    training_epochs = 1000
    display_step = 50

    # For now select only one FEATURE column as x-axis and LABEL as y-axis.
    train_X = np.asarray(dataset[FEATURE].values)
    train_Y = np.asarray(dataset[LABEL].values)

    # Normalize to zero mean and standard deviation one.
    train_X = normalize_array(train_X)
    train_Y = normalize_array(train_Y)
    n_samples = train_X.shape[0]
    
    # Tensorflow graph inputs.
    X = tf.placeholder("float")
    Y = tf.placeholder("float")

    # Set model weights.
    W = tf.Variable(rng.randn(), name="weight")
    b = tf.Variable(rng.randn(), name="bias")

    # Construct a linear model.
    pred = tf.add(tf.multiply(X, W), b)

    # Mean squared error
    cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
    # Gradient descent
    #  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()


    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        # Fit all training data
        for epoch in range(training_epochs):
            for (x, y) in zip(train_X, train_Y):
                sess.run(optimizer, feed_dict={X: x, Y: y})

            # Display logs per epoch step
            if (epoch+1) % display_step == 0:
                c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                    "W=", sess.run(W), "b=", sess.run(b))

        print("Optimization Finished!")
        training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
        print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

        # Graphic display
        plt.plot(train_X, train_Y, 'ro', label='Original data')
        plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
        plt.legend()
        plt.show()


# More pedestrian linear regression that gives more detailed information.
def linreg_multidim(dataset, splitpercentage, FEATURES, LABEL):

    # Parameters:
    rng = np.random
    learning_rate = 0.3
    training_epochs = 1000
    display_step = 50

    # For now select only one FEATURE column as x-axis and LABEL as y-axis.
    train_X = np.asarray(dataset.drop(columns=LABEL).values)
    train_Y = np.asarray(dataset[LABEL].values)
    n_samples = train_X.shape[0]
    n_features = train_X.shape[1]

    # Normalize input and output to zero mean and standard deviation one.
    # Important to normalize column by column!
    for i in range(n_features):
        norm_column = normalize_array(train_X[:,i])
        # build normalized matrix up column by column
        if i==0:
            train_X_norm = norm_column
        else:
            train_X_norm = np.column_stack((train_X_norm,norm_column))
    train_X = train_X_norm
    train_Y = normalize_array(train_Y)

    # Bring train_y into column vector form.
    train_Y = train_Y.reshape(-1,1)

    # Tensorflow graph input.
    X = tf.placeholder(tf.float32, [None, n_features], "X_in")
    Y = tf.placeholder(tf.float32, [None, 1], "y_in")

    # Set model weights.
    W = tf.Variable(tf.random_normal((n_features, 1)), name="weights")
    b = tf.Variable(tf.constant(0.1, shape=[]), name="bias")

    # Construct a linear model, leave out bias for normalized data.
    # TODO(mrumm): What do you want to do.
    pred = tf.add(tf.matmul(X, W), b, name="pred")
    #pred = tf.matmul(X, W, name="pred")

    # Mean squared error
    cost = tf.reduce_mean(tf.square(tf.subtract(Y, pred)), name="loss")
    # Gradient descent. Note: minimize() knows to modify W and b because
    # Variable objects are trainable=True by default.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # Fit all training data.
        for epoch in range(training_epochs):
            sess.run(optimizer, feed_dict={
              X: train_X,
              Y: train_Y
            })
            W_computed = sess.run(W)
            b_computed = sess.run(b)

            # Display logs per every new epoch.
            if (epoch+1) % display_step == 0:
                c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                    "W = [%s]" % ', '.join(['%.5f' % x for x in W_computed.flatten()]), "b=", b_computed)

        print("Optimization Finished!")
        training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
        print("Training cost=", training_cost, \
                "W = [%s]" % ', '.join(['%.5f' % x for x in W_computed.flatten()]), "b=", b_computed, '\n')

        # Graphic display
        for ifeat in range(n_features):
            plt.plot(train_X[:,ifeat], train_Y, 'ro', label='Original data '+FEATURES[ifeat])
            plt.plot(train_X[:,ifeat], W_computed[ifeat] * train_X[:,ifeat] + b_computed, label='Fitted line')
            plt.legend()
            plt.show()

   
# Define input data sets for train and test data.
def input_fn(data_set, features, label):
    feature_cols = {k: tf.constant(data_set[k].values) for k in features}
    labels = tf.constant(data_set[label].values)
    return feature_cols, labels

# Define input data sets for prediction data.
def input_fn_pred(data_set, features):
    feature_cols = {k: tf.constant(data_set[k].values) for k in features}
    return feature_cols

# Normalize array to mean zero and standard deviation one.
def normalize_array(A):
    A = np.array(A)
    mu = np.mean(A, axis=0)
    sigma = np.std(A, axis=0)
    return (A - mu)/sigma
