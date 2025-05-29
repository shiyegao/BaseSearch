# AdaBoost config

# model hyperparameters
n_estimators = [1, 2, 5, 10, 20, 50, 100, 200]  # number of weak learners
learning_rate = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]  # learning rate
base_regressor_max_depth = [1, 2, 3, 5, 7, 10]  # the max depth of the decision tree

name = "adaboost_reg"
seed = 42

batch_size = 32
