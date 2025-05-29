# rf config

# model hyperparameters
n_estimators = [1, 2, 3, 5, 7, 10, 20, 50, 100, 200]  # number of trees
max_depth = [
    1,
    2,
    3,
    5,
    7,
    10,
    15,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
]  # max depth of the trees

name = "rf_reg"
seed = 42

batch_size = 32
