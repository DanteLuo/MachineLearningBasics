from hyperopt import hp, tpe, fmin, STATUS_OK, Trials, space_eval
from random_forest_classifier import StockRandomForest
import time
import numpy as np


def objective(args):
    test_error = StockRandomForest(args)

    return {'loss':test_error,
            'status': STATUS_OK,
            'eval_time': time.time()}


def optimize():
    space = \
        {
            'max_depth': hp.choice('max_depth',np.arange(200,400+1,dtype=int)),
            'n_estimators': hp.choice('n_estimators',np.arange(20,40+1,dtype=int))
        }

    trials = Trials()
    best_model = fmin(objective,space,algo=tpe.suggest,max_evals=150,trials=trials)

    # print(best_model)
    print(space_eval(space,best_model))

    return space_eval(space,best_model)


if __name__ == '__main__':
    best_model = optimize()
    StockRandomForest(best_model,verbose=True)