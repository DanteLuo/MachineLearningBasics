from hyperopt import hp, tpe, fmin, STATUS_OK, Trials, space_eval
from random_forest_classifier import StockRandomForest
import time
import numpy as np
import pickle


def objective(args):
    test_error, _ = StockRandomForest(args)

    return {'loss':test_error,
            'status': STATUS_OK,
            'eval_time': time.time()}


def optimize():
    space = \
        {
            'max_depth': hp.choice('max_depth',np.arange(350,400+1,dtype=int)),
            'n_estimators': hp.choice('n_estimators',np.arange(60,80+1,dtype=int)),
            'max_features': hp.choice('max_features',['sqrt','log2',None]),
            'tree_criterion': hp.choice('criterion',['gini','entropy'])
        }

    trials = Trials()
    best_model = fmin(objective,space,algo=tpe.suggest,max_evals=150,trials=trials)

    print(space_eval(space,best_model))

    return space_eval(space,best_model)


def optRF():
    best_model = optimize()
    StockRandomForest(best_model, verbose=True)
    pickle_out = open('dict.best_model','wb')
    pickle.dump(best_model,pickle_out)
    pickle_out.close()


if __name__ == '__main__':
    import timeit
    start_time = timeit.default_timer()
    optRF()
    print('The time of execution is',timeit.default_timer()-start_time)
