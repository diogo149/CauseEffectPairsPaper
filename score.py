#!/bin/env python
import sys
import os

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor

from boomlet.storage import joblib_load
from boomlet.transform.preprocessing import PercentileScaler
from autocause.challenge import target_score


CONFIGS_FOLDER = 'configs'
RESULTS_FILE = "results.txt"
CV_PERCENT = 0.2
y = None


def score(data, clf):
    cv_size = int(y.shape[0] * CV_PERCENT)
    train, valid = data[:-cv_size], data[-cv_size:]
    # reads in nonlocal variable y
    y_train, y_valid = y[:-cv_size], y[-cv_size:]

    clf.fit(train, y_train)
    pred = clf.predict(valid)
    return target_score(y_valid, pred)

def gbm_score(data):
    gbm = GradientBoostingRegressor(max_depth=2, max_features='sqrt', random_state=0)
    return score(data, gbm)

def sgd_score(data):
    # huber loss to prevent over/under-flow
    sgd = SGDRegressor(loss='huber', random_state=0, n_iter=20, shuffle=True)
    # squashing to prevent very high values from taking over
    return score(PercentileScaler(squash=True).fit_transform(data), sgd)


def write_score(filename):
    assert target.endswith(".py.pkl")
    assert target.startswith(CONFIGS_FOLDER)

    # initializing global y
    global y
    if y is None:
        y = joblib_load("y.pkl")

    data = joblib_load(target)
    assert data.shape[0] == y.shape[0]

    gbm = gbm_score(data)
    sgd = sgd_score(data)

    results = "\t".join(map(str, [target, data.shape[1], gbm, sgd]))
    print(results)
    with open(RESULTS_FILE, 'a') as outfile:
        outfile.write(results)
        outfile.write('\n')


def write_scores():
    configs = os.listdir(CONFIGS_FOLDER)
    pickles = filter(lambda x: x.endswith(".py.pkl"), configs)
    pickle_paths = [os.path.join(CONFIGS_FOLDER, x) for x in pickles]
    with open(RESULTS_FILE) as infile:
        lines = infile.readlines()
    old_paths = set([x.split('\t')[0] for x in lines])
    new_paths = filter(lambda x: x not in old_paths, pickle_paths)
    map(write_score, new_paths)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
        write_score(target)
    else:
        write_scores()
