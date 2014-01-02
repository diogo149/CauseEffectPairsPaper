from time import time

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor

from boomlet.storage import joblib_load
from boomlet.utils import timer
from boomlet.transform.preprocessing import PercentileScaler
from autocause.challenge import target_score


CV_PERCENT = 0.2

old = joblib_load("old.pkl")
new = joblib_load("new.pkl")
y = joblib_load("y.pkl")


def score(data, clf):
    start = time()

    cv_size = int(y.shape[0] * CV_PERCENT)
    train, valid = data[:-cv_size], data[-cv_size:]
    y_train, y_valid = y[:-cv_size], y[-cv_size:]

    with timer("fit"):
        clf.fit(train, y_train)
    pred = clf.predict(valid)
    print(target_score(y_valid, pred))

def gbm_score(data):
    gbm = GradientBoostingRegressor(max_depth=2, max_features='sqrt', random_state=0)
    score(data, gbm)

def sgd_score(data):
    # huber loss to prevent over/under-flow
    sgd = SGDRegressor(loss='huber', random_state=0, n_iter=20, shuffle=True)
    # squashing to prevent very high values from taking over
    score(PercentileScaler(squash=True).fit_transform(data), sgd)
