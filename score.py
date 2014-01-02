from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor

from boomlet.storage import joblib_load
from boomlet.transform.preprocessing import PercentileScaler
from autocause.challenge import target_score


CV_PERCENT = 0.2


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


if __name__ == "__main__":
    import sys
    y = joblib_load("y.pkl")
    target = sys.argv[1]
    data = joblib_load(target)
    assert data.shape[0] == y.shape[0]

    gbm = gbm_score(data)
    sgd = sgd_score(data)

    print("\t".join(map(str, [target, data.shape[1], gbm, sgd])))
