import pandas as pd
from sklearn.metrics import accuracy_score
from xg import X_train, X_test, y_train, y_test
import joblib
import numpy as np
from pathlib import Path
from optuna.integration.lightgbm import lgb

if __name__ == "__main__":
    import pdb

    pdb.set_trace()
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_test, label=y_test)

    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "verbosity": 2,
        "boosting_type": "dart",
    }

    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dval],
        verbose_eval=100,
        early_stopping_rounds=100,
    )
    preds = np.rint(model.predict(X_test, num_iteration=model.best_iteration))

    accuracy = accuracy_score(y_test, preds)
    joblib.dump(model, "lgbm_optmized.pkl")
    print("model dumped, and ready to predict")
