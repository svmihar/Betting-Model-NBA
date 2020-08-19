from sklearn.metrics import accuracy_score
import numpy as np
import joblib
from optuna.integration.lightgbm import lgb
from data import X_train, X_test, y_train, y_test

if __name__ == "__main__":
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
