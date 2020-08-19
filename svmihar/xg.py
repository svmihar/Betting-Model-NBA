import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import optuna
from data import X_train, X_test, y_train, y_test


def objective(trial):
    dtrain = xgb.DMatrix(X_train.values, label=y_train.values)
    dvalid = xgb.DMatrix(X_test.values, label=y_test.values)

    param = {
        "objective": trial.suggest_categorical(
            "objective", ["reg:squaredlogerror", "reg:squarederror"]
        ),
        "eval_metric": "mae",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
    }
    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 50)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        )
    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical(
            "sample_type", ["uniform", "weighted"]
        )
        param["normalize_type"] = trial.suggest_categorical(
            "normalize_type", ["tree", "forest"]
        )
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
    model = xgb.train(param, dtrain)
    preds = model.predict(dvalid)
    pred_labels = np.rint(preds)
    accuracy = mean_absolute_error(y_test, pred_labels)
    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10000)
    print(study.best_trial)
