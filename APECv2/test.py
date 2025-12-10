import optuna

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    print(x)
    x = trial.suggest_float('x', -10, 10)
    print(x)
    print("next")
    out_features = trial.suggest_int("head_units_{}".format(2), 4, 22)
    print(out_features)
    out_features = trial.suggest_int("head_units_{}".format(2), 4, 22)
    print(out_features)

    return (x - 2) ** 2

study = optuna.create_study()
study.optimize(objective, n_trials=100)

study.best_params  # E.g. {'x': 2.002108042}