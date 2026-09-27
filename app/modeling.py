import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.dummy import DummyRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (roc_auc_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from lightgbm import LGBMClassifier, LGBMRegressor

# LightGBM logs from its C++ core, not through python's logging module
# verbose=-1 on the estimator is the only thing that silences the training.
LGBM_Q = {'verbose': -1}

# the pipeline hands LightGBM a plain array, so this warning is noise
warnings.filterwarnings('ignore', message='X does not have valid feature names')


def _grid_size(params):
    total = 1
    for values in params.values():
        if not hasattr(values, '__len__'):
            return None
        total *= len(values)
    return total


def build_pipeline(model, num_cols, cat_cols):
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('scaler',  StandardScaler())
    ])
    prep = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
    return Pipeline([('prep', prep), ('model', model)])


def make_search(pipeline, params, scoring, X_train, y_train, X_valid, y_valid,
                n_iter, random_state):
    # refit=False, the winner is refitted on train only so valid stays clean for calibration.
    X = pd.concat([X_train, X_valid])
    y = pd.concat([y_train, y_valid])
    folds = PredefinedSplit(np.r_[np.full(len(X_train), -1), np.zeros(len(X_valid))])

    size = _grid_size(params)
    search = RandomizedSearchCV( # instead of gridsearch to minimise memory usage
        pipeline, params,
        n_iter=n_iter if size is None else min(n_iter, size),
        scoring=scoring, cv=folds, refit=False, n_jobs=1,
        random_state=random_state
    )
    search.fit(X, y)

    best = clone(pipeline).set_params(**search.best_params_).fit(X_train, y_train)
    return best, search.best_score_, search.best_params_


class ClassificationTrainer:
    def __init__(self, num_cols, cat_cols, random_state=42):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.random_state = random_state
        self.results = {}
        self.best_model = None
        self.best_name = None
        self.calibrated_model = None

    def base_models(self):
        return {
            'Logistic Regression': LogisticRegression(max_iter=2000, class_weight='balanced'),
            'LightGBM': LGBMClassifier(random_state=self.random_state, n_jobs=-1,
                                       class_weight='balanced', **LGBM_Q)
        }

    def hyperparameter_search(self, X_train, y_train, X_valid, y_valid, n_iter=10):
        param_grids = {
            'Logistic Regression': {
                'model__C': [0.001, 0.01, 0.1, 1, 10]
            },
            'LightGBM': {
                'model__num_leaves':        [15, 31, 63],
                'model__learning_rate':     [0.03, 0.05, 0.1],
                'model__n_estimators':      [200, 400],
                'model__min_child_samples': [50, 200]
            }
        }

        for name, model in self.base_models().items():
            pipe = build_pipeline(model, self.num_cols, self.cat_cols)
            best, score, params = make_search(pipe, param_grids[name], 'roc_auc',
                                              X_train, y_train, X_valid, y_valid,
                                              n_iter, self.random_state)
            self.results[name] = {
                'best_estimator': best,
                'valid_auc':      score,
                'best_params':    {k.replace('model__', ''): v for k, v in params.items()}
            }
            print(f"[{name}] Valid AUC: {score:.4f} | {self.results[name]['best_params']}")

        self.best_name  = max(self.results, key=lambda n: self.results[n]['valid_auc'])
        self.best_model = self.results[self.best_name]['best_estimator']
        print(f"Best model: {self.best_name}")
        return self.results

    def calibrate(self, X_valid, y_valid):
        self.calibrated_model = CalibratedClassifierCV(
            FrozenEstimator(self.best_model), method='sigmoid'
        ).fit(X_valid, y_valid)
        return self.calibrated_model

    def evaluate(self, X_test, y_test):
        return {name: roc_auc_score(y_test, res['best_estimator'].predict_proba(X_test)[:, 1])
                for name, res in self.results.items()}


class RegressionTrainer:
    def __init__(self, num_cols, cat_cols, random_state=42):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.random_state = random_state
        self.results = {}
        self.best_model = None
        self.best_name = None

    def base_models(self):
        return {
            'Mean baseline': DummyRegressor(strategy='mean'),
            'Linear Regression': LinearRegression(),
            # cross_entropy accepts labels in [0, 1]
            'LightGBM': LGBMRegressor(objective='cross_entropy', random_state=self.random_state,
                                               n_jobs=-1, **LGBM_Q)
        }

    def fit(self, X_train, y_train, X_valid, y_valid, params_lgbm, n_iter=10):
        # RMSE, not MAE: expected loss needs the mean LGD, not the median
        for name, model in self.base_models().items():
            pipe = build_pipeline(model, self.num_cols, self.cat_cols)
            if name == 'LightGBM':
                best, _, params = make_search(pipe, params_lgbm, 'neg_root_mean_squared_error',
                                              X_train, y_train, X_valid, y_valid,
                                              n_iter, self.random_state)
            else:
                best, params = pipe.fit(X_train, y_train), {}

            y_pred = np.clip(best.predict(X_valid), 0, 1)
            self.results[name] = {
                'best_estimator': best,
                'valid_rmse':     mean_squared_error(y_valid, y_pred) ** 0.5,
                'best_params':    {k.replace('model__', ''): v for k, v in params.items()}
            }
            print(f"[{name}] Valid RMSE: {self.results[name]['valid_rmse']:.4f} | {self.results[name]['best_params']}")

        self.best_name  = min(self.results, key=lambda n: self.results[n]['valid_rmse'])
        self.best_model = self.results[self.best_name]['best_estimator']
        print(f"Best regression model: {self.best_name}")
        return self.results

    def evaluate(self, X_test, y_test):
        rows = []
        for name, res in self.results.items():
            y_pred = np.clip(res['best_estimator'].predict(X_test), 0, 1)
            rows.append({
                'model':     name,
                'mae':       mean_absolute_error(y_test, y_pred),
                'rmse':      mean_squared_error(y_test, y_pred) ** 0.5,
                'r2':        r2_score(y_test, y_pred),
                'mean_pred': y_pred.mean()
            })
        return pd.DataFrame(rows).sort_values('rmse')
