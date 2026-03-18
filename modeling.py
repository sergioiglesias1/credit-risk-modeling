import logging
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             recall_score, precision_score, confusion_matrix,
                             classification_report, mean_absolute_error,
                             mean_squared_error, r2_score)
from lightgbm import LGBMClassifier, LGBMRegressor

logging.getLogger('lightgbm').setLevel(logging.WARNING)


class ClassificationTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = {}
        self.best_model = None
        self.best_name = None

    def base_models(self):
        return {
            'LightGBM': LGBMClassifier(random_state=self.random_state, n_jobs=-1),
            'Random Forest': RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=self.random_state)
        }

    def fit_base(self, X_train, y_train, X_test, y_test):
        rows = []

        for name, model in self.base_models().items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            rows.append({
                'Modelo': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1': f1_score(y_test, y_pred),
                'ROC_AUC': roc_auc_score(y_test, y_proba)
            })
        return pd.DataFrame(rows).sort_values('ROC_AUC', ascending=False)

    def hyperparameter_search(self, X_train, y_train):
        stf = StratifiedKFold(n_splits=2, shuffle=True, random_state=self.random_state)

        param_grids = {
            'LightGBM': {
                'max_depth':     [5, 8, 11],
                'learning_rate': [0.05, 0.1, 0.25],
                'n_estimators':  [100, 200]
            },
            'Logistic Regression': {
                'C':       [0.001, 0.01, 0.1, 1],
                'penalty': ['l1', 'l2'],
                'solver':  ['liblinear', 'saga']
            },
            'Random Forest': {
                'n_estimators':    [100, 300],
                'max_depth':       [None, 10],
                'min_samples_split': [2, 5],
                'min_samples_leaf':  [1, 3]
            },
            'Decision Tree': {
                'max_depth':         [None, 5, 10],
                'min_samples_split': [2, 5],
                'min_samples_leaf':  [1, 4]
            }
        }

        base_models = self.base_models()

        for name, params in param_grids.items():
            grid = GridSearchCV(
                base_models[name], params,
                scoring='roc_auc', cv=stf, n_jobs=-1, verbose=0
            )
            grid.fit(X_train, y_train)
            self.results[name] = {
                'best_estimator': grid.best_estimator_,
                'best_score':     grid.best_score_,
                'best_params':    grid.best_params_
            }
            print(f"[{name}] Best AUC: {grid.best_score_:.4f} | {grid.best_params_}")

        return self.results

    def evaluate(self, X_test, y_test):
        auc_results = {}
        for name, res in self.results.items():
            model   = res['best_estimator']
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred  = model.predict(X_test)
            auc     = roc_auc_score(y_test, y_proba)
            auc_results[name] = auc

            print(f"\n{'='*60}")
            print(f"MODEL: {name} (ROC-AUC: {auc:.4f})")
            print('='*60)
            print(classification_report(y_test, y_pred,
                                        target_names=['No Default', 'Defaulter']))
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            print(f"TN: {tn}  FP: {fp}  FN: {fn}  TP: {tp}")

        self.best_name  = max(auc_results, key=auc_results.get)
        self.best_model = self.results[self.best_name]['best_estimator']
        print(f"\nBest model: {self.best_name} (AUC={auc_results[self.best_name]:.4f})")
        return auc_results


class RegressionTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results      = {}
        self.best_model   = None

    def base_models(self):
        return {
            'Linear Regression': LinearRegression(),
            'Decision Tree':     DecisionTreeRegressor(random_state=self.random_state),
            'Random Forest':     RandomForestRegressor(random_state=self.random_state, n_jobs=-1),
            'LightGBM':          LGBMRegressor(random_state=self.random_state, n_jobs=-1)
        }

    def fit_base(self, X_train, y_train, X_test, y_test):
        rows = []
        for name, model in self.base_models().items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rows.append({
                'Modelo':   name,
                'MAE':      mean_absolute_error(y_test, y_pred),
                'RMSE':     mean_squared_error(y_test, y_pred) ** 0.5,
                'R2 Score': r2_score(y_test, y_pred)
            })
        return pd.DataFrame(rows).sort_values('MAE')

    def hyperparameter_search(self, X_train, y_train, params_lgbm, params_rf, params_dt):
        kf = KFold(n_splits=2, shuffle=True, random_state=self.random_state)

        searches = {
            'LightGBM':      (LGBMRegressor(random_state=self.random_state, n_jobs=-1), params_lgbm),
            'Random Forest': (RandomForestRegressor(random_state=self.random_state, n_jobs=-1), params_rf),
            'Decision Tree': (DecisionTreeRegressor(random_state=self.random_state), params_dt)
        }

        for name, (model, params) in searches.items():
            grid = GridSearchCV(
                model, params,
                scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1, verbose=0
            )
            grid.fit(X_train, y_train)
            self.results[name] = {
                'best_estimator': grid.best_estimator_,
                'best_score':     -grid.best_score_,
                'best_params':    grid.best_params_
            }
            print(f"[{name}] Best MAE: {-grid.best_score_:.4f} | {grid.best_params_}")

        return self.results

    def final_models(self, X_train, y_train, X_test, y_test):
        tuned = {
            'Linear Regression': LinearRegression(),
            'Decision Tree':     self.results['Decision Tree']['best_estimator'],
            'Random Forest':     self.results['Random Forest']['best_estimator'],
            'LightGBM':          self.results['LightGBM']['best_estimator']
        }
        rows = []
        for name, model in tuned.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rows.append({
                'Modelo':   name,
                'MAE':      mean_absolute_error(y_test, y_pred),
                'RMSE':     mean_squared_error(y_test, y_pred) ** 0.5,
                'R2 Score': r2_score(y_test, y_pred)
            })

        df_res = pd.DataFrame(rows).sort_values('MAE')
        self.best_model = tuned[df_res.iloc[0]['Modelo']]
        self.best_model.fit(X_train, y_train)
        print(f"\nBest regression model: {df_res.iloc[0]['Modelo']} "
              f"(MAE={df_res.iloc[0]['MAE']:.4f})")
        return df_res, tuned
