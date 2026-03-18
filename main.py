import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from modeling import ClassificationTrainer, RegressionTrainer
from visualization import Visualizer
from utils import ModelSaver, ThresholdAnalyzer, ExpectedLossCalculator

DATA_PATH = "Data/cleaned_data.csv"
CLF_MODEL_PATH = "Models/best_clf_model.pkl"
REG_MODEL_PATH = "Models/best_reg_model.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.2

PARAMS_LGBM = {
    'max_depth':     [5, 8, 11],
    'learning_rate': [0.05, 0.1, 0.25],
    'n_estimators':  [100, 200]
}
PARAMS_RF = {
    'n_estimators':      [100, 300],
    'max_depth':         [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf':  [1, 3]
}
PARAMS_DT = {
    'max_depth':         [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf':  [1, 4]
}

def main():
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {DATA_PATH}")
        print("Run EDA.ipynb first to generate cleaned_data.csv")
        return

    viz = Visualizer()

    y = df['PD']
    X = df.drop(columns=['PD', 'LGD', 'EAD']) # Leakage if only dropping PD

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    ssc = StandardScaler()
    X_train = ssc.fit_transform(X_train)
    X_test  = ssc.transform(X_test)

    viz.class_imbalance(df)
    viz.interest_rate_vs_default(df)

    clf_trainer = ClassificationTrainer(random_state=RANDOM_STATE)

    df_clf_base = clf_trainer.fit_base(X_train, y_train, X_test, y_test)
    print(df_clf_base.to_string(index=False))

    clf_trainer.hyperparameter_search(X_train, y_train)
    clf_trainer.evaluate(X_test, y_test)

    best_clf  = clf_trainer.best_model
    best_name = clf_trainer.best_name

    y_proba = best_clf.predict_proba(X_test)[:, 1]
    ta = ThresholdAnalyzer()
    df_thresh = ta.sweep(y_test, y_proba)
    print(f"\nThreshold Sweep — {best_name}")
    print(df_thresh.to_string(index=False))

    y_pred_thresh = (y_proba >= 0.10).astype(int)
    morosos_detectados = y_pred_thresh.sum()
    morosos_reales = y_test.sum()
    print(f"\nMorosos reales en test: {morosos_reales}")
    print(f"Morosos detectados (t=10): {morosos_detectados}")
    print(f"Recall: {morosos_detectados / morosos_reales:.2%}")

    df_reg = df[df['PD'] == 1].copy()
    y_reg = df_reg['LGD']
    X_reg = df_reg.drop(columns=['LGD', 'PD', 'EAD'])

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    reg_trainer = RegressionTrainer(random_state=RANDOM_STATE)

    df_reg_base = reg_trainer.fit_base(X_train_r, y_train_r, X_test_r, y_test_r)
    print(df_reg_base.to_string(index=False))

    reg_trainer.hyperparameter_search(X_train_r, y_train_r, PARAMS_LGBM, PARAMS_RF, PARAMS_DT)
    df_reg_final, _ = reg_trainer.final_models(X_train_r, y_train_r, X_test_r, y_test_r)
    print(df_reg_final.to_string(index=False))

    best_reg = reg_trainer.best_model

    _, X_test_final, _, y_test_final = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_test_final_scaled = ssc.transform(X_test_final)

    EAD_test = df.loc[y_test_final.index, 'EAD'].values
    PD_proba = best_clf.predict_proba(X_test_final_scaled)[:, 1]
    LGD_pred = best_reg.predict(X_test_final_scaled)

    elc = ExpectedLossCalculator()
    result = elc.compute(PD_proba, LGD_pred, EAD_test)
    elc.print_summary(result)

    ms = ModelSaver()
    ms.save_model(best_clf, CLF_MODEL_PATH)
    ms.save_model(best_reg, REG_MODEL_PATH)

if __name__ == "__main__":
    main()
