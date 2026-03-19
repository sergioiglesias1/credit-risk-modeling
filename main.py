import numpy as np
import json
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn import __version__

from modeling import ClassificationTrainer, RegressionTrainer
from visualization import Visualizer
from utils import ThresholdAnalyzer, ExpectedLossCalculator
warnings.filterwarnings('ignore')

DATA_PATH = "Data/cleaned_data.csv"
META_PATH = "Models/metadata.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
PD_THRESHOLD = 0.10

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
    
    print(f"\nThreshold Sweep: {best_name}")
    print(df_thresh.to_string(index=False))

    y_pred_thresh = (y_proba >= 0.10).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_thresh, average='binary', zero_division=0)
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

    lgd_best_model  = df_reg_final.iloc[0]['Modelo']
    lgd_best_params = reg_trainer.results[lgd_best_model]['best_params'] if lgd_best_model != 'Linear Regression' else {}
 
    metadata = {
        "pd_model":             best_name,
        "lgd_model":            lgd_best_model,
        "trained_on":           pd.Timestamp.today().strftime("%Y-%m"),
        "dataset":              "Lending Club Loan Data (Kaggle)",
        "pd_cv_auc":            round(clf_trainer.results[best_name]['best_score'], 4),
        "pd_best_params":       clf_trainer.results[best_name]['best_params'],
        "lgd_best_params":      lgd_best_params,
        "pd_threshold":         PD_THRESHOLD,
        "pd_test_precision":    round(precision, 3),
        "pd_test_recall":       round(recall, 3),
        "pd_test_f1":           round(f1, 3),
        "lgd_mae":              round(df_reg_final.iloc[0]['MAE'], 4),
        "lgd_rmse":             round(df_reg_final.iloc[0]['RMSE'], 4),
        "portfolio_value":      round(result['cartera_total'], 2),
        "expected_loss":        round(result['perdida_total'], 2),
        "expected_loss_pct":    round(result['pct_perdida'], 4),
        "positive_class":       1,
        "positive_label":       "default",
        "scikit_learn_version": __version__
    }
 
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved at {META_PATH}")

if __name__ == "__main__":
    main()
