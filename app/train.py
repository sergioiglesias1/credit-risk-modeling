import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.metrics import roc_auc_score
from lightgbm import __version__ as lightgbm_version

from .config import (PD_MODEL_PATH, LGD_MODEL_PATH, METRICS_PATH, FIGURES_DIR,
                     RANDOM_STATE, N_ITER, THRESHOLDS, MARGIN, MARGIN_GRID,
                     NUM_FEATURES, CAT_FEATURES, APP_NUM_FEATURES, APP_CAT_FEATURES,
                     APP_FEATURES)
from .data import load_processed
from .modeling import ClassificationTrainer, RegressionTrainer
from .plots import Visualizer
from .utils import pd_metrics, decile_table, ThresholdAnalyzer, ExpectedLossCalculator

PARAMS_LGBM_REG = {
    'model__num_leaves':        [15, 31],
    'model__learning_rate':     [0.03, 0.1],
    'model__n_estimators':      [100, 300],
    'model__min_child_samples': [50, 200]
}


def records(df):
    return json.loads(df.round(4).to_json(orient='records'))


def rounded(d):
    return {k: round(float(v), 4) for k, v in d.items()}


def main():
    try:
        df = load_processed()
    except FileNotFoundError:
        print("[ERROR] Processed data not found. Run `python -m app.data` first")
        return

    train, valid, test = (df[df['split'] == s] for s in ['train', 'valid', 'test'])
    viz = Visualizer(FIGURES_DIR)

    # PD: full feature set vs the application-time subset served by the API
    feature_sets = {
        'full': (NUM_FEATURES, CAT_FEATURES),
        'app':  (APP_NUM_FEATURES, APP_CAT_FEATURES)
    }
    pd_runs = {}
    for label, (num_cols, cat_cols) in feature_sets.items():
        print(f"\n=== PD model | {label} features ({len(num_cols) + len(cat_cols)})")
        cols = num_cols + cat_cols
        trainer = ClassificationTrainer(num_cols, cat_cols, random_state=RANDOM_STATE)
        trainer.hyperparameter_search(train[cols], train['default'],
                                      valid[cols], valid['default'], n_iter=N_ITER)
        model = trainer.calibrate(valid[cols], valid['default'])
        proba = model.predict_proba(test[cols])[:, 1]
        test_auc = trainer.evaluate(test[cols], test['default'])

        pd_runs[label] = {'trainer': trainer, 'model': model, 'proba': proba,
                          'metrics': pd_metrics(test['default'], proba), 'test_auc': test_auc}
        print(f"Test AUC (calibrated {trainer.best_name}): {pd_runs[label]['metrics']['auc']:.4f}")

    pd_app  = pd_runs['app']
    proba   = pd_app['proba']
    y_test  = test['default'].values

    # Benchmark: Lending Club's own sub-grade (A1 < A2 < ... < G5) used directly as a score
    grade_auc = roc_auc_score(y_test, test['grade'].rank(method='dense'))
    print(f"Sub-grade only AUC: {grade_auc:.4f}")

    # LGD: defaulted loans only, same application-time features
    train_d, valid_d, test_d = (s[s['lgd'].notna()] for s in [train, valid, test])
    print("\n=== LGD model")
    reg_trainer = RegressionTrainer(APP_NUM_FEATURES, APP_CAT_FEATURES, random_state=RANDOM_STATE)
    reg_trainer.fit(train_d[APP_FEATURES], train_d['lgd'], valid_d[APP_FEATURES], valid_d['lgd'],
                    PARAMS_LGBM_REG, n_iter=N_ITER)
    df_lgd = reg_trainer.evaluate(test_d[APP_FEATURES], test_d['lgd'])
    print(df_lgd.to_string(index=False))

    # EAD = CCF * funded, CCF estimated on training defaults
    train_def = train[train['default'] == 1]
    ccf = float((train_def['ead_default'] / train_def['funded_amnt']).mean())
    lgd_mean = float(train_d['lgd'].mean())

    # Expected loss on the 2015 test portfolio vs what was actually lost
    lgd_pred = np.clip(reg_trainer.best_model.predict(test[APP_FEATURES]), 0, 1)
    elc = ExpectedLossCalculator(ccf)
    result = elc.compute(proba, lgd_pred, test['funded_amnt'].values)
    realized = elc.realized_loss(y_test, test['ead_default'].values, test['recoveries'].values)
    elc.print_summary(result)
    print(f"  Realized loss         : ${realized.sum():>15,.2f} ({realized.sum() / result['portfolio']:.2%})")

    # Decision policy
    ta = ThresholdAnalyzer(THRESHOLDS)
    t_star = ta.optimal_threshold(MARGIN, lgd_mean, ccf)
    df_thresh = ta.sweep(y_test, proba)
    df_margin = ta.margin_sensitivity(y_test, proba, MARGIN_GRID, lgd_mean, ccf)
    policy = ta.evaluate(y_test, proba, t_star)
    print(f"\nt* = {t_star:.4f} (margin={MARGIN}, LGD={lgd_mean:.3f}, CCF={ccf:.3f})")
    print(df_margin.round(4).to_string(index=False))

    # Decile tables
    dec = decile_table(proba, default=y_test, el=result['el_per_loan'],
                       realized=realized, funded=test['funded_amnt'].values)
    by_decile = dec.groupby('decile').agg(
        loans=('default', 'size'),
        mean_pd=('proba', 'mean'),
        default_rate=('default', 'mean'),
        el=('el', 'sum'),
        realized=('realized', 'sum'),
        funded=('funded', 'sum')
    ).reset_index()
    by_decile['el_pct'] = by_decile['el'] / by_decile['funded']
    by_decile['realized_pct'] = by_decile['realized'] / by_decile['funded']

    # Charts
    full_auc = pd_runs['full']['metrics']['auc']
    viz.roc(y_test, {
        f"Deployed, {len(APP_FEATURES)} features (AUC {pd_app['metrics']['auc']:.3f})": proba,
        f"Full, {len(NUM_FEATURES) + len(CAT_FEATURES)} features (AUC {full_auc:.3f})": pd_runs['full']['proba']
    })
    viz.calibration(by_decile)
    viz.el_distribution(result['el_per_loan'] / test['funded_amnt'].values)
    viz.backtest(by_decile)

    # Save models
    os.makedirs(os.path.dirname(PD_MODEL_PATH), exist_ok=True)
    joblib.dump(pd_app['model'], PD_MODEL_PATH)
    joblib.dump(reg_trainer.best_model, LGD_MODEL_PATH)

    splits = df[df['split'].isin(['train', 'valid', 'test'])].groupby('split').agg(
        loans=('default', 'size'),
        default_rate=('default', 'mean'),
        first_issue=('issue_period', 'min'),
        last_issue=('issue_period', 'max')
    ).reindex(['train', 'valid', 'test']).reset_index()

    metrics = {
        "trained_on": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "dataset": "Lending Club loans 2007-2015, 36-month term",
        "splits": records(splits),
        "features": {
            "numeric": APP_NUM_FEATURES,
            "categorical": APP_CAT_FEATURES,
            "categories": {c: sorted(train[c].dropna().unique().tolist()) for c in APP_CAT_FEATURES}
        },
        "pd": {
            "model": pd_app['trainer'].best_name,
            "params": pd_app['trainer'].results[pd_app['trainer'].best_name]['best_params'],
            "calibration": "sigmoid on validation vintage (class_weight='balanced' in training)",
            "test": rounded(pd_app['metrics']),
            "full_model_test": rounded(pd_runs['full']['metrics']),
            "auc_cost_vs_full": round(full_auc - pd_app['metrics']['auc'], 4),
            "grade_only_auc": round(grade_auc, 4),
            "candidates": [
                {"feature_set": label, "model": name,
                 "valid_auc": round(res['valid_auc'], 4),
                 "test_auc": round(run['test_auc'][name], 4)}
                for label, run in pd_runs.items()
                for name, res in run['trainer'].results.items()
            ],
            "deciles": records(by_decile[['decile', 'loans', 'mean_pd', 'default_rate']])
        },
        "lgd": {
            "model": reg_trainer.best_name,
            "params": reg_trainer.results[reg_trainer.best_name]['best_params'],
            "definition": "1 - recoveries / EAD at default, charged-off loans, clipped to [0, 1]",
            "test_loans": int(len(test_d)),
            "test_mean_observed": round(float(test_d['lgd'].mean()), 4),
            "test": records(df_lgd)
        },
        "policy": {
            "margin": MARGIN,
            "lgd_mean_train": round(lgd_mean, 4),
            "ccf_train": round(ccf, 4),
            "threshold": round(t_star, 4),
            "at_threshold": rounded(policy),
            "threshold_table": records(df_thresh),
            "margin_sensitivity": records(df_margin)
        },
        "expected_loss": {
            "portfolio_funded": round(float(result['portfolio']), 2),
            "expected_loss": round(float(result['expected_loss']), 2),
            "expected_loss_pct": round(float(result['el_pct']), 4),
            "realized_loss": round(float(realized.sum()), 2),
            "realized_loss_pct": round(float(realized.sum() / result['portfolio']), 4),
            "by_decile": records(by_decile[['decile', 'el_pct', 'realized_pct']])
        },
        "versions": {"scikit-learn": sklearn_version, "lightgbm": lightgbm_version}
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nModels saved at {PD_MODEL_PATH}, {LGD_MODEL_PATH}")
    print(f"Metrics saved at {METRICS_PATH}")


if __name__ == "__main__":
    main()
