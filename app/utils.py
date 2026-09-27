import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss


def ks_statistic(y_true, proba):
    fpr, tpr, _ = roc_curve(y_true, proba)
    return float(np.max(tpr - fpr))


def pd_metrics(y_true, proba):
    auc = roc_auc_score(y_true, proba)
    return {
        'auc':          auc,
        'gini':         2 * auc - 1,
        'ks':           ks_statistic(y_true, proba),
        'brier':        brier_score_loss(y_true, proba),
        'mean_pd':      float(np.mean(proba)),
        'default_rate': float(np.mean(y_true))
    }


def decile_table(proba, **columns):
    # decile 1 = lowest PD
    df = pd.DataFrame({'proba': proba, **columns})
    df['decile'] = pd.qcut(df['proba'].rank(method='first'), 10, labels=range(1, 11)).astype(int)
    return df


class ThresholdAnalyzer:
    def __init__(self, thresholds=None):
        self.thresholds = thresholds or [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    def evaluate(self, y_true, proba, t):
        # loans with PD >= t are rejected
        y_true   = np.asarray(y_true)
        approved = proba < t
        rejected = ~approved
        return {
            'threshold':         t,
            'approval_rate':     approved.mean(),
            'bad_rate_approved': y_true[approved].mean() if approved.any() else 0.0,
            'recall':            y_true[rejected].sum() / y_true.sum(),
            'precision':         y_true[rejected].mean() if rejected.any() else 0.0
        }

    def sweep(self, y_true, proba) -> pd.DataFrame:
        return pd.DataFrame([self.evaluate(y_true, proba, t) for t in self.thresholds])

    @staticmethod
    def optimal_threshold(margin, lgd, ccf):
        # approve if (1 - PD) * margin > PD * LGD * CCF
        return margin / (margin + lgd * ccf)

    def margin_sensitivity(self, y_true, proba, margins, lgd, ccf) -> pd.DataFrame:
        rows = []
        for m in margins:
            t = self.optimal_threshold(m, lgd, ccf)
            rows.append({'margin': m, **self.evaluate(y_true, proba, t)})
        return pd.DataFrame(rows)


class ExpectedLossCalculator:
    def __init__(self, ccf):
        self.ccf = ccf

    def compute(self, pd_proba, lgd_pred, funded) -> dict:
        ead = self.ccf * funded
        el_per_loan = pd_proba * np.clip(lgd_pred, 0, 1) * ead

        portfolio     = funded.sum()
        expected_loss = el_per_loan.sum()

        return {
            'el_per_loan':   el_per_loan,
            'portfolio':     portfolio,
            'expected_loss': expected_loss,
            'el_pct':        expected_loss / portfolio
        }

    @staticmethod
    def realized_loss(default, ead_default, recoveries):
        return default * np.clip(ead_default - recoveries, 0, None)

    def print_summary(self, result: dict):
        print("=" * 55)
        print(f"  Portfolio (funded)    : ${result['portfolio']:>15,.2f}")
        print(f"  Expected Loss         : ${result['expected_loss']:>15,.2f}")
        print(f"  % of portfolio        :  {result['el_pct']:>14.2%}")
