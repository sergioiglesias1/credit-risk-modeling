import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_curve

class ThresholdAnalyzer:
    def __init__(self, thresholds=None):
        self.thresholds = thresholds or [0.05, 0.10, 0.15, 0.20, 0.25,
                                          0.30, 0.35, 0.40, 0.45, 0.50]

    def sweep(self, y_test, probabilities) -> pd.DataFrame:
        _, _, thresh_arr = roc_curve(y_test, probabilities)

        results = []
        for t in self.thresholds:
            y_pred = (probabilities >= t).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='binary', zero_division=0
            )
            results.append({
                'threshold': t,
                'precision': round(precision, 3),
                'recall':    round(recall, 3),
                'f1':        round(f1, 3)
            })

        return pd.DataFrame(results)


class ExpectedLossCalculator:
    def compute(self, pd_proba, lgd_pred, ead) -> dict:
        lgd_clipped = np.clip(lgd_pred, 0, 100)
        el_per_loan = pd_proba * (lgd_clipped / 100) * ead

        cartera_total  = ead.sum()
        perdida_total  = el_per_loan.sum()
        pct_perdida    = perdida_total / cartera_total

        return {
            'el_per_loan':    el_per_loan,
            'cartera_total':  cartera_total,
            'perdida_total':  perdida_total,
            'pct_perdida':    pct_perdida
        }

    def print_summary(self, result: dict):
        print("=" * 55)
        print(f"  Portfolio total value : ${result['cartera_total']:>15,.2f}")
        print(f"  Expected Loss         : ${result['perdida_total']:>15,.2f}")
        print(f"  % of portfolio        :  {result['pct_perdida']:>14.2%}")
