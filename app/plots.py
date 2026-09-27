import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator
from sklearn.metrics import roc_curve

SURFACE = '#fcfcfb'
ACCENT  = '#1c5cab'
GREY    = '#a3a29c'
INK     = '#52514e'
GRID    = '#e1e0d9'
AXIS    = '#c3c2b7'

plt.rcParams.update({
    'font.family':       ['Segoe UI', 'Arial', 'DejaVu Sans'],
    'font.size':         10,
    'svg.fonttype':      'none',   # text stays text: the browser renders it with the page fonts
    'axes.edgecolor':    AXIS,
    'axes.labelcolor':   INK,
    'axes.titlesize':    11,
    'axes.titlecolor':   INK,
    'axes.titlelocation': 'left',
    'xtick.color':       INK,
    'ytick.color':       INK,
    'legend.frameon':    False,
    'figure.facecolor':  SURFACE,
    'axes.facecolor':    SURFACE,
    'savefig.facecolor': SURFACE
})


class Visualizer:
    def __init__(self, out_dir, figsize=(6.4, 4.2)):
        self.out_dir = out_dir
        self.figsize = figsize
        os.makedirs(out_dir, exist_ok=True)

    def _axes(self):
        fig, ax = plt.subplots(figsize=self.figsize)
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        return fig, ax

    @staticmethod
    def _percent(axis):
        # ticks on 1/2/5 steps so labels stay round (no 2.5% steps shown as 3%)
        axis.set_major_locator(MaxNLocator(steps=[1, 2, 5, 10]))
        axis.set_major_formatter(PercentFormatter(1, decimals=0))

    def _save(self, fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, f"{name}.svg"))
        plt.close(fig)

    def roc(self, y_true, curves: dict):
        # curves: {label: proba}, the first one is drawn with the accent
        fig, ax = self._axes()
        ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
        for i, (label, proba) in enumerate(curves.items()):
            fpr, tpr, _ = roc_curve(y_true, proba)
            ax.plot(fpr, tpr, color=ACCENT if i == 0 else GREY, linewidth=2, label=label,
                    zorder=3 if i == 0 else 2)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc='lower right', labelcolor=INK)
        self._save(fig, 'roc')

    def calibration(self, table):
        # table: one row per PD decile with mean predicted PD and observed default rate
        fig, ax = self._axes()
        top = max(table['mean_pd'].max(), table['default_rate'].max()) * 1.1
        ax.plot([0, top], [0, top], color=AXIS, linewidth=1)
        ax.plot(table['mean_pd'], table['default_rate'], color=ACCENT, linewidth=2,
                marker='o', markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.set_xlabel("Mean predicted PD (by decile)")
        ax.set_ylabel("Observed default rate")
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        self._percent(ax.xaxis)
        self._percent(ax.yaxis)
        self._save(fig, 'calibration')

    def el_distribution(self, el_pct):
        fig, ax = self._axes()
        upper = np.quantile(el_pct, 0.995)
        ax.hist(el_pct, bins=40, range=(0, upper), color=ACCENT, edgecolor=SURFACE, linewidth=1.5)
        ax.axvline(el_pct.mean(), color=INK, linewidth=1)
        ax.annotate(f"Mean {el_pct.mean():.1%}", xy=(el_pct.mean(), 1), xycoords=('data', 'axes fraction'),
                    xytext=(6, -12), textcoords='offset points', color=INK)
        ax.set_xlabel("Expected loss as % of funded amount, per loan (top 0.5% not shown)")
        ax.set_ylabel("Loans")
        self._percent(ax.xaxis)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
        ax.grid(False, axis='x')
        self._save(fig, 'el_distribution')

    def backtest(self, table):
        # table: one row per PD decile with expected and realized loss as % of funded
        fig, ax = self._axes()
        x = np.arange(len(table))
        w = 0.38
        ax.bar(x - w / 2, table['el_pct'], w, color=ACCENT, label="Expected loss")
        ax.bar(x + w / 2, table['realized_pct'], w, color=GREY, label="Realized loss")
        ax.set_xticks(x, table['decile'])
        ax.set_xlabel("PD decile (1 = lowest risk)")
        ax.set_ylabel("Loss as % of funded amount")
        self._percent(ax.yaxis)
        ax.grid(False, axis='x')
        ax.legend(loc='upper left', labelcolor=INK)
        self._save(fig, 'backtest')
