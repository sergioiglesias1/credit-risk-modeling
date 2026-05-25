import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    def __init__(self, figsize=(10, 6), fontweight="bold"):
        self.figsize = figsize
        self.fontweight = fontweight

    def class_imbalance(self, df, target_col='PD'):
        counts = df[target_col].value_counts()
        labels = ['No Default (0)', 'Default (1)']

        plt.figure(figsize=(6, 5))
        sns.barplot(x=labels, y=counts.values, palette='viridis')
        plt.title("Class Imbalance — Default vs Non-Default", fontweight=self.fontweight)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("viz/class_imbalance.png", dpi=150)
        plt.show()

    def interest_rate_vs_default(self, df, rate_col='interest_rate', target_col='PD'):
        plt.figure(figsize=self.figsize)
        sns.boxplot(data=df, x=target_col, y=rate_col, palette='viridis',
                    hue=target_col, legend=False)
        plt.xticks([0, 1], ['No Default', 'Default'])
        plt.title("Interest Rate vs Default Status", fontweight=self.fontweight)
        plt.xlabel("Default")
        plt.ylabel("Interest Rate (%)")
        plt.tight_layout()
        plt.savefig("viz/interest_rate_vs_default.png", dpi=150)
        plt.show()
