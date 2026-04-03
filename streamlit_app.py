import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import json

st.set_page_config(
    page_title="Credit Risk Project",
    page_icon="💳",
    layout="wide"
)

# ── Dark theme override ────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background-color: #07111c; }
  [data-testid="stSidebar"] { background-color: #0d1b2a; }
  [data-testid="stHeader"] { background-color: #07111c; }
  .block-container { padding-top: 1.5rem; }
  h1,h2,h3,h4 { color: #00e5ff !important; font-family: 'Courier New', monospace !important; }
  p, li, label { color: #cdd9e5 !important; }
  .stMetric label { color: #6b8fa8 !important; font-size: 11px !important; }
  .stMetric [data-testid="stMetricValue"] { color: #00e5ff !important; font-family: monospace !important; }
  .stDataFrame { background: #0d1b2a; }
  thead tr th { background: #1e2d3d !important; color: #6b8fa8 !important; font-size: 12px !important; }
  tbody tr td { color: #cdd9e5 !important; font-size: 12px !important; }
  .stTabs [data-baseweb="tab"] { color: #6b8fa8; font-family: monospace; font-size: 12px; }
  .stTabs [aria-selected="true"] { color: #00e5ff !important; border-bottom-color: #00e5ff !important; }
  .stSlider [data-testid="stSlider"] > div { color: #00e5ff; }
</style>
""", unsafe_allow_html=True)

# ── Synthetic data ─────────────────────────────────────────────────────────
CLASS_IMBALANCE = pd.DataFrame({
    "Class": ["No Default", "Default"],
    "Count": [28435, 7315],
    "Pct": ["79.5%", "20.5%"]
})

PD_BASE = pd.DataFrame([
    {"Model": "LightGBM",            "ROC_AUC": 0.7521, "F1": 0.621, "Recall": 0.724, "Precision": 0.543},
    {"Model": "Random Forest",       "ROC_AUC": 0.7318, "F1": 0.598, "Recall": 0.691, "Precision": 0.526},
    {"Model": "Logistic Regression", "ROC_AUC": 0.7204, "F1": 0.574, "Recall": 0.667, "Precision": 0.505},
    {"Model": "Decision Tree",       "ROC_AUC": 0.5512, "F1": 0.431, "Recall": 0.512, "Precision": 0.371},
])

PD_TUNED = pd.DataFrame([
    {"Model": "LightGBM",            "CV_AUC": 0.7891, "Best Params": "depth=8, lr=0.1, n=200"},
    {"Model": "Random Forest",       "CV_AUC": 0.7634, "Best Params": "n=300, depth=10"},
    {"Model": "Logistic Regression", "CV_AUC": 0.7389, "Best Params": "C=0.1, l2"},
    {"Model": "Decision Tree",       "CV_AUC": 0.5721, "Best Params": "depth=5"},
])

fpr_arr = np.linspace(0, 1, 50)
tpr_arr = np.minimum(1, fpr_arr + (1 - fpr_arr) * (0.79 * np.power(1 - fpr_arr, 0.4)))
ROC = pd.DataFrame({"FPR": fpr_arr, "TPR": tpr_arr, "Random": fpr_arr})

THR = pd.DataFrame([
    {"Threshold": 0.05, "Precision": 0.321, "Recall": 0.982, "F1": 0.484},
    {"Threshold": 0.10, "Precision": 0.541, "Recall": 0.951, "F1": 0.690},
    {"Threshold": 0.15, "Precision": 0.634, "Recall": 0.902, "F1": 0.745},
    {"Threshold": 0.20, "Precision": 0.712, "Recall": 0.843, "F1": 0.772},
    {"Threshold": 0.25, "Precision": 0.769, "Recall": 0.781, "F1": 0.775},
    {"Threshold": 0.30, "Precision": 0.821, "Recall": 0.712, "F1": 0.763},
    {"Threshold": 0.35, "Precision": 0.863, "Recall": 0.631, "F1": 0.729},
    {"Threshold": 0.40, "Precision": 0.891, "Recall": 0.542, "F1": 0.675},
    {"Threshold": 0.45, "Precision": 0.912, "Recall": 0.451, "F1": 0.604},
    {"Threshold": 0.50, "Precision": 0.934, "Recall": 0.361, "F1": 0.521},
])

LGD = pd.DataFrame([
    {"Model": "Random Forest",     "MAE": 0.2489, "RMSE": 1.371, "R2": 0.621},
    {"Model": "LightGBM",          "MAE": 0.3891, "RMSE": 1.452, "R2": 0.583},
    {"Model": "Decision Tree",     "MAE": 0.5201, "RMSE": 2.512, "R2": 0.441},
    {"Model": "Linear Regression", "MAE": 11.394, "RMSE": 20.09, "R2": 0.089},
])

np.random.seed(42)
el_per_loan = np.random.exponential(scale=800, size=5000)

TOP_LOANS = pd.DataFrame({
    "Loan ID":        [f"L-{1000 + i*137}" for i in range(10)],
    "EAD ($)":        [45000 - i*3800 for i in range(10)],
    "Expected Loss":  [12400 - i*1100 for i in range(10)],
    "EL %":           [round((12400 - i*1100) / (45000 - i*3800) * 100, 1) for i in range(10)],
})

META = {
    "pd_model": "LightGBM",
    "lgd_model": "Random Forest",
    "trained_on": "2025-04",
    "dataset": "Lending Club Loan Data",
    "pd_cv_auc": 0.7891,
    "pd_best_params": {"max_depth": 8, "learning_rate": 0.1, "n_estimators": 200},
    "lgd_best_params": {"n_estimators": 300, "max_depth": 10},
    "pd_threshold": 0.10,
    "pd_test_precision": 0.541,
    "pd_test_recall": 0.951,
    "pd_test_f1": 0.690,
    "lgd_mae": 0.2489,
    "lgd_rmse": 1.371,
    "portfolio_value": 174008000,
    "expected_loss": 7815756,
    "expected_loss_pct": 0.0449,
}

BG     = "#07111c"
CARD   = "#0d1b2a"
DIM    = "#1e2d3d"
CYAN   = "#00e5ff"
RED    = "#ff4d6d"
GREEN  = "#00ffa3"
YELLOW = "#ffd166"
MUTED  = "#6b8fa8"
TEXT   = "#cdd9e5"

def styled_fig():
    fig, ax = plt.subplots()
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    for spine in ax.spines.values():
        spine.set_edgecolor(DIM)
    return fig, ax

def styled_fig2(ncols=2):
    fig, axes = plt.subplots(1, ncols, figsize=(10, 4))
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(CARD)
        ax.tick_params(colors=MUTED, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(DIM)
    return fig, axes

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#0d1b2a,#07111c);
  border-bottom:1px solid #1e3a52;padding:16px 0 10px;margin-bottom:8px'>
  <span style='font-size:28px'>💳</span>
  <span style='color:#00e5ff;font-family:monospace;font-size:20px;font-weight:700;
    letter-spacing:0.1em;margin-left:10px'>CREDIT RISK PIPELINE</span>
  <span style='color:#6b8fa8;font-size:12px;margin-left:12px'>Lending Club · Synthetic Demo</span>
  <span style='float:right;background:#00ffa322;color:#00ffa3;border:1px solid #00ffa355;
    border-radius:20px;padding:3px 14px;font-size:12px;font-family:monospace'>✓ Trained</span>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 EDA", "🤖 PD Models", "🎚️ Threshold", "📉 LGD", "💰 Exp. Loss", "📋 Metadata"
])

# ══════════════════════════════════════════════════════════
# TAB 1 — EDA
# ══════════════════════════════════════════════════════════
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Loans", "35,750")
    c2.metric("Features", "28")
    c3.metric("Default Rate", "20.5%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Class Imbalance")
        fig, ax = styled_fig()
        fig.set_size_inches(5, 3.5)
        bars = ax.bar(["No Default", "Default"], [28435, 7315],
                      color=[CYAN, RED], width=0.5, edgecolor="none")
        ax.set_ylabel("Count", color=MUTED)
        ax.set_title("Default vs Non-Default", color=CYAN,
                     fontfamily="monospace", fontsize=11)
        for bar, val in zip(bars, [28435, 7315]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
                    f"{val:,}", ha="center", color=TEXT, fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("#### Interest Rate vs Default")
        int_data = {
            "No Default": {"Min":5.3,"Q1":9.2,"Median":12.4,"Q3":16.1,"Max":24.8},
            "Default":    {"Min":6.1,"Q1":13.7,"Median":17.9,"Q3":22.3,"Max":29.5},
        }
        rows = []
        for cls, stats in int_data.items():
            for stat, val in stats.items():
                rows.append({"Class": cls, "Stat": stat, "Rate (%)": val})
        df_int = pd.DataFrame(rows)
        fig2, ax2 = styled_fig()
        fig2.set_size_inches(5, 3.5)
        stats_order = ["Min","Q1","Median","Q3","Max"]
        x = np.arange(len(stats_order))
        w = 0.35
        nd = [int_data["No Default"][s] for s in stats_order]
        df = [int_data["Default"][s]    for s in stats_order]
        ax2.bar(x - w/2, nd, w, color=CYAN,  label="No Default", alpha=0.85)
        ax2.bar(x + w/2, df, w, color=RED,   label="Default",    alpha=0.85)
        ax2.set_xticks(x)
        ax2.set_xticklabels(stats_order, color=MUTED, fontsize=9)
        ax2.set_ylabel("Interest Rate (%)", color=MUTED)
        ax2.set_title("Interest Rate Stats", color=CYAN,
                      fontfamily="monospace", fontsize=11)
        ax2.legend(facecolor=CARD, edgecolor=DIM, labelcolor=TEXT, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig2)

# ══════════════════════════════════════════════════════════
# TAB 2 — PD Models
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### Baseline Models — Default Hyperparameters")
    st.dataframe(
        PD_BASE.style
            .highlight_max(subset=["ROC_AUC","F1","Recall","Precision"], color="#00e5ff33")
            .format("{:.4f}", subset=["ROC_AUC","F1","Recall","Precision"]),
        use_container_width=True, hide_index=True
    )

    st.markdown("---")
    st.markdown("#### After GridSearchCV Tuning")
    st.dataframe(
        PD_TUNED.style
            .highlight_max(subset=["CV_AUC"], color="#00ffa333")
            .format("{:.4f}", subset=["CV_AUC"]),
        use_container_width=True, hide_index=True
    )
    st.success("🏆 Best model: **LightGBM** — CV AUC 0.7891")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ROC Curve — LightGBM")
        fig, ax = styled_fig()
        fig.set_size_inches(5, 4)
        ax.plot(ROC["FPR"], ROC["TPR"], color=CYAN, lw=2.5, label="AUC = 0.7891")
        ax.plot([0,1],[0,1], color=MUTED, lw=1, linestyle="--", label="Random")
        ax.set_xlabel("False Positive Rate", color=MUTED)
        ax.set_ylabel("True Positive Rate", color=MUTED)
        ax.set_title("ROC — LightGBM", color=CYAN, fontfamily="monospace", fontsize=11)
        ax.legend(facecolor=CARD, edgecolor=DIM, labelcolor=TEXT, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("#### CV AUC Comparison")
        fig2, ax2 = styled_fig()
        fig2.set_size_inches(5, 4)
        colors = [GREEN if m == "LightGBM" else CYAN+"88" for m in PD_TUNED["Model"]]
        ax2.barh(PD_TUNED["Model"], PD_TUNED["CV_AUC"], color=colors, edgecolor="none")
        ax2.set_xlim(0.5, 0.85)
        ax2.set_xlabel("CV ROC-AUC", color=MUTED)
        ax2.set_title("Model Comparison", color=CYAN, fontfamily="monospace", fontsize=11)
        for i, v in enumerate(PD_TUNED["CV_AUC"]):
            ax2.text(v + 0.003, i, f"{v:.4f}", va="center", color=TEXT, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig2)

# ══════════════════════════════════════════════════════════
# TAB 3 — Threshold
# ══════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### Precision / Recall / F1 vs Threshold")
    fig, ax = styled_fig()
    fig.set_size_inches(9, 3.5)
    ax.plot(THR["Threshold"], THR["Precision"], marker="o", color=CYAN,   label="Precision", lw=2)
    ax.plot(THR["Threshold"], THR["Recall"],    marker="o", color=RED,    label="Recall",    lw=2)
    ax.plot(THR["Threshold"], THR["F1"],        marker="o", color=GREEN,  label="F1",        lw=2, linestyle="--")
    ax.axvline(0.10, color=YELLOW, linestyle=":", lw=2, label="t=0.10 chosen")
    ax.set_xlabel("Threshold", color=MUTED)
    ax.set_ylabel("Score", color=MUTED)
    ax.set_title("Precision / Recall / F1 vs Threshold", color=CYAN, fontfamily="monospace", fontsize=11)
    ax.legend(facecolor=CARD, edgecolor=DIM, labelcolor=TEXT, fontsize=9)
    ax.grid(True, alpha=0.15, color=MUTED)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("#### Interactive Threshold Selector")
    t_idx = st.slider("Threshold", min_value=0, max_value=9, value=1,
                      format="",
                      help="Slide to explore different thresholds")
    t_labels = [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50]
    selected = THR.iloc[t_idx]

    st.markdown(f"<h2 style='text-align:center;color:#00e5ff;font-family:monospace'>"
                f"t = {t_labels[t_idx]:.2f}"
                f"{'  <span style=\"color:#ffd166;font-size:16px\">★ chosen</span>' if t_idx==1 else ''}"
                f"</h2>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Precision", f"{selected['Precision']:.3f}")
    c2.metric("Recall",    f"{selected['Recall']:.3f}")
    c3.metric("F1",        f"{selected['F1']:.3f}")

    st.info("💡 **t=0.10** → Recall 0.951 — 95% of defaulters caught. "
            "Missing a defaulter (FN) costs far more than rejecting a good client (FP).")

    st.markdown("---")
    st.markdown("#### Full Threshold Table")
    st.dataframe(
        THR.style
            .highlight_max(subset=["Recall"],    color="#ff4d6d33")
            .highlight_max(subset=["Precision"], color="#00e5ff33")
            .highlight_max(subset=["F1"],        color="#00ffa333")
            .format("{:.3f}", subset=["Precision","Recall","F1"]),
        use_container_width=True, hide_index=True
    )

# ══════════════════════════════════════════════════════════
# TAB 4 — LGD
# ══════════════════════════════════════════════════════════
with tab4:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Regression Models (Tuned)")
        st.dataframe(
            LGD.style
                .highlight_min(subset=["MAE","RMSE"], color="#00ffa333")
                .highlight_max(subset=["R2"],         color="#00e5ff33")
                .format("{:.4f}", subset=["MAE","RMSE","R2"]),
            use_container_width=True, hide_index=True
        )
        st.success("🏆 Best: **Random Forest** — MAE = 0.2489")

    with col2:
        st.markdown("#### MAE Comparison")
        lgd_plot = LGD[LGD["MAE"] < 5]
        fig, ax = styled_fig()
        fig.set_size_inches(5, 3.5)
        colors = [GREEN if m == "Random Forest" else CYAN+"88" for m in lgd_plot["Model"]]
        ax.barh(lgd_plot["Model"], lgd_plot["MAE"], color=colors, edgecolor="none")
        ax.set_xlabel("MAE", color=MUTED)
        ax.set_title("MAE by Model", color=CYAN, fontfamily="monospace", fontsize=11)
        for i, v in enumerate(lgd_plot["MAE"]):
            ax.text(v + 0.005, i, f"{v:.4f}", va="center", color=TEXT, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

# ══════════════════════════════════════════════════════════
# TAB 5 — Expected Loss
# ══════════════════════════════════════════════════════════
with tab5:
    c1, c2, c3 = st.columns(3)
    c1.metric("Portfolio Value", "$174,008,000")
    c2.metric("Expected Loss",   "$7,815,756")
    c3.metric("EL % of Portfolio", "4.49%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Portfolio Breakdown")
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        wedges, texts, autotexts = ax.pie(
            [7815756, 174008000 - 7815756],
            labels=["Expected Loss", "Safe Capital"],
            colors=[RED, GREEN],
            autopct="%1.2f%%",
            startangle=90,
            wedgeprops={"edgecolor": BG, "linewidth": 2},
            textprops={"color": TEXT, "fontsize": 11}
        )
        for at in autotexts:
            at.set_color(BG)
            at.set_fontweight("bold")
        ax.set_title("Portfolio Composition", color=CYAN,
                     fontfamily="monospace", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("#### EL Distribution per Loan")
        fig2, ax2 = styled_fig()
        fig2.set_size_inches(5, 4)
        ax2.hist(el_per_loan, bins=40, color=RED, edgecolor="none", alpha=0.85)
        ax2.axvline(el_per_loan.mean(), color=YELLOW, lw=1.5, linestyle="--",
                    label=f"Mean: ${el_per_loan.mean():,.0f}")
        ax2.set_xlabel("Expected Loss ($)", color=MUTED)
        ax2.set_ylabel("Frequency", color=MUTED)
        ax2.set_title("EL Distribution", color=CYAN, fontfamily="monospace", fontsize=11)
        ax2.legend(facecolor=CARD, edgecolor=DIM, labelcolor=TEXT, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig2)

    st.markdown("---")
    st.markdown("#### Top 10 Riskiest Loans")
    st.dataframe(
        TOP_LOANS.style
            .highlight_max(subset=["Expected Loss"], color="#ff4d6d33")
            .format({"EAD ($)": "${:,.0f}", "Expected Loss": "${:,.0f}", "EL %": "{}%"}),
        use_container_width=True, hide_index=True
    )

# ══════════════════════════════════════════════════════════
# TAB 6 — Metadata
# ══════════════════════════════════════════════════════════
with tab6:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Models Selected")
        c1, c2, c3 = st.columns(3)
        c1.metric("PD Model",  META["pd_model"])
        c2.metric("LGD Model", META["lgd_model"])
        c3.metric("Trained",   META["trained_on"])

        st.markdown("#### PD Performance")
        c1, c2, c3 = st.columns(3)
        c1.metric("CV AUC",    META["pd_cv_auc"])
        c2.metric("Recall",    META["pd_test_recall"])
        c3.metric("Precision", META["pd_test_precision"])
        c1b, c2b = st.columns(2)
        c1b.metric("F1",        META["pd_test_f1"])
        c2b.metric("Threshold", META["pd_threshold"])

    with col2:
        st.markdown("#### LGD Performance")
        c1, c2 = st.columns(2)
        c1.metric("MAE",  META["lgd_mae"])
        c2.metric("RMSE", META["lgd_rmse"])

        st.markdown("#### Expected Loss")
        c1, c2, c3 = st.columns(3)
        c1.metric("Portfolio",  f"${META['portfolio_value']:,}")
        c2.metric("EL ($)",     f"${META['expected_loss']:,}")
        c3.metric("EL %",       f"{META['expected_loss_pct']:.2%}")

    st.markdown("---")
    st.markdown("#### metadata.json")
    st.json(META)
    st.download_button(
        "⬇️ Download metadata.json",
        data=json.dumps(META, indent=2),
        file_name="metadata.json",
        mime="application/json"
    )
