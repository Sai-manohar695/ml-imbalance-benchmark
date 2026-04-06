import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="ML Imbalance Benchmark",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_results():
    engine = create_engine(os.getenv("DATABASE_URL"))
    query = """
        SELECT
            d.name as dataset,
            e.classifier,
            e.sampling_strategy,
            r.fold,
            r.f1,
            r.auc_roc,
            r.mcc,
            r.precision,
            r.recall
        FROM results r
        JOIN experiments e ON r.experiment_id = e.id
        JOIN datasets d ON e.dataset_id = d.id
        ORDER BY d.name, e.classifier, e.sampling_strategy, r.fold
    """
    return pd.read_sql(query, engine)

@st.cache_data
def load_statistical_tests():
    engine = create_engine(os.getenv("DATABASE_URL"))
    query = """
        SELECT * FROM statistical_tests
        ORDER BY metric, p_value
    """
    return pd.read_sql(query, engine)

def avg_metrics(df):
    return df.groupby(["dataset", "classifier", "sampling_strategy"]).agg(
        f1=("f1", "mean"),
        auc_roc=("auc_roc", "mean"),
        mcc=("mcc", "mean"),
        precision=("precision", "mean"),
        recall=("recall", "mean")
    ).reset_index()

# ── HEADER ──────────────────────────────────────────────
st.title("📊 ML Imbalance Benchmark")
st.markdown("**5 classifiers × 4 sampling strategies × 3 datasets × 5-fold CV**")
st.markdown("Every result backed by Wilcoxon signed-rank and Friedman statistical tests.")
st.divider()

# ── LOAD DATA ───────────────────────────────────────────
with st.spinner("Loading results from database..."):
    df_raw = load_results()
    df_tests = load_statistical_tests()
    df = avg_metrics(df_raw)

# ── SIDEBAR FILTERS ─────────────────────────────────────
st.sidebar.header("Filters")
selected_dataset = st.sidebar.selectbox(
    "Dataset",
    ["All"] + sorted(df["dataset"].unique().tolist())
)
selected_metric = st.sidebar.selectbox(
    "Metric",
    ["f1", "auc_roc", "mcc", "precision", "recall"],
    format_func=lambda x: x.upper().replace("_", "-")
)
selected_classifiers = st.sidebar.multiselect(
    "Classifiers",
    sorted(df["classifier"].unique().tolist()),
    default=sorted(df["classifier"].unique().tolist())
)
selected_strategies = st.sidebar.multiselect(
    "Sampling Strategies",
    sorted(df["sampling_strategy"].unique().tolist()),
    default=sorted(df["sampling_strategy"].unique().tolist())
)

# Apply filters
df_filtered = df.copy()
if selected_dataset != "All":
    df_filtered = df_filtered[df_filtered["dataset"] == selected_dataset]
df_filtered = df_filtered[df_filtered["classifier"].isin(selected_classifiers)]
df_filtered = df_filtered[df_filtered["sampling_strategy"].isin(selected_strategies)]

# ── KPI CARDS ───────────────────────────────────────────
st.subheader("Overview")
col1, col2, col3, col4 = st.columns(4)

best_row = df_filtered.loc[df_filtered["f1"].idxmax()] if not df_filtered.empty else None

col1.metric("Total Runs", len(df_raw))
col2.metric("Best F1", f"{df_filtered['f1'].max():.4f}" if not df_filtered.empty else "—")
col3.metric(
    "Best Classifier (F1)",
    best_row["classifier"].replace("_", " ").title() if best_row is not None else "—"
)
col4.metric(
    "Best Strategy (F1)",
    best_row["sampling_strategy"].replace("_", " ").title() if best_row is not None else "—"
)

st.divider()

# ── TAB LAYOUT ──────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Metric Comparison",
    "🔥 Heatmap",
    "⚠️ Metric Gap Analysis",
    "🧪 Statistical Tests"
])

# ── TAB 1: METRIC COMPARISON ────────────────────────────
with tab1:
    st.subheader(f"{selected_metric.upper().replace('_', '-')} by Classifier & Sampling Strategy")

    fig = px.bar(
        df_filtered,
        x="classifier",
        y=selected_metric,
        color="sampling_strategy",
        barmode="group",
        facet_col="dataset" if selected_dataset == "All" else None,
        labels={
            selected_metric: selected_metric.upper().replace("_", "-"),
            "classifier": "Classifier",
            "sampling_strategy": "Sampling Strategy"
        },
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
        legend_title="Sampling Strategy",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 2: HEATMAP ──────────────────────────────────────
with tab2:
    st.subheader(f"{selected_metric.upper().replace('_', '-')} Heatmap")

    dataset_options = df["dataset"].unique().tolist()
    selected_heatmap_dataset = st.selectbox(
        "Select Dataset", dataset_options, key="heatmap_dataset"
    )

    pivot = df[df["dataset"] == selected_heatmap_dataset].pivot(
        index="classifier",
        columns="sampling_strategy",
        values=selected_metric
    )

    fig2 = px.imshow(
        pivot,
        text_auto=".3f",
        color_continuous_scale="RdYlGn",
        aspect="auto",
        labels={"color": selected_metric.upper()}
    )
    fig2.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
        height=400
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── TAB 3: METRIC GAP ANALYSIS ──────────────────────────
with tab3:
    st.subheader("AUC-ROC vs F1 Gap — The Core Finding")
    st.markdown(
        "Random undersampling scores high on AUC-ROC while F1 and MCC collapse. "
        "This tab shows that gap directly."
    )

    gap_dataset = st.selectbox(
        "Select Dataset", df["dataset"].unique().tolist(), key="gap_dataset"
    )

    df_gap = df[df["dataset"] == gap_dataset].copy()
    df_gap["gap"] = df_gap["auc_roc"] - df_gap["f1"]
    df_gap = df_gap.sort_values("gap", ascending=False)

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        name="AUC-ROC",
        x=[f"{r['classifier']}<br>{r['sampling_strategy']}" for _, r in df_gap.iterrows()],
        y=df_gap["auc_roc"],
        marker_color="#00cc96"
    ))
    fig3.add_trace(go.Bar(
        name="F1",
        x=[f"{r['classifier']}<br>{r['sampling_strategy']}" for _, r in df_gap.iterrows()],
        y=df_gap["f1"],
        marker_color="#ef553b"
    ))
    fig3.update_layout(
        barmode="group",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
        height=500,
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("**Largest gaps (AUC-ROC − F1):**")
    st.dataframe(
        df_gap[["classifier", "sampling_strategy", "auc_roc", "f1", "mcc", "gap"]]
        .round(4)
        .head(10),
        use_container_width=True
    )

# ── TAB 4: STATISTICAL TESTS ────────────────────────────
with tab4:
    st.subheader("Statistical Significance — Wilcoxon Signed-Rank Test")

    metric_filter = st.selectbox(
        "Metric", ["f1", "auc_roc", "mcc"], key="test_metric",
        format_func=lambda x: x.upper().replace("_", "-")
    )

    df_sig = df_tests[df_tests["metric"] == metric_filter].copy()

    if not df_sig.empty:
        df_sig["significant"] = df_sig["is_significant"].map({True: "✓ Yes", False: "✗ No"})
        df_sig["p_value"] = df_sig["p_value"].round(4)

        sig_count = df_sig["is_significant"].sum()
        total = len(df_sig)
        st.markdown(f"**{sig_count} of {total} pairs are statistically significant** (p < 0.05)")

        st.dataframe(
            df_sig[["classifier_a", "classifier_b", "metric", "statistic", "p_value", "significant"]]
            .sort_values("p_value"),
            use_container_width=True
        )
    else:
        st.info("No statistical test results found.")

st.divider()
st.caption("Built with Streamlit · PostgreSQL on Neon · Results from 300 experiment runs")