"""Streamlit dashboard for the seeded traffic and slicing experiments.

Run from the repository root:
    streamlit run streamlit_app.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
SLICES = {
    "eMBB": "embb_mbps",
    "URLLC": "urllc_mbps",
    "IoT": "iot_mbps",
}
APPROACH_FILES = {
    "Static": ROOT / "data" / "processed" / "static_results.csv",
    "Dynamic-MA": ROOT / "data" / "processed" / "dynamic_results.csv",
    "Dynamic-RF": ROOT / "data" / "processed" / "dynamic_rf_results.csv",
}
METRIC_DEFINITIONS = {
    "MAE": ("Mean Absolute Error: average absolute prediction error in Mbps.", "Lower is better."),
    "RMSE": ("Root Mean Squared Error: penalizes large prediction errors more strongly.", "Lower is better."),
    "latency": ("Time required for traffic to be served, in milliseconds.", "Lower is better."),
    "throughput": ("Traffic successfully served, in Mbps.", "Higher is better."),
    "packet_loss": ("Fraction of traffic lost during delivery.", "Lower is better."),
    "qos_violations": ("Count of observations that violated the configured QoS target.", "Lower is better."),
    "utilization": ("Demand divided by allocated capacity.", "Higher is generally better, but excessive utilization can increase violations."),
    "fairness": ("How evenly resources are shared across slices; 1.0 is perfectly fair.", "Higher is better."),
    "reconfigurations": ("Number of allocation changes made by the dynamic controller.", "Lower is more stable, but zero is not automatically best."),
    "reconfiguration_cost": ("Total modelled resource movement caused by reconfiguration.", "Lower is better."),
    "oscillations": ("Direction changes in allocation decisions.", "Lower is more stable."),
}
COMPARISON_RULES = {
    "latency": "min",
    "packet_loss": "min",
    "qos_violations": "min",
    "throughput": "max",
    "fairness": "max",
    "utilization": "context",
    "reconfigurations": "min_dynamic_only",
    "reconfiguration_cost": "min_dynamic_only",
    "oscillations": "min_dynamic_only",
}


def artifact_path(relative_path):
    return ROOT / relative_path


@st.cache_data
def read_csv(path):
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def read_multiseed_summary(path):
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(
        path,
        header=None,
        names=["metric", "statistic", "Dynamic-MA", "Dynamic-RF", "Static"],
        skiprows=1,
    )


@st.cache_data
def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_json(path) if path.suffix == ".json" else None


def require(data, label):
    if data is None:
        st.error(f"{label} was not found. Run the corresponding project experiment first.")
        return False
    return True


def line_chart(data, x, columns, title, y_label):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for label, column in columns.items():
        if column in data.columns:
            ax.plot(data[x], data[column], label=label, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def traffic_view():
    data = read_csv(artifact_path("data/ml/ml_traffic_runs.csv"))
    if not require(data, "Seeded traffic dataset"):
        return

    st.subheader("Seeded traffic patterns")
    st.caption(
        "Synthetic traffic generated from the project's six scenario templates. "
        "These are not measurements from a real 5G network."
    )

    split = st.sidebar.selectbox("Dataset split", ["All", "train", "val", "test"])
    scenarios = sorted(data["scenario"].unique())
    selected_scenarios = st.sidebar.multiselect("Scenarios", scenarios, default=scenarios)
    run_ids = sorted(data["run_id"].unique())
    selected_run = st.sidebar.selectbox("Run", ["All runs"] + run_ids)

    filtered = data[data["scenario"].isin(selected_scenarios)].copy()
    if split != "All":
        filtered = filtered[filtered["split"] == split]
    if selected_run != "All runs":
        filtered = filtered[filtered["run_id"] == selected_run]

    st.dataframe(
        filtered.groupby(["split", "scenario"]).agg(
            runs=("run_id", "nunique"),
            observations=("time", "size"),
            eMBB_mean_mbps=("embb_mbps", "mean"),
            URLLC_mean_mbps=("urllc_mbps", "mean"),
            IoT_mean_mbps=("iot_mbps", "mean"),
        ).round(3),
        use_container_width=True,
    )

    if selected_run == "All runs":
        grouped = filtered.groupby("time", as_index=False)[list(SLICES.values())].mean()
        line_chart(grouped, "time", SLICES, "Mean seeded traffic by interval", "Demand (Mbps)")
    elif not filtered.empty:
        line_chart(filtered, "time", SLICES, f"Traffic pattern: {selected_run}", "Demand (Mbps)")


def prediction_view():
    data = read_csv(artifact_path("results/ml/prediction_accuracy.csv"))
    if not require(data, "Prediction accuracy results"):
        return

    st.subheader("Prediction accuracy")
    st.info(
        "**MAE** is the average absolute prediction error. "
        "**RMSE** penalizes large errors more heavily. "
        "Both are measured in Mbps, and lower values are better."
    )
    dataset = st.selectbox("Evaluation dataset", sorted(data["dataset"].unique()))
    selected = data[(data["dataset"] == dataset) & (data["group_type"].isin(["overall", "slice"]))]
    pivot = (
        selected.pivot_table(
            index=["group_type", "group"],
            columns="predictor",
            values="mae_mbps",
        )
        .reset_index()
    )
    chart_data = pivot.set_index("group").drop(columns=["group_type"], errors="ignore")
    st.bar_chart(chart_data, y_label="MAE (Mbps)")
    st.dataframe(selected.round(4), use_container_width=True)

    st.markdown("#### Existing generated plots")
    plot_dir = ROOT / "plots" / "ml"
    for plot in sorted(plot_dir.glob("*.png")):
        st.image(str(plot), caption=plot.name, use_container_width=True)


def comparison_view():
    comparison = read_csv(artifact_path("results/comparison_three_way.csv"))
    if not require(comparison, "Three-way comparison results"):
        return

    st.subheader("Static vs Dynamic-MA vs Dynamic-RF")
    metric = st.selectbox("Metric", comparison["Metric"].tolist())
    metric_key = metric.lower().replace("eMBB", "embb").replace("URLLC", "urllc").replace("IoT", "iot")
    definition = next(
        (value for key, value in METRIC_DEFINITIONS.items() if key in metric_key),
        ("Result metric from the project's experiment output.", "Interpret using the project context."),
    )
    st.info(f"**What it means:** {definition[0]} **How to compare:** {definition[1]}")
    row = comparison[comparison["Metric"] == metric].set_index("Metric").T
    st.bar_chart(row)
    values = comparison.loc[
        comparison["Metric"] == metric, ["Static", "Dynamic-MA", "Dynamic-RF"]
    ].iloc[0].astype(float)
    rule = next((value for key, value in COMPARISON_RULES.items() if key in metric_key), "context")
    if rule == "min_dynamic_only":
        dynamic_values = values[["Dynamic-MA", "Dynamic-RF"]]
        winner = dynamic_values.idxmin()
        st.success(
            f"Best dynamic approach: **{winner}** ({dynamic_values[winner]:.4f}). "
            "Static is not applicable because it never reallocates."
        )
    elif rule == "min":
        winner = values.idxmin()
        st.success(f"Best for this metric: **{winner}** ({values[winner]:.4f})")
    elif rule == "max":
        winner = values.idxmax()
        st.success(f"Best for this metric: **{winner}** ({values[winner]:.4f})")
    else:
        st.warning("This metric requires context; higher or lower is not universally better.")
    st.dataframe(comparison.round(4), use_container_width=True)

    st.markdown("#### Evidence supporting Dynamic-RF")
    st.write(
        "The original experiment outputs support RF as the preferred predictive controller "
        "overall, but not as the winner of every individual metric."
    )
    evidence = pd.DataFrame(
        [
            ["Held-out prediction MAE", "Dynamic-RF", "0.878 Mbps vs 2.442 Mbps for Dynamic-MA", "Lower is better"],
            ["Held-out prediction RMSE", "Dynamic-RF", "2.940 Mbps vs 5.230 Mbps for Dynamic-MA", "Lower is better"],
            ["10-seed mean QoS violations", "Dynamic-RF", "131.2 vs 141.3 for Dynamic-MA and 138.6 for Static", "Lower is better"],
            ["10-seed QoS result", "Dynamic-RF", "Fewer violations than Dynamic-MA in 8 of 10 seeds", "More consistent outcome"],
        ],
        columns=["Evidence", "Observed leader", "Original result", "Interpretation"],
    )
    st.dataframe(evidence, hide_index=True, use_container_width=True)

    st.markdown("#### Allocation traces")
    for approach, path in APPROACH_FILES.items():
        data = read_csv(path)
        if data is None:
            st.warning(f"{approach} results are missing: {path}")
            continue
        columns = {
            "eMBB": "embb_allocation_pct",
            "URLLC": "urllc_allocation_pct",
            "IoT": "iot_allocation_pct",
        }
        st.markdown(f"**{approach}**")
        line_chart(data, "time", columns, f"{approach} allocation", "Allocation (%)")


def multiseed_view():
    summary = read_multiseed_summary(artifact_path("results/multiseed/summary.csv"))
    per_seed = read_csv(artifact_path("results/multiseed/per_seed_metrics.csv"))
    if not require(summary, "Multi-seed summary"):
        return

    st.subheader("10-seed comparison")
    st.info(
        "**Primary decision rule:** choose the approach with the lowest mean total QoS violations "
        "across the independent seeds. Secondary metrics show trade-offs such as fairness, latency, "
        "throughput, and controller stability."
    )
    summary_mean = summary[summary["statistic"] == "mean"].copy()
    st.dataframe(summary_mean, use_container_width=True)
    qos = summary_mean[summary_mean["metric"] == "total_qos_violations"]
    if not qos.empty:
        values = qos.set_index("metric")[["Static", "Dynamic-MA", "Dynamic-RF"]].iloc[0]
        winner = values.idxmin()
        st.success(
            f"Overall recommendation from the 10-seed QoS criterion: **{winner}** "
            f"with {values[winner]:.1f} mean total QoS violations."
        )
        st.markdown(
            "**Why Dynamic-RF is recommended:** it has the lowest mean QoS violations "
            "(131.2), beats Dynamic-MA in 8 of 10 seeds, and also has lower held-out "
            "prediction error than MA. This recommendation does not claim RF wins every "
            "latency, fairness, or stability metric."
        )
        st.caption(
            "This is a synthetic workload result. It is not a claim that the selected approach "
            "will be best on real 5G traffic."
        )

    if per_seed is not None and "total_qos_violations" in per_seed.columns:
        chart = per_seed.pivot(index="seed", columns="approach", values="total_qos_violations")
        st.line_chart(chart, y_label="Total QoS violations")
        st.dataframe(per_seed, use_container_width=True)


def artifact_view():
    metadata = artifact_path("models/rf_traffic_predictor_metadata.json")
    run_info = artifact_path("data/processed/dynamic_rf_results.csv.run_info.json")
    st.subheader("Model and run artifacts")
    for path in (metadata, run_info):
        if path.exists():
            st.markdown(f"**{path.relative_to(ROOT)}**")
            st.json(path.read_text(encoding="utf-8"))
        else:
            st.warning(f"Missing artifact: {path.relative_to(ROOT)}")


def main():
    st.set_page_config(page_title="5G Slicing Results", page_icon="📊", layout="wide")
    st.title("5G Network Slicing: Traffic and Predictor Results")
    st.sidebar.header("Views")
    view = st.sidebar.radio(
        "Open dashboard view",
        ["Seeded traffic", "Prediction accuracy", "Three-way comparison", "Multi-seed comparison", "Artifacts"],
    )

    if view == "Seeded traffic":
        traffic_view()
    elif view == "Prediction accuracy":
        prediction_view()
    elif view == "Three-way comparison":
        comparison_view()
    elif view == "Multi-seed comparison":
        multiseed_view()
    else:
        artifact_view()


if __name__ == "__main__":
    main()
