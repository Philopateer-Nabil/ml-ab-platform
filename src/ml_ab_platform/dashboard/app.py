"""Streamlit dashboard with six tabs.

Launch with::

    mlab dashboard

Reads directly from the SQLite DB populated by the FastAPI gateway — the
dashboard is a passive viewer, it doesn't own any state.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ml_ab_platform.analysis import StatisticalAnalyzer
from ml_ab_platform.config import get_settings
from ml_ab_platform.experiments import ExperimentStore
from ml_ab_platform.storage import get_conn

st.set_page_config(page_title="ML A/B Platform", layout="wide")

settings = get_settings()
store = ExperimentStore()


def _refresh_every(seconds: int) -> None:
    """Auto-rerun the app every N seconds without external dependencies."""
    key = "_refresh_last"
    if key not in st.session_state:
        st.session_state[key] = datetime.utcnow()
    # Streamlit will re-execute this file on each interaction; combined with
    # the `st_autorefresh` helper below we get a live-ish feel.
    try:
        from streamlit_autorefresh import st_autorefresh  # type: ignore
        st_autorefresh(interval=seconds * 1000, key="autorefresh")
    except ImportError:
        st.caption(f"(Manual refresh: press R or the button) refresh interval={seconds}s")


# --------------------------------------------------------------------------- #
def _load_predictions(exp_id: str) -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM predictions WHERE experiment_id = ?", conn, params=(exp_id,),
        )
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _load_feedback(exp_id: str) -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query(
            """SELECT f.*, p.prediction FROM feedback f
               JOIN predictions p USING(request_id)
               WHERE f.experiment_id = ?""",
            conn, params=(exp_id,),
        )
    if not df.empty:
        df["correct"] = (df["prediction"] == df["ground_truth"]).astype(int)
        df["received_at"] = pd.to_datetime(df["received_at"])
    return df


def _load_bandit_history(exp_id: str) -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM bandit_history WHERE experiment_id = ? ORDER BY id",
            conn, params=(exp_id,),
        )
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# --------------------------------------------------------------------------- #
st.title("ML Model A/B Testing Platform")
st.caption("Gateway + experiments + statistical analysis — all local.")

with st.sidebar:
    st.subheader("Settings")
    refresh = st.slider(
        "Auto-refresh (seconds)", 2, 60, value=settings.dashboard.refresh_seconds, step=1,
    )
    experiments = store.list()
    if not experiments:
        st.warning("No experiments yet. Create one via the CLI or API.")
        st.stop()

    # Default to the active experiment if any, else the most-recent one.
    active = store.active()
    default_idx = 0
    if active is not None:
        for i, e in enumerate(experiments):
            if e.id == active.id:
                default_idx = i
                break
    exp_selected = st.selectbox(
        "Experiment",
        experiments,
        index=default_idx,
        format_func=lambda e: f"{e.name} [{e.status.value}] ({e.id[:8]})",
    )

_refresh_every(refresh)

exp = exp_selected
preds = _load_predictions(exp.id)
fb = _load_feedback(exp.id)
bandit_hist = _load_bandit_history(exp.id)
analyzer = StatisticalAnalyzer()
try:
    analysis = analyzer.analyze(exp.id).to_dict()
except Exception as exc:
    st.error(f"Analysis failed: {exc}")
    analysis = {}

tabs = st.tabs([
    "Active Experiment",
    "Performance",
    "Statistical Results",
    "Traffic Over Time",
    "History",
    "Bandit vs Fixed Split",
])

# ============================ Tab 1: Active ================================ #
with tabs[0]:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", exp.status.value.upper())
    col1.caption(f"ID: {exp.id}")
    col2.metric("Strategy", exp.routing_strategy)
    col3.metric("Target metric", exp.target_metric)
    col4.metric("Min sample size", exp.minimum_sample_size)

    if preds.empty:
        st.info("No prediction traffic yet.")
    else:
        pie_df = preds.groupby("model_version").size().reset_index(name="count")
        fig_pie = px.pie(pie_df, names="model_version", values="count",
                         title="Traffic split (actual)")
        st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("Sample size vs target")
        required = analysis.get("required_sample_size") or exp.minimum_sample_size
        c1, c2 = st.columns(2)
        for col, v in zip((c1, c2), ("A", "B"), strict=True):
            n = int(fb[fb["model_version"] == v].shape[0]) if not fb.empty else 0
            col.metric(f"Model {v} feedback", n, delta=f"required ≈ {required}")
            pct = min(1.0, n / max(required, 1))
            col.progress(pct, text=f"{pct*100:.0f}% of required")

# ============================ Tab 2: Performance =========================== #
with tabs[1]:
    if fb.empty:
        st.info("No feedback yet.")
    else:
        m_a = analysis["model_a"]
        m_b = analysis["model_b"]
        rows = []
        for m in (m_a, m_b):
            rows.append({
                "model": m["version"],
                "accuracy": m["accuracy"],
                "positive_rate": m["positive_rate"],
                "p50 latency": m["latency_p50"],
                "p95 latency": m["latency_p95"],
                "p99 latency": m["latency_p99"],
                "throughput (qps)": m["throughput_qps"],
                "n_predictions": m["n_predictions"],
                "n_feedback": m["n_feedback"],
            })
        st.dataframe(pd.DataFrame(rows).set_index("model"))

        acc_test = analysis.get("accuracy_test") or {}
        if acc_test:
            # CI bars for accuracy difference
            lo, hi = acc_test["ci_low"], acc_test["ci_high"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=["Accuracy diff (B - A)"],
                y=[acc_test["diff"]],
                error_y={
                    "type": "data",
                    "array": [hi - acc_test["diff"]],
                    "arrayminus": [acc_test["diff"] - lo],
                },
                mode="markers", marker={"size": 14},
            ))
            fig.add_hline(y=0, line_dash="dot")
            fig.update_layout(
                title=f"Accuracy difference with {int((1 - analysis['alpha']) * 100)}% CI",
                yaxis_title="Difference (B − A)",
            )
            st.plotly_chart(fig, use_container_width=True)

        if not preds.empty:
            fig_lat = px.histogram(
                preds, x="latency_ms", color="model_version",
                nbins=50, barmode="overlay", opacity=0.55,
                title="Latency distribution (overlapping)",
            )
            st.plotly_chart(fig_lat, use_container_width=True)

# ============================ Tab 3: Statistics ============================ #
with tabs[2]:
    if not analysis:
        st.info("Not enough data for analysis yet.")
    else:
        verdict = analysis.get("verdict", "unknown")
        reason = analysis.get("verdict_reason", "")
        color = {
            "significant": "success",
            "not_enough_data": "info",
            "not_yet": "info",
            "no_difference": "warning",
            "promising": "warning",
        }.get(verdict, "info")
        getattr(st, color)(f"**Verdict:** {verdict.upper().replace('_', ' ')} — {reason}")

        c1, c2, c3 = st.columns(3)
        acc = analysis.get("accuracy_test") or {}
        if acc:
            c1.metric("p-value", f"{acc['p_value']:.4f}", delta=None)
            c2.metric("Effect size (Cohen's h)", f"{acc['cohen_h']:.3f}")
            c3.metric("Observed Δ accuracy", f"{acc['diff']*100:.2f} pp")

        c1, c2, c3 = st.columns(3)
        c1.metric("Current power", f"{(analysis.get('current_power') or 0):.2%}")
        c2.metric("Required sample size (per arm)",
                  analysis.get("required_sample_size") or "—")
        c3.metric("O'Brien-Fleming critical z",
                  f"{(analysis.get('obrien_fleming_critical_z') or 0):.3f}")

        if analysis.get("sequential_significant"):
            st.success("Sequential test crossed the O'Brien-Fleming boundary.")
        elif analysis.get("obrien_fleming_critical_z") is not None:
            st.info("Sequential test has not yet crossed the O'Brien-Fleming boundary.")

        for w in analysis.get("warnings", []):
            st.warning(w)

        latency = analysis.get("latency_test")
        if latency:
            st.subheader("Latency — Welch's t-test")
            st.json({k: round(v, 4) if isinstance(v, (int, float)) else v
                     for k, v in latency.items()})

# ============================ Tab 4: Traffic =============================== #
with tabs[3]:
    if preds.empty:
        st.info("No traffic yet.")
    else:
        preds["minute"] = preds["timestamp"].dt.floor("min")
        agg = preds.groupby(["minute", "model_version"]).size().reset_index(name="count")
        fig = px.line(agg, x="minute", y="count", color="model_version",
                      markers=True, title="Requests per minute")
        st.plotly_chart(fig, use_container_width=True)

        if not bandit_hist.empty:
            bandit_hist["cum_count"] = (
                bandit_hist.groupby("model_version").cumcount() + 1
            )
            totals = bandit_hist.groupby("timestamp").size().cumsum()
            share = (bandit_hist.groupby(["timestamp", "model_version"]).size()
                     .groupby(level=1).cumsum().reset_index(name="n"))
            fig = px.line(share, x="timestamp", y="n", color="model_version",
                          title="Bandit allocation over time (cumulative)")
            st.plotly_chart(fig, use_container_width=True)

# ============================ Tab 5: History =============================== #
with tabs[4]:
    rows = []
    for e in store.list():
        with get_conn() as conn:
            c = conn.execute(
                "SELECT COUNT(*) AS n FROM feedback WHERE experiment_id = ?",
                (e.id,),
            ).fetchone()
        pval = None
        if e.conclusion:
            acc = e.conclusion.get("accuracy_test") or {}
            pval = acc.get("p_value")
        duration = None
        if e.started_at and (e.stopped_at or e.concluded_at):
            end = e.concluded_at or e.stopped_at
            duration = (end - e.started_at).total_seconds() / 60
        rows.append({
            "id": e.id[:8],
            "name": e.name,
            "status": e.status.value,
            "winner": e.winner or "—",
            "feedback labels": c["n"],
            "duration (min)": round(duration, 1) if duration else None,
            "final p-value": round(pval, 4) if pval is not None else None,
        })
    st.dataframe(pd.DataFrame(rows))

# ============================ Tab 6: Bandit vs Fixed ======================= #
with tabs[5]:
    st.write("""
    This panel compares cumulative **reward** and **regret** across experiments.
    For bandit experiments, regret is logged in ``bandit_history``; for fixed-split
    experiments we approximate regret against the empirically-better arm.
    """)
    rows = []
    for e in store.list():
        strat = e.routing_strategy
        with get_conn() as conn:
            fb_rows = conn.execute(
                """SELECT f.model_version, p.prediction, f.ground_truth
                   FROM feedback f JOIN predictions p USING(request_id)
                   WHERE f.experiment_id = ? ORDER BY f.id""",
                (e.id,),
            ).fetchall()
        if not fb_rows:
            continue
        # empirical winner = model with higher accuracy; regret per choice =
        # (winner_acc - observed_reward) summed cumulatively
        per_model = {"A": [0, 0], "B": [0, 0]}
        for r in fb_rows:
            ver = r["model_version"]
            reward = int(r["prediction"] == r["ground_truth"])
            per_model[ver][0] += reward
            per_model[ver][1] += 1
        accs = {v: (k[0] / k[1]) if k[1] else 0 for v, k in per_model.items()}
        best_acc = max(accs.values())
        cum_reward = 0
        cum_regret = 0.0
        series = []
        for i, r in enumerate(fb_rows):
            ver = r["model_version"]
            reward = int(r["prediction"] == r["ground_truth"])
            cum_reward += reward
            cum_regret += best_acc - reward
            series.append({"i": i, "cum_reward": cum_reward,
                           "cum_regret": cum_regret, "strategy": strat,
                           "exp": e.name[:16]})
        rows.extend(series)
    if not rows:
        st.info("No completed feedback yet across experiments.")
    else:
        df = pd.DataFrame(rows)
        fig1 = px.line(df, x="i", y="cum_reward", color="exp",
                       line_dash="strategy", title="Cumulative reward")
        fig2 = px.line(df, x="i", y="cum_regret", color="exp",
                       line_dash="strategy", title="Cumulative regret")
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)

st.caption(f"Last rendered: {datetime.utcnow().isoformat()}  •  DB: {settings.database.path}")
