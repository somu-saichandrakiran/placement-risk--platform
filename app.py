# app.py
import streamlit as st
import requests
import plotly.graph_objects as go

API_URL = "http://localhost:8000/v1"

st.set_page_config(
    page_title="Placement Risk Intelligence",
    page_icon="📊",
    layout="wide"
)

# ── Styling ──────────────────────────────────────────────
st.markdown("""
<style>
.big-metric { font-size: 3rem; font-weight: 700; text-align: center; }
.section-header { font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; }
.disclaimer { font-size: 0.75rem; color: #888; margin-top: 1rem; }
.stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────
st.title("📊 Placement Risk Intelligence Platform")
st.caption("ML-powered · SHAP-explained · Calibrated probabilities")

# ── Sidebar — Student Input ───────────────────────────────
with st.sidebar:
    st.header("🎓 Student Profile")

    st.subheader("Academic")
    tenth   = st.slider("10th Percentage",  40.0, 100.0, 78.5, 0.5)
    twelfth = st.slider("12th Percentage",  40.0, 100.0, 72.0, 0.5)
    cgpa    = st.slider("Degree CGPA (%)",  40.0, 100.0, 67.5, 0.5)
    mba     = st.slider("MBA Percentage",   40.0, 100.0, 62.0, 0.5)
    etest   = st.slider("Employability Score", 40.0, 100.0, 74.0, 0.5)

    st.subheader("Experience")
    internships = st.number_input("Internships",    0, 10, 1)
    projects    = st.number_input("Projects",       0, 20, 3)
    hackathons  = st.number_input("Hackathons",     0, 10, 1)
    backlogs    = st.number_input("Active Backlogs",0, 20, 0)
    workex      = st.selectbox("Work Experience", ["No", "Yes"])

    st.subheader("Profile")
    stream  = st.selectbox("12th Stream",    ["Science", "Commerce", "Arts"])
    degree  = st.selectbox("Degree Type",    ["Sci&Tech", "Comm&Mgmt", "Others"])
    spec    = st.selectbox("MBA Specialisation", ["Mkt&Fin", "Mkt&HR"])
    tier    = st.selectbox("College Tier",   ["T1", "T2", "T3"])

    predict_btn = st.button("🔍 Predict Placement", type="primary", use_container_width=True)

# ── Main Panel ───────────────────────────────────────────
if predict_btn:
    payload = {
        "tenth_percent":       tenth,
        "twelfth_percent":     twelfth,
        "cgpa":                cgpa,
        "employability_score": etest,
        "mba_percent":         mba,
        "internship_count":    internships,
        "project_count":       projects,
        "hackathon_count":     hackathons,
        "active_backlogs":     backlogs,
        "work_exp_flag":       1 if workex == "Yes" else 0,
        "specialization_12th": stream,
        "degree_type":         degree,
        "specialization":      spec,
        "college_tier":        tier,
    }

    with st.spinner("Analysing profile..."):
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
            data = resp.json()
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

    prob    = data["placement"]["probability"]
    ci_low  = data["placement"]["confidence_interval"][0]
    ci_high = data["placement"]["confidence_interval"][1]
    salary  = data["salary"]
    drivers = data["top_drivers"]
    recs    = data["top_recommendations"]

    # ── Row 1: Gauge + Salary ────────────────────────────
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Placement Probability")

        color = (
            "#2d6a4f" if prob >= 0.70 else
            "#b5622a" if prob >= 0.45 else
            "#c0392b"
        )
        label = (
            "High Chance" if prob >= 0.70 else
            "Moderate"    if prob >= 0.45 else
            "Lower Chance"
        )

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 1),
            number={"suffix": "%", "font": {"size": 52, "color": color}},
            title={"text": f"{label}<br><sub>{ci_low:.0%} – {ci_high:.0%} confidence range</sub>"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": color},
                "steps": [
                    {"range": [0,  45], "color": "#fdecea"},
                    {"range": [45, 70], "color": "#fef3e2"},
                    {"range": [70, 100],"color": "#e8f5e9"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 3},
                    "thickness": 0.75,
                    "value": prob * 100,
                },
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=60, b=0, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.markdown("### Expected Salary")
        if salary["p50_inr"]:
            s_col1, s_col2, s_col3 = st.columns(3)
            s_col1.metric("Conservative (P10)", f"₹{salary['p10_inr']/100000:.1f}L")
            s_col2.metric("Expected (P50)",     f"₹{salary['p50_inr']/100000:.1f}L")
            s_col3.metric("Optimistic (P90)",   f"₹{salary['p90_inr']/100000:.1f}L")

            fig_sal = go.Figure()
            fig_sal.add_trace(go.Bar(
                x=["Conservative\n(P10)", "Expected\n(P50)", "Optimistic\n(P90)"],
                y=[salary["p10_inr"]/100000,
                   salary["p50_inr"]/100000,
                   salary["p90_inr"]/100000],
                marker_color=["#aed6f1", "#2980b9", "#1a5276"],
                text=[
                    f"₹{salary['p10_inr']/100000:.1f}L",
                    f"₹{salary['p50_inr']/100000:.1f}L",
                    f"₹{salary['p90_inr']/100000:.1f}L",
                ],
                textposition="outside",
            ))
            fig_sal.update_layout(
                height=240,
                margin=dict(t=20, b=20, l=10, r=10),
                yaxis_title="Salary (Lakhs ₹)",
                showlegend=False,
            )
            st.plotly_chart(fig_sal, use_container_width=True)
        else:
            st.info(salary.get("note", "Salary estimate unavailable"))

    st.divider()

    # ── Row 2: SHAP Waterfall + Recommendations ──────────
    col3, col4 = st.columns([1.2, 1])

    with col3:
        st.markdown("### What's driving this prediction")
        features = [d["feature"].replace("_", " ").title() for d in drivers]
        impacts  = [d["shap_value"] for d in drivers]
        colors   = ["#2d6a4f" if v > 0 else "#c0392b" for v in impacts]
        labels   = [f"+{v:.1%}" if v > 0 else f"{v:.1%}" for v in impacts]

        fig_shap = go.Figure(go.Bar(
            x=impacts,
            y=features,
            orientation="h",
            marker_color=colors,
            text=labels,
            textposition="outside",
        ))
        fig_shap.add_vline(x=0, line_width=1, line_color="gray")
        fig_shap.update_layout(
            height=300,
            margin=dict(t=10, b=10, l=10, r=80),
            xaxis_title="Impact on placement probability",
            xaxis=dict(zeroline=True),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    with col4:
        st.markdown("### Top actions to improve")
        if recs:
            for i, rec in enumerate(recs, 1):
                gain = rec["probability_gain"]
                if gain > 0:
                    st.success(
                        f"**{i}. {rec['action']}**  \n"
                        f"↑ +{gain:.1%} improvement"
                    )
                else:
                    st.info(
                        f"**{i}. {rec['action']}**  \n"
                        f"→ {gain:.1%} change"
                    )
        else:
            st.info("No actionable recommendations found.")

    st.divider()

    # ── Row 3: What-If Simulator ──────────────────────────
    st.markdown("### 🔬 What-If Simulator")
    st.caption("Adjust values below to see how changes affect your probability")

    w1, w2, w3, w4 = st.columns(4)
    new_internships = w1.slider("Internships",     0, 10, internships, key="wi1")
    new_projects    = w2.slider("Projects",        0, 20, projects,    key="wi2")
    new_backlogs    = w3.slider("Active Backlogs", 0, 20, backlogs,    key="wi3")
    new_hackathons  = w4.slider("Hackathons",      0, 10, hackathons,  key="wi4")

    if st.button("♻️ Recalculate", use_container_width=True):
        modified = {
            **payload,
            "internship_count": new_internships,
            "project_count":    new_projects,
            "active_backlogs":  new_backlogs,
            "hackathon_count":  new_hackathons,
        }
        with st.spinner("Recalculating..."):
            resp2 = requests.post(f"{API_URL}/predict", json=modified, timeout=15)
            data2 = resp2.json()

        new_prob = data2["placement"]["probability"]
        delta    = new_prob - prob

        r1, r2, r3 = st.columns(3)
        r1.metric("Original probability", f"{prob:.1%}")
        r2.metric("New probability",      f"{new_prob:.1%}")
        r3.metric("Change",               f"{delta:+.1%}",
                  delta_color="normal" if delta >= 0 else "inverse")

    st.divider()

    # ── Footer ────────────────────────────────────────────
    st.markdown(
        f"<div class='disclaimer'>⚠️ {data['bias_disclaimer']} "
        f"| Model: {data['model_version']} "
        f"| Latency: {data['latency_ms']}ms</div>",
        unsafe_allow_html=True
    )

else:
    # Landing state
    st.info(
        "👈 Fill in the student profile in the sidebar and click "
        "**Predict Placement** to get started."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Model",     "XGBoost + Calibrated")
    c2.metric("ROC-AUC",   "0.865")
    c3.metric("Explainer", "SHAP TreeExplainer")