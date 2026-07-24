"""
Dark Satellite Hunter - Interactive SDA Vetting Dashboard (`app/streamlit_app.py`)

Dark-mode dashboard for vetting uncorrelated optical transients: candidate
catalog, phase-folded light-curve inspection with glint highlighting, and a
2-D projection of the object representation space.

Scientific integrity: every score shown here is computed from the photometric
and astrometric data alone (domain features + positional cross-matching).
Simulation ground truth appears ONLY in the clearly marked validation panel,
after scoring - exactly how one would validate against a labeled holdout.
"""
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure the repository root is importable when run via `streamlit run`.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

st.set_page_config(
    page_title="Dark Satellite Hunter | SDA Vetting Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    from src.data.mock_generator import build_reference_catalog, generate_mock_ztf_dataset
    from src.models.baselines import DomainFeatureExtractor, infer_glint_material
    from src.pipeline.anomaly_scorer import HybridAnomalyScorer
    from src.pipeline.crossmatch import CrossmatchEngine
    from src.pipeline.orbital_fit import estimate_glint_periodicity, fit_preliminary_orbit
except Exception as exc:  # pragma: no cover - surfaced in the UI
    st.error(
        f"Failed to import the dark-satellite-hunter package: {exc}\n\n"
        "Run the app from the repository root: `streamlit run app/streamlit_app.py`"
    )
    st.stop()

# ----------------------------------------------------------------------------- #
# Theme
# ----------------------------------------------------------------------------- #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
.stApp { background-color: #0B0E14; font-family: 'Outfit', sans-serif; color: #E2E8F0; }
h1, h2, h3 { font-family: 'Outfit', sans-serif; font-weight: 700; letter-spacing: -0.5px; }
.glow-title {
    background: linear-gradient(90deg, #00F0FF 0%, #7000FF 50%, #FF5E00 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 2.8rem; font-weight: 700; margin-bottom: 0px;
}
.sub-title { color: #94A3B8; font-size: 1.1rem; margin-bottom: 2rem; }
.metric-card {
    background: rgba(20, 24, 36, 0.7); border: 1px solid rgba(0, 240, 255, 0.2);
    border-radius: 12px; padding: 1.2rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); backdrop-filter: blur(8px);
    transition: all 0.3s ease;
}
.metric-card:hover { border-color: #00F0FF; transform: translateY(-2px); }
.metric-label { font-size: 0.85rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { font-size: 1.8rem; font-weight: 700; color: #00F0FF; margin-top: 0.3rem; }
.metric-highlight { color: #FF5E00; }
section[data-testid="stSidebar"] { background-color: #0F131D; border-right: 1px solid rgba(255,255,255,0.08); }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- #
# Discovery pipeline (cached) - mirrors run_discovery.py, label-free
# ----------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def run_discovery_pipeline(n_clean: int = 300, n_debris: int = 60, seed: int = 42,
                           anomaly_threshold: float = 2.0):
    records = generate_mock_ztf_dataset(n_clean=n_clean, n_debris=n_debris, seed=seed)
    catalog = build_reference_catalog(records)
    N = len(records)

    # 1. Representations from data only.
    feats = DomainFeatureExtractor().fit_transform(records).astype(np.float64)
    recon_proxy = feats[:, 10]  # log10 reduced chi^2 vs constant source

    # 2. Positional cross-match -> reference population.
    engine = CrossmatchEngine(reference_catalog=catalog, offline_mode=True,
                              anomaly_threshold=anomaly_threshold)
    mean_ra = np.array([float(np.mean(r["ra"])) for r in records])
    mean_dec = np.array([float(np.mean(r["dec"])) for r in records])
    matched = np.array([
        engine.check_reference_catalog(mean_ra[i], mean_dec[i])[0] for i in range(N)
    ])

    ref_idx = np.flatnonzero(matched) if matched.sum() >= 30 else np.arange(N)

    # 3. Calibrate scorer on the catalog-matched population; score everyone.
    scorer = HybridAnomalyScorer(alpha=0.5, beta=0.5, density_method="mahalanobis")
    scorer.fit(feats[ref_idx], recon_proxy[ref_idx])
    surprisal = scorer.score(recon_proxy, feats)["surprisal_score"]

    # 4. Tag + kinematics per object.
    rows = []
    for i, rec in enumerate(records):
        period_info = estimate_glint_periodicity(rec["t"], rec["mag"])
        kin = fit_preliminary_orbit(rec["t"], rec["ra"], rec["dec"])
        tag = engine.tag_candidate(
            object_id=rec["object_id"], ra_deg=mean_ra[i], dec_deg=mean_dec[i],
            mjd_obs=float(np.median(rec["raw_t"])), surprisal_score=float(surprisal[i]),
        )
        rows.append({
            "object_id": rec["object_id"],
            "RA": round(mean_ra[i], 5),
            "Dec": round(mean_dec[i], 5),
            "Surprisal_Score": round(float(surprisal[i]), 3),
            "Status": tag["status"],
            "Matched_Catalog": tag["matched_catalog"],
            "Period_Days": round(period_info["period_days"], 4),
            "Period_Hours": round(period_info["period_hours"], 2),
            "Phase_Sharpness": round(period_info["phase_sharpness"], 3),
            "Period_FAP": round(period_info["false_alarm_prob"], 4),
            "Num_Glints": int(period_info["num_glints"]),
            "Material_Inference": infer_glint_material(
                float(feats[i, 14]), int(period_info["num_glints"])
            ),
            "Rate_deg_day": round(kin["angular_rate_deg_day"], 5),
            "Orbital_Regime": kin["regime"],
            # Simulation truth - used ONLY in the validation panel below.
            "True_Label": int(rec["label"]),
            "True_Class": rec.get("true_class", "?"),
            # Raw arrays for the light-curve viewer.
            "t": rec["t"], "band": rec["band"],
            "raw_mag": rec["raw_mag"], "raw_magerr": rec["raw_magerr"],
        })

    df = pd.DataFrame(rows).sort_values("Surprisal_Score", ascending=False).reset_index(drop=True)

    # 5. Validation vs simulation truth (post-hoc, never feeds the scores).
    from sklearn.metrics import roc_auc_score
    val_auc = float(roc_auc_score(df["True_Label"], df["Surprisal_Score"]))
    flagged = df["Status"] == "UNCORRELATED_ANOMALY"
    precision = float(df.loc[flagged, "True_Label"].mean()) if flagged.any() else 0.0
    recall = float(flagged[df["True_Label"] == 1].mean()) if (df["True_Label"] == 1).any() else 0.0

    validation = {"roc_auc": val_auc, "precision": precision, "recall": recall,
                  "n_flagged": int(flagged.sum())}
    return df, feats.astype(np.float32), validation


# ----------------------------------------------------------------------------- #
# Header + sidebar
# ----------------------------------------------------------------------------- #
st.markdown('<div class="glow-title">DARK SATELLITE HUNTER</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Label-free anomaly discovery on irregular optical '
            'time-series - domain features + positional cross-matching '
            '(swap in PatchTST-MAE latents via <code>run_discovery.py --ckpt_path</code>)</div>',
            unsafe_allow_html=True)

st.sidebar.markdown("### 🌌 Survey Simulation")
n_clean = st.sidebar.slider("Cataloged sources", 100, 800, 300, step=50)
n_debris = st.sidebar.slider("Uncataloged debris", 20, 200, 60, step=10)
seed = st.sidebar.number_input("Random seed", value=42, step=1)
anomaly_threshold = st.sidebar.slider("Anomaly threshold (robust z)", 0.5, 6.0, 2.0, step=0.25)

with st.spinner("Running label-free discovery pipeline..."):
    df, latents, validation = run_discovery_pipeline(
        n_clean=int(n_clean), n_debris=int(n_debris), seed=int(seed),
        anomaly_threshold=float(anomaly_threshold),
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Candidate Filters")
status_options = sorted(df["Status"].unique().tolist())
default_status = [s for s in ("UNCORRELATED_ANOMALY", "KNOWN_SATELLITE") if s in status_options] or status_options
status_filter = st.sidebar.multiselect("Cross-Match Status", options=status_options, default=default_status)
min_surprisal = st.sidebar.slider("Min Surprisal Score", -2.0, 20.0, 1.0, step=0.5)
glint_filter = st.sidebar.checkbox("Only candidates with >1 specular glint", value=False)

filtered_df = df[df["Status"].isin(status_filter) & (df["Surprisal_Score"] >= min_surprisal)]
if glint_filter:
    filtered_df = filtered_df[filtered_df["Num_Glints"] > 1]

# ----------------------------------------------------------------------------- #
# KPI cards
# ----------------------------------------------------------------------------- #
c1, c2, c3, c4 = st.columns(4)
uncorrelated = df[df["Status"] == "UNCORRELATED_ANOMALY"]
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Uncorrelated Anomalies</div>'
                f'<div class="metric-value metric-highlight">{len(uncorrelated)}</div></div>',
                unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Max Surprisal</div>'
                f'<div class="metric-value">{df["Surprisal_Score"].max():.2f} σ</div></div>',
                unsafe_allow_html=True)
with c3:
    avg_period = uncorrelated["Period_Hours"].mean()
    period_text = f"{avg_period:.2f} hrs" if np.isfinite(avg_period) else "—"
    st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Tumbling Period</div>'
                f'<div class="metric-value">{period_text}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Light Curves Scanned</div>'
                f'<div class="metric-value">{len(df)}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Validation panel: truth used only AFTER scoring.
with st.expander("🧪 Validation against simulation ground truth (labels never used in scoring)",
                 expanded=True):
    v1, v2, v3 = st.columns(3)
    v1.metric("Surprisal ROC-AUC vs truth", f"{validation['roc_auc']:.4f}")
    v2.metric("Flag precision", f"{validation['precision']:.1%}")
    v3.metric("Flag recall", f"{validation['recall']:.1%}")
    st.caption(
        "The pipeline scores every object using photometry + astrometry only; the simulation's "
        "ground-truth labels are compared against those scores afterwards, exactly like a "
        "labeled-holdout validation on real injected data."
    )

# ----------------------------------------------------------------------------- #
# Tabs
# ----------------------------------------------------------------------------- #
tab1, tab2, tab3 = st.tabs([
    "📋 Candidate Discovery Catalog",
    "🔬 Phase-Folding Light Curve Viewer",
    "🌌 Representation Space",
])

with tab1:
    st.markdown("#### High-Surprisal Candidates")
    show_truth = st.checkbox("Show simulation ground truth columns (validation only)", value=False)
    display_cols = ["object_id", "Surprisal_Score", "Status", "Matched_Catalog",
                    "Period_Hours", "Phase_Sharpness", "Period_FAP", "Num_Glints",
                    "Material_Inference", "Rate_deg_day", "Orbital_Regime", "RA", "Dec"]
    if show_truth:
        display_cols += ["True_Class", "True_Label"]

    st.dataframe(
        filtered_df[display_cols],
        use_container_width=True, height=420,
        column_config={
            "Surprisal_Score": st.column_config.NumberColumn("Surprisal σ", format="%.2f"),
            "Period_Hours": st.column_config.NumberColumn("Tumbling (hrs)", format="%.2f"),
            "Rate_deg_day": st.column_config.NumberColumn("Rate (°/day)", format="%.5f"),
        },
    )
    st.download_button(
        "📥 Download Filtered Candidates CSV",
        data=filtered_df[display_cols].to_csv(index=False),
        file_name="candidates_final.csv", mime="text/csv", type="primary",
    )

with tab2:
    st.markdown("#### Interactive Photometric Inspector & Phase Folding")
    if len(filtered_df) == 0:
        st.warning("No candidates match the current filters - adjust the sidebar.")
    else:
        col_sel, col_ctrl = st.columns([1, 2])
        with col_sel:
            selected = st.selectbox("Candidate Object ID:", filtered_df["object_id"].tolist())
            obj = filtered_df[filtered_df["object_id"] == selected].iloc[0]
            st.markdown("##### Candidate Metadata")
            st.markdown(f"**Status:** `{obj['Status']}`")
            st.markdown(f"**Surprisal:** `{obj['Surprisal_Score']} σ`")
            st.markdown(f"**Catalog match:** `{obj['Matched_Catalog']}`")
            st.markdown(f"**Tumbling period:** `{obj['Period_Hours']} h` "
                        f"(sharpness `{obj['Phase_Sharpness']}`, FAP `{obj['Period_FAP']}`)")
            st.markdown(f"**Material hypothesis:** `{obj['Material_Inference']}`")
            st.markdown(f"**Kinematics:** `{obj['Orbital_Regime']}` at `{obj['Rate_deg_day']}°/day`")

        with col_ctrl:
            fold_mode = st.radio("Display Mode:",
                                 ["Raw Time-Series", "Phase-Folded (Rotation Period)"],
                                 horizontal=True)
            t_data = np.asarray(obj["t"], dtype=np.float64)
            mag_data = np.asarray(obj["raw_mag"], dtype=np.float64)
            err_data = np.asarray(obj["raw_magerr"], dtype=np.float64)
            band_data = np.asarray(obj["band"])

            fold_period = None
            if fold_mode == "Phase-Folded (Rotation Period)":
                fold_period = st.slider(
                    "Fold Period (days):", 0.02, 3.00,
                    float(np.clip(obj["Period_Days"], 0.02, 3.0)), 0.001, format="%.3f",
                )

            def x_axis(times: np.ndarray) -> np.ndarray:
                return (times % fold_period) / fold_period if fold_period else times

            fig = go.Figure()
            for b, name, color in ((0, "g-band", "#00F0FF"), (1, "r-band", "#FF5E00")):
                m = band_data == b
                fig.add_trace(go.Scatter(
                    x=x_axis(t_data[m]), y=mag_data[m],
                    error_y=dict(type="data", array=err_data[m], visible=True,
                                 color=f"rgba(148,163,184,0.35)"),
                    mode="markers", name=name,
                    marker=dict(color=color, symbol="circle" if b else "square",
                                size=8, line=dict(width=1, color="#0B0E14")),
                ))

            # Glint highlighting from robust bright outliers (same rule as the
            # feature extractor - no ground truth involved).
            med = np.median(mag_data)
            robust_sigma = 1.4826 * np.median(np.abs(mag_data - med)) + 1e-6
            glint_mask = (med - mag_data) > 3.0 * robust_sigma
            if np.any(glint_mask):
                fig.add_trace(go.Scatter(
                    x=x_axis(t_data[glint_mask]), y=mag_data[glint_mask],
                    mode="markers", name="Specular glints (>3σ bright)",
                    marker=dict(color="#FFD700", symbol="star", size=14,
                                line=dict(width=1.5, color="#FF5E00")),
                ))

            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,19,29,0.8)",
                xaxis_title="Tumbling phase (0-1)" if fold_period else "Days since first epoch",
                yaxis_title="Magnitude (bright ↓)",
                yaxis=dict(autorange="reversed"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=60, b=40), height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("#### Object Representation Space (12-D domain features)")
    st.markdown(
        "2-D projection of the per-object feature vectors used for Mahalanobis scoring. "
        "Uncorrelated anomalies separate from the astrophysical locus because their "
        "variability statistics (χ², glint ratio, skew) are unlike any cataloged class."
    )
    try:
        import umap
        proj = umap.UMAP(n_neighbors=25, min_dist=0.1, n_components=2,
                         random_state=int(seed)).fit_transform(latents)
        proj_name = "UMAP"
    except Exception:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        proj = PCA(n_components=2, random_state=int(seed)).fit_transform(
            StandardScaler().fit_transform(latents))
        proj_name = "PCA"

    plot_df = pd.DataFrame({
        "Dim_1": proj[:, 0], "Dim_2": proj[:, 1],
        "Status": df["Status"], "object_id": df["object_id"],
        "Surprisal_Score": df["Surprisal_Score"],
    })
    fig_latent = px.scatter(
        plot_df, x="Dim_1", y="Dim_2", color="Status",
        hover_data=["object_id", "Surprisal_Score"],
        color_discrete_map={
            "UNCORRELATED_ANOMALY": "#FF5E00",
            "KNOWN_SATELLITE": "#00F0FF",
            "KNOWN_STAR_OR_QSO": "#00E676",
            "KNOWN_ASTEROID": "#E040FB",
            "UNMATCHED_LOW_SIGNIFICANCE": "#94A3B8",
        },
        labels={"Dim_1": f"{proj_name} 1", "Dim_2": f"{proj_name} 2"},
    )
    fig_latent.update_traces(marker=dict(size=9, line=dict(width=1, color="#0B0E14")))
    fig_latent.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,19,29,0.8)",
        margin=dict(l=40, r=40, t=40, b=40), height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_latent, use_container_width=True)
