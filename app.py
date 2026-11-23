"""BANA 212 Team 6: El Niño in the TAO buoy record, NOAA's ONI and Los Angeles winter rain.

The app only reads files saved by notebooks/elnino_analysis_v4.ipynb: data/summary.json (results frozen on 22 Nov 2026),
the small tables in data/, and the scikit-learn pipelines plus metadata.json in models/.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
DATA, MODELS = ROOT / "data", ROOT / "models"

INK, MUTED, GRID = "#1d1d1f", "#6e6e73", "#ececf0"
BLUE, WARM, AMBER, RAIN, GREY = "#0a84ff", "#ff5a36", "#ff9f0a", "#32ade6", "#c7c7cc"
PHASES = ["La Niña", "Neutral", "El Niño"]
PHASE_COLORS = {"La Niña": RAIN, "Neutral": GREY, "El Niño": WARM}
MODEL_COLORS = {"persistence": "#8e8e93", "logistic": BLUE, "random_forest": AMBER}
MODEL_SHORT = {"persistence": "Persistence (latest ONI)", "logistic": "Logistic regression", "random_forest": "Random forest"}
LRO, RFO, PERSIST = "Logistic, buoys + latest ONI", "Random forest, buoys + latest ONI", "Persistence (latest ONI)"
SUMMARY_KEY = {"persistence": PERSIST, "logistic": LRO, "random_forest": RFO}
SANS = "Inter, -apple-system, 'Helvetica Neue', Arial, sans-serif"
MONO = "'IBM Plex Mono', ui-monospace, Menlo, monospace"
HORIZON_TEXT = {0: "this month", 1: "1 month ahead", 2: "2 months ahead", 3: "3 months ahead", 6: "6 months ahead"}

st.set_page_config(page_title="El Niño signals · BANA 212 Team 6", layout="wide")
st.markdown(f"""<style>
[data-testid="stMetricValue"], [data-testid="stMetricDelta"] {{ font-family: {MONO}; }}
.small-label {{ font-family: {MONO}; font-size: 0.78rem; letter-spacing: 0.04em; text-transform: uppercase; color: {MUTED}; }}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------------------------- loading
@st.cache_data
def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


@st.cache_resource
def load_models() -> dict:
    meta = load_json("models/metadata.json")
    return {(int(k[1:]), kind): joblib.load(ROOT / info["file"])
            for k, block in meta["live_models"]["by_horizon"].items() for kind, info in block["models"].items()}


S = load_json("data/summary.json")
META = load_json("models/metadata.json")
FEATURES = META["features"]
FEATURE_NAMES = [f["name"] for f in FEATURES]
HORIZONS = META["horizons"]


# ------------------------------------------------------------------------------------------------------------- helpers
def month_label(ym: str | pd.Timestamp) -> str:
    return f"{pd.Timestamp(str(ym)[:7] + '-01'):%b %Y}"


def add_months(ym: str, k: int) -> str:
    return f"{pd.Timestamp(ym + '-01') + pd.DateOffset(months=k):%Y-%m}"


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def ratio(text: str) -> float:
    a, b = text.split("/")
    return int(a) / int(b)


def style(fig: go.Figure, height: int = 380, legend: bool = True) -> go.Figure:
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=8, r=8, t=36, b=8),
                      font=dict(family=SANS, color=INK, size=13), hoverlabel=dict(font_family=MONO),
                      paper_bgcolor="white", plot_bgcolor="white", showlegend=legend,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, tickfont=dict(family=MONO, size=11), linecolor="#d2d2d7")
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="#d2d2d7", tickfont=dict(family=MONO, size=11))
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def frozen_q2(block: dict) -> pd.DataFrame:
    """summary.json Q2 block {k0: {model: {...}}} -> one row per horizon and model."""
    rows = [{"k": int(k[1:]), "model": kind, "balanced_accuracy": v[SUMMARY_KEY[kind]]["balanced_accuracy"],
             "changes": f"{v[SUMMARY_KEY[kind]]['changes_caught']}/{v[SUMMARY_KEY[kind]]['phase_changes']}"}
            for k, v in block.items() for kind in SUMMARY_KEY]
    return pd.DataFrame(rows)


def setting_text(kind: str, p: dict) -> str:
    if kind == "logistic":
        return f"C = {p['C']:g}"
    depth = "none" if p["max_depth"] is None else p["max_depth"]
    return f"{p['n_estimators']} trees · max depth {depth} · min leaf {p['min_samples_leaf']}"


# ------------------------------------------------------------------------------------------------------------- header
st.markdown('<p class="small-label">BANA 212 · Team 6 · results frozen 22 Nov 2026</p>', unsafe_allow_html=True)
st.title("El Niño: from Pacific buoys to Los Angeles rain")
st.caption("Can the equatorial Pacific buoy array tell us early that El Niño is coming, and what does that mean for winter rain in "
           "Los Angeles?")

tab_exec, tab_desc, tab_perf, tab_pred = st.tabs(
    ["Executive Summary", "Descriptive Analytics", "Model Performance", "Interactive Prediction"])

# ===================================================================================================== Executive Summary
with tab_exec:
    d, ext, la, q1, q2, q3 = S["dataset"], S["extension"], S["la_rain"], S["q1"], S["q2"], S["q3"]
    wf3, ms3 = ext["q2_walk_forward"]["k3"], q2["main_split"]["k3"]
    gaps = q1["gap_elnino_minus_lanina_sd"]
    eo = q3["elnino_vs_other"]
    w = q3["winter_2026_27"]
    out = ext["outlook_2026_27"]

    st.subheader("The question")
    st.markdown(
        "El Niño warms the central and eastern Pacific and tends to steer more winter storms into Southern California. We asked "
        "three questions:\n"
        "1. **Q1** How do ocean and air conditions along the equator differ between El Niño, La Niña and neutral months?\n"
        "2. **Q2** Which buoy readings signal the phase best, and can they call it earlier than NOAA's official index (ONI)?\n"
        "3. **Q3** After an El Niño fall, how much more likely is a wet, stormy winter in Los Angeles, and what does that mean "
        "for winter 2026–27?")

    st.subheader("The data")
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.markdown('<p class="small-label">Ocean buoys</p>', unsafe_allow_html=True)
        st.markdown(f"**UCI El Niño dataset**: {d['rows']:,} daily readings from {d['buoy_sites']} TAO buoys, "
                    f"{d['date_start'][:4]}–{d['date_end'][:4]}. Extended with **NOAA PMEL GTMBA** to "
                    f"{month_label(ext['date_end'])}: {ext['combined_rows']:,} buoy-days in total.")
    with c2.container(border=True):
        st.markdown('<p class="small-label">ENSO index</p>', unsafe_allow_html=True)
        st.markdown(f"**NOAA CPC Oceanic Niño Index (ONI)**, {S['oni_file']['first_month'][:4]}–"
                    f"{month_label(S['oni_file']['last_month'])}: El Niño if ONI ≥ +0.5, La Niña if ≤ −0.5. "
                    "The Relative ONI (RONI) is used as a check.")
    with c3.container(border=True):
        st.markdown('<p class="small-label">Los Angeles rain</p>', unsafe_allow_html=True)
        st.markdown(f"**NOAA GHCN-Daily {la['station_id']}** (Los Angeles Downtown, USC): {la['n_seasons']} rain seasons "
                    f"(July–June), {la['seasons_first']}–{la['seasons_last']}. A wet winter = above the median "
                    f"{la['median_inches']:.2f} in.")

    st.subheader("What we found")
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.markdown('<p class="small-label">Q1 · El Niño vs La Niña</p>', unsafe_allow_html=True)
        st.metric("Sea surface temperature gap", f"{gaps['sst_anom']['gap']:.2f} SD")
        gn = ext["q1_gap_sd"]["gtmba_1998_on"]
        st.markdown(f"Air temperature {gaps['air_temp_anom']['gap']:.2f} SD, zonal wind {gaps['zon_wind_anom']['gap']:.2f} SD "
                    f"(weaker trade winds in El Niño); humidity ({gaps['humidity_anom']['gap']:+.2f}) and meridional wind "
                    f"({gaps['mer_wind_anom']['gap']:+.2f}) barely move. On the new {ext['q1_periods']['gtmba_1998_on'].split(' ')[0]} "
                    f"buoy data: {gn['sst_anom']['gap']:.2f}, {gn['air_temp_anom']['gap']:.2f} and {gn['zon_wind_anom']['gap']:.2f} SD.")
    with c2.container(border=True):
        st.markdown('<p class="small-label">Q2 · Calling the phase 3 months ahead</p>', unsafe_allow_html=True)
        st.metric(f"Balanced accuracy, test years 1999–{ext['last_full_month'][:4]}", f"{wf3[LRO]['balanced_accuracy']:.2f}",
                  f"{wf3[LRO]['balanced_accuracy'] - wf3[PERSIST]['balanced_accuracy']:+.2f} vs persistence "
                  f"{wf3[PERSIST]['balanced_accuracy']:.2f}")
        st.markdown(f"Logistic regression on buoy readings plus the latest ONI catches {wf3[LRO]['changes_caught']} of "
                    f"{wf3[LRO]['phase_changes']} phase changes. On the 1995–98 test: {ms3[LRO]['balanced_accuracy']:.2f} vs "
                    f"{ms3[PERSIST]['balanced_accuracy']:.2f}. 0.33 = guessing.")
    with c3.container(border=True):
        st.markdown('<p class="small-label">Q3 · LA winter after an El Niño fall</p>', unsafe_allow_html=True)
        st.metric("Chance of a wetter-than-median winter", pct(ratio(eo["wet_elnino"])),
                  f"{(ratio(eo['wet_elnino']) - ratio(eo['wet_other'])) * 100:+.0f} pts vs other falls", delta_color="off")
        wet_en, n_en = eo["wet_elnino"].split("/")
        wet_ot, n_ot = eo["wet_other"].split("/")
        st.markdown(f"{wet_en} of {n_en} winters after an El Niño fall were wetter than the median, vs {wet_ot} of {n_ot} after "
                    f"other falls ({pct(ratio(eo['wet_other']))}). Heavy-rain days (≥ 1 inch) per season: "
                    f"{eo['heavy_mean_elnino']:g} vs {eo['heavy_mean_other']:g}.")

    st.subheader("This winter: 2026–27")
    lro_calls = sorted((c for c in out["calls"] if c["model"] == LRO), key=lambda c: c["k"])
    all_en = all(c["call"] == "El Niño" for c in lro_calls)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"ONI, {S['oni_file']['latest_season']}", f"{w['aso_oni']:+.2f}",
              f"#{w['aso_oni_rank_since_1950']} since 1950", delta_color="off", delta_arrow="off", border=True)
    c2.metric(f"Niño 3.4 buoys, {month_label(out['issue_month'])}", f"{out['nino34_sst_anom_c']:+.2f} °C",
              f"{out['buoys_reporting']} buoys reporting", delta_color="off", delta_arrow="off", border=True)
    c3.metric("Model call", "El Niño" if all_en else lro_calls[-1]["call"],
              f"through {month_label(lro_calls[-1]['target_month'])}" if all_en else "see Model Performance",
              delta_color="off", delta_arrow="off", border=True)
    c4.metric(f"LA rain since 1 Jul (to {pd.Timestamp(w['through']):%-d %b})", f"{w['season_to_date_in']:.2f} in",
              f"median by this date {w['median_to_date_1950_2025_in']:.2f} in", delta_color="off", delta_arrow="off", border=True)
    dry_strong = [f["winter"] for f in q3["strong_falls"] if not f["wet"]]
    st.markdown(
        f"This fall's ONI ranks #{w['aso_oni_rank_since_1950']} of {la['aso_rank_out_of']} falls since 1950 (RONI "
        f"{w['aso_roni']:+.2f}, #{w['aso_roni_rank_since_1950']}). After past El Niño falls, {pct(ratio(eo['wet_elnino']))} "
        "of LA winters were wet, so the odds lean towards a wet, stormy winter. This is a **seasonal risk outlook, not a "
        f"forecast**: {sum(f['wet'] for f in q3['strong_falls'])} of {len(q3['strong_falls'])} strong falls since 1950 "
        "(ONI or RONI ≥ 1.5) brought a wet winter"
        + (f", but {' and '.join(dry_strong)} stayed dry." if dry_strong else "."))

# ================================================================================================= Descriptive Analytics
with tab_desc:
    buoys = load_csv("buoys_monthly.csv")
    buoys["date"] = pd.to_datetime(buoys["month"] + "-01")
    oni = load_csv("oni_monthly.csv")
    oni["date"] = pd.to_datetime(oni["month"] + "-01")
    seasons = load_csv("la_rain_seasons.csv")

    st.subheader("1 · How many buoys report each month")
    fig = go.Figure()
    for src, color in [("UCI", GREY), ("NOAA PMEL GTMBA", BLUE)]:
        b = buoys[buoys["source"] == src]
        fig.add_bar(x=b["date"], y=b["sst_buoys"], name=f"{src} file", marker_color=color, marker_line_width=0,
                    hovertemplate="%{x|%b %Y}: %{y} buoys<extra></extra>")
    fig.add_hline(y=5, line_dash="dot", line_color=INK, annotation_text="5 buoys: minimum for a monthly value",
                  annotation_position="top left", annotation_font=dict(size=11, color=MUTED))
    fig.update_layout(bargap=0, yaxis_title="Buoys reporting SST")
    show(style(fig, 330))
    st.caption("The UCI file covers 1980–June 1998; NOAA PMEL's GTMBA files continue the same moorings. The array thinned after 2012.")

    st.subheader("2 · ONI since 1950, coloured by phase")
    c1, c2 = st.columns([3, 1])
    years = c1.slider("Years", 1950, int(oni["date"].dt.year.max()), (1980, int(oni["date"].dt.year.max())), key="oni_years")
    overlay = c2.checkbox("Add the buoys' Niño 3.4 SST anomaly", value=True, key="oni_overlay")
    o = oni[oni["date"].dt.year.between(*years)]
    fig = go.Figure()
    for p in PHASES:
        op = o[o["phase"] == p]
        fig.add_bar(x=op["date"], y=op["oni"], name=p, marker_color=PHASE_COLORS[p], marker_line_width=0,
                    hovertemplate="%{x|%b %Y}: ONI %{y:+.2f}<extra>" + p + "</extra>")
    if overlay:
        b = buoys[buoys["date"].dt.year.between(*years)]
        fig.add_scatter(x=b["date"], y=b["nino34_buoy_sst_anom"], mode="lines", name="Buoy SST anomaly, Niño 3.4 box",
                        line=dict(color=INK, width=1.2), hovertemplate="%{x|%b %Y}: %{y:+.2f} °C<extra>buoys</extra>")
    for y in (0.5, -0.5):
        fig.add_hline(y=y, line_dash="dot", line_color=MUTED, line_width=1)
    fig.update_layout(bargap=0, barmode="overlay", yaxis_title="°C")
    show(style(fig, 360))
    st.caption("ONI = 3-month mean sea surface temperature anomaly in the Niño 3.4 region (NOAA CPC). Dotted lines: ±0.5 °C "
               "phase thresholds. The buoy line uses months with at least 3 buoys inside the box.")

    st.subheader("3 · El Niño vs La Niña along the equator")
    prof = S["q1"]["equator_profile"]
    var = st.radio("Variable", ["Sea surface temperature", "Zonal wind"], horizontal=True, key="profile_var")
    block, unit = (prof["sst_by_lon"], "°C") if var == "Sea surface temperature" else (prof["zonal_wind_by_lon"], "m/s")
    lons = list(block)
    fig = go.Figure()
    for p in PHASES:
        fig.add_scatter(x=lons, y=[block[lon][p] for lon in lons], mode="lines+markers", name=p,
                        line=dict(color=PHASE_COLORS[p], width=2.5), marker=dict(size=8),
                        hovertemplate="%{x}: %{y:+.2f} " + unit + "<extra>" + p + "</extra>")
    fig.add_hline(y=0, line_color="#d2d2d7")
    fig.update_layout(xaxis_title="Buoy longitude along the equator (2°S–2°N), west → east",
                      yaxis_title=f"Mean anomaly ({unit}{', + = westerly' if unit == 'm/s' else ''})")
    show(style(fig, 360))
    st.caption(f"1980–98 daily anomalies vs each buoy's calendar-month mean. The SST gap peaks at {prof['sst_gap_max_lon']} "
               f"({prof['sst_gap_max_c']:+.2f} °C); the zonal-wind gap peaks near the dateline at {prof['zonal_wind_gap_max_lon']} "
               f"({prof['zonal_wind_gap_max_ms']:+.2f} m/s).")

    st.subheader("4 · Does the El Niño − La Niña gap hold on new years?")
    q1g, periods = S["extension"]["q1_gap_sd"], S["extension"]["q1_periods"]
    var_names = {"sst_anom": "Sea surface temp", "air_temp_anom": "Air temp", "zon_wind_anom": "Zonal wind",
                 "humidity_anom": "Humidity", "mer_wind_anom": "Meridional wind"}
    fig = go.Figure()
    for (key, label), color in zip(periods.items(), [GREY, BLUE, INK]):
        g = q1g[key]
        names = list(var_names)
        fig.add_scatter(y=[var_names[v] for v in names], x=[g[v]["gap"] for v in names], mode="markers", name=label,
                        marker=dict(color=color, size=10),
                        error_x=dict(type="data", symmetric=False, color=color,
                                     array=[g[v]["ci95"][1] - g[v]["gap"] for v in names],
                                     arrayminus=[g[v]["gap"] - g[v]["ci95"][0] for v in names]),
                        hovertemplate="%{y}: %{x:+.2f} SD<extra>" + label + "</extra>")
    fig.add_vline(x=0, line_color="#d2d2d7")
    fig.update_layout(xaxis_title="El Niño − La Niña gap, in SDs of each variable (95% CI, bootstrap over years)",
                      yaxis=dict(autorange="reversed"))
    show(style(fig, 340))

    st.subheader("5 · Los Angeles rain after El Niño falls")
    idx = st.radio("Label each fall with", ["ONI", "RONI"], horizontal=True, key="rain_index")
    col = "aso_oni" if idx == "ONI" else "aso_roni"
    phase_of = np.select([seasons[col] >= 0.5, seasons[col] <= -0.5], ["El Niño", "La Niña"], default="Neutral")
    c1, c2 = st.columns([1.6, 1])
    fig = go.Figure()
    for p in PHASES:
        sp = seasons[phase_of == p]
        fig.add_scatter(x=sp[col], y=sp["inches"], mode="markers", name=f"{p} fall ({len(sp)})",
                        marker=dict(color=PHASE_COLORS[p], size=10, line=dict(color="white", width=1)),
                        customdata=np.stack([sp["season"], sp["season"] + 1, sp["heavy"]], axis=1),
                        hovertemplate="%{customdata[0]}–%{customdata[1]}: %{y:.1f} in, %{customdata[2]} heavy days<extra></extra>")
    fig.add_hline(y=S["la_rain"]["median_inches"], line_dash="dash", line_color=INK, line_width=1,
                  annotation_text=f"median {S['la_rain']['median_inches']:.1f} in", annotation_position="top right")
    fig.add_vline(x=S["q3"]["winter_2026_27"][col], line_color=WARM, line_width=2,
                  annotation_text=f"Fall 2026: {S['q3']['winter_2026_27'][col]:+.2f}", annotation_position="top left",
                  annotation_font=dict(color=WARM))
    fig.update_layout(xaxis_title=f"Fall (Aug–Oct) {idx} (°C)", yaxis_title="Season rain, July–June (inches)")
    with c1:
        show(style(fig, 400))
    shares = seasons.assign(phase=phase_of, wet=seasons["wet"].astype(bool)).groupby("phase").agg(
        n=("season", "size"), wet=("wet", "sum"), heavy=("heavy", "mean")).reindex(PHASES)
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.18,
                        subplot_titles=("Wetter-than-median winters", "Heavy-rain days (≥ 1 in) per season"))
    fig.add_bar(x=PHASES, y=shares["wet"] / shares["n"], marker_color=[PHASE_COLORS[p] for p in PHASES], row=1, col=1,
                text=[f"{pct(w_ / n)} ({w_}/{n})" for w_, n in zip(shares["wet"], shares["n"])], textposition="outside",
                hovertemplate="%{x}: %{y:.0%}<extra></extra>")
    fig.add_bar(x=PHASES, y=shares["heavy"], marker_color=[PHASE_COLORS[p] for p in PHASES], row=2, col=1,
                text=[f"{h:.1f}" for h in shares["heavy"]], textposition="outside",
                hovertemplate="%{x}: %{y:.2f} days<extra></extra>")
    fig.update_yaxes(range=[0, 1], tickformat=".0%", row=1, col=1)
    fig.update_yaxes(range=[0, shares["heavy"].max() * 1.35], row=2, col=1)
    with c2:
        show(style(fig, 400, legend=False))
    st.caption(f"{S['la_rain']['n_seasons']} seasons {S['la_rain']['seasons_first']}–{S['la_rain']['seasons_last']}, "
               f"Los Angeles Downtown (USC). Wet = above the {S['la_rain']['median_inches']:.2f} in median of all seasons.")

# ===================================================================================================== Model Performance
with tab_perf:
    tests = pd.DataFrame(META["test_scores"]["rows"])
    runs = META["tuning"]["runs"]
    st.markdown(
        "**Set-up.** A forecast is issued at the start of month *t + 1* from the buoy anomalies of month *t* and the latest "
        "official ONI (month *t − 1*). The target is the ONI phase of month *t + k*. Score: **balanced accuracy**, the "
        "average of the hit rates for La Niña, Neutral and El Niño (0.33 = guessing). **Persistence** assumes the latest ONI "
        "phase continues. Two held-out tests: the **main split** (train to 1994, test 1995–June 1998) and a **walk-forward** "
        "test on 1999–2026, where each year is predicted by models fitted on the 30 years before it.")

    st.subheader("Balanced accuracy by lead time (frozen results)")
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
                        subplot_titles=("Main split: test 1995–98",
                                        f"Walk-forward: test 1999–{S['extension']['last_full_month'][:4]}"))
    for col, block in [(1, S["q2"]["main_split"]), (2, S["extension"]["q2_walk_forward"])]:
        fq = frozen_q2(block)
        for kind in SUMMARY_KEY:
            r = fq[fq["model"] == kind].sort_values("k")
            fig.add_scatter(x=r["k"], y=r["balanced_accuracy"], mode="lines+markers", name=MODEL_SHORT[kind],
                            legendgroup=kind, showlegend=col == 1, customdata=r["changes"],
                            line=dict(color=MODEL_COLORS[kind], width=2.5, dash="dash" if kind == "persistence" else "solid"),
                            marker=dict(size=8), row=1, col=col,
                            hovertemplate="k=%{x}: %{y:.2f} · phase changes caught %{customdata}<extra>"
                                          + MODEL_SHORT[kind] + "</extra>")
        fig.add_hline(y=1 / 3, line_dash="dot", line_color=MUTED, line_width=1, row=1, col=col)
    fig.update_xaxes(title_text="Months ahead (k)", tickvals=[0, 1, 2, 3, 6])
    fig.update_yaxes(range=[0.2, 1], title_text="Balanced accuracy", row=1, col=1)
    show(style(fig, 380))
    st.caption("Logistic regression and the random forest use the buoy readings plus the latest ONI with the fixed settings "
               "of the report (C = 1; 500 trees, min 3 months per leaf). Hover a point for the phase changes caught.")

    st.subheader("What the final models rely on")
    models = load_models()
    k_sel = st.radio("Horizon", HORIZONS, horizontal=True, key="coef_k", format_func=lambda k: f"k = {k}")
    labels = [f["label"] for f in FEATURES]
    final_params = META["live_models"]["params"]
    c1, c2 = st.columns(2)
    lr = models[(k_sel, "logistic")][-1]
    coef = pd.DataFrame(lr.coef_, index=lr.classes_, columns=labels).reindex(PHASES)
    fig = go.Figure()
    for p in ["El Niño", "La Niña"]:
        fig.add_bar(y=labels, x=coef.loc[p], orientation="h", name=p, marker_color=PHASE_COLORS[p],
                    hovertemplate="%{y}: %{x:+.2f}<extra>" + p + "</extra>")
    fig.update_layout(barmode="group", yaxis=dict(autorange="reversed"),
                      xaxis_title="Coefficient per SD of the input (log-odds)",
                      title=dict(text=f"Logistic regression ({setting_text('logistic', final_params['logistic'])})",
                                 font=dict(size=13)))
    with c1:
        show(style(fig, 420))
    imp = pd.Series(models[(k_sel, "random_forest")][-1].feature_importances_, index=labels).sort_values()
    fig = go.Figure(go.Bar(y=imp.index, x=imp.values, orientation="h", marker_color=AMBER,
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(xaxis_title="Impurity-based importance (sums to 1)",
                      title=dict(text=f"Random forest ({setting_text('random_forest', final_params['random_forest'])})",
                                 font=dict(size=13)))
    with c2:
        show(style(fig, 420, legend=False))
    info = META["live_models"]["by_horizon"][f"k{k_sel}"]
    st.caption(f"Final models for k = {k_sel} (used in the Interactive Prediction tab; the logistic ones produced this "
               "winter's outlook): "
               f"fitted on {info['train_months']} months, issue months {month_label(info['first_issue_month'])}–"
               f"{month_label(info['last_issue_month'])}. Inputs are standardised before the logistic regression, so its "
               "coefficients compare across inputs: a positive bar makes that phase more likely as the input rises (the "
               "Neutral coefficients are not shown).")

    st.divider()
    st.subheader("Hyperparameter tuning (time-based CV on training years only)")
    st.markdown(META["tuning"]["conclusion"])
    st.markdown(
        "**Method.** Expanding-window folds by year: each setting is fitted on the years before a validation year and scored "
        "on that year, with balanced accuracy pooled over 7 validation years; no test month is used. Ties go to the simpler "
        "setting. Grids: logistic C ∈ {0.01, 0.03, 0.1, 0.3, 1, 3, 10}; random forest 200/500 trees × max depth 4/8/none × "
        "min leaf 1/3/5/10. The walk-forward settings were tuned once, on the first window, and refitted every test year.")

    st.markdown("##### Best hyperparameters per model and horizon")
    run_key = st.radio("Tuning run", list(runs), horizontal=True, key="tuning_run", format_func=lambda r: runs[r]["label"])
    run = runs[run_key]

    def fixed_mask(cv: pd.DataFrame, kind: str) -> np.ndarray:
        fixed = META["tuning"]["fixed"][kind]
        return np.all([cv[p].isna().to_numpy() if v is None else cv[p].astype(object).eq(v).to_numpy()
                       for p, v in fixed.items()], axis=0)

    rows = []
    for k, best in run["best"].items():
        row = {"Horizon": f"k = {k[1:]} ({HORIZON_TEXT[int(k[1:])]})"}
        for kind in ["logistic", "random_forest"]:
            cv = pd.DataFrame(run["cv"][k][kind])
            row[f"{MODEL_SHORT[kind]}: best setting"] = setting_text(kind, best[kind])
            row[f"{MODEL_SHORT[kind]}: CV score"] = cv["cv_balanced_accuracy"].max()
            row[f"{MODEL_SHORT[kind]}: CV score, fixed setting"] = float(cv.loc[fixed_mask(cv, kind), "cv_balanced_accuracy"].iloc[0])
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.2f") for c in rows[0] if "CV score" in c})
    st.caption(f"Validation years {run['validation_years'][0]}–{run['validation_years'][-1]}, targets up to "
               f"{month_label(run['last_target_month'])}; validation months per horizon: "
               + ", ".join(f"k={k[1:]}: {v['validation_months']}" for k, v in run["folds"].items()) + ".")

    st.markdown("##### Cross-validation table")
    c1, c2 = st.columns(2)
    cv_k = c1.radio("Horizon", list(run["cv"]), horizontal=True, key="cv_k", format_func=lambda k: f"k = {k[1:]}")
    cv_kind = c2.radio("Model", ["logistic", "random_forest"], horizontal=True, key="cv_kind", format_func=MODEL_SHORT.get)
    cv = pd.DataFrame(run["cv"][cv_k][cv_kind])
    cv_view = pd.DataFrame({"Setting": [setting_text(cv_kind, {c: (None if pd.isna(v) else int(v) if c != "C" else v)
                                                               for c, v in r.items() if c in META["tuning"]["fixed"][cv_kind]})
                                        for r in cv.to_dict(orient="records")],
                            "CV balanced accuracy": cv["cv_balanced_accuracy"], "CV accuracy": cv["cv_accuracy"],
                            "Fixed setting of the report": np.where(fixed_mask(cv, cv_kind), "yes", "")})
    st.dataframe(cv_view.sort_values("CV balanced accuracy", ascending=False, kind="stable"), hide_index=True, width="stretch",
                 height=min(38 + 35 * len(cv_view), 420),
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["CV balanced accuracy", "CV accuracy"]})

    with st.expander("Cross-validation curve of the logistic model"):
        fig = go.Figure()
        for (k, block), color in zip(run["cv"].items(), ["#0a84ff", "#32ade6", "#5e5ce6", "#1d1d1f", "#8e8e93"]):
            curve = pd.DataFrame(block["logistic"])
            fig.add_scatter(x=curve["C"], y=curve["cv_balanced_accuracy"], mode="lines+markers", name=f"k = {k[1:]}",
                            line=dict(color=color, width=2), hovertemplate="C=%{x}: %{y:.3f}<extra>k=" + k[1:] + "</extra>")
        fig.update_xaxes(type="log", title_text="C (log scale; smaller = stronger shrinkage)")
        fig.update_yaxes(title_text="Cross-validated balanced accuracy")
        show(style(fig, 340))

    st.markdown("##### Tuned vs fixed settings on the held-out tests")
    ev = st.radio("Test", ["main_split", "walk_forward"], horizontal=True, key="tuned_eval",
                  format_func=lambda e: {"main_split": "Main split 1995–98",
                                         "walk_forward": f"Walk-forward 1999–{S['extension']['last_full_month'][:4]}"}[e])
    t = tests[tests["evaluation"] == ev]
    rows = []
    for k in sorted(t["k"].unique()):
        tk = t[t["k"] == k].set_index(["model", "setting"])
        row = {"k": int(k), "Persistence": tk.loc[("persistence", "baseline"), "balanced_accuracy"]}
        for kind in ["logistic", "random_forest"]:
            fx, tu = tk.loc[(kind, "fixed")], tk.loc[(kind, "tuned")]
            row[f"{MODEL_SHORT[kind]}: fixed"] = fx["balanced_accuracy"]
            row[f"{MODEL_SHORT[kind]}: tuned"] = tu["balanced_accuracy"]
            row[f"{MODEL_SHORT[kind]}: change"] = tu["balanced_accuracy"] - fx["balanced_accuracy"]
            row[f"{MODEL_SHORT[kind]}: tuned, phase changes caught"] = f"{int(tu['changes_caught'])}/{int(tu['phase_changes'])}"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%+.2f" if "change" in c and "caught" not in c
                                                                 else "%.2f")
                                for c in rows[0] if c != "k" and "caught" not in c})
    tuned_from = "main_split" if ev == "main_split" else "walk_forward"
    st.caption(f"Tuned settings from the “{runs[tuned_from]['label']}” run. Fixed = the frozen results used in the report, "
               "the Executive Summary and the final models.")
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
                        subplot_titles=(MODEL_SHORT["logistic"], MODEL_SHORT["random_forest"]))
    for col, kind in [(1, "logistic"), (2, "random_forest")]:
        p = t[t["model"] == "persistence"].sort_values("k")
        fig.add_scatter(x=p["k"], y=p["balanced_accuracy"], mode="lines+markers", name="Persistence", legendgroup="p",
                        showlegend=col == 1, line=dict(color=MODEL_COLORS["persistence"], dash="dash"), row=1, col=col)
        for setting, dash, fill in [("fixed", "solid", MODEL_COLORS[kind]), ("tuned", "dot", "white")]:
            r = t[(t["model"] == kind) & (t["setting"] == setting)].sort_values("k")
            fig.add_scatter(x=r["k"], y=r["balanced_accuracy"], mode="lines+markers", name=f"{setting.capitalize()} setting",
                            legendgroup=setting, showlegend=col == 1, row=1, col=col,
                            line=dict(color=MODEL_COLORS[kind], dash=dash, width=2.5),
                            marker=dict(size=9, color=fill, line=dict(color=MODEL_COLORS[kind], width=2)))
    fig.update_xaxes(title_text="Months ahead (k)", tickvals=sorted(t["k"].unique()))
    fig.update_yaxes(range=[0.2, 1], title_text="Balanced accuracy", row=1, col=1)
    show(style(fig, 340))

# ================================================================================================ Interactive Prediction
with tab_pred:
    inputs = load_csv("model_inputs_monthly.csv").set_index("issue_month")
    models = load_models()
    medians = dict(zip(FEATURE_NAMES, models[(HORIZONS[0], "logistic")][0].statistics_))
    latest_month = META["live_models"]["latest_issue_month"]
    months = list(inputs.index[::-1])
    lo = inputs[FEATURE_NAMES].min().apply(lambda v: np.floor(v * 2) / 2)
    hi = inputs[FEATURE_NAMES].max().apply(lambda v: np.ceil(v * 2) / 2)

    def preset_values(month: str) -> dict:
        row = inputs.loc[month, FEATURE_NAMES]
        return {f: round(float(medians[f] if pd.isna(row[f]) else row[f]), 2) for f in FEATURE_NAMES}

    def apply_preset() -> None:
        for f, v in preset_values(st.session_state["preset"]).items():
            st.session_state[f"x_{f}"] = v

    if "preset" not in st.session_state:
        st.session_state["preset"] = latest_month
        apply_preset()

    st.markdown(
        "Set one month's buoy readings and the latest official ONI; the saved models (the report's fixed settings; the "
        "logistic ones are the models behind this winter's outlook) return the chance of each phase 0, 1, 3 and 6 months later. Readings are "
        "**anomalies**: the month's value minus that buoy's average for the same calendar month over "
        f"{META['live_models']['setup'].split('over ')[-1]}. The sliders start at the latest observed month.")
    c1, c2 = st.columns([1, 2])
    c1.selectbox("Start from the readings of", months, key="preset", on_change=apply_preset,
                 format_func=lambda m: month_label(m) + (" (latest)" if m == latest_month else ""))
    missing = [f["label"] for f in FEATURES if pd.isna(inputs.loc[st.session_state["preset"], f["name"]])]
    if missing:
        c2.info("Not measured that month, set to the training median: " + ", ".join(missing))

    groups = [("Temperature", ["sst_n34", "sst_all", "air_all"]), ("Wind", ["zw_all", "zw_west", "mw_all"]),
              ("Change since last month, and ONI", ["d_sst_n34", "d_zw_west", "oni_lag1"])]
    by_name = {f["name"]: f for f in FEATURES}
    cols = st.columns(3)
    for col, (title, names) in zip(cols, groups):
        with col.container(border=True):
            st.markdown(f'<p class="small-label">{title}</p>', unsafe_allow_html=True)
            for n in names:
                f = by_name[n]
                st.slider(f"{f['label']} ({f['unit']})", float(lo[n]), float(hi[n]), step=0.01, key=f"x_{n}",
                          help=f"Range = the lowest and highest monthly value since {month_label(inputs.index[0])}.")

    x = pd.DataFrame([{n: st.session_state[f"x_{n}"] for n in FEATURE_NAMES}])[FEATURE_NAMES]
    issue = st.session_state["preset"]
    edited = any(abs(x.iloc[0][n] - v) > 1e-9 for n, v in preset_values(issue).items())
    model_kind = st.radio("Model", ["logistic", "random_forest"], horizontal=True, key="pred_model",
                          format_func=lambda m: MODEL_SHORT[m] + (" (main model)" if m == "logistic" else ""))
    rows = []
    for k in HORIZONS:
        m = models[(k, model_kind)]
        prob = dict(zip(m.classes_, m.predict_proba(x)[0]))
        target = add_months(issue, k)
        recorded = oni.loc[oni["month"] == target, "phase"]
        rows.append({"k": k, "label": f"{HORIZON_TEXT[k].capitalize()} · {month_label(target)}", "target": target,
                     **{p: prob.get(p, 0.0) for p in PHASES}, "call": max(prob, key=prob.get),
                     "recorded": "" if edited or recorded.empty else recorded.iloc[0]})
    res = pd.DataFrame(rows)

    st.subheader(f"Phase chances, readings of {month_label(issue)}" + (" (edited)" if edited else ""))
    fig = go.Figure()
    for p in PHASES:
        fig.add_bar(y=res["label"], x=res[p], orientation="h", name=p, marker_color=PHASE_COLORS[p],
                    text=[pct(v) if v >= 0.06 else "" for v in res[p]], textposition="inside", insidetextanchor="middle",
                    textfont=dict(family=MONO, color="white" if p != "Neutral" else INK),
                    hovertemplate="%{y}: %{x:.1%}<extra>" + p + "</extra>")
    fig.update_layout(barmode="stack", xaxis=dict(range=[0, 1], tickformat=".0%", title_text="Probability"),
                      yaxis=dict(autorange="reversed"))
    show(style(fig, 300))

    oni_now = float(x.iloc[0]["oni_lag1"])
    persist = "El Niño" if oni_now >= 0.5 else "La Niña" if oni_now <= -0.5 else "Neutral"
    table = res[["label", *PHASES, "call"]].rename(columns={"label": "Target month", "call": "Most likely"})
    table["Persistence"] = persist
    if res["recorded"].astype(bool).any():
        table["NOAA recorded"] = res["recorded"]
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={p: st.column_config.ProgressColumn(p, format="percent", min_value=0.0, max_value=1.0)
                                for p in PHASES})
    first_issue = META["live_models"]["by_horizon"]["k0"]["first_issue_month"]
    last_issue = max(b["last_issue_month"] for b in META["live_models"]["by_horizon"].values())
    st.caption(f"Persistence = the phase of the latest ONI you entered ({oni_now:+.2f}). The models were fitted on months "
               f"{month_label(first_issue)}–{month_label(last_issue)}, so for those months the recorded phase is not an "
               "out-of-sample check; see Model Performance for the held-out tests. Readings far outside the historical range "
               "give probabilities near 0 or 1 that mean “far from a phase boundary”, not certainty.")

    st.subheader("What it means for Los Angeles")
    call_now = res.loc[res["k"] == 0, "call"].iloc[0]
    odds = S["q3"]["wet_odds_by_oni_phase"]
    st.markdown(f"Wet-winter odds after a fall (Aug–Oct) in each phase, Los Angeles Downtown {S['la_rain']['seasons_first']}–"
                f"{S['la_rain']['seasons_last']}. The model's call for {month_label(issue)} is **{call_now}**.")
    cols = st.columns(3)
    for col, p in zip(cols, ["El Niño", "Neutral", "La Niña"]):
        o_ = odds[p]
        with col.container(border=True):
            st.markdown(f'<p class="small-label">After a {p} fall{" · current call" if p == call_now else ""}</p>',
                        unsafe_allow_html=True)
            st.metric("Wetter-than-median winter", pct(o_["share"]), f"{o_['wet']} of {o_['n']} seasons", delta_color="off",
                      delta_arrow="off")
            st.markdown(f"95% CI {pct(o_['ci95'][0])}–{pct(o_['ci95'][1])} · {o_['heavy_mean']:.1f} heavy-rain days per season · "
                        f"median {o_['median_in']:.1f} in")
    strong = S["q3"]["wet_odds_by_strength"]["ONI"]["Strong El Niño (≥ 1.5)"]
    if oni_now >= 1.5:
        st.info(f"The latest ONI you entered ({oni_now:+.2f}) is a strong El Niño level (≥ 1.5). After strong El Niño falls, "
                f"{strong['wet']} of {strong['n']} LA winters were wet ({strong['heavy_mean']:.1f} heavy-rain days per season); "
                "with so few cases the 95% interval is wide "
                f"({pct(strong['ci95'][0])}–{pct(strong['ci95'][1])}).")
    st.caption("These odds come from the rain record, not from the buoy models: the models say which phase is likely, the rain "
               "record says what that phase has meant for LA. A seasonal risk outlook, not a weather forecast.")
