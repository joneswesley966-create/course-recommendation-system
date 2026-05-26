"""
France Top 50 Spotify Playlist - Audience Sensitivity, Content Compliance
& Format Preference Analysis
For: Atlantic Recording Corporation | Unified Mentor Internship Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="France Top 50 | Atlantic RC",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0d0f14; }
    section[data-testid="stSidebar"] { background-color: #161b22; }
    section[data-testid="stSidebar"] * { color: #e6edf3 !important; }
    [data-testid="metric-container"] {
        background: #161b22; border: 1px solid #21262d;
        border-radius: 10px; padding: 14px 18px; border-left: 3px solid #e63946;
    }
    [data-testid="metric-container"] label { color: #7d8590 !important; font-size: 12px !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #e63946 !important; font-size: 26px !important; font-weight: 700 !important;
    }
    h1, h2, h3 { color: #e6edf3 !important; }
    .main-header {
        background: linear-gradient(135deg, #1A1A2E 0%, #0F3460 100%);
        padding: 24px 32px; border-radius: 12px;
        border-left: 4px solid #e63946; margin-bottom: 24px;
    }
    .main-header h1 { color: white !important; margin: 0; font-size: 22px; }
    .main-header p  { color: #E2B04A !important; margin: 6px 0 0; font-size: 13px; }
    .section-header {
        background: #0F3460; color: white !important;
        padding: 10px 16px; border-radius: 8px; margin: 18px 0 12px;
        font-weight: 700; font-size: 14px; letter-spacing: 1px;
    }
    .insight-red {
        background: #1c2128; border: 1px solid #e63946;
        border-left: 4px solid #e63946; border-radius: 8px;
        padding: 12px 16px; color: #e6edf3;
        font-size: 13px; line-height: 1.6; margin-top: 8px;
    }
    .insight-gold {
        background: #1c2128; border: 1px solid #E2B04A;
        border-left: 4px solid #E2B04A; border-radius: 8px;
        padding: 12px 16px; color: #e6edf3;
        font-size: 13px; line-height: 1.6; margin-top: 8px;
    }
    .insight-teal {
        background: #1c2128; border: 1px solid #2ec4b6;
        border-left: 4px solid #2ec4b6; border-radius: 8px;
        padding: 12px 16px; color: #e6edf3;
        font-size: 13px; line-height: 1.6; margin-top: 8px;
    }
    .rec-card {
        background: #161b22; border: 1px solid #21262d;
        border-radius: 10px; padding: 16px; margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Atlantic_France.csv")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df["duration_min"] = df["duration_ms"] / 60000
    df["is_explicit"] = df["is_explicit"].astype(str).str.upper() == "TRUE"
    df["album_type"] = df["album_type"].str.lower().str.strip()
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    df["duration_bucket"] = pd.cut(
        df["duration_min"], bins=[0, 2.5, 4.0, 99],
        labels=["Short (<2.5 min)", "Medium (2.5-4 min)", "Long (>4 min)"]
    )
    df["album_size"] = df["total_tracks"].apply(
        lambda x: "Large (>15 tracks)" if x > 15 else "Small (<=15 tracks)"
    )
    df["rank_tier"] = pd.cut(
        df["position"], bins=[0, 10, 25, 50],
        labels=["Top 10", "Top 11-25", "Top 26-50"]
    )
    return df

df = load_data()

COLORS = {
    "explicit": "#e63946", "clean": "#2ec4b6",
    "album": "#f4a261",    "single": "#457b9d", "accent": "#ffd166",
}

def styled_fig(fig, height=360):
    fig.update_layout(
        paper_bgcolor="#161b22", plot_bgcolor="#0d0f14",
        font=dict(color="#e6edf3", family="monospace", size=11),
        margin=dict(l=20, r=20, t=40, b=20), height=height,
        legend=dict(bgcolor="#21262d", bordercolor="#21262d"),
    )
    fig.update_xaxes(gridcolor="#21262d", zerolinecolor="#21262d")
    fig.update_yaxes(gridcolor="#21262d", zerolinecolor="#21262d")
    return fig

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎵 France Top 50")
    st.markdown("**Atlantic Recording Corporation**")
    st.markdown("---")
    st.markdown("### 🔧 Filters")

    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    date_range = st.date_input("📅 Date Range", value=(min_date, max_date),
                                min_value=min_date, max_value=max_date)

    rank_filter = st.multiselect("🏆 Rank Tier",
        options=["Top 10", "Top 11-25", "Top 26-50"],
        default=["Top 10", "Top 11-25", "Top 26-50"])

    explicit_filter = st.selectbox("🔞 Content Type",
        options=["All", "Explicit Only", "Clean Only"])

    album_filter = st.multiselect("💿 Album Type",
        options=["single", "album"], default=["single", "album"],
        format_func=lambda x: x.capitalize())

    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.markdown(f"**Total Records:** {len(df):,}")
    st.markdown(f"**Date Range:** {min_date} → {max_date}")
    st.markdown(f"**Unique Days:** {df['date'].nunique()}")
    st.markdown(f"**Unique Artists:** {df['artist'].nunique()}")
    st.markdown("---")
    st.markdown("*Unified Mentor Internship*")
    st.markdown("*Analyst: Jones Wesley*")

# ── APPLY FILTERS ─────────────────────────────────────────────────────────────
fdf = df.copy()
if len(date_range) == 2:
    fdf = fdf[(fdf["date"].dt.date >= date_range[0]) & (fdf["date"].dt.date <= date_range[1])]
if rank_filter:
    fdf = fdf[fdf["rank_tier"].isin(rank_filter)]
if explicit_filter == "Explicit Only":
    fdf = fdf[fdf["is_explicit"] == True]
elif explicit_filter == "Clean Only":
    fdf = fdf[fdf["is_explicit"] == False]
if album_filter:
    fdf = fdf[fdf["album_type"].isin(album_filter)]

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎵 France Top 50 · Content Intelligence Dashboard</h1>
    <p>Audience Sensitivity, Content Compliance & Format Preference Analysis · Atlantic Recording Corporation</p>
</div>
""", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "🔞 Explicit Analysis", "💿 Format Preference",
    "⏱ Duration Analysis", "🎤 Artists", "🎯 Recommendations",
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">📊 KEY PERFORMANCE INDICATORS</div>', unsafe_allow_html=True)

    total     = len(fdf)
    exp_count = fdf["is_explicit"].sum()
    cln_count = total - exp_count
    exp_pct   = exp_count / total * 100 if total else 0
    exp_pop   = fdf[fdf["is_explicit"]]["popularity"].mean() if exp_count else 0
    cln_pop   = fdf[~fdf["is_explicit"]]["popularity"].mean() if cln_count else 0
    avg_dur   = fdf["duration_min"].mean()
    single_pct= (fdf["album_type"] == "single").mean() * 100

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Explicit Share",      f"{exp_pct:.1f}%",     "Audience sensitivity")
    c2.metric("Clean Avg Popularity",f"{cln_pop:.1f}",      f"+{cln_pop-exp_pop:.1f} vs explicit")
    c3.metric("Single Format Share", f"{single_pct:.1f}%",  "Format preference")
    c4.metric("Avg Song Duration",   f"{avg_dur:.2f} min",  "Structural norm")
    c5.metric("Filtered Records",    f"{total:,}",          "Active dataset")
    c6.metric("Explicit Avg Pop",    f"{exp_pop:.1f}",      "vs Clean benchmark")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Popularity by Content Type**")
        pop_data = pd.DataFrame({
            "Category":   ["Explicit", "Clean", "Album", "Single"],
            "Popularity": [
                fdf[fdf["is_explicit"]]["popularity"].mean(),
                fdf[~fdf["is_explicit"]]["popularity"].mean(),
                fdf[fdf["album_type"]=="album"]["popularity"].mean(),
                fdf[fdf["album_type"]=="single"]["popularity"].mean(),
            ],
            "Color": [COLORS["explicit"], COLORS["clean"], COLORS["album"], COLORS["single"]],
        })
        fig = px.bar(pop_data, x="Category", y="Popularity", color="Category",
                     color_discrete_map={r["Category"]: r["Color"] for _, r in pop_data.iterrows()},
                     text="Popularity", template="plotly_dark")
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(showlegend=False, yaxis_range=[65, 86])
        st.plotly_chart(styled_fig(fig), use_container_width=True)
        st.markdown('<div class="insight-teal">💡 Clean content averages <b>80.9</b> vs <b>73.3</b> for explicit — a 10.4% popularity premium. Singles (80.75) consistently outperform albums (72.99).</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("**Monthly Explicit Content Trend**")
        monthly = fdf.groupby("year_month")["is_explicit"].agg(["sum","count"]).reset_index()
        monthly["explicit_pct"] = monthly["sum"] / monthly["count"] * 100
        monthly = monthly.sort_values("year_month")
        fig2 = px.line(monthly, x="year_month", y="explicit_pct",
                       markers=True, template="plotly_dark",
                       labels={"year_month": "Month", "explicit_pct": "Explicit %"})
        fig2.update_traces(line=dict(color=COLORS["explicit"], width=2.5), marker=dict(size=7))
        fig2.add_hline(y=monthly["explicit_pct"].mean(), line_dash="dash",
                       line_color=COLORS["accent"],
                       annotation_text=f"Avg {monthly['explicit_pct'].mean():.1f}%")
        st.plotly_chart(styled_fig(fig2), use_container_width=True)
        st.markdown('<div class="insight-gold">💡 Explicit content share fluctuates monthly but remains above 50% consistently, peaking in Sep–Oct 2024 (~67%).</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Popularity Distribution Overview**")
    fig3 = px.histogram(fdf, x="popularity", color="is_explicit",
                        barmode="overlay", nbins=40, template="plotly_dark",
                        color_discrete_map={True: COLORS["explicit"], False: COLORS["clean"]},
                        labels={"popularity": "Popularity Score", "is_explicit": "Explicit"})
    fig3.update_layout(bargap=0.05)
    st.plotly_chart(styled_fig(fig3, height=260), use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2 — EXPLICIT ANALYSIS
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🔞 EXPLICIT CONTENT SENSITIVITY ANALYSIS</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        exp_df = pd.DataFrame({
            "Type":  ["Explicit", "Clean"],
            "Count": [fdf["is_explicit"].sum(), (~fdf["is_explicit"]).sum()],
        })
        fig = px.pie(exp_df, names="Type", values="Count", hole=0.55,
                     color="Type",
                     color_discrete_map={"Explicit": COLORS["explicit"], "Clean": COLORS["clean"]},
                     template="plotly_dark")
        fig.update_traces(textfont=dict(color="white"), pull=[0.03, 0])
        fig.update_layout(annotations=[dict(
            text=f"{fdf['is_explicit'].mean()*100:.1f}%<br>Explicit",
            font=dict(size=15, color="white"), showarrow=False)])
        st.markdown("**Explicit vs Clean Split**")
        st.plotly_chart(styled_fig(fig), use_container_width=True)

    with col2:
        pop_comp = pd.DataFrame({
            "Type":       ["Explicit", "Clean"],
            "Popularity": [
                fdf[fdf["is_explicit"]]["popularity"].mean(),
                fdf[~fdf["is_explicit"]]["popularity"].mean(),
            ],
        })
        fig2 = px.bar(pop_comp, x="Type", y="Popularity", color="Type",
                      color_discrete_map={"Explicit": COLORS["explicit"], "Clean": COLORS["clean"]},
                      text="Popularity", template="plotly_dark")
        fig2.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig2.update_layout(showlegend=False, yaxis_range=[68, 86])
        st.markdown("**Avg Popularity: Explicit vs Clean**")
        st.plotly_chart(styled_fig(fig2), use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        tier_exp = fdf.groupby("rank_tier", observed=True)["is_explicit"].agg(["sum","count"]).reset_index()
        tier_exp["Explicit %"] = tier_exp["sum"] / tier_exp["count"] * 100
        tier_exp["Clean %"]    = 100 - tier_exp["Explicit %"]
        tier_exp["rank_tier"]  = tier_exp["rank_tier"].astype(str)
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name="Explicit", x=tier_exp["rank_tier"], y=tier_exp["Explicit %"],
                               marker_color=COLORS["explicit"], text=tier_exp["Explicit %"].round(1),
                               texttemplate="%{text}%"))
        fig3.add_trace(go.Bar(name="Clean", x=tier_exp["rank_tier"], y=tier_exp["Clean %"],
                               marker_color=COLORS["clean"], text=tier_exp["Clean %"].round(1),
                               texttemplate="%{text}%"))
        fig3.update_layout(barmode="group", template="plotly_dark")
        st.markdown("**Explicit % by Rank Tier**")
        st.plotly_chart(styled_fig(fig3), use_container_width=True)
        st.markdown('<div class="insight-red">💡 Top 10 has the highest explicit share (62.4%). French listeners do not penalise explicit content in rankings — but clean tracks still score higher in popularity.</div>', unsafe_allow_html=True)

    with col4:
        fmt_exp = fdf.groupby("album_type")["is_explicit"].agg(["sum","count"]).reset_index()
        fmt_exp["Explicit %"] = fmt_exp["sum"] / fmt_exp["count"] * 100
        fmt_exp["Clean %"]    = 100 - fmt_exp["Explicit %"]
        fmt_exp["album_type"] = fmt_exp["album_type"].str.capitalize()
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(name="Explicit", y=fmt_exp["album_type"], x=fmt_exp["Explicit %"],
                               orientation="h", marker_color=COLORS["explicit"],
                               text=fmt_exp["Explicit %"].round(1), texttemplate="%{text}%"))
        fig4.add_trace(go.Bar(name="Clean", y=fmt_exp["album_type"], x=fmt_exp["Clean %"],
                               orientation="h", marker_color=COLORS["clean"],
                               text=fmt_exp["Clean %"].round(1), texttemplate="%{text}%"))
        fig4.update_layout(barmode="stack", template="plotly_dark", xaxis_title="Percentage (%)")
        st.markdown("**Explicit Split by Album Type**")
        st.plotly_chart(styled_fig(fig4), use_container_width=True)
        st.markdown('<div class="insight-red">💡 <b>72.4%</b> of album tracks are explicit vs only <b>38.2%</b> for singles. Apply stricter compliance checks to album releases targeting France.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📋 Explicit Content Summary Table**")
    summary = fdf.groupby("is_explicit").agg(
        Count=("popularity","count"),
        Avg_Popularity=("popularity","mean"),
        Avg_Duration_min=("duration_min","mean"),
        Avg_Position=("position","mean"),
    ).reset_index()
    summary["is_explicit"] = summary["is_explicit"].map({True:"Explicit", False:"Clean"})
    summary = summary.rename(columns={"is_explicit":"Content Type"}).round(2)
    st.dataframe(summary, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# TAB 3 — FORMAT PREFERENCE
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">💿 RELEASE FORMAT PREFERENCE ANALYSIS</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fmt_df = fdf["album_type"].str.capitalize().value_counts().reset_index()
        fmt_df.columns = ["Format","Count"]
        fig = px.pie(fmt_df, names="Format", values="Count", hole=0.5,
                     color="Format",
                     color_discrete_map={"Single": COLORS["single"], "Album": COLORS["album"]},
                     template="plotly_dark")
        fig.update_traces(textfont=dict(color="white"))
        st.markdown("**Single vs Album Distribution**")
        st.plotly_chart(styled_fig(fig), use_container_width=True)

    with col2:
        fmt_pop = fdf.groupby("album_type")["popularity"].mean().reset_index()
        fmt_pop["album_type"] = fmt_pop["album_type"].str.capitalize()
        fmt_pop.columns = ["Format","Avg Popularity"]
        fig2 = px.bar(fmt_pop, x="Format", y="Avg Popularity", color="Format",
                      color_discrete_map={"Single": COLORS["single"], "Album": COLORS["album"]},
                      text="Avg Popularity", template="plotly_dark")
        fig2.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig2.update_layout(showlegend=False, yaxis_range=[68, 86])
        st.markdown("**Format Popularity Comparison**")
        st.plotly_chart(styled_fig(fig2), use_container_width=True)

    st.markdown('<div class="insight-teal">💡 Singles outperform albums by <b>+10.6 popularity points</b> (80.75 vs 72.99) — a 14.5% premium. Singles also carry half the explicit content rate of albums.</div>', unsafe_allow_html=True)

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        tier_fmt = fdf.groupby(["rank_tier","album_type"], observed=True).size().reset_index(name="count")
        tier_fmt["album_type"] = tier_fmt["album_type"].str.capitalize()
        tier_fmt["rank_tier"]  = tier_fmt["rank_tier"].astype(str)
        total_per_tier = tier_fmt.groupby("rank_tier")["count"].transform("sum")
        tier_fmt["pct"] = tier_fmt["count"] / total_per_tier * 100
        fig3 = px.bar(tier_fmt, x="rank_tier", y="pct", color="album_type",
                      barmode="group", text="pct",
                      color_discrete_map={"Single": COLORS["single"], "Album": COLORS["album"]},
                      template="plotly_dark",
                      labels={"rank_tier":"Rank Tier","pct":"Percentage (%)","album_type":"Format"})
        fig3.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.markdown("**Format % by Rank Tier**")
        st.plotly_chart(styled_fig(fig3), use_container_width=True)

    with col4:
        album_df = fdf[fdf["album_type"]=="album"].copy()
        size_pop = album_df.groupby("album_size")["popularity"].mean().reset_index()
        size_pop.columns = ["Album Size","Avg Popularity"]
        fig4 = px.bar(size_pop, x="Album Size", y="Avg Popularity", color="Album Size",
                      color_discrete_map={
                          "Small (<=15 tracks)": COLORS["clean"],
                          "Large (>15 tracks)": COLORS["explicit"],
                      },
                      text="Avg Popularity", template="plotly_dark")
        fig4.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig4.update_layout(showlegend=False, yaxis_range=[67, 78])
        st.markdown("**Album Size Impact on Popularity**")
        st.plotly_chart(styled_fig(fig4), use_container_width=True)
        st.markdown('<div class="insight-red">💡 Album dilution confirmed: large albums (>15 tracks) avg <b>70.91</b> vs <b>74.48</b> for smaller ones (−4.8%). France rewards curated, focused albums.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Track Count Distribution**")
    track_dist = fdf["total_tracks"].value_counts().head(15).reset_index()
    track_dist.columns = ["Total Tracks","Count"]
    track_dist = track_dist.sort_values("Total Tracks")
    fig5 = px.bar(track_dist, x="Total Tracks", y="Count",
                  color="Count", color_continuous_scale="reds", template="plotly_dark")
    st.plotly_chart(styled_fig(fig5, height=250), use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 4 — DURATION ANALYSIS
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">⏱ SONG DURATION PREFERENCE ANALYSIS</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Duration",    f"{fdf['duration_min'].mean():.2f} min")
    c2.metric("Median Duration", f"{fdf['duration_min'].median():.2f} min")
    c3.metric("Min Duration",    f"{fdf['duration_min'].min():.2f} min")
    c4.metric("Max Duration",    f"{fdf['duration_min'].max():.2f} min")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(fdf, x="duration_min", nbins=50,
                           color_discrete_sequence=[COLORS["single"]], template="plotly_dark",
                           labels={"duration_min":"Duration (minutes)"})
        fig.add_vline(x=fdf["duration_min"].mean(), line_dash="dash", line_color=COLORS["accent"],
                      annotation_text=f"Mean: {fdf['duration_min'].mean():.2f}m")
        st.markdown("**Duration Distribution (Histogram)**")
        st.plotly_chart(styled_fig(fig), use_container_width=True)

    with col2:
        bucket_df = fdf["duration_bucket"].value_counts().reset_index()
        bucket_df.columns = ["Bucket","Count"]
        fig2 = px.bar(bucket_df, x="Bucket", y="Count", color="Bucket",
                      color_discrete_map={
                          "Short (<2.5 min)":   COLORS["single"],
                          "Medium (2.5-4 min)": COLORS["clean"],
                          "Long (>4 min)":       COLORS["album"],
                      },
                      text="Count", template="plotly_dark")
        fig2.update_traces(textposition="outside")
        fig2.update_layout(showlegend=False)
        st.markdown("**Duration Bucket Distribution**")
        st.plotly_chart(styled_fig(fig2), use_container_width=True)

    st.markdown('<div class="insight-gold">💡 <b>81.7%</b> of chart entries fall in the 2.5–4 min range. French listeners strongly prefer the classic pop/rap song format. Avg duration = <b>3.09 min</b>.</div>', unsafe_allow_html=True)

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        dur_pop = fdf.groupby("duration_bucket", observed=True)["popularity"].mean().reset_index()
        dur_pop.columns = ["Bucket","Avg Popularity"]
        fig3 = px.bar(dur_pop, x="Bucket", y="Avg Popularity", color="Bucket",
                      color_discrete_map={
                          "Short (<2.5 min)":   COLORS["single"],
                          "Medium (2.5-4 min)": COLORS["clean"],
                          "Long (>4 min)":       COLORS["album"],
                      },
                      text="Avg Popularity", template="plotly_dark")
        fig3.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig3.update_layout(showlegend=False, yaxis_range=[74, 80])
        st.markdown("**Duration Bucket vs Avg Popularity**")
        st.plotly_chart(styled_fig(fig3), use_container_width=True)
        st.markdown('<div class="insight-teal">💡 Duration has minimal impact on popularity (range: 76.53–77.79). Content type and format matter far more than song length.</div>', unsafe_allow_html=True)

    with col4:
        sample = fdf.sample(min(2000, len(fdf)), random_state=42)
        fig4 = px.scatter(sample, x="duration_min", y="popularity", color="is_explicit",
                          color_discrete_map={True: COLORS["explicit"], False: COLORS["clean"]},
                          labels={"duration_min":"Duration (min)","popularity":"Popularity","is_explicit":"Explicit"},
                          template="plotly_dark", opacity=0.6)
        st.markdown("**Duration vs Popularity (Scatter)**")
        st.plotly_chart(styled_fig(fig4), use_container_width=True)

    st.markdown("---")
    st.markdown("**Duration by Rank Tier (Box Plot)**")
    fig5 = px.box(fdf, x="rank_tier", y="duration_min", color="rank_tier",
                  color_discrete_sequence=[COLORS["explicit"], COLORS["clean"], COLORS["album"]],
                  template="plotly_dark",
                  labels={"rank_tier":"Rank Tier","duration_min":"Duration (min)"},
                  category_orders={"rank_tier":["Top 10","Top 11-25","Top 26-50"]})
    fig5.update_layout(showlegend=False)
    st.plotly_chart(styled_fig(fig5, height=280), use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 5 — ARTISTS
# ══════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">🎤 ARTIST LANDSCAPE ANALYSIS</div>', unsafe_allow_html=True)

    top_n = st.slider("Number of top artists to show", 5, 20, 10)

    top_artists = fdf["artist"].value_counts().head(top_n).reset_index()
    top_artists.columns = ["Artist","Appearances"]

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = px.bar(top_artists, x="Appearances", y="Artist", orientation="h",
                     color="Appearances", color_continuous_scale=["#457b9d","#e63946"],
                     text="Appearances", template="plotly_dark")
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis=dict(categoryorder="total ascending"),
                          showlegend=False, coloraxis_showscale=False)
        st.markdown(f"**Top {top_n} Artists by Chart Appearances**")
        st.plotly_chart(styled_fig(fig, height=400), use_container_width=True)

    with col2:
        st.markdown("**Catalogue Stats**")
        top3_share  = top_artists.head(3)["Appearances"].sum() / len(fdf) * 100
        top10_share = top_artists.head(10)["Appearances"].sum() / len(fdf) * 100
        st.metric("Top 3 Artists Share",  f"{top3_share:.1f}%",  "of all chart days")
        st.metric("Top 10 Artists Share", f"{top10_share:.1f}%", "of all chart days")
        st.metric("Unique Artists",       f"{fdf['artist'].nunique():,}")
        st.metric("Days Analysed",        f"{fdf['date'].nunique()}")
        st.markdown('<div class="insight-red">💡 French rap dominates: Werenoi, PLK, SDM, Jul, Ninho lead chart longevity. International acts appear via high-popularity single spikes, not sustained albums.</div>', unsafe_allow_html=True)

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        artist_stats = fdf.groupby("artist").agg(
            appearances=("popularity","count"),
            explicit_pct=("is_explicit","mean"),
            avg_pop=("popularity","mean"),
        ).reset_index()
        artist_stats["explicit_pct"] *= 100
        artist_stats = artist_stats.sort_values("appearances", ascending=False).head(15)
        fig2 = px.scatter(artist_stats, x="explicit_pct", y="avg_pop",
                          size="appearances", text="artist",
                          color="explicit_pct", color_continuous_scale=["#2ec4b6","#e63946"],
                          template="plotly_dark",
                          labels={"explicit_pct":"Explicit %","avg_pop":"Avg Popularity","appearances":"Chart Days"})
        fig2.update_traces(textposition="top center", textfont=dict(size=9))
        fig2.update_layout(coloraxis_showscale=False)
        st.markdown("**Top Artist Explicit Profile (Bubble Chart)**")
        st.plotly_chart(styled_fig(fig2, height=350), use_container_width=True)

    with col4:
        top10_names = fdf["artist"].value_counts().head(10).index
        artist_pop  = fdf[fdf["artist"].isin(top10_names)].groupby("artist")["popularity"].mean().reset_index()
        artist_pop.columns = ["Artist","Avg Popularity"]
        artist_pop = artist_pop.sort_values("Avg Popularity")
        fig3 = px.bar(artist_pop, x="Avg Popularity", y="Artist", orientation="h",
                      color="Avg Popularity", color_continuous_scale=["#457b9d","#2ec4b6"],
                      text="Avg Popularity", template="plotly_dark")
        fig3.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig3.update_layout(coloraxis_showscale=False, xaxis_range=[60, 92])
        st.markdown("**Top 10 Artists — Avg Popularity**")
        st.plotly_chart(styled_fig(fig3, height=350), use_container_width=True)

    st.markdown("---")
    st.markdown("**📋 Full Artist Data Table (Top 30)**")
    artist_tbl = fdf.groupby("artist").agg(
        Chart_Days=("popularity","count"),
        Avg_Popularity=("popularity","mean"),
        Avg_Position=("position","mean"),
        Explicit_Rate=("is_explicit","mean"),
    ).reset_index().sort_values("Chart_Days", ascending=False).head(30)
    artist_tbl["Explicit_Rate"] = (artist_tbl["Explicit_Rate"]*100).round(1).astype(str) + "%"
    artist_tbl = artist_tbl.round(2)
    st.dataframe(artist_tbl, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# TAB 6 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">🎯 STRATEGIC RECOMMENDATIONS — ATLANTIC RECORDING CORPORATION</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-red" style="margin-bottom:20px; font-size:14px;">
    <b>Executive Summary:</b> Based on 27,800 daily chart entries from France Top 50 Spotify playlist,
    Atlantic Recording Corporation should adopt a <b>clean-singles-first, lean-album</b> release strategy.
    While explicit content is prevalent (56.3%), clean tracks command a +10.4 point popularity premium.
    The French market heavily favours domestic rap artists with deep catalogues over one-off hits.
    </div>
    """, unsafe_allow_html=True)

    recs = [
        ("R1","Prioritise Clean Singles for France Releases",
         "Clean singles score 80.75 vs 73.33 for explicit tracks. Always lead France campaigns with clean-edit singles. "
         "The 10.4-point popularity premium translates directly to improved editorial playlist placement probability.",
         "HIGH"),
        ("R2","Singles-First Rollout Strategy",
         "Release 2-3 clean singles ahead of any album drop to build chart presence. Singles outperform albums "
         "by 14.5% in popularity and carry half the explicit risk of album-format releases.",
         "HIGH"),
        ("R3","Cap Albums at 15 Tracks for France",
         "Albums >15 tracks average 70.91 vs 74.48 for leaner projects (-4.8%). Every additional track beyond 15 "
         "statistically dilutes individual track performance in the French market.",
         "HIGH"),
        ("R4","Mandatory Clean Versions for All Album Tracks",
         "72.4% of album tracks are explicit. Proactively produce clean versions of every track before French "
         "market submission to reduce compliance friction and maximise playlist eligibility.",
         "HIGH"),
        ("R5","Target 2.5-4 Minute Duration Window",
         "81.7% of chart entries fall here. While duration does not drive popularity directly, tracks outside "
         "this window are heavily under-represented. Avoid extreme outliers for France submissions.",
         "MEDIUM"),
        ("R6","Pursue French Rap Collaboration Opportunities",
         "Werenoi (2,206 days), PLK, SDM, Jul, and Ninho dominate long-term chart presence. Collaboration "
         "opportunities with these artists unlock sustained playlist positioning unavailable to foreign acts alone.",
         "MEDIUM"),
    ]

    col1, col2 = st.columns(2)
    for i, (tag, title, body, impact) in enumerate(recs):
        border_col = "#e63946" if impact == "HIGH" else "#E2B04A"
        tag_bg     = "#e63946" if impact == "HIGH" else "#E2B04A"
        tag_color  = "white"   if impact == "HIGH" else "#1A1A2E"
        imp_label  = "● HIGH IMPACT" if impact == "HIGH" else "◐ MEDIUM IMPACT"
        col = col1 if i % 2 == 0 else col2
        with col:
            st.markdown(f"""
            <div class="rec-card" style="border-top: 3px solid {border_col};">
                <span style="background:{tag_bg}; color:{tag_color}; border-radius:4px;
                      padding:2px 8px; font-size:11px; font-weight:700;">{tag}</span>
                <span style="float:right; font-size:11px; color:{border_col}; font-weight:700;">{imp_label}</span>
                <div style="color:#e6edf3; font-weight:700; font-size:14px; margin:8px 0 6px;">{title}</div>
                <div style="color:#7d8590; font-size:12px; line-height:1.6;">{body}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">📊 KPI COMPLIANCE SUMMARY</div>', unsafe_allow_html=True)

    top10_fdf = fdf[fdf["rank_tier"] == "Top 10"]
    kpi_data = {
        "KPI": [
            "Explicit Content Share",
            "Clean Content Dominance Ratio",
            "Single vs Album Track Ratio",
            "Average Song Duration",
            "Album Size Impact Index",
            "Content Acceptance Score",
        ],
        "Value": [
            f"{fdf['is_explicit'].mean()*100:.1f}%",
            f"{(~fdf['is_explicit']).mean()*100:.1f}%",
            f"{(fdf['album_type']=='single').mean()*100:.1f}% / {(fdf['album_type']=='album').mean()*100:.1f}%",
            f"{fdf['duration_min'].mean():.2f} min",
            f"Small={fdf[fdf['album_size']=='Small (<=15 tracks)']['popularity'].mean():.2f} | Large={fdf[fdf['album_size']=='Large (>15 tracks)']['popularity'].mean():.2f}",
            f"{(~top10_fdf['is_explicit']).mean()*100:.1f}% clean in Top 10" if len(top10_fdf) else "N/A",
        ],
        "Description": [
            "Audience sensitivity indicator",
            "Compliance preference",
            "Format preference",
            "Structural norm",
            "Catalog strategy insight",
            "Rank-aligned preference",
        ],
    }
    st.dataframe(pd.DataFrame(kpi_data), use_container_width=True, hide_index=True)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#7d8590; font-size:11px; padding:8px;">
    Audience Sensitivity, Content Compliance & Format Preference Analysis ·
    Unified Mentor Internship · Atlantic Recording Corporation · France Top 50 Spotify Playlist
</div>
""", unsafe_allow_html=True)
