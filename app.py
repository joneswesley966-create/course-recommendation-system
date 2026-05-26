import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduPro | Student Segmentation",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

:root {
    --primary: #6C63FF;
    --secondary: #FF6584;
    --accent: #43C6AC;
    --bg: #0F0F1A;
    --card: #1A1A2E;
    --text: #E8E8F0;
    --muted: #9090A8;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.main { background-color: var(--bg); }
.block-container { padding: 1.5rem 2rem; }

h1, h2, h3 { font-family: 'Space Mono', monospace; }

.stMetric {
    background: var(--card);
    border-radius: 12px;
    padding: 1rem;
    border-left: 4px solid var(--primary);
}

.segment-card {
    background: var(--card);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    border: 1px solid rgba(108,99,255,0.25);
}

.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
    margin-right: 6px;
}

.stButton > button {
    background: var(--primary);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
}

.stButton > button:hover {
    background: #5a52d5;
    transform: translateY(-1px);
}

div[data-testid="stSidebarNav"] { background: var(--card); }

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6C63FF, #43C6AC);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

.hero-sub {
    color: var(--muted);
    font-size: 1rem;
    margin-bottom: 1.5rem;
}

.rec-card {
    background: linear-gradient(135deg, #1A1A2E, #16213e);
    border: 1px solid rgba(108,99,255,0.3);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.7rem;
}

.cluster-pill-0 { background: rgba(108,99,255,0.2); color: #6C63FF; border: 1px solid #6C63FF; }
.cluster-pill-1 { background: rgba(67,198,172,0.2); color: #43C6AC; border: 1px solid #43C6AC; }
.cluster-pill-2 { background: rgba(255,101,132,0.2); color: #FF6584; border: 1px solid #FF6584; }
.cluster-pill-3 { background: rgba(255,200,87,0.2); color: #FFC857; border: 1px solid #FFC857; }

.stSelectbox div { background: var(--card); color: var(--text); }
.stSlider .st-bf { background: var(--primary); }

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── CLUSTER NAMES & COLORS ──────────────────────────────────────────────────
CLUSTER_NAMES = {
    0: "🔵 Casual Explorers",
    1: "🟢 Power Learners",
    2: "🔴 Budget Browsers",
    3: "🟡 Focused Achievers"
}
CLUSTER_COLORS = ['#6C63FF', '#43C6AC', '#FF6584', '#FFC857']
CLUSTER_DESCRIPTIONS = {
    0: "Moderate engagement, wide interests. Like variety but don't commit deeply.",
    1: "High enrollment, high spending. Avid learners hungry for advanced content.",
    2: "Low spending, selective enrollment. Value-conscious with niche preferences.",
    3: "Focused learners. Mid-range spending, consistent in preferred domains."
}

# ─── DATA LOADING ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    users = pd.read_csv('users.csv')
    courses = pd.read_csv('courses.csv')
    trans = pd.read_csv('transactions.csv')
    features = pd.read_csv('learner_features_clustered.csv')
    return users, courses, trans, features

@st.cache_data
def build_clustered_features(features_df):
    le_cat = LabelEncoder()
    le_lev = LabelEncoder()
    le_gen = LabelEncoder()
    df = features_df.copy()
    df['preferred_category_enc'] = le_cat.fit_transform(df['preferred_category'])
    df['preferred_level_enc'] = le_lev.fit_transform(df['preferred_level'])
    df['gender_enc'] = le_gen.fit_transform(df['Gender'])
    numeric_cols = ['total_courses_enrolled', 'total_spending', 'avg_spending_per_course',
                    'category_diversity', 'avg_course_rating_enrolled', 'learning_depth_index',
                    'Age', 'preferred_category_enc', 'preferred_level_enc', 'gender_enc']
    X = df[numeric_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Elbow
    inertias, silhouettes = {}, {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        silhouettes[k] = round(silhouette_score(X_scaled, labels), 4)
        inertias[k] = round(km.inertia_, 2)
    return X_scaled, scaler, inertias, silhouettes

users, courses, trans, features = load_data()
merged = trans.merge(courses, on='CourseID').merge(users, on='UserID')

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title">EduPro</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Student Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Overview & EDA",
        "🔬 Clustering Analysis",
        "👤 Learner Profile Explorer",
        "🎯 Course Recommender",
        "📈 Segment Comparison"
    ])
    st.markdown("---")
    st.markdown(f"**📦 Dataset Stats**")
    st.markdown(f"- Users: `{len(users):,}`")
    st.markdown(f"- Courses: `{len(courses):,}`")
    st.markdown(f"- Transactions: `{len(trans):,}`")
    st.markdown("---")
    st.markdown('<span style="color:#9090A8;font-size:12px;">Built for EduPro · Jones Wesley</span>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW & EDA
# ════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview & EDA":
    st.markdown('<div class="hero-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Understanding the EduPro learner landscape</div>', unsafe_allow_html=True)

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Learners", f"{len(users):,}")
    with col2:
        st.metric("Total Courses", f"{len(courses):,}")
    with col3:
        st.metric("Transactions", f"{len(trans):,}")
    with col4:
        st.metric("Avg Courses/User", f"{len(trans)/len(users):.1f}")

    st.markdown("---")

    # Row 1: Gender & Age
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Gender Distribution")
        gender_counts = users['Gender'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#1A1A2E')
        ax.set_facecolor('#1A1A2E')
        wedges, texts, autotexts = ax.pie(
            gender_counts.values,
            labels=gender_counts.index,
            autopct='%1.1f%%',
            colors=['#6C63FF', '#FF6584'],
            startangle=90,
            wedgeprops=dict(width=0.6, edgecolor='#0F0F1A', linewidth=2)
        )
        for t in texts + autotexts:
            t.set_color('#E8E8F0')
            t.set_fontsize(11)
        ax.set_title('Gender Split', color='#E8E8F0', fontsize=13, fontweight='bold', pad=10)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#1A1A2E')
        ax.set_facecolor('#1A1A2E')
        ax.hist(users['Age'], bins=20, color='#6C63FF', edgecolor='#0F0F1A', alpha=0.9)
        ax.set_xlabel('Age', color='#9090A8')
        ax.set_ylabel('Count', color='#9090A8')
        ax.tick_params(colors='#9090A8')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2A2A3E')
        ax.set_title('Age Distribution', color='#E8E8F0', fontsize=13, fontweight='bold')
        ax.axvline(users['Age'].mean(), color='#FF6584', linestyle='--', linewidth=2, label=f'Mean: {users["Age"].mean():.1f}')
        ax.legend(facecolor='#1A1A2E', labelcolor='#E8E8F0')
        st.pyplot(fig)
        plt.close()

    # Row 2: Category & Level
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Enrollments by Category")
        cat_counts = merged['CourseCategory'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#1A1A2E')
        ax.set_facecolor('#1A1A2E')
        bars = ax.barh(cat_counts.index, cat_counts.values,
                       color=CLUSTER_COLORS * 2, edgecolor='#0F0F1A', height=0.7)
        ax.set_xlabel('Enrollments', color='#9090A8')
        ax.tick_params(colors='#9090A8')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2A2A3E')
        ax.set_title('Course Category Popularity', color='#E8E8F0', fontsize=13, fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Course Level Distribution")
        level_counts = merged['CourseLevel'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#1A1A2E')
        ax.set_facecolor('#1A1A2E')
        bars = ax.bar(level_counts.index, level_counts.values,
                      color=['#43C6AC', '#6C63FF', '#FF6584'],
                      edgecolor='#0F0F1A', width=0.6)
        ax.set_ylabel('Count', color='#9090A8')
        ax.tick_params(colors='#9090A8')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2A2A3E')
        ax.set_title('Level Distribution', color='#E8E8F0', fontsize=13, fontweight='bold')
        for bar, val in zip(bars, level_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{val:,}', ha='center', color='#E8E8F0', fontsize=10)
        st.pyplot(fig)
        plt.close()

    # Row 3: Spending & Ratings
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Spending Distribution per Transaction")
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#1A1A2E')
        ax.set_facecolor('#1A1A2E')
        ax.hist(trans['Amount'], bins=30, color='#FFC857', edgecolor='#0F0F1A', alpha=0.9)
        ax.set_xlabel('Amount ($)', color='#9090A8')
        ax.set_ylabel('Frequency', color='#9090A8')
        ax.tick_params(colors='#9090A8')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2A2A3E')
        ax.set_title('Transaction Amounts', color='#E8E8F0', fontsize=13, fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Course Ratings Distribution")
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#1A1A2E')
        ax.set_facecolor('#1A1A2E')
        ax.hist(courses['CourseRating'], bins=20, color='#FF6584', edgecolor='#0F0F1A', alpha=0.9)
        ax.set_xlabel('Rating', color='#9090A8')
        ax.set_ylabel('Count', color='#9090A8')
        ax.tick_params(colors='#9090A8')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2A2A3E')
        ax.set_title('Course Rating Distribution', color='#E8E8F0', fontsize=13, fontweight='bold')
        st.pyplot(fig)
        plt.close()

    # Heatmap: Category x Level
    st.subheader("Category × Level Enrollment Heatmap")
    heatmap_data = merged.groupby(['CourseCategory', 'CourseLevel']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#1A1A2E')
    ax.set_facecolor('#1A1A2E')
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='Blues',
                ax=ax, linewidths=0.5, linecolor='#0F0F1A',
                cbar_kws={'shrink': 0.8})
    ax.set_title('Enrollment Heatmap by Category & Level', color='#E8E8F0', fontsize=13, fontweight='bold', pad=10)
    ax.tick_params(colors='#9090A8')
    ax.set_xlabel('Course Level', color='#9090A8')
    ax.set_ylabel('Category', color='#9090A8')
    st.pyplot(fig)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2: CLUSTERING ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Clustering Analysis":
    st.markdown('<div class="hero-title">Clustering Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">K-Means learner segmentation with Elbow & Silhouette validation</div>', unsafe_allow_html=True)

    X_scaled, scaler, inertias, silhouettes = build_clustered_features(features)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Elbow Method")
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#1A1A2E')
        ax.set_facecolor('#1A1A2E')
        ks = list(inertias.keys())
        vals = list(inertias.values())
        ax.plot(ks, vals, 'o-', color='#6C63FF', linewidth=2.5, markersize=8)
        ax.axvline(4, color='#FF6584', linestyle='--', linewidth=2, alpha=0.7, label='k=4 selected')
        ax.set_xlabel('Number of Clusters (k)', color='#9090A8')
        ax.set_ylabel('Inertia', color='#9090A8')
        ax.tick_params(colors='#9090A8')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2A2A3E')
        ax.legend(facecolor='#1A1A2E', labelcolor='#E8E8F0')
        ax.set_title('Elbow Curve', color='#E8E8F0', fontsize=13, fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Silhouette Scores")
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#1A1A2E')
        ax.set_facecolor('#1A1A2E')
        ks = list(silhouettes.keys())
        svals = list(silhouettes.values())
        bars = ax.bar(ks, svals, color=['#6C63FF' if k != 4 else '#43C6AC' for k in ks],
                      edgecolor='#0F0F1A', width=0.6)
        ax.set_xlabel('k', color='#9090A8')
        ax.set_ylabel('Silhouette Score', color='#9090A8')
        ax.tick_params(colors='#9090A8')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2A2A3E')
        ax.set_title('Silhouette Scores by k', color='#E8E8F0', fontsize=13, fontweight='bold')
        for bar, val in zip(bars, svals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{val:.3f}', ha='center', color='#E8E8F0', fontsize=9)
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Cluster Size Distribution")
    cluster_counts = features['Cluster'].value_counts().sort_index()
    cols = st.columns(4)
    for i, (col, (c, cnt)) in enumerate(zip(cols, cluster_counts.items())):
        with col:
            st.markdown(f"""
            <div class="segment-card">
                <div class="badge cluster-pill-{c}">{CLUSTER_NAMES[c]}</div><br><br>
                <span style="font-size:2rem;font-weight:700;font-family:'Space Mono'">{cnt:,}</span><br>
                <span style="color:#9090A8;font-size:0.85rem;">learners</span><br><br>
                <span style="color:#9090A8;font-size:0.8rem;">{CLUSTER_DESCRIPTIONS[c]}</span>
            </div>
            """, unsafe_allow_html=True)

    # Cluster profiles table
    st.subheader("Cluster Profiles")
    profile = features.groupby('Cluster').agg(
        Count=('UserID', 'count'),
        Avg_Courses=('total_courses_enrolled', 'mean'),
        Avg_Spending=('total_spending', 'mean'),
        Avg_Diversity=('category_diversity', 'mean'),
        Avg_Depth=('learning_depth_index', 'mean'),
        Avg_Age=('Age', 'mean'),
        Top_Category=('preferred_category', lambda x: x.mode()[0]),
        Top_Level=('preferred_level', lambda x: x.mode()[0])
    ).round(2)
    profile.index = [CLUSTER_NAMES[i] for i in profile.index]
    st.dataframe(profile, use_container_width=True)

    # Feature comparison radar-style bar chart
    st.subheader("Feature Comparison Across Clusters")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor='#1A1A2E')
    metrics = [('total_courses_enrolled', 'Avg Courses Enrolled'),
               ('total_spending', 'Avg Total Spending ($)'),
               ('category_diversity', 'Avg Category Diversity')]

    for ax, (col, title) in zip(axes, metrics):
        ax.set_facecolor('#1A1A2E')
        vals = [features[features['Cluster'] == c][col].mean() for c in range(4)]
        labels = [CLUSTER_NAMES[c].split(' ', 1)[1] for c in range(4)]
        bars = ax.bar(range(4), vals, color=CLUSTER_COLORS, edgecolor='#0F0F1A', width=0.6)
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, fontsize=8, color='#9090A8', rotation=15, ha='right')
        ax.set_ylabel(title, color='#9090A8', fontsize=9)
        ax.tick_params(colors='#9090A8')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2A2A3E')
        ax.set_title(title, color='#E8E8F0', fontsize=10, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3: LEARNER PROFILE EXPLORER
# ════════════════════════════════════════════════════════════════════════════
elif page == "👤 Learner Profile Explorer":
    st.markdown('<div class="hero-title">Learner Profile Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">View assigned segment and behavioral profile for any learner</div>', unsafe_allow_html=True)

    uid_list = features['UserID'].tolist()
    selected_uid = st.selectbox("Select Learner ID", uid_list)

    learner = features[features['UserID'] == selected_uid].iloc[0]
    user_info = users[users['UserID'] == selected_uid].iloc[0]
    cluster = int(learner['Cluster'])

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
        <div class="segment-card">
            <div class="badge cluster-pill-{cluster}">{CLUSTER_NAMES[cluster]}</div>
            <br><br>
            <b style="font-size:1.1rem">{user_info['UserName']}</b><br>
            <span style="color:#9090A8">ID: {selected_uid}</span><br><br>
            <b>Age:</b> {int(user_info['Age'])}<br>
            <b>Gender:</b> {user_info['Gender']}<br>
            <b>Email:</b> {user_info['Email'][:28]}...<br>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="segment-card">
            <b>🎯 Segment Description</b><br><br>
            <span style="color:#9090A8">{CLUSTER_DESCRIPTIONS[cluster]}</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Courses Enrolled", int(learner['total_courses_enrolled']))
        m2.metric("Total Spent", f"${learner['total_spending']:.0f}")
        m3.metric("Avg per Course", f"${learner['avg_spending_per_course']:.0f}")
        m4.metric("Category Diversity", int(learner['category_diversity']))

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="segment-card">
                <b>📚 Learning Preferences</b><br><br>
                🗂️ <b>Fav Category:</b> {learner['preferred_category']}<br>
                📶 <b>Fav Level:</b> {learner['preferred_level']}<br>
                ⭐ <b>Avg Rating Enrolled:</b> {learner['avg_course_rating_enrolled']:.2f}<br>
                📐 <b>Depth Index:</b> {learner['learning_depth_index']:.2f}
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            # Mini spider-chart for learner vs cluster avg
            cluster_avg = features[features['Cluster'] == cluster]
            fig, ax = plt.subplots(figsize=(4, 3), facecolor='#1A1A2E')
            ax.set_facecolor('#1A1A2E')
            dims = ['Courses', 'Spending\n(÷100)', 'Diversity', 'Depth×10']
            learner_vals = [
                learner['total_courses_enrolled'],
                learner['total_spending'] / 100,
                learner['category_diversity'],
                learner['learning_depth_index'] * 10
            ]
            cluster_vals = [
                cluster_avg['total_courses_enrolled'].mean(),
                cluster_avg['total_spending'].mean() / 100,
                cluster_avg['category_diversity'].mean(),
                cluster_avg['learning_depth_index'].mean() * 10
            ]
            x = range(4)
            ax.plot(x, learner_vals, 'o-', color=CLUSTER_COLORS[cluster], linewidth=2, label='Learner', markersize=6)
            ax.plot(x, cluster_vals, 's--', color='#9090A8', linewidth=1.5, label='Cluster Avg', markersize=5)
            ax.set_xticks(x)
            ax.set_xticklabels(dims, color='#9090A8', fontsize=8)
            ax.tick_params(colors='#9090A8')
            for spine in ax.spines.values():
                spine.set_edgecolor('#2A2A3E')
            ax.legend(facecolor='#1A1A2E', labelcolor='#E8E8F0', fontsize=8)
            ax.set_title('Learner vs Cluster Avg', color='#E8E8F0', fontsize=10, fontweight='bold')
            st.pyplot(fig)
            plt.close()

    # Learner's enrolled courses
    st.subheader("Enrolled Courses")
    learner_trans = trans[trans['UserID'] == selected_uid].merge(courses, on='CourseID')
    if not learner_trans.empty:
        st.dataframe(learner_trans[['CourseID', 'CourseCategory', 'CourseLevel', 'CourseRating', 'Amount', 'TransactionDate']].reset_index(drop=True), use_container_width=True)
    else:
        st.info("No transactions found for this learner.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4: COURSE RECOMMENDER
# ════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Course Recommender":
    st.markdown('<div class="hero-title">Personalized Course Recommender</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Cluster-aware recommendations using content-based filtering & rating-weighted relevance</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        selected_uid = st.selectbox("Select Learner", features['UserID'].tolist())
        filter_level = st.selectbox("Filter by Level", ['Any', 'Beginner', 'Intermediate', 'Advanced'])
        n_recs = st.slider("Number of Recommendations", 3, 10, 5)

    learner = features[features['UserID'] == selected_uid].iloc[0]
    cluster = int(learner['Cluster'])
    pref_cat = learner['preferred_category']
    pref_level = learner['preferred_level']

    with col1:
        st.markdown(f"""
        <div class="segment-card" style="margin-top:1rem">
            <b>Learner Segment</b><br>
            <div class="badge cluster-pill-{cluster}">{CLUSTER_NAMES[cluster]}</div><br><br>
            <span style="color:#9090A8;font-size:0.85rem">Preferred: {pref_cat} · {pref_level}</span>
        </div>
        """, unsafe_allow_html=True)

    # Recommendation logic
    enrolled = trans[trans['UserID'] == selected_uid]['CourseID'].tolist()
    # Cluster peers
    cluster_peers = features[features['Cluster'] == cluster]['UserID'].tolist()
    peer_enrollments = trans[trans['UserID'].isin(cluster_peers) & ~trans['CourseID'].isin(enrolled)]
    popular_in_cluster = peer_enrollments['CourseID'].value_counts().reset_index()
    popular_in_cluster.columns = ['CourseID', 'peer_count']

    # Score = category match + level match + rating + popularity
    not_enrolled = courses[~courses['CourseID'].isin(enrolled)].copy()
    not_enrolled = not_enrolled.merge(popular_in_cluster, on='CourseID', how='left').fillna({'peer_count': 0})

    not_enrolled['score'] = (
        (not_enrolled['CourseCategory'] == pref_cat).astype(int) * 3 +
        (not_enrolled['CourseLevel'] == pref_level).astype(int) * 2 +
        not_enrolled['CourseRating'] / 5.0 +
        not_enrolled['peer_count'] / (not_enrolled['peer_count'].max() + 1)
    )

    if filter_level != 'Any':
        not_enrolled = not_enrolled[not_enrolled['CourseLevel'] == filter_level]

    recs = not_enrolled.sort_values('score', ascending=False).head(n_recs)

    with col2:
        st.subheader(f"Top {n_recs} Recommendations")
        for _, row in recs.iterrows():
            score_pct = min(int((row['score'] / 7.0) * 100), 100)
            stars = '⭐' * round(row['CourseRating'])
            st.markdown(f"""
            <div class="rec-card">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <span style="font-family:'Space Mono';font-weight:700">{row['CourseID']}</span>
                    <span class="badge cluster-pill-{cluster}">{row['CourseLevel']}</span>
                </div>
                <div style="margin:0.4rem 0">
                    🗂️ <b>{row['CourseCategory']}</b> &nbsp;·&nbsp; 📺 {row['CourseType']} &nbsp;·&nbsp; {stars} {row['CourseRating']}
                </div>
                <div style="background:#0F0F1A;border-radius:6px;height:6px;margin-top:8px">
                    <div style="background:{CLUSTER_COLORS[cluster]};height:6px;border-radius:6px;width:{score_pct}%"></div>
                </div>
                <span style="color:#9090A8;font-size:0.75rem">Relevance: {score_pct}% &nbsp;·&nbsp; 👥 {int(row['peer_count'])} cluster peers enrolled</span>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5: SEGMENT COMPARISON
# ════════════════════════════════════════════════════════════════════════════
elif page == "📈 Segment Comparison":
    st.markdown('<div class="hero-title">Segment Comparison Panel</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Side-by-side analysis of all learner segments</div>', unsafe_allow_html=True)

    # Summary cards
    cols = st.columns(4)
    for i, col in enumerate(cols):
        seg = features[features['Cluster'] == i]
        with col:
            st.markdown(f"""
            <div class="segment-card">
                <div class="badge cluster-pill-{i}">{CLUSTER_NAMES[i]}</div><br><br>
                <b>Size:</b> {len(seg):,}<br>
                <b>Avg Courses:</b> {seg['total_courses_enrolled'].mean():.1f}<br>
                <b>Avg Spend:</b> ${seg['total_spending'].mean():.0f}<br>
                <b>Top Category:</b> {seg['preferred_category'].mode()[0]}<br>
                <b>Top Level:</b> {seg['preferred_level'].mode()[0]}<br>
                <b>Avg Age:</b> {seg['Age'].mean():.1f}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Spending distribution per cluster
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Spending Distribution by Segment")
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1A1A2E')
        ax.set_facecolor('#1A1A2E')
        for i in range(4):
            seg_data = features[features['Cluster'] == i]['total_spending']
            ax.hist(seg_data, bins=30, alpha=0.6,
                    color=CLUSTER_COLORS[i], label=CLUSTER_NAMES[i].split(' ', 1)[1], edgecolor='none')
        ax.set_xlabel('Total Spending ($)', color='#9090A8')
        ax.set_ylabel('Count', color='#9090A8')
        ax.tick_params(colors='#9090A8')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2A2A3E')
        ax.legend(facecolor='#1A1A2E', labelcolor='#E8E8F0', fontsize=8)
        ax.set_title('Spending Overlap', color='#E8E8F0', fontsize=13, fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Category Preference by Segment")
        cat_cluster = features.groupby(['Cluster', 'preferred_category']).size().reset_index(name='count')
        cat_pivot = cat_cluster.pivot(index='preferred_category', columns='Cluster', values='count').fillna(0)
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1A1A2E')
        ax.set_facecolor('#1A1A2E')
        x = range(len(cat_pivot))
        width = 0.2
        for i in range(4):
            if i in cat_pivot.columns:
                offset = (i - 1.5) * width
                ax.bar([xi + offset for xi in x], cat_pivot[i],
                       width=width, color=CLUSTER_COLORS[i], label=CLUSTER_NAMES[i].split(' ',1)[1], edgecolor='#0F0F1A')
        ax.set_xticks(range(len(cat_pivot)))
        ax.set_xticklabels(cat_pivot.index, rotation=30, ha='right', color='#9090A8', fontsize=8)
        ax.set_ylabel('Count', color='#9090A8')
        ax.tick_params(colors='#9090A8')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2A2A3E')
        ax.legend(facecolor='#1A1A2E', labelcolor='#E8E8F0', fontsize=7)
        ax.set_title('Category Preference Breakdown', color='#E8E8F0', fontsize=13, fontweight='bold')
        st.pyplot(fig)
        plt.close()

    # Age × Spending scatter
    st.subheader("Age vs Total Spending (by Segment)")
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#1A1A2E')
    ax.set_facecolor('#1A1A2E')
    for i in range(4):
        seg = features[features['Cluster'] == i]
        ax.scatter(seg['Age'], seg['total_spending'],
                   color=CLUSTER_COLORS[i], alpha=0.4, s=15, label=CLUSTER_NAMES[i].split(' ',1)[1])
    ax.set_xlabel('Age', color='#9090A8')
    ax.set_ylabel('Total Spending ($)', color='#9090A8')
    ax.tick_params(colors='#9090A8')
    for spine in ax.spines.values():
        spine.set_edgecolor('#2A2A3E')
    ax.legend(facecolor='#1A1A2E', labelcolor='#E8E8F0')
    ax.set_title('Age vs Spending by Segment', color='#E8E8F0', fontsize=13, fontweight='bold')
    st.pyplot(fig)
    plt.close()

    # Engagement lift table
    st.subheader("Engagement Lift (Segment vs Overall Average)")
    overall_avg = features[['total_courses_enrolled', 'total_spending', 'category_diversity', 'learning_depth_index']].mean()
    lift_rows = []
    for i in range(4):
        seg = features[features['Cluster'] == i]
        seg_avg = seg[['total_courses_enrolled', 'total_spending', 'category_diversity', 'learning_depth_index']].mean()
        lift = ((seg_avg - overall_avg) / (overall_avg + 0.001) * 100).round(1)
        lift_rows.append({
            'Segment': CLUSTER_NAMES[i],
            'Courses Lift %': f"{lift['total_courses_enrolled']:+.1f}%",
            'Spending Lift %': f"{lift['total_spending']:+.1f}%",
            'Diversity Lift %': f"{lift['category_diversity']:+.1f}%",
            'Depth Lift %': f"{lift['learning_depth_index']:+.1f}%"
        })
    st.dataframe(pd.DataFrame(lift_rows).set_index('Segment'), use_container_width=True)
