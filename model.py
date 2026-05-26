# ============================================================
# EduPro - Student Segmentation & Course Recommendation
# analysis.py - Standalone Analysis Script
# Author: Jones Wesley
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ── Output folder for charts ──────────────────────────────────
import os
os.makedirs("outputs", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
print("=" * 55)
print("STEP 1: Loading Data")
print("=" * 55)

users    = pd.read_csv("users.csv")
courses  = pd.read_csv("courses.csv")
trans    = pd.read_csv("transactions.csv")
trans["TransactionDate"] = pd.to_datetime(trans["TransactionDate"])

print(f"Users        : {users.shape}")
print(f"Courses      : {courses.shape}")
print(f"Transactions : {trans.shape}")

# Merge all
merged = trans.merge(courses, on="CourseID").merge(users, on="UserID")
print(f"Merged shape : {merged.shape}")


# ─────────────────────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 2: Exploratory Data Analysis")
print("=" * 55)

# --- Basic stats ---
print("\n-- Users --")
print(users.describe())
print("\nGender distribution:\n", users["Gender"].value_counts())
print("\nAge range:", users["Age"].min(), "–", users["Age"].max())

print("\n-- Courses --")
print(courses["CourseCategory"].value_counts())
print(courses["CourseLevel"].value_counts())
print(f"Avg course rating: {courses['CourseRating'].mean():.2f}")

print("\n-- Transactions --")
print(f"Total revenue     : ${trans['Amount'].sum():,.2f}")
print(f"Avg per transaction: ${trans['Amount'].mean():.2f}")
print(f"Avg enrollments/user: {len(trans)/len(users):.1f}")

# --- Missing values ---
print("\n-- Missing Values --")
for df, name in [(users,"Users"), (courses,"Courses"), (trans,"Transactions")]:
    print(f"{name}: {df.isnull().sum().sum()} missing")

# ── Plot 1: Gender Pie ──
fig, ax = plt.subplots(figsize=(5, 4))
users["Gender"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax,
    colors=["#6C63FF","#FF6584"], startangle=90, wedgeprops=dict(width=0.6))
ax.set_title("Gender Distribution")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig("outputs/eda_01_gender.png", dpi=150)
plt.close()
print("Saved: outputs/eda_01_gender.png")

# ── Plot 2: Age Histogram ──
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(users["Age"], bins=20, color="#6C63FF", edgecolor="white", alpha=0.85)
ax.axvline(users["Age"].mean(), color="#FF6584", linestyle="--",
           label=f"Mean Age: {users['Age'].mean():.1f}")
ax.set_xlabel("Age"); ax.set_ylabel("Count")
ax.set_title("Age Distribution of Learners")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/eda_02_age.png", dpi=150)
plt.close()
print("Saved: outputs/eda_02_age.png")

# ── Plot 3: Enrollment by Category ──
cat_counts = merged["CourseCategory"].value_counts()
fig, ax = plt.subplots(figsize=(7, 4))
cat_counts.plot.barh(ax=ax, color="#43C6AC", edgecolor="white")
ax.set_title("Enrollments by Course Category")
ax.set_xlabel("Count")
plt.tight_layout()
plt.savefig("outputs/eda_03_category.png", dpi=150)
plt.close()
print("Saved: outputs/eda_03_category.png")

# ── Plot 4: Course Level ──
level_counts = merged["CourseLevel"].value_counts()
fig, ax = plt.subplots(figsize=(5, 4))
level_counts.plot.bar(ax=ax, color=["#6C63FF","#43C6AC","#FF6584"], edgecolor="white")
ax.set_title("Enrollments by Course Level")
ax.set_ylabel("Count"); ax.set_xticklabels(level_counts.index, rotation=0)
plt.tight_layout()
plt.savefig("outputs/eda_04_level.png", dpi=150)
plt.close()
print("Saved: outputs/eda_04_level.png")

# ── Plot 5: Spending Distribution ──
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(trans["Amount"], bins=30, color="#FFC857", edgecolor="white", alpha=0.85)
ax.set_title("Transaction Amount Distribution")
ax.set_xlabel("Amount ($)"); ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/eda_05_spending.png", dpi=150)
plt.close()
print("Saved: outputs/eda_05_spending.png")

# ── Plot 6: Category × Level Heatmap ──
heatmap_data = merged.groupby(["CourseCategory","CourseLevel"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="Blues", ax=ax, linewidths=0.5)
ax.set_title("Enrollment Heatmap: Category × Level")
plt.tight_layout()
plt.savefig("outputs/eda_06_heatmap.png", dpi=150)
plt.close()
print("Saved: outputs/eda_06_heatmap.png")


# ─────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 3: Feature Engineering")
print("=" * 55)

# --- Engagement Features ---
engagement = trans.groupby("UserID").agg(
    total_courses_enrolled = ("CourseID", "count"),
    enrollment_frequency   = ("TransactionDate", "nunique"),
    total_spending         = ("Amount", "sum"),
    avg_spending_per_course= ("Amount", "mean")
).reset_index()

# --- Preference Features ---
cat_pref = merged.groupby(["UserID","CourseCategory"]).size().reset_index(name="count")
cat_pivot = cat_pref.pivot(index="UserID", columns="CourseCategory", values="count").fillna(0)
cat_pivot["preferred_category"] = cat_pivot.idxmax(axis=1)
cat_pivot["category_diversity"] = (cat_pivot.drop("preferred_category", axis=1) > 0).sum(axis=1)
cat_simple = cat_pivot[["preferred_category","category_diversity"]].reset_index()

level_pref = merged.groupby(["UserID","CourseLevel"]).size().reset_index(name="count")
level_pivot = level_pref.pivot(index="UserID", columns="CourseLevel", values="count").fillna(0)
level_pivot["preferred_level"] = level_pivot.idxmax(axis=1)
level_simple = level_pivot[["preferred_level"]].reset_index()

avg_rating = merged.groupby("UserID")["CourseRating"].mean().reset_index()
avg_rating.columns = ["UserID","avg_course_rating_enrolled"]

# --- Behavioral Features ---
adv = level_pivot.get("Advanced", pd.Series(0, index=level_pivot.index))
beg = level_pivot.get("Beginner", pd.Series(0, index=level_pivot.index))
depth_df = pd.DataFrame({
    "UserID": level_pivot.index,
    "learning_depth_index": (adv / (beg + 1)).values
}).reset_index(drop=True)

# --- Combine ---
features = (engagement
    .merge(cat_simple,   on="UserID")
    .merge(level_simple, on="UserID")
    .merge(avg_rating,   on="UserID")
    .merge(depth_df,     on="UserID")
    .merge(users[["UserID","Age","Gender"]], on="UserID")
)

print(f"Feature matrix shape : {features.shape}")
print("Columns:", features.columns.tolist())
print(features.describe())


# ─────────────────────────────────────────────────────────────
# 4. DATA PREPROCESSING
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 4: Data Preprocessing")
print("=" * 55)

le_cat = LabelEncoder()
le_lev = LabelEncoder()
le_gen = LabelEncoder()

features["preferred_category_enc"] = le_cat.fit_transform(features["preferred_category"])
features["preferred_level_enc"]    = le_lev.fit_transform(features["preferred_level"])
features["gender_enc"]             = le_gen.fit_transform(features["Gender"])

numeric_cols = [
    "total_courses_enrolled", "total_spending", "avg_spending_per_course",
    "category_diversity", "avg_course_rating_enrolled", "learning_depth_index",
    "Age", "preferred_category_enc", "preferred_level_enc", "gender_enc"
]

X = features[numeric_cols].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaled feature matrix shape:", X_scaled.shape)
print("Normalization complete. All features on unit scale.")


# ─────────────────────────────────────────────────────────────
# 5. LEARNER SEGMENTATION — K-MEANS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 5: K-Means Clustering")
print("=" * 55)

inertias, silhouettes = {}, {}
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias[k]    = km.inertia_
    silhouettes[k] = silhouette_score(X_scaled, labels)
    print(f"  k={k} | Inertia: {km.inertia_:,.0f} | Silhouette: {silhouettes[k]:.4f}")

best_k = max(silhouettes, key=silhouettes.get)
print(f"\nBest Silhouette at k={best_k} → using k=4 for interpretability")

# ── Plot 7: Elbow ──
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(list(inertias.keys()), list(inertias.values()), "o-", color="#6C63FF", linewidth=2.5, markersize=8)
ax.axvline(4, color="#FF6584", linestyle="--", label="k=4 selected")
ax.set_xlabel("Number of Clusters (k)"); ax.set_ylabel("Inertia")
ax.set_title("Elbow Method for Optimal k")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/cluster_01_elbow.png", dpi=150)
plt.close()
print("Saved: outputs/cluster_01_elbow.png")

# ── Plot 8: Silhouette ──
fig, ax = plt.subplots(figsize=(6, 4))
ks = list(silhouettes.keys())
svals = list(silhouettes.values())
colors = ["#43C6AC" if k == 4 else "#6C63FF" for k in ks]
bars = ax.bar(ks, svals, color=colors, edgecolor="white", width=0.6)
for bar, val in zip(bars, svals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"{val:.3f}", ha="center", fontsize=9)
ax.set_xlabel("k"); ax.set_ylabel("Silhouette Score")
ax.set_title("Silhouette Scores by k  (green = selected)")
plt.tight_layout()
plt.savefig("outputs/cluster_02_silhouette.png", dpi=150)
plt.close()
print("Saved: outputs/cluster_02_silhouette.png")

# Final K-Means with k=4
km_final = KMeans(n_clusters=4, random_state=42, n_init=10)
features["Cluster"] = km_final.fit_predict(X_scaled)
print("\nCluster sizes:\n", features["Cluster"].value_counts().sort_index())


# ─────────────────────────────────────────────────────────────
# 6. HIERARCHICAL CLUSTERING (Validation)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 6: Hierarchical Clustering (Validation)")
print("=" * 55)

hc = AgglomerativeClustering(n_clusters=4, linkage="ward")
features["HC_Cluster"] = hc.fit_predict(X_scaled)

hc_sil = silhouette_score(X_scaled, features["HC_Cluster"])
km_sil = silhouette_score(X_scaled, features["Cluster"])
print(f"K-Means     Silhouette: {km_sil:.4f}")
print(f"Hierarchical Silhouette: {hc_sil:.4f}")
print("→ K-Means selected as primary model (consistent performance)")


# ─────────────────────────────────────────────────────────────
# 7. CLUSTER PROFILES
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 7: Cluster Profiles")
print("=" * 55)

CLUSTER_NAMES = {
    0: "Casual Explorers",
    1: "Power Learners",
    2: "Budget Browsers",
    3: "Focused Achievers"
}

profile = features.groupby("Cluster").agg(
    Count                   = ("UserID", "count"),
    Avg_Courses_Enrolled    = ("total_courses_enrolled", "mean"),
    Avg_Total_Spending      = ("total_spending", "mean"),
    Avg_Category_Diversity  = ("category_diversity", "mean"),
    Avg_Learning_Depth      = ("learning_depth_index", "mean"),
    Avg_Age                 = ("Age", "mean"),
    Top_Category            = ("preferred_category", lambda x: x.mode()[0]),
    Top_Level               = ("preferred_level",    lambda x: x.mode()[0])
).round(2)

profile.index = [CLUSTER_NAMES[i] for i in profile.index]
print(profile.to_string())

# ── Plot 9: Cluster feature comparison ──
CLUSTER_COLORS = ["#6C63FF","#43C6AC","#FF6584","#FFC857"]
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
metrics = [
    ("total_courses_enrolled", "Avg Courses Enrolled"),
    ("total_spending",         "Avg Total Spending ($)"),
    ("category_diversity",     "Avg Category Diversity")
]
for ax, (col, title) in zip(axes, metrics):
    vals   = [features[features["Cluster"] == c][col].mean() for c in range(4)]
    labels = [CLUSTER_NAMES[c] for c in range(4)]
    ax.bar(range(4), vals, color=CLUSTER_COLORS, edgecolor="white", width=0.6)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_title(title, fontweight="bold")
plt.suptitle("Cluster Feature Comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/cluster_03_profiles.png", dpi=150)
plt.close()
print("Saved: outputs/cluster_03_profiles.png")


# ─────────────────────────────────────────────────────────────
# 8. EVALUATION METRICS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 8: Evaluation & Validation")
print("=" * 55)

km_sil_final = silhouette_score(X_scaled, features["Cluster"])
print(f"Silhouette Score (k=4)       : {km_sil_final:.4f}")
print(f"Inertia (k=4)                : {km_final.inertia_:,.2f}")

# Intra-cluster similarity (avg pairwise distance proxy)
for c in range(4):
    seg = X_scaled[features["Cluster"] == c]
    center = seg.mean(axis=0)
    avg_dist = np.mean(np.linalg.norm(seg - center, axis=1))
    print(f"  Cluster {c} ({CLUSTER_NAMES[c]}): avg dist to centroid = {avg_dist:.4f}")

# Engagement Lift
overall_avg = features[["total_courses_enrolled","total_spending","category_diversity"]].mean()
print("\n-- Engagement Lift vs Overall Average --")
for c in range(4):
    seg_avg = features[features["Cluster"] == c][["total_courses_enrolled","total_spending","category_diversity"]].mean()
    lift = ((seg_avg - overall_avg) / (overall_avg + 0.001) * 100).round(1)
    print(f"  {CLUSTER_NAMES[c]}: Courses {lift['total_courses_enrolled']:+.1f}%, "
          f"Spending {lift['total_spending']:+.1f}%, "
          f"Diversity {lift['category_diversity']:+.1f}%")


# ─────────────────────────────────────────────────────────────
# 9. SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 9: Saving Outputs")
print("=" * 55)

features.to_csv("learner_features_clustered.csv", index=False)
profile.to_csv("outputs/cluster_profiles.csv")
print("Saved: learner_features_clustered.csv")
print("Saved: outputs/cluster_profiles.csv")

print("\n✅ Analysis complete. Run 'streamlit run app.py' for the dashboard.")