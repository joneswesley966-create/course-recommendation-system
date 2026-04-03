import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import random

# Load users
users = pd.read_csv("EduPro Online Platform.xlsx - Users (1).csv")

# Create courses
courses = pd.DataFrame({
    "CourseID": [101,102,103,104,105],
    "CourseName": ["Python", "Data Science", "Web Dev", "Machine Learning", "AI"],
    "Category": ["Programming", "Data", "Web", "ML", "AI"]
})

# Create transactions
transactions = []

for user in users["UserID"]:
    for i in range(random.randint(1,3)):
        transactions.append({
            "UserID": user,
            "CourseID": random.choice(courses["CourseID"])
        })

transactions = pd.DataFrame(transactions)

# Feature Engineering
user_course_count = transactions.groupby("UserID")["CourseID"].count().reset_index()
user_course_count.columns = ["UserID", "course_count"]

user_data = pd.merge(users, user_course_count, on="UserID", how="left")
user_data["course_count"] = user_data["course_count"].fillna(0)

# Convert Gender
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
user_data["Gender"] = le.fit_transform(user_data["Gender"])

# Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(user_data[["Age", "Gender", "course_count"]])

# Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
user_data["Cluster"] = kmeans.fit_predict(scaled_data)

print(user_data.head())

# Recommendation Function
def recommend_courses(user_id):
    user_cluster = user_data[user_data["UserID"] == user_id]["Cluster"].values[0]

    similar_users = user_data[user_data["Cluster"] == user_cluster]["UserID"]

    recommended = transactions[transactions["UserID"].isin(similar_users)]

    top_courses = recommended["CourseID"].value_counts().head(3)

    return top_courses.index.tolist()

def recommend_courses(user_id):
    user_cluster = user_data[user_data["UserID"] == user_id]["Cluster"].values[0]

    similar_users = user_data[user_data["Cluster"] == user_cluster]["UserID"]

    recommended = transactions[transactions["UserID"].isin(similar_users)]

    top_courses = recommended["CourseID"].value_counts().head(3).index.tolist()

    # Convert ID → Name
    result = courses[courses["CourseID"].isin(top_courses)]

    return result[["CourseName", "Category"]]

def recommend_courses(user_id):
    # Check user exists
    if user_id not in user_data["UserID"].values:
        return "User not found"

    user_cluster = user_data[user_data["UserID"] == user_id]["Cluster"].values[0]

    similar_users = user_data[user_data["Cluster"] == user_cluster]["UserID"]

    recommended = transactions[transactions["UserID"].isin(similar_users)]

    top_courses = recommended["CourseID"].value_counts().head(3).index.tolist()

    result = courses[courses["CourseID"].isin(top_courses)]

    return result[["CourseName", "Category"]]

print(user_data["UserID"].head())