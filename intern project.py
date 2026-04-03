import pandas as pd
import random

# Load data
users = pd.read_csv("EduPro Online Platform.xlsx - Users (1).csv")

print("Users Data:")
print(users.head())

# Create courses
courses = pd.DataFrame({
    "CourseID": [101,102,103,104,105],
    "CourseName": ["Python", "Data Science", "Web Dev", "Machine Learning", "AI"],
    "Category": ["Programming", "Data", "Web", "ML", "AI"]
})

print("\nCourses:")
print(courses)

# Create transactions
transactions = []

for user in users["UserID"]:
    for i in range(random.randint(1,3)):
        transactions.append({
            "UserID": user,
            "CourseID": random.choice(courses["CourseID"])
        })

transactions = pd.DataFrame(transactions)

print("\nTransactions:")
print(transactions.head())
