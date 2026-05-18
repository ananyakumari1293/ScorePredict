import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("student_data.csv")

# Features (inputs)
X = df[['study_hours',
        'mid_marks',
        'attendance',
        'assignments_completed',
        'sleep_hours']]

# Target (output)
y = df['final_marks']

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# User inputs
study_hours = float(input("Enter study hours: "))
mid_marks = float(input("Enter mid sem marks: "))
attendance = float(input("Enter attendance percentage: "))
assignments_completed = float(input("Enter assignments completed: "))
sleep_hours = float(input("Enter average sleep hours: "))

# Predict final marks
prediction = model.predict([[
    study_hours,
    mid_marks,
    attendance,
    assignments_completed,
    sleep_hours
]])

# Show prediction
print(f"\nPredicted Final Marks: {prediction[0]:.2f}")

# Calculate model accuracy
predictions = model.predict(X)

score = r2_score(y, predictions)

print(f"Model Accuracy: {score * 100:.2f}%")

# ---------------- GRAPH SECTION ---------------- #

plt.figure(figsize=(8,5))

# Scatter plot
sns.scatterplot(
    x='study_hours',
    y='final_marks',
    data=df
)

# Sort data for smooth line
sorted_df = df.sort_values(by='study_hours')

# Prediction line
plt.plot(
    sorted_df['study_hours'],
    model.predict(sorted_df[[
        'study_hours',
        'mid_marks',
        'attendance',
        'assignments_completed',
        'sleep_hours'
    ]]),
    color='red'
)

# Labels and title
plt.title("Study Hours vs Final Marks")
plt.xlabel("Study Hours")
plt.ylabel("Final Marks")

# Show graph
plt.show()