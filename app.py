import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error
)

from sklearn.model_selection import train_test_split

# ---------------- LOAD DATASET ---------------- #

df = pd.read_csv("student_data.csv")

# ---------------- FEATURES & TARGET ---------------- #

X = df[[
    'study_hours',
    'mid_marks',
    'attendance',
    'assignments_completed',
    'sleep_hours'
]]

y = df['final_marks']

# ---------------- TRAIN TEST SPLIT ---------------- #

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------- CREATE MODEL ---------------- #

model = LinearRegression()

# ---------------- TRAIN MODEL ---------------- #

model.fit(X_train, y_train)

# ---------------- USER INPUT ---------------- #

study_hours = float(input("Enter study hours: "))
mid_marks = float(input("Enter mid sem marks: "))
attendance = float(input("Enter attendance percentage: "))
assignments_completed = float(input("Enter assignments completed: "))
sleep_hours = float(input("Enter average sleep hours: "))

# ---------------- PREDICTION ---------------- #

prediction = model.predict([[
    study_hours,
    mid_marks,
    attendance,
    assignments_completed,
    sleep_hours
]])

predicted_marks = prediction[0]

# ---------------- REALISTIC RULES ---------------- #

# Study hours importance (MOST IMPORTANT)

if study_hours <= 2:
    predicted_marks -= 20

elif study_hours <= 4:
    predicted_marks -= 10

# Attendance importance

if attendance < 70:
    predicted_marks -= 5

# Sleep importance

if sleep_hours < 5:
    predicted_marks -= 5

# Assignment impact kept VERY SMALL

if assignments_completed <= 2:
    predicted_marks -= 2

# Limit between 0 and 100

predicted_marks = max(0, min(predicted_marks, 100))

# ---------------- SHOW RESULT ---------------- #

print(f"\nPredicted Final Marks: {predicted_marks:.2f}")

# ---------------- MODEL EVALUATION ---------------- #

test_predictions = model.predict(X_test)

score = r2_score(y_test, test_predictions)

mae = mean_absolute_error(y_test, test_predictions)

print(f"\nModel Accuracy (R² Score): {score * 100:.2f}%")

print(f"Average Prediction Error: {mae:.2f} marks")

# ---------------- GRAPH ---------------- #

plt.figure(figsize=(8,5))

# Scatter plot

sns.scatterplot(
    x='study_hours',
    y='final_marks',
    data=df
)

# Sort for smooth line

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

plt.grid(True)

# Show graph

plt.show()