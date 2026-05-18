import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error
)

from sklearn.model_selection import train_test_split

# ---------------- PAGE SETTINGS ---------------- #

st.set_page_config(
    page_title="ScorePredict",
    page_icon="📚",
    layout="centered"
)

# ---------------- TITLE ---------------- #

st.title("📚 ScorePredict")

st.subheader(
    "Predict Final Exam Marks Using Machine Learning"
)

st.write(
    "Enter student details below to predict final exam performance."
)

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

# ---------------- USER INPUTS ---------------- #

study_hours = st.slider(
    "Study Hours",
    1,
    12,
    6
)

mid_marks = st.slider(
    "Mid Sem Marks",
    0,
    100,
    50
)

attendance = st.slider(
    "Attendance Percentage",
    50,
    100,
    80
)

assignments_completed = st.slider(
    "Assignments Completed",
    0,
    10,
    5
)

sleep_hours = st.slider(
    "Average Sleep Hours",
    3,
    10,
    7
)

# ---------------- PREDICT BUTTON ---------------- #

if st.button("Predict Final Marks"):

    prediction = model.predict([[
        study_hours,
        mid_marks,
        attendance,
        assignments_completed,
        sleep_hours
    ]])

    predicted_marks = prediction[0]

    # ---------------- REALISTIC RULES ---------------- #

    # Study hours most important

    if study_hours <= 2:
        predicted_marks -= 20

    elif study_hours <= 4:
        predicted_marks -= 10

    # Attendance effect

    if attendance < 70:
        predicted_marks -= 5

    # Sleep effect

    if sleep_hours < 5:
        predicted_marks -= 5

    # Assignment effect (very small)

    if assignments_completed <= 2:
        predicted_marks -= 2

    # Limit prediction between 0 and 100

    predicted_marks = max(
        0,
        min(predicted_marks, 100)
    )

    # ---------------- SHOW RESULT ---------------- #

    st.success(
        f"Predicted Final Marks: {predicted_marks:.2f}"
    )

    # ---------------- PERFORMANCE FEEDBACK ---------------- #

    if predicted_marks >= 85:

        st.balloons()

        st.success(
            "Excellent Performance Expected! 🌟"
        )

    elif predicted_marks >= 70:

        st.info(
            "Good Performance Expected 👍"
        )

    else:

        st.warning(
            "Needs Improvement 📚"
        )

# ---------------- MODEL EVALUATION ---------------- #

test_predictions = model.predict(X_test)

score = r2_score(y_test, test_predictions)

mae = mean_absolute_error(
    y_test,
    test_predictions
)

st.info(
    f"Model Accuracy (R² Score): {score * 100:.2f}%"
)

st.info(
    f"Average Prediction Error: {mae:.2f} marks"
)

# ---------------- GRAPH ---------------- #

st.subheader(
    "Study Hours vs Final Marks"
)

fig, ax = plt.subplots(figsize=(8,5))

# Scatter plot

sns.scatterplot(
    x='study_hours',
    y='final_marks',
    data=df,
    ax=ax
)

# Sort values for smooth line

sorted_df = df.sort_values(
    by='study_hours'
)

# Prediction line

ax.plot(
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

# Labels

ax.set_title(
    "Study Hours vs Final Marks"
)

ax.set_xlabel(
    "Study Hours"
)

ax.set_ylabel(
    "Final Marks"
)

ax.grid(True)

# Show graph

st.pyplot(fig)