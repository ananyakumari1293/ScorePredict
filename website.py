import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------------- PAGE SETTINGS ---------------- #

st.set_page_config(
    page_title="ScorePredict",
    page_icon="📚",
    layout="centered"
)

# ---------------- TITLE ---------------- #

st.title("📚 ScorePredict")
st.subheader("Predict Final Exam Marks Using Machine Learning")

st.write(
    "Enter student details below to predict final exam performance."
)

# ---------------- LOAD DATASET ---------------- #

df = pd.read_csv("student_data.csv")

# Features (inputs)
X = df[[
    'study_hours',
    'mid_marks',
    'attendance',
    'assignments_completed',
    'sleep_hours'
]]

# Target (output)
y = df['final_marks']

# ---------------- TRAIN MODEL ---------------- #

model = LinearRegression()
model.fit(X, y)

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

    st.success(
        f"Predicted Final Marks: {predicted_marks:.2f}"
    )

    # Performance feedback

    if predicted_marks >= 85:
        st.balloons()
        st.success("Excellent Performance Expected! 🌟")

    elif predicted_marks >= 70:
        st.info("Good Performance Expected 👍")

    else:
        st.warning("Needs Improvement 📚")

# ---------------- MODEL ACCURACY ---------------- #

predictions = model.predict(X)

score = r2_score(y, predictions)

st.info(f"Model Accuracy: {score * 100:.2f}%")

# ---------------- GRAPH ---------------- #

st.subheader("Study Hours vs Final Marks")

fig, ax = plt.subplots(figsize=(8,5))

# Scatter plot
sns.scatterplot(
    x='study_hours',
    y='final_marks',
    data=df,
    ax=ax
)

# Sort values for smooth line
sorted_df = df.sort_values(by='study_hours')

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

# Labels and title
ax.set_title("Study Hours vs Final Marks")
ax.set_xlabel("Study Hours")
ax.set_ylabel("Final Marks")

# Show graph
st.pyplot(fig)