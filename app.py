
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Page configuration
st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom blue theme styling
st.markdown("""
<style>
.stApp {
    background-color: #F4F7FC;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

[data-testid="stSidebar"] {
    background-color: #172554;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

.hero {
    background: linear-gradient(120deg, #1D4ED8, #2563EB, #60A5FA);
    padding: 28px;
    border-radius: 18px;
    color: white;
    margin-bottom: 24px;
}

.hero h1 {
    color: white;
    margin-bottom: 8px;
}

.hero p {
    color: #E0EAFF;
    margin-bottom: 0;
    font-size: 16px;
}

div.stButton > button[kind="primary"] {
    background: #2563EB;
    border-radius: 10px;
    border: none;
    color: white;
    font-weight: 600;
    min-height: 48px;
}

div.stButton > button[kind="primary"]:hover {
    background: #1D4ED8;
    color: white;
    border: none;
}

[data-testid="stMetric"] {
    background: white;
    padding: 20px;
    border: 1px solid #DBEAFE;
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🎓 Student AI")
    st.caption("MACHINE LEARNING PROJECT")
    st.divider()

    st.subheader("📌 About")
    st.write(
        "Predict student final scores using study hours, "
        "attendance, and previous marks."
    )

    st.divider()
    st.subheader("🛠️ Technology")
    st.write("🐍 Python")
    st.write("🐼 Pandas")
    st.write("🤖 Scikit-learn")
    st.write("🎨 Streamlit")

    st.divider()
    st.caption("Student Performance Prediction v1.0")

# Dashboard header
st.markdown("""
<div class="hero">
    <h1>Student Performance Dashboard</h1>
    <p>Machine Learning Based Final Score Prediction</p>
</div>
""", unsafe_allow_html=True)

# Load dataset
data = pd.read_csv("student_data.csv")

# Separate features and target
X = data[["study_hours", "attendance", "previous_marks"]]
y = data["final_score"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Student input section
st.subheader("📝 Student Details")
st.caption("Enter the student's details below to generate a prediction.")

with st.container(border=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        study_hours = st.number_input(
            "📚 Study Hours Per Day",
            min_value=0.0,
            max_value=24.0,
            value=6.0,
            step=0.5
        )

    with col2:
        attendance = st.number_input(
            "📅 Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=85.0,
            step=1.0
        )

    with col3:
        previous_marks = st.number_input(
            "📝 Previous Marks",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0
        )

    predict_button = st.button(
        "✨ Predict Final Score",
        type="primary",
        use_container_width=True
    )

# Prediction result
if predict_button:
    new_student = pd.DataFrame({
        "study_hours": [study_hours],
        "attendance": [attendance],
        "previous_marks": [previous_marks]
    })

    prediction = float(model.predict(new_student)[0])

    # Keep displayed score within 0–100
    prediction = max(0.0, min(100.0, prediction))

    st.divider()
    st.subheader("🎯 Prediction Results")

    result_col1, result_col2 = st.columns([1, 2])

    with result_col1:
        st.metric(
            "Predicted Final Score",
            f"{prediction:.2f} / 100"
        )

    with result_col2:
        st.write("**Predicted Score Progress**")
        st.progress(prediction / 100)
        st.caption(f"{prediction:.1f}% of the 100-mark scale")

    if prediction >= 80:
        st.success("🌟 Excellent predicted score!")
        st.balloons()
    elif prediction >= 60:
        st.info("👍 Good predicted score!")
    elif prediction >= 40:
        st.warning("📘 Average predicted score. Improvement may help.")
    else:
        st.error("📚 Low predicted score. Additional support may help.")

# Footer
st.divider()
st.caption(
    "Note: This model uses a small synthetic practice dataset. "
    "Predictions are for learning purposes and are not guaranteed "
    "to reflect real student performance."
)
