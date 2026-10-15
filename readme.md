# 🎓 Student Performance Prediction using Machine Learning

## 📌 Project Overview

Student Performance Prediction is a Machine Learning project that predicts a student's final score based on study hours, attendance percentage, and previous marks.

The project uses **Linear Regression**, a supervised machine learning algorithm, to estimate the student's final score. A simple interactive web application is developed using Streamlit.

## ✨ Features

- Predict student final scores.
- Enter study hours, attendance, and previous marks.
- View the predicted score through a simple user interface.
- Use a trained Linear Regression model.
- Interactive web application built with Streamlit.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Git and GitHub

## 📂 Project Structure

```text
student-performance-prediction/
├── app.py
├── main.py
├── student_data.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## 📊 Dataset

The project uses a small synthetic dataset created for practice and learning purposes.

Dataset columns:

- `study_hours` — Hours spent studying.
- `attendance` — Attendance percentage.
- `previous_marks` — Previous examination marks.
- `final_score` — Final score used as the prediction target.

**Note:** This is a practice project using synthetic data. Its predictions should not be treated as validated real-world student performance assessments.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/vimalprakash2345-pixel/student-performance-prediction.git
```

### 2. Navigate to the project folder

```bash
cd student-performance-prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install the required packages

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🤖 Machine Learning Workflow

1. Load the dataset using Pandas.
2. Select input features and the target variable.
3. Split the data into training and testing sets.
4. Train a Linear Regression model.
5. Accept user inputs through the Streamlit interface.
6. Predict the student's estimated final score.

## 🎯 Learning Outcomes

- Understanding supervised machine learning.
- Working with Pandas and Scikit-learn.
- Training a Linear Regression model.
- Understanding training and testing data.
- Building an interactive ML application using Streamlit.

## 👨‍💻 Author

**Vimalathithan J**

GitHub: [vimalprakash2345-pixel](https://github.com/vimalprakash2345-pixel)
