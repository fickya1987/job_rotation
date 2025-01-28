import streamlit as st
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Title and description
st.title("Job Rotation/Promotion Prediction App")
st.write("This app predicts the likelihood of job rotation or promotion based on employee data.")

# Section to upload CSV file
st.subheader("Upload Employee Data:")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

# Function to create and train the model
@st.cache_resource
def train_model(data):
    # Encode categorical columns
    label_encoders = {}
    for column in ["Current Job Role", "Department", "Education Level", "Preferred Job Role"]:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    
    # Define features (X) and target (y)
    X = data.drop(["Employee ID", "Skills", "Certifications", "Target"], axis=1, errors="ignore")
    y = data["Target"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RandomForest model
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    return model, label_encoders

# Function to generate dummy data
def generate_dummy_data():
    dummy_data = {
        "Employee ID": [f"E{1000 + i}" for i in range(100)],
        "Current Job Role": [random.choice(["Analyst", "Manager", "Developer", "Consultant", "Specialist"]) for _ in range(100)],
        "Department": [random.choice(["HR", "Finance", "IT", "Operations", "Sales", "Marketing"]) for _ in range(100)],
        "Monthly Income": [random.randint(3000, 15000) for _ in range(100)],
        "Skills": [", ".join(random.sample(["Python", "Excel", "SQL", "Leadership", "Project Management", "Java", "Cloud"], random.randint(2, 5))) for _ in range(100)],
        "Education Level": [random.choice(["High School", "Bachelor's", "Master's", "PhD"]) for _ in range(100)],
        "Certifications": [", ".join(random.sample(["PMP", "AWS Certified", "Google Cloud Certified", "Scrum Master", "ITIL"], random.randint(1, 3))) for _ in range(100)],
        "Promotion Last 3 Years": [random.randint(0, 3) for _ in range(100)],
        "Job Performance Rating": [random.randint(1, 5) for _ in range(100)],
        "Preferred Job Role": [random.choice(["Team Lead", "Senior Manager", "Architect", "Consultant", "Director"]) for _ in range(100)],
        "Target": [random.randint(0, 1) for _ in range(100)]  # Dummy target column
    }
    return pd.DataFrame(dummy_data)

# Load data
if uploaded_file is not None:
    try:
        # Load the uploaded data
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.write(data.head())
    except Exception as e:
        st.error(f"Error loading file: {e}")
        data = None
else:
    st.write("No file uploaded. Using dummy data for demonstration.")
    data = generate_dummy_data()
    st.write(data.head())

# Train the model
if data is not None:
    required_columns = ["Current Job Role", "Department", "Monthly Income", 
                        "Education Level", "Promotion Last 3 Years", 
                        "Job Performance Rating", "Preferred Job Role", "Target"]

    missing_cols = set(required_columns) - set(data.columns)
    if missing_cols:
        st.error(f"The dataset is missing the following columns: {', '.join(missing_cols)}")
    else:
        model, encoders = train_model(data)
        st.success("Model trained successfully!")

        # Make predictions
        st.subheader("Prediction Results:")
        features = data.drop(["Employee ID", "Skills", "Certifications", "Target"], axis=1, errors="ignore")
        predictions = model.predict(features)
        data["Prediction"] = ["Promotion/Rotation Likely" if pred == 1 else "No Promotion/Rotation" for pred in predictions]
        st.write(data[["Employee ID", "Prediction"]])

