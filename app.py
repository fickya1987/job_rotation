import streamlit as st
import pandas as pd
import pickle

# Title and description
st.title("Job Rotation/Promotion Prediction App")
st.write("Upload a CSV file containing employee data to predict job rotation or promotion.")

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'model.pkl' is in the same directory as this script.")
        return None

model = load_model()

# Section to upload CSV file
st.subheader("Upload Employee Data:")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Load the uploaded data
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.write(data.head())

        # Preprocess the data
        required_columns = [
            "Current Job Role", "Department", "Monthly Income", 
            "Education Level", "Promotion Last 3 Years", 
            "Job Performance Rating", "Preferred Job Role"
        ]

        missing_cols = set(required_columns) - set(data.columns)
        if missing_cols:
            st.error(f"The uploaded file is missing the following columns: {', '.join(missing_cols)}")
        else:
            # Make predictions
            predictions = model.predict(data[required_columns])
            data["Prediction"] = ["Promotion/Rotation Likely" if pred == 1 else "No Promotion/Rotation" for pred in predictions]
            
            st.subheader("Prediction Results:")
            st.write(data[["Employee ID", "Prediction"]])
    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
