import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib
import json
import io

# Set page configuration
st.set_page_config(
    page_title="Network Attack Detection",
    page_icon="🔒",
    layout="wide"
)

# Title and description
st.title("🔒 Network Attack Detection System")
st.markdown("This application uses a K-Nearest Neighbors model to detect network attacks based on traffic parameters.")

# Load the saved metrics
try:
    with open('model_metrics.json', 'r') as f:
        metrics = json.load(f)
    
    # Display metrics in the sidebar with proper formatting
    st.sidebar.header("Model Performance")
    st.sidebar.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    st.sidebar.metric("Precision", f"{metrics['precision']:.2%}")
    st.sidebar.metric("Recall", f"{metrics['recall']:.2%}")
    st.sidebar.metric("F1 Score", f"{metrics['f1']:.2%}")
except Exception as e:
    st.sidebar.error("Could not load model metrics. Please train the model first.")

# Create tabs for different input methods
tab1, tab2 = st.tabs(["Manual Input", "CSV Upload"])

with tab1:
    # Create columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        # Input fields for numerical features
        st.subheader("Network Traffic Parameters")
        duration = st.number_input("Duration", min_value=0)
        src_bytes = st.number_input("Source Bytes", min_value=0)
        dst_bytes = st.number_input("Destination Bytes", min_value=0)
        count = st.number_input("Count", min_value=0)
        srv_count = st.number_input("Service Count", min_value=0)

    with col2:
        # Input fields for categorical features
        protocol_type = st.selectbox("Protocol Type", ["tcp", "udp", "icmp"])
        service = st.selectbox("Service", ["http", "private", "domain_u", "smtp", "ftp_data"])
        flag = st.selectbox("Flag", ["SF", "S0", "REJ", "RSTR", "SH"])
        
        # Additional numerical features
        wrong_fragment = st.number_input("Wrong Fragment", min_value=0)
        urgent = st.number_input("Urgent", min_value=0)

    # Create a button to make predictions
    if st.button("Detect Attack"):
        try:
            # Load the saved model and preprocessors
            model = joblib.load('knn_model.pkl')
            label_encoders = joblib.load('label_encoders.pkl')
            scaler = joblib.load('scaler.pkl')
            
            # Create a dataframe with the input values
            input_data = pd.DataFrame({
                'duration': [duration],
                'protocol_type': [protocol_type],
                'service': [service],
                'flag': [flag],
                'src_bytes': [src_bytes],
                'dst_bytes': [dst_bytes],
                'wrong_fragment': [wrong_fragment],
                'urgent': [urgent],
                'count': [count],
                'srv_count': [srv_count],
                'land': [0],
                'hot': [0],
                'num_failed_logins': [0],
                'logged_in': [0],
                'num_compromised': [0],
                'root_shell': [0],
                'su_attempted': [0],
                'num_root': [0],
                'num_file_creations': [0],
                'num_shells': [0],
                'num_access_files': [0],
                'num_outbound_cmds': [0],
                'is_host_login': [0],
                'is_guest_login': [0],
                'serror_rate': [0],
                'srv_serror_rate': [0],
                'rerror_rate': [0],
                'srv_rerror_rate': [0],
                'same_srv_rate': [0],
                'diff_srv_rate': [0],
                'srv_diff_host_rate': [0],
                'dst_host_count': [0],
                'dst_host_srv_count': [0],
                'dst_host_same_srv_rate': [0],
                'dst_host_diff_srv_rate': [0],
                'dst_host_same_src_port_rate': [0],
                'dst_host_srv_diff_host_rate': [0],
                'dst_host_serror_rate': [0],
                'dst_host_srv_serror_rate': [0],
                'dst_host_rerror_rate': [0],
                'dst_host_srv_rerror_rate': [0],
                'level': [0]
            })
            
            # Encode categorical variables
            categorical_columns = ['protocol_type', 'service', 'flag']
            for col in categorical_columns:
                input_data[col] = label_encoders[col].transform(input_data[col])
                
            # Scale the features
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)
            attack_type = label_encoders['attack'].inverse_transform(prediction)[0]
            
            # Display result with styling
            st.success(f"Detected Attack Type: {attack_type}")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please make sure you have trained and saved the model first by running the training script.")

with tab2:
    st.subheader("Upload Test Data")
    st.info("""
    Please upload a CSV file with the following format:
    - No header row
    - 41 columns in the NSL-KDD format
    - You can use a subset of the test dataset
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            test_data = pd.read_csv(uploaded_file, header=None)
            
            # Check the number of columns
            if len(test_data.columns) != 41:
                st.error(f"Invalid number of columns. Your file has {len(test_data.columns)} columns, but the model expects 41 columns.")
                st.info("Please make sure your CSV file follows the NSL-KDD dataset format.")
                st.stop()
            
            # Set the column names
            columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
                      'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
                      'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
                      'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                      'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                      'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                      'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                      'dst_host_srv_rerror_rate']
            test_data.columns = columns
            
            # Load the model and preprocessors
            model = joblib.load('knn_model.pkl')
            label_encoders = joblib.load('label_encoders.pkl')
            scaler = joblib.load('scaler.pkl')
            
            # Create a copy for predictions
            X_test = test_data.copy()
            
            # Encode categorical variables
            categorical_columns = ['protocol_type', 'service', 'flag']
            for col in categorical_columns:
                try:
                    X_test[col] = label_encoders[col].transform(X_test[col])
                except ValueError as e:
                    st.error(f"Error encoding {col}: Found unknown categories. Make sure your data only contains values present in the training set.")
                    st.info(f"Allowed values for {col}: {list(label_encoders[col].classes_)}")
                    st.stop()
            
            # Scale the features
            X_test_scaled = scaler.transform(X_test)
            
            # Make predictions
            predictions = model.predict(X_test_scaled)
            attack_types = label_encoders['attack'].inverse_transform(predictions)
            
            # Add predictions to the dataframe
            results = test_data.copy()
            results['Predicted Attack'] = attack_types
            
            # Display results
            st.write("### Prediction Results")
            st.write(f"Total records processed: {len(results)}")
            
            # Show prediction distribution
            st.write("### Attack Type Distribution")
            attack_counts = pd.Series(attack_types).value_counts()
            st.bar_chart(attack_counts)
            
            # Display the results table
            st.write("### Detailed Results")
            st.write(results)
            
            # Add download button for results
            csv = results.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="prediction_results.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")
            st.info("""
            Please make sure your CSV file matches the NSL-KDD format:
            - 41 columns
            - No header row
            - Correct data types for each column
            """)

# Add information about the model
st.sidebar.header("About")
st.sidebar.info("""
This application uses a K-Nearest Neighbors (KNN) classifier to detect network attacks.
The model has been trained on the NSL-KDD dataset, which is a benchmark dataset for 
network intrusion detection systems.
""") 