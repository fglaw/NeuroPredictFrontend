import streamlit as st
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import os

# Set page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="CSV Processor App",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem !important;
        color: #4682B4;
    }
    .sub-header {
        font-size: 1.5rem !important;
        color: #708090;
    }
    .api-section {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .plot-container {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border: 1px solid #d1e3fa;
    }
    .status-placeholder {
        background-color: #e6e6e6;
        padding: 8px;
        border-radius: 4px;
        color: #666;
        margin-top: 10px;
    }
    .prediction {
        padding: 8px;
        border-radius: 4px;
        color: #333;
        font-weight: 500;
        margin-top: 10px;
    }
    .prediction-healthy {
        background-color: #90EE90; /* Light green */
    }
    .prediction-tumor {
        background-color: #FFFF99; /* Light yellow */
    }
    .prediction-seizure {
        background-color: #FFCCCC; /* Light red */
    }
    .button-container {
        display: flex;
        gap: 10px;
        margin: 20px 0;
    }
    .stButton>button {
        flex: 1;
        background-color: #4682B4;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #357ABD;
    }
    .summary-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border: 1px solid #dee2e6;
    }
    .model-info {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        color: #1976d2;
    }
</style>
""", unsafe_allow_html=True)

# Main app
st.markdown('<h1 class="main-header">CSV Processor 📊</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analyzing EEG data from random_test_samples.csv</p>', unsafe_allow_html=True)

# Add a divider
st.markdown("---")

# API Connection section
st.markdown('<div class="api-section">', unsafe_allow_html=True)
st.subheader("Process CSV File")

api_url = st.text_input("FastAPI URL", "https://neuropredict-api-773733892552.us-central1.run.app/")

# Initialize session state for API response and selected model
if 'api_response' not in st.session_state:
    st.session_state.api_response = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None

# Load the CSV file directly
try:
    df = pd.read_csv("data/random_test_samples.csv")
    st.write("Preview of the CSV data:")
    st.dataframe(df.head(5))
    
    # Add model selection buttons
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Model 1", key="model1"):
            st.session_state.selected_model = 1
            st.session_state.api_response = None
    
    with col2:
        if st.button("Model 2", key="model2"):
            st.session_state.selected_model = 2
            st.session_state.api_response = None
    
    with col3:
        if st.button("Model 3", key="model3"):
            st.session_state.selected_model = 3
            st.session_state.api_response = None
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show selected model
    if st.session_state.selected_model:
        st.info(f"Selected Model: {st.session_state.selected_model}")
    
    # Process the file with the API when a model is selected
    if st.session_state.selected_model and not st.session_state.api_response:
        try:
            # Create a temporary file for API upload
            temp_file = "temp_random_test_samples.csv"
            df.to_csv(temp_file, index=False)
            
            with open(temp_file, 'rb') as f:
                files = {"csv_file": f}
                data = {"model": st.session_state.selected_model}
                with st.spinner("Processing CSV with the API..."):
                    response = requests.post(api_url, files=files, data=data)
            
            # Clean up temporary file
            os.remove(temp_file)
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"CSV processed successfully!")
                st.session_state.api_response = result
                
                # Display API response information
                if result.get("status") == "success":
                    # Show model information
                    st.markdown(f'<div class="model-info">Model Used: {result.get("model_used", "Unknown")}</div>', unsafe_allow_html=True)
                    
                    # Show data summary
                    if "data_summary" in result:
                        summary = result["data_summary"]
                        st.markdown("### Data Summary")
                        st.markdown(f'<div class="summary-box">', unsafe_allow_html=True)
                        st.write(f"Total Rows: {summary.get('total_rows', 'N/A')}")
                        st.write(f"Numeric Columns: {', '.join(summary.get('numeric_columns', []))}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show predictions
                    st.markdown("### Predictions")
                    if "predictions" in result and "prediction_labels" in result:
                        for i, (pred, label) in enumerate(zip(result["predictions"], result["prediction_labels"])):
                            st.write(f"Row {i+1}: {pred} ({label})")
                
                st.json(result)
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Error connecting to API: {e}")
    
    # Plot rows as individual plots
    st.subheader("Row Signal Visualizations")
    
    # Filter to only include numeric columns for plotting
    numeric_df = df.select_dtypes(include=np.number)
    
    if not numeric_df.empty:
        # Get the first 6 rows for visualization
        rows_to_plot = min(6, len(numeric_df))
        
        # Create time points starting from 1
        time_points = np.arange(1, len(numeric_df.columns) + 1)
        
        # Create individual plots for each row with status directly below
        for i in range(rows_to_plot):
            # Create container for this row's plot and status
            st.markdown(f'<div class="plot-container">', unsafe_allow_html=True)
            
            # Row identifier
            if 'name' in df.columns or 'id' in df.columns or 'title' in df.columns:
                id_col = next((col for col in df.columns if col.lower() in ['name', 'id', 'title']), None)
                row_name = f"Row {i+1}: {df.iloc[i][id_col]}"
            else:
                row_name = f"Row {i+1}"
            
            st.write(f"**{row_name}**")
            
            # Create plot for this row
            fig, ax = plt.subplots(figsize=(10, 4))
            
            # Get row values
            row_values = numeric_df.iloc[i].values
            
            # Plot as a line (signal-like visualization) using time points
            ax.plot(time_points, row_values, linewidth=2)
            
            # Set x-ticks for all time points but only show labels at positions 1, 10, 20, etc.
            ax.set_xticks(time_points)
            
            # Create custom tick labels showing only at intervals of 10
            tick_labels = []
            for t in time_points:
                if t == 1 or t % 10 == 0:
                    tick_labels.append(str(int(t)))
                else:
                    tick_labels.append('')
            
            ax.set_xticklabels(tick_labels)
            
            # Add grid for better readability
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Set title and labels
            ax.set_title(f'Signal for {row_name}', fontsize=12)
            ax.set_ylabel('Amplitude', fontsize=10)
            ax.set_xlabel('Time', fontsize=10)
            
            # Adjust layout
            plt.tight_layout()
            
            # Display the plot in this container
            st.pyplot(fig)
            plt.close(fig)  # Close to avoid the warning
            
            # Add the prediction or placeholder based on API response
            if st.session_state.api_response and 'predictions' in st.session_state.api_response and i < len(st.session_state.api_response['predictions']):
                prediction = st.session_state.api_response['predictions'][i]
                label = st.session_state.api_response['prediction_labels'][i]
                
                # Determine the appropriate CSS class based on prediction content
                prediction_class = "prediction "
                if "healthy" in prediction.lower():
                    prediction_class += "prediction-healthy"
                elif "tumor-induced seizure" in prediction.lower():
                    prediction_class += "prediction-seizure"
                elif "tumor" in prediction.lower():
                    prediction_class += "prediction-tumor"
                
                st.markdown(f'<div class="{prediction_class}">{prediction} ({label})</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-placeholder">Waiting for prediction...</div>', unsafe_allow_html=True)
            
            # Close the plot container
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.write("No numeric columns found for plotting row signals.")
except Exception as e:
    st.error(f"Error loading CSV file: {e}")

st.markdown('</div>', unsafe_allow_html=True)




