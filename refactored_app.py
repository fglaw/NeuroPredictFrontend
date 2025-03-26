# app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(
    page_title="NeuroPredict: ML EEG Classification",
    page_icon="📊",
    layout="centered"
)

# Custom CSS stays unchanged
st.markdown("""
<style>
    .main-header { font-size:3rem!important; color:#000000; font-weight:bold; }
    .sub-header { font-size:1.5rem!important; color:#FFFFFF; font-weight:bold; }
    .plot-container { background:#f0f8ff; padding:15px; border-radius:8px; margin-bottom:15px; border:1px solid #d1e3fa; }
    .prediction { padding:8px; border-radius:4px; color:#333; font-weight:500; margin-top:10px; }
    .prediction-healthy { background:#90EE90; }
    .prediction-tumor { background:#FFFF99; }
    .prediction-seizure { background:#FFCCCC; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("https://res.cloudinary.com/dim47nr4g/image/upload/a_-90/a_hflip/v1742997775/background_image_for_NeuroPredict_vgcolz.png");
        background-size: cover;
        background-repeat: no-repeat;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Title and headers
st.markdown('<h1 class="main-header">NeuroPredict: Machine Learning EEG Classification 📊</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Select an ML model to analyze EEG data</p>', unsafe_allow_html=True)


st.markdown("---")

# Button interactions for selecting ML model
model_selected = None
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🔵 XGBoost"):
        model_selected = "XGBoost"
with col2:
    if st.button("🟡 Rocket"):
        model_selected = "Rocket"
with col3:
    if st.button("🔴 LTSM-Rocket Hybrid"):
        model_selected = "Hybrid"

# Container for plotting and API response
response_container = st.container()

if model_selected:
    try:
        with st.spinner(f"Processing data using {model_selected.capitalize()} Model..."):
            # API request sending model selection
            response = requests.get("https://neuropred-631627542868.europe-west1.run.app/predict", params={"model": model_selected})

            if response.status_code == 200:
                result = response.json()
                predictions = result["predictions"]
                #eeg_data = result["X_pred"]

                # ensure eeg_data is numeric DataFrame
                #data_df = pd.DataFrame(eeg_data).select_dtypes(include=np.number)

                with response_container:
                    st.success(f"Results for {model_selected.capitalize()} Model")

                    # rows_to_plot = min(6, len(data_df))
                    # time_points = np.arange(1, len(data_df.columns) + 1)

                    # for idx in range(rows_to_plot):
                    #     st.markdown('<div class="plot-container">', unsafe_allow_html=True)

                    #     # Plotting each EEG Data row
                    #     fig, ax = plt.subplots(figsize=(10, 4))
                    #     ax.plot(time_points, data_df.iloc[idx], linewidth=2)

                    #     ax.set_xticks(time_points[::10])  # Show every 10 ticks for clarity
                    #     ax.grid(True, linestyle="--", alpha=0.7)
                    #     ax.set_xlabel("Time")
                    #     ax.set_ylabel("Amplitude")
                    #     ax.set_title(f"EEG Signal Row {idx + 1}")

                    #     st.pyplot(fig)
                    #     plt.close(fig)

                    # displaying the prediction underneath the plot
                    prediction = predictions[0]
                    prediction_class = "prediction "
                    if "healthy" in prediction.lower():
                        prediction_class += "prediction-healthy"
                    elif "tumor-induced seizure" in prediction.lower():
                        prediction_class += "prediction-seizure"
                    elif "tumor" in prediction.lower():
                        prediction_class += "prediction-tumor"

                    st.markdown(f'<div class="{prediction_class}">{prediction}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            else:
                st.error(f"API Error: {response.status_code}: {response.text}")

    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")

else:
    st.markdown('<h2 style="; color:#000000; font-weight:bold;">Please select an ML model above 👆🏼 to get started.</h2>', unsafe_allow_html=True)
