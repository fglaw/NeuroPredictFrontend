# app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Page setup
st.set_page_config(
    page_title="NeuroPredict: ML EEG Classification",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main-header {
        text-align: center;
        font-size: 32px; /* Adjust size if needed */
        font-weight: bold;
        color: white !important; /* Change color if necessary */
        margin-top: 20px; /* Adjust spacing */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="main-header">NeuroPredict: Machine Learning EEG Classification</h1>', unsafe_allow_html=True)
# # Title and headers
# st.markdown('<h1 class="main-header">NeuroPredict: Machine Learning EEG Classification </h1>', unsafe_allow_html=True)
# #st.markdown('<p class="sub-header">Select an ML model to analyze EEG data</p>', unsafe_allow_html=True)



# Custom CSS to move the title to the top-left corner
# st.markdown(
#     """
#     <style>
#     .title {
#         position: absolute;
#         top: 10px;
#         left: 20px;
#         font-size: 30px;
#         font-weight: bold;
#         color: black;
#     }
#     </style>
#     <div class="title">NeuroPredict: ML EEG Classification</div>
#     """,
#     unsafe_allow_html=True
# )



# Custom CSS stays unchanged
st.markdown("""
<style>
    .main-header { font-size:3rem!important; color:white !important; font-weight:bold; }
    .sub-header { font-size:1.5rem!important; color:white !important; font-weight:bold; }
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
        background:
            /* Background image */
            url("https://res.cloudinary.com/dim47nr4g/image/upload/v1748879468/NeuroPredict_Streamlit_background_image_rzptbj.png")
            no-repeat center center fixed;
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown("---")


# # # Button interactions for selecting ML model
model_selected = None
col1, col2, col3, col4, col5 = st.columns(5)
with col2:
    if st.button("🔵 XGBoost"):
        model_selected = "XGBoost"
with col3:
    if st.button("🟡 Rocket"):
        model_selected = "Rocket"
with col4:
    if st.button("🔴 LSTM-Rocket Hybrid"):
        model_selected = "Hybrid"


# For this example, we'll plot Sample 1 (the 0/3/5 row of the DataFrame)
prediction_data = pd.read_csv('./data/prediction_data.csv')
sample1 = prediction_data.iloc[0]
sample2 = prediction_data.iloc[3]
sample3 = prediction_data.iloc[5]
scale_factor = 178
time_points = prediction_data.columns
time_in_seconds = [i / scale_factor*60 for i in range(len(time_points))]

# Create a Plotly figure
fig1 = go.Figure()
fig2 = go.Figure()
fig3 = go.Figure()


##########################################################################
#####################    Seaborn  #######################################
#########################################################################

# Extract three samples (for demonstration)
sample1 = prediction_data.iloc[0]
sample2 = prediction_data.iloc[3]
sample3 = prediction_data.iloc[5]

# Define the scale factor and convert time points to "ms" (or seconds as needed)
scale_factor = 178
time_points = prediction_data.columns
time_in_seconds = [i / scale_factor * 60 for i in range(len(time_points))]

# Set the Seaborn theme to "white" for a clean look.
sns.set_theme(style="white")

# Helper function to create a custom EEG plot
def create_eeg_plot(x, y, title,  width=5, height=5,y_min=None, y_max=None):  #(10,6)
    fig, ax = plt.subplots(figsize=(width, height))

    # Plot the data: black line with turquoise circular markers
    ax.plot(x, y,
            color='black', linewidth=2,
            marker='o', markersize=4, markerfacecolor='turquoise')

    # Customize title and axis labels
    ax.set_title(title)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    # Remove gridlines
    ax.grid(False)
     # Set custom y-axis limits if provided
    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)

    # Remove all spines (plot borders)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Optionally add zero lines (dashed)
    ax.axhline(0, color='black', linewidth=1, linestyle='--')
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    fig.tight_layout()
    return fig, ax

# Create three separate figures with different titles
fig1, ax1 = create_eeg_plot(time_in_seconds, sample1, "EEG Signal for tumor-induced seizure", y_min=-1500, y_max=1000)
fig2, ax2 = create_eeg_plot(time_in_seconds, sample2, "EEG Signal for tumor baseline", y_min=-500, y_max=500)
fig3, ax3 = create_eeg_plot(time_in_seconds, sample3, "EEG Signal for healthy baseline", y_min=-500, y_max=500)

#################################################################
################# Create ✅ / ❌ signs ##########################
##################################################################

model_performance = {
    "XGBoost": [True, True, False],
    "Rocket": [True, False, True],
    "Hybrid": [True, True, False]
}

# Helper function to add an icon based on correctness.
def add_icon(text, is_correct):
    if is_correct:
        icon = '<span style="color: #00FF00 !important; font-weight: bold;">✅</span>'
    else:
        icon = '<span style="color:red;">❌</span>'
    return f"{icon} {text}"


# ########################################################################
#######################     Plotly                 #######################
##########################################################################
# Plot the line graph for Sample 1
# fig1.add_trace(go.Scatter(
#     x=time_in_seconds,  # The time points (columns)
#     y=sample1,  # The EEG data for Sample 1
#     mode='lines+markers',  # Line plot with markers
#     #name='Sample 1',  # Name for the legend
#     line=dict(color='black', width=2),  # Line color and width
#     marker=dict(size=4, color='turquoise', symbol='circle'),  # Marker style
# ))

# fig2.add_trace(go.Scatter(
#     x=time_in_seconds,  # The time points (columns)
#     y=sample2,  # The EEG data for Sample 1
#     mode='lines+markers',  # Line plot with markers
#     #name='Sample 1',  # Name for the legend
#     line=dict(color='black', width=2),  # Line color and width
#     marker=dict(size=4, color='turquoise', symbol='circle'),  # Marker style
# ))

# fig3.add_trace(go.Scatter(
#     x=time_in_seconds,  # The time points (columns)
#     y=sample3,  # The EEG data for Sample 1
#     mode='lines+markers',  # Line plot with markers
#     #name='Sample 1',  # Name for the legend
#     line=dict(color='black', width=2),  # Line color and width
#     marker=dict(size=4, color='turquoise', symbol='circle'),  # Marker style
# ))

# # Customize layout
# def customize_layout(fig, title, width=400, height=300):
#     fig.update_layout(
#     title=title,
#     xaxis_title="Time (ms)",
#     yaxis_title="Amplitude (µV)",
#     template="plotly_white",  # Optional: Choose a dark theme for the plot
#     hovermode="closest",  # Show the nearest data point on hover
#     showlegend=False, # Show legend
#     xaxis=dict(
#         showgrid=False,  zeroline=True),
#     yaxis=dict(showgrid=False,  zeroline=True),
#     width=width,
#     height=height
#     )

# customize_layout(fig1,"EEG Signal for tumor-induced seizure")
# customize_layout(fig2,"EEG Signal for tumor baseline ")
# customize_layout(fig3,"EEG Signal for healthy baseline ")

###########################################################################
###########################################################################

# # Streamlit layout: Display 3 plots horizontally in one row
# col1, col2, col3 = st.columns(3)

# # Display each figure in its respective column
# with col1:
#     st.plotly_chart(fig1)

# with col2:
#     st.plotly_chart(fig2)

# with col3:
#     st.plotly_chart(fig3)

st.markdown("---")

# Container for plotting and API response
response_container = st.container()

if model_selected:
    try:
        with st.spinner(f"Processing data using {model_selected.capitalize()} Model..."):
            # API request sending model selection
            response = requests.get("https://neuropred-631627542868.europe-west1.run.app/predict", params={"model": model_selected})

        if response.status_code == 200:
            result = response.json()
            prediction_texts = result["predictions"]

        if prediction_texts:
            # Get the correctness status list for the chosen model.
            correctness = model_performance.get(model_selected, [False, False, False])

            with st.container():  # Response container for the layout
                # Display 3 plots horizontally in one row.
                col1, col2, col3 = st.columns(3)

            with col1:
                st.pyplot(fig1)
                st.markdown(f'<div>{add_icon(prediction_texts[0], correctness[0])}</div>',
                unsafe_allow_html=True
                )

            with col2:
                st.pyplot(fig2)
                st.markdown(f'<div>{add_icon(prediction_texts[3], correctness[1])}</div>',
                unsafe_allow_html=True
                )

            with col3:
                st.pyplot(fig3)
                st.markdown(f'<div>{add_icon(prediction_texts[5], correctness[2])}</div>',
                unsafe_allow_html=True
                )

        # # Ensure the response contains predictions
        # if prediction_texts:
        #     with st.container():  # Response container for the layout
        #     # Display 3 plots horizontally in one row
        #         col1, col2, col3 = st.columns(3)

        #     # Display each figure in its respective column
        #     with col1:
        #         st.pyplot(fig1)    #  fig1 Seaborn figure
        #         #st.plotly_chart(fig1)  #  fig1 Plotly figure
        #         st.markdown(f'<div class="{prediction_texts[0]}">{prediction_texts[0]}</div>', unsafe_allow_html=True)

        #     with col2:
        #         st.pyplot(fig2)
        #         #st.plotly_chart(fig2)  # Assuming fig2 is your Plotly figure
        #         st.markdown(f'<div class="{prediction_texts[3]}">{prediction_texts[3]}</div>', unsafe_allow_html=True)

        #     with col3:
        #         st.pyplot(fig3)
        #        # st.plotly_chart(fig3)  # Assuming fig3 is your Plotly figure
        #         st.markdown(f'<div class="{prediction_texts[5]}">{prediction_texts[5]}</div>', unsafe_allow_html=True)

        else:
            st.error(f"API Error: {response.status_code}: {response.text}")

    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")

else:
    st.markdown(
    """
    <style>
    .small-centered-text {
        text-align: center;   /* Center the text */
        font-size: 24px;      /* Make the font smaller (adjust as needed) */
        color: black;         /* Customize color */
    }
    </style>
    """,
        unsafe_allow_html=True
        )

    st.markdown('<h2 style="text-align: center; color: white;">Please select an ML model above 👆🏼 to get started.</h2>', unsafe_allow_html=True)
    #st.markdown('<h2 style="; color:#000000; font-weight:bold;">Please select an ML model above  to get started.</h2>', unsafe_allow_html=True)
