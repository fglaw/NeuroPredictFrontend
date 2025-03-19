import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set page title
st.title("Hello World Streamlit App")

# Add a subheader
st.subheader("Welcome to my first Streamlit application!")

# Add some text
st.write("This is a simple example of a Streamlit app.")

# Add a sidebar
st.sidebar.header("Sidebar")
st.sidebar.write("You can add controls here.")

# Display some interactive elements
if st.button("Click Me!"):
    st.balloons()
    st.success("You clicked the button!")

# Add a simple input
user_input = st.text_input("Enter your name")
if user_input:
    st.write(f"Hello, {user_input}!")

# Add a section for matplotlib plots
st.header("Data Visualization")
st.write("Here are 6 plots from our sample data")

# Load the CSV data
try:
    df = pd.read_csv('data/random_test_samples.csv')
    
    # Create a 3x2 grid of plots
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    fig.tight_layout(pad=4.0)
    
    # Flatten the axes array for easier iteration
    axes = axes.flatten()
    
    # For each of the first 6 rows, create a line plot
    for i in range(min(6, len(df))):
        # Get the data for this row
        row_data = df.iloc[i].values
        
        # Create x-axis values (column indices)
        x = np.arange(len(row_data))
        
        # Plot the data
        axes[i].plot(x, row_data, 'b-')
        axes[i].set_title(f'Sample {i+1}')
        axes[i].set_xlabel('Feature Index')
        axes[i].set_ylabel('Value')
        axes[i].grid(True)
    
    # Display the plots in Streamlit
    st.pyplot(fig)
    
except Exception as e:
    st.error(f"Error loading or plotting data: {e}") 