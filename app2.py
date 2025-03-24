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
st.write("Individual plots from our sample data")

# Load the CSV data
try:
    df = pd.read_csv('data/random_test_samples.csv')
    
    # For each of the first 6 rows, create a separate plot
    for i in range(min(6, len(df))):
        # Create a section for this plot
        st.subheader(f"Sample {i+1}")
        
        # Get the data for this row
        row_data = df.iloc[i].values
        
        # Create x-axis values (column indices)
        x = np.arange(len(row_data))
        
        # Create a figure for this plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the data
        ax.plot(x, row_data, 'b-')
        ax.set_title(f'Sample {i+1} Data')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Value')
        ax.grid(True)
        
        # Display the plot
        st.pyplot(fig)
        
        # Calculate statistics for status
        min_val = np.min(row_data)
        max_val = np.max(row_data)
        mean_val = np.mean(row_data)
        median_val = np.median(row_data)
        std_val = np.std(row_data)
        
        # Display status information beneath the plot
        st.info(f"**Status Information:**  \n"
                f"Min: {min_val:.2f}  \n"
                f"Max: {max_val:.2f}  \n"
                f"Mean: {mean_val:.2f}  \n"
                f"Median: {median_val:.2f}  \n"
                f"Standard Deviation: {std_val:.2f}")
        
        # Add a separator between plots
        st.markdown("---")
    
except Exception as e:
    st.error(f"Error loading or plotting data: {e}") 