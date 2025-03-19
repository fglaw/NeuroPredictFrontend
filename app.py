import streamlit as st

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