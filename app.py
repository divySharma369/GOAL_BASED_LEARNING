import streamlit as st
from inference import predict

st.title("Goal-Driven Student AI")

st.write("Enter student features:")

gender = st.selectbox("Gender", [0, 1])
race = st.selectbox("Race", [0, 1, 2, 3, 4])
parent_edu = st.selectbox("Parent Education", [0, 1, 2, 3, 4, 5])
lunch = st.selectbox("Lunch", [0, 1])
test_prep = st.selectbox("Test Prep", [0, 1])

if st.button("Predict"):
    input_data = [gender, race, parent_edu, lunch, test_prep]
    result = predict(input_data)
    st.write("Result:", result)