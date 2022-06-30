import streamlit as st
from predict_page import show_predict_page
from explore_page import show_explore_page
from About_page import show_about_page

page = st.sidebar.selectbox("Select Activicties", ("About", "Predict", "Explore"))

if page == "About":
    show_about_page()
elif page== "Predict":
    show_predict_page()
else:
    show_explore_page()
