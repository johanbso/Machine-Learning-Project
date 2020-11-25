import streamlit as st
import pandas as pd

def getUserInput(features):
    # Header
    st.sidebar.header("Enter Patient Data")

    # Store a dictionary into a variable
    user_data = {}

    for feat in features[:-1]:
        try:
            # Feature name
            featName = feat
            # Text fields
            feature = float(st.sidebar.text_input('Enter ' + featName))
            user_data_element = {featName:feature}
            user_data.update(user_data_element)
        except:
            print('Exception')

    # Transform the data into a data frame
    features = pd.DataFrame(user_data, index=[0])
    return features