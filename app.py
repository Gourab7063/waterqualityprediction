#importing necessary library
import numpy as np
import pandas as pd
import joblib
import pickle
import streamlit as st
#load the model and structure
model = joblib.load("pollution_model.pkl")
model_cols=joblib.load("model_columns.pkl")
#Lets create a user interface 
st.title("Water Pollution Predictor")
st.write("Predict the water pollutants based on Year & Station ID")
#User inputs
year_input=st.number_input("Enter year",min_value=2000,max_value=2100,value=2022)
station_id=st.text_input("Enter station ID",value='1')
#encode and predict
if st.button("Predict"):
    if not station_id:
        st.warning("Please enter a station ID")
    else:
        # prepare the input
        input_df = pd.DataFrame({'year': [year_input], 'station_id': [station_id]})
        input_encoded = pd.get_dummies(input_df, columns=[ 'station_id'])
        #align with cols
        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_cols]
        #predict
        predicted_pollutants=model.predict(input_encoded)[0]
        pollutants=['O2','NO3','NO2','SO2','PO4','CL']
        st.subheader(f'Predicted pollutant level for the station {station_id} in the year {year_input}:')
        for p, val in zip(pollutants, predicted_pollutants):
            st.write(f'{p}: {val:.2f}')
    #     streamlit run app.py

