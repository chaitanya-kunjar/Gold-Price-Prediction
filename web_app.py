# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 21:12:01 2026

@author: Chaitanya Kunjar
"""
import numpy as np
import pickle
import streamlit as st

# Loading the saved model
loaded_model = pickle.load(open('gold_price_prediction.sav', 'rb'))

# Creating a function for prediction
def gold_price_prediction(input_data):
    # Changing the input data to a NumPy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array (For one instance)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    # Make prediction
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    
    return prediction

def main():
    # Title
    st.title('🪙 GOLD PRICE PREDICTION 🪙')
    
    SPX = st.text_input('Stocks Price')
    USO = st.text_input('US Oil Prices')
    SLV = st.text_input('Silver Price')
    EUR_USD = st.text_input('Euro/US Dollar Ratio')
    
    price = 0.0
    
    if st.button('CALCULATE'):
        price = gold_price_prediction((SPX, USO, SLV, EUR_USD))
    
    st.success(price)

if __name__ == '__main__':
    main()
