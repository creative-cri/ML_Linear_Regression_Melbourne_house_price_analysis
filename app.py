
# stream streamlit run app.py


import os
from pathlib import Path
os.chdir(Path(__file__).parent)

import pickle
import pandas as pd
import streamlit as st

def load_model():
    with open('./models/model_linear_reg_melb.pkl', mode= 'rb') as file:
        model = pickle.load(file)
    return model

def get_dataframe():
    df = pd.read_csv('./predicted_melbourne_prices.csv')
    return df


def main():

    # 1. Load the model
    model = load_model()

    # 2. write an App Tite

    st.title('Welcome by the Melbourne houses price prediction application')
    st.image('./image.jpg', caption = 'This is melbourne city', width = 680)

    # 3. Add input field
    st.header('Calculate the house price')

    Rooms = int(st.text_input("Rooms", 5))
    house_types = int(st.text_input('House_types', 0))
    region_name = int(st.text_input('Region_name', 2)) 
    

    # 4. Add a prediction Button

    if st.button("predict"):
        predicted_price =  round (float( model.predict([[Rooms, house_types, region_name]])),1)

        st.success( f' The price for the provided parameters is: {predicted_price}')

    st.image('./villa.jpg', caption = 'This is a villa in the Southern Metropolitan district of Melbourne city', width = 680)
    
    df = get_dataframe()
    st.markdown('---')

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader('Dataframe')
        st.dataframe(df)
        st.markdown('---')


    with right_col:
        st.subheader('Explanation of the house types and region names')
        json = {
        'House_types': {
            'house, cottage, villa, semi, terrace': 0,
            'unit, duplex': 1,
            'townhouse' : 2
            },
        'Region_names': {
            'Northern Metropolitan': 0, 
            'Western Metropolitan': 1, 
            'Southern Metropolitan': 2, 
            'South-Eastern Metropolitan': 3, 
            'Eastern Metropolitan': 4, 
            'Northern Victoria': 5, 
            'Western Victoria': 6, 
             'Eastern Victoria': 7
             }
        }
        st.json(json)

if __name__ =="__main__":
    main()