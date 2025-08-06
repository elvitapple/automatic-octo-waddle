pip install joblib

# app.py
import streamlit as st
import pandas as pd
import joblib


# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Paris Housing Price Prediction")

# Feature columns (same as dataset features)
features = ['squareMeters', 'numberOfRooms', 'hasYard', 'hasPool', 'floors',
       'cityCode', 'cityPartRange', 'numPrevOwners', 'made', 'isNewBuilt',
       'hasStormProtector', 'basement', 'attic', 'garage', 'hasStorageRoom',
       'hasGuestRoom', 'PropertyAge']
names = ['Square Meters', 'Number Of Rooms', 'Is there a Yard (0-no, 1-yes)', 'Is there a Pool (0-no, 1-yes)', 'Number of floors',
       'City Code', 'City Part Range (0 (less exclusive)-10 (more exclusive)), ', 'Number of Prevous Owners', 'Built in (year)', 'Is Newly Built',
       'Has a Storm Protector', 'Area of basement', 'Area of attic', 'Area of garage', 'Has Storage Room (0-no, 1-yes)','Number of Guest Rooms', 'Property Age (in Years)']


user_data = []
for feature in names:
    value = st.number_input(f"Enter value for {feature}", step=1.0)
    user_data.append(value)

input_df = pd.DataFrame([user_data], columns=features)
    
if st.button("Predict Price"):
    # Predict
    prediction = model.predict(input_df)[0]
    df=input_df.values.tolist()
    st.subheader(f"Predicted Price: €{prediction:,.2f}")



