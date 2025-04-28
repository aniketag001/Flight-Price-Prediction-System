import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import joblib
import seaborn as sns

# Load your data (replace with your actual loading method)

model = joblib.load('Price_model.pkl')
scaler = joblib.load('scaler.pkl')

# Perform the same preprocessing steps as before
#df['date'] = pd.to_datetime(df['flight_date'])
# df['day'] = df['date'].dt.day
# df['month'] = df['date'].dt.month
# df['year'] = df['date'].dt.year
# df['duration_min'] = df['duration_minutes'] # No change needed here, its already correct
# df['price'] = df['price'].str.replace('₹', '').str.replace(',', '').astype(float)

le_airline = LabelEncoder()
df['line_encoded'] = df['airline_encoded'] # No change needed here, its already correct
le_from = LabelEncoder()
df['from_encoded'] = df['from_encoded']  # No change needed here, its already correct
le_to = LabelEncoder()
df['to_encoded'] = df['to_encoded'] # No change needed here, its already correct
le_dep_time = LabelEncoder()
df['dep_time_encoded'] = df['dep_time_encoded'] # No change needed here, its already correct
le_arr_time = LabelEncoder()
df['arr_time_encoded'] = df['arr_time_encoded'] # No change needed here, its already correct
le_stops = LabelEncoder()
df['stops_encoded'] = df['stops_encoded'] # No change needed here, its already correct
le_class = LabelEncoder()
df['class_encoded'] = df['class_encoded'] # No change needed here, its already correct

X = df[['airline_encoded', 'from_encoded', 'to_encoded', 'dep_time_encoded', 'arr_time_encoded', 'duration_minutes', 'stops_encoded', 'class_encoded', 'day', 'month', 'year']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the best model (let's assume Random Forest is the best after comparison)
rf_model = RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_leaf=1, min_samples_split=2, random_state=42) # Using some example best parameters
rf_model.fit(X_train, y_train)

st.title('Flight Price Prediction App')

# User input section
st.sidebar.header('Enter Flight Details')

airline_options = df['airline_encoded'].unique() # changed
airline = st.sidebar.selectbox('Airline', airline_options) # changed

from_options = df['from_encoded'].unique() # changed
departure_city = st.sidebar.selectbox('Departure City', from_options) # changed

to_options = df['to_encoded'].unique() # changed
destination_city = st.sidebar.selectbox('Destination City', to_options) # changed

class_options = df['class_encoded'].unique() # changed
travel_class = st.sidebar.selectbox('Class', class_options) # changed

stops_options = df['stops_encoded'].unique() # changed
stops = st.sidebar.selectbox('Number of Stops', stops_options) # changed

date = st.sidebar.date_input('Date of Travel')

dep_time_options = df['dep_time_encoded'].unique() # changed
dep_time = st.sidebar.selectbox('Departure Time', dep_time_options) # changed

arr_time_options = df['arr_time_encoded'].unique() # changed
arr_time = st.sidebar.selectbox('Arrival Time', arr_time_options) # changed

duration_input = st.sidebar.text_input('Duration (e.g., 2h 30m)')

if st.sidebar.button('Predict Price'):
    # Preprocess user input
    user_data = {
        'airline_encoded': [airline], # changed
        'from_encoded': [departure_city], # changed
        'to_encoded': [destination_city], # changed
        'class_encoded': [travel_class], # changed
        'stops_bin': [stops], # changed
        'flight date': [pd.to_datetime(date)],
        'dep_time_encoded': [dep_time], # changed
        'arr_time_encoded': [arr_time], # changed
        'duration_minutes': [duration_input]
    }
    user_df = pd.DataFrame(user_data)

    user_df['day'] = user_df['date'].dt.day
    user_df['month'] = user_df['date'].dt.month
    user_df['year'] = user_df['date'].dt.year
    user_df['duration_min'] = user_df['duration_minutes']
    user_df['airline_enco'] = user_df['airline_encoded'] # changed
    user_df['from_encoded'] = user_df['from_encoded'] # changed
    user_df['to_encoded'] = user_df['to_encoded'] # changed
    user_df['dep_time_encoded'] = user_df['dep_time_encoded'] # changed
    user_df['arr_time_encoded'] = user_df['arr_time_encoded'] # changed
    user_df['stops_encoded'] = user_df['stops_encoded'] # changed
    user_df['class_encoded'] = user_df['class_encoded'] # changed

    # Make prediction
    prediction = rf_model.predict(user_df[['airline_encoded', 'from_encoded', 'to_encoded', 'dep_time_encoded', 'arr_time_encoded', 'duration_minutes', 'stops_encoded', 'class_encoded', 'day', 'month', 'year']])[0]

    st.subheader(f'Predicted Flight Price: ₹{prediction:.2f}')

    # Visualize Model Accuracy (example: scatter plot of actual vs predicted on test set)
    st.subheader('Model Performance Visualization (Test Data)')
    test_predictions = rf_model.predict(X_test)
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=test_predictions, ax=ax)
