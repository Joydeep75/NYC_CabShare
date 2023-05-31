import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


header = st.container()
dataset = st.container()
features = st.container()
model_trainings = st.container()

# Creation of background color
st.markdown(
    """
    <style>
     .main {
     background-color: #F5F5F5;

     }
    </style>
    """,
    unsafe_allow_html=True
)

# Using Caching to load data, only once
@st.cache_data
def get_data(filename):
    cab_data = pd.read_parquet(filename)

    return cab_data


with header:
    st.title("Analyze NYC Yellow Cab Ride!!")
    st.text('This is a Streamlit based Web App and here I look into the transactions of NYC Cabs')

with dataset:
    st.header("NYC Yellow Cab Trip Data")
    st.text('Dataset Location: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page')
    cab_data = get_data('data/yellow_tripdata_2023-02.parquet')
    st.write(cab_data.head())

    # How many times, from which pickup location, passenger was picked up
    # Add Title
    st.subheader('Pickup Location ID distribution on the NYC Cab Dataset')

    # Note: pulocation_dist means pulocation_distribution
    # pulocation_dist = pd.DataFrame(cab_data['PULocationID'].value_counts().head(50)) # OR
    pulocation_dist = pd.DataFrame(cab_data['PULocationID'].value_counts()).head(50)
    st.bar_chart(pulocation_dist)

with features:
    st.header("Features that has been created/considered")

    # Create a new Feature on top of the existing dataset columns
    st.markdown('* **First Feature:** I created this feature because of this...I calculated it using this logic...')
    st.markdown('* **Second Feature:** I created this feature because of this...I calculated it using this logic...')

with model_trainings:
    st.header("Time to train the model!")
    st.text('Here user get to choose the hyperparameters of the model and see how the performance changes!!')

    # Create User Input areas in the form of columns
    # Note: sel_col = Selection Column &  disp_col = Display Columns; columns(2) = Two columns
    # disp_col --> This is used to display the performance of the model

    sel_col, disp_col = st.columns(2)

    # Create Sliders
    max_depth = sel_col.slider('What should be the max depth of the model?', min_value=5, max_value=100, value=20, step=5)
    num_estimators = sel_col.selectbox('How many trees should there be?', options = [50, 100, 150, 200, 250, 300, 'No Limit'], index=0)

    # The list of ALL input features within the dataset
    sel_col.text('The list of features in the dataset')
    sel_col.write(cab_data.columns)

    # Default input feature, based on which the model will be created, out of the box
    input_feature = sel_col.text_input('Which feature should be used as the input feature?', 'PULocationID')

    # Now with the Model and it's training
    if num_estimators == 'No Limit':
        regsr = RandomForestRegressor(max_depth=max_depth)
    else:
        regsr = RandomForestRegressor(max_depth=max_depth, n_estimators=num_estimators)

    X = cab_data[[input_feature]]
    y = cab_data[['trip_distance']]

    regsr.fit(X, y.values.ravel())
    prediction = regsr.predict(y)

disp_col.subheader('Mean Absolute Error of the model is:' )
disp_col.write(mean_absolute_error(y, prediction))

disp_col.subheader('Mean Squared Error, MSE of the model is:')
disp_col.write(mean_squared_error(y, prediction))

disp_col.subheader('Mean Absolute Error (R^2) of the model is:')
disp_col.write(r2_score(y, prediction))