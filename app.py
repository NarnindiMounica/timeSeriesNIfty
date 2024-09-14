#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

#converting dataset to dataframe
data = pd.read_csv('nifty_historical_data.csv')

# dtype of date to datetime
data['Date']= pd.to_datetime(data['Date'])

# dropping 'Adj Close' since its same as 'Close'
data.drop(['Adj Close'], axis=1, inplace=True)

#changing feature names to ds and y, here y="Close" feature
pr_data= data[['Date', 'Close']]
pr_data.columns = ['ds', 'y']

#training fbprophet model
p_model = Prophet()
p_model.fit(pr_data)

# to forecast Close values of next 365 days (1 year)
future = p_model.make_future_dataframe(periods=365)

#prediction dataframe
prediction= p_model.predict(future)

#saving predictions in a csv file
prediction[['ds','yhat']].to_csv('pred.csv')

#streamlit framework
st.title('Closing Price Forecasting')

given_date=st.text_input('Enter the date to get closing price forecasting: ')

#getting date from user and showing predicted close value
predicted_price=prediction[['yhat']].loc[(prediction['ds'] == given_date)].values
st.write(' Forecasted Closing Price', predicted_price)