import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import tensorflow as tf
import xgboost as xgb
import warnings 
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, TimeDistributed, Flatten, Dropout, RepeatVector
from tensorflow.keras.models import Sequential
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

st.title('Electricity Forecasting App')

# Load datasets
df_weather = pd.read_csv(r"F:\Aditi Final Year Projectt\Final Code\weather_features.csv", parse_dates=['dt_iso'])
df_energy = pd.read_csv(r"F:\Aditi Final Year Projectt\Final Code\energy_dataset.csv", parse_dates=['time'])

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"]) 
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.write(df.head())
    
    # Data Visualization
    st.write("## Data Visualization")
    fig, ax = plt.subplots()
    sns.lineplot(data=df, ax=ax)
    st.pyplot(fig)
    
    # Data Preprocessing
    st.write("## Data Preprocessing")
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df.iloc[:, 1:].values)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(df_scaled[:, :-1], df_scaled[:, -1], test_size=0.2, random_state=42)
    
    # Model Training
    st.write("## Train a LSTM Model")
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(np.expand_dims(X_train, axis=2), y_train, epochs=10, batch_size=16, verbose=1)
    
    st.write("### Model Training Completed")
    
    # Forecasting
    st.write("## Forecasting")
    predictions = model.predict(np.expand_dims(X_test, axis=2))
    fig2, ax2 = plt.subplots()
    ax2.plot(y_test, label='Actual')
    ax2.plot(predictions, label='Predicted')
    ax2.legend()
    st.pyplot(fig2)
    
    st.write("### Forecasting Completed")
