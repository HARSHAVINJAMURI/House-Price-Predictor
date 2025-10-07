import pandas as pd
import re
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso, Ridge, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer

# Load the house price data (replace with your data path)
df = pd.read_csv(r"C:\Users\vinja\Downloads\FSDS\ML\28_August\Test\done\HPP\House-Price-Predictor\Bengaluru_House_Data.csv")

# Clean and preprocess the data
def preprocess_data(df):
    # Handle missing values
    df['bath'].fillna(df['bath'].median(), inplace=True)
    df['balcony'].fillna(df['balcony'].median(), inplace=True)
    
    # Function to extract integer size from string
    def extract_int_size(text):
        if isinstance(text, str):
            match = re.search(r'\d+', text)
            if match:
                return int(match.group())
        return None

    df['size'] = df['size'].apply(extract_int_size)

    # Function to extract integers from total_sqft
    def extract_integer(value):
        match = re.search(r'\d+', str(value))
        if match:
            return float(match.group())
        else:
            return None

    df['total_sqft'] = df['total_sqft'].apply(extract_integer)
    
    # Drop unnecessary columns
    df.drop(columns=['society','area_type', 'availability','location'], inplace=True)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    return df

# Preprocess the data
df = preprocess_data(df)

# Select relevant features and target variable
X = df[['bath', 'balcony', 'size', 'total_sqft']]
y = df['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values in training set
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Initialize MinMaxScaler and scale features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the RandomForestRegressor model
rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_train, y_train)

# Function to predict house price
def predict_price(bathrooms, balconies, size, total_sqft):
    # Ensure integer input for bathrooms and balconies
    bathrooms = int(round(bathrooms))
    balconies = int(round(balconies))
    
    # Scale the input features
    scaled_features = scaler.transform([[bathrooms, balconies, size, total_sqft]])

    # Predict the price
    predicted_price = rf_regressor.predict(scaled_features)[0]
    return predicted_price

# Streamlit app configuration
st.title("House Price Prediction App")
st.subheader("Predict the price of your Dream house based on your requirenments.")

# User input for bathrooms, balconies, size, and total_sqft
bathrooms_input = st.number_input("Number of Bathrooms:", min_value=0, step=1)
balconies_input = st.number_input("Number of Balconies:", min_value=0, step=1)
size_input = st.number_input("Size (in terms of bedrooms):", min_value=0, step=1)
total_sqft_input = st.number_input("Total Square Feet:", min_value=0, step=100)

# Predict button
predict_button = st.button("Predict Price")

# Make prediction and display result if button is clicked
if predict_button:
    predicted_price = predict_price(bathrooms_input, balconies_input, size_input, total_sqft_input)
    st.success(f"Predicted Price in lakhs: ₹{predicted_price:.2f}")

