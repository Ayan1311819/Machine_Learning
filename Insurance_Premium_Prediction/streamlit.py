import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

model = joblib.load('Insurance_Premium_Prediction/my_model.joblib') 

def run_streamlit():
    st.title('Insurance Premium Prediction')
    st.write('Enter the following information to predict insurance premium:')
    
    # Create input fields
    age = st.number_input('Age', min_value=0, max_value=120, value=30)
    sex = st.selectbox('Sex', ['male', 'female'])
    bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0)
    children = st.number_input('Number of Children', min_value=0, max_value=10, value=0)
    smoker = st.selectbox('Smoker', ['yes', 'no'])
    region = st.selectbox('Region', ['northeast', 'northwest', 'southeast', 'southwest'])

    if st.button('Predict Premium'):
        # Create a dataframe from input
        data = {
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'children': children,
            'smoker': smoker,
            'region': region
        }
        df = pd.DataFrame(data, index=[0])

        # Perform the same preprocessing steps as in your training code
        df['sex'] = df['sex'].map({'male': 1, 'female': 0})
        df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
        # Manually create one-hot encoded columns for region
        df['region_northeast'] = (df['region'] == 'northeast').astype(int)
        df['region_northwest'] = (df['region'] == 'northwest').astype(int)
        df['region_southeast'] = (df['region'] == 'southeast').astype(int)
        df['region_southwest'] = (df['region'] == 'southwest').astype(int)
    
    # Drop the original 'region' column
        df = df.drop('region', axis=1)
    
    # Ensure all columns are float
        df = df.astype(float)
        prediction = model.predict(df)
        st.success(f'Predicted Insurance Premium: ${prediction[0]:.2f}')

if __name__ == "__main__":
    run_streamlit()
