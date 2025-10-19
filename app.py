# app.py

import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

with open('models/logistic_regression_diabetes_model.pkl', 'rb') as f:
    log_model = pickle.load(f)
with open('models/random_forest_diabetes_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('models/xgboost_diabetes_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

models = {"Logistic Regression":log_model,"Random Forest":rf_model,"XGBoost":xgb_model}
scaler = StandardScaler()
st.set_page_config(page_title="Diabetes Prediction App",layout="wide")
st.title("Diabetes Prediction App")
st.write("Select the model and enter patient data to predict diabetes:")
st.header("Pregnancy & Age Info")
pregnancies = st.selectbox("Number of Pregnancies",options=list(range(0,18)))
age = st.selectbox("Age",options=list(range(21,82)))
st.header("Blood Test Data")
glucose=st.selectbox("Glucose Level",options=list(range(0,201)))
blood_pressure=st.selectbox("Blood Pressure",options=list(range(0,123)))
skin_thickness=st.selectbox("Skin Thickness",options=list(range(0,100)))
insulin=st.selectbox("Insulin Level",options=list(range(0,847)))
st.header("Continuous Features")
bmi=st.number_input("BMI",min_value=0.0,max_value=67.1,value=25.0,step=0.1,format="%.1f")
dpf=st.number_input("Diabetes Pedigree Function",min_value=0.08, max_value=2.42,value=0.5,step=0.01,format="%.2f")
input_data = np.array([[pregnancies,glucose,blood_pressure,skin_thickness,insulin,bmi,dpf,age]])
try:
    input_data_scaled=scaler.fit_transform(input_data)
except Exception as e:
    st.error(f"Error in scaling input: {e}")
if st.button("Predict"):
    try:
        pred_log=log_model.predict(input_data_scaled)[0]
        pred_rf=rf_model.predict(input_data_scaled)[0]
        pred_xgb=xgb_model.predict(input_data_scaled)[0]
        avg_pred=round((pred_log+pred_rf+pred_xgb)/3)
        result="🟢Non-Diabetic" if avg_pred==0 else "🔴Diabetic"
        st.success(f"Average Prediction:{result}")
        with st.expander("Show Individual Model Predictions"):
            st.info(f"Logistic Regression:{'Non-Diabetic' if pred_log==0 else 'Diabetic'}")
            st.info(f"Random Forest:{'Non-Diabetic' if pred_rf==0 else 'Diabetic'}")
            st.info(f"XGBoost:{'Non-Diabetic' if pred_xgb==0 else 'Diabetic'}")
    except Exception as e:
        st.error(f"Prediction failed:{e}")
