import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import pickle

model= tf.keras.models.load_model('model.h5')

#loading the model and the scaler and one hot encoder file
st.title("CUSTOMER CHURN PREDICTION")

with open('one_hot_encoder_geography.pkl','rb') as file:
     one_hot_encoder_geo1=pickle.load(file)

with open('label_encode_gender.pkl','rb') as file:
     lab_encode_gender=pickle.load(file)

with open('scalling.pkl','rb') as file:
     scalling_new=pickle.load(file)

 #streamlit app input data


geograpy=st.selectbox('Geography',one_hot_encoder_geo1.categories_[0])
gender=st.selectbox('Gender',lab_encode_gender.classes_) 
age=st.slider('Age',19,82)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10) 
num_of_products=st.slider('NumOfProducts',1,4)
has_creditcard=st.selectbox('HasCrCard',[0,1])
is_active_member= st.selectbox('Is Active Member',[0,1]) 



#input data
input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[lab_encode_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_creditcard],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
    
})

geography_encoded= one_hot_encoder_geo1.transform([[geograpy]]).toarray()
geo_encoded_df= pd.DataFrame(geography_encoded,columns=one_hot_encoder_geo1.get_feature_names_out(['Geography']))


input_data_df=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)
input_data_df

input_data_scalled= scalling_new.transform(input_data_df)

prediction=model.predict(input_data_scalled)
prediction_probab=prediction[0][0]


if prediction_probab>0.5:
    st.write("Customer likely to churn")
else:
     st.write("Customer not likely to churn")  
