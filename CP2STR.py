import pip
import streamlit as st
st.__file__
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

pickle_in = open('C:/Users/Aniss/PycharmProjects/pythonProject6/str2.pkl', 'rb')
str2 = pickle.load(pickle_in)


def welcome():
    return 'Welcome all'


def predict_BA(country, year, location_type, cellphone_access, household_size, age_of_respondent,
               gender_of_respondent, relationship_with_head,
               marital_status, education_level, job_type):
    prediction = str2.predict([[country, year, location_type, cellphone_access, household_size,
                                age_of_respondent, gender_of_respondent, relationship_with_head,
                                marital_status, education_level, job_type]])
    print(prediction)
    return prediction


def main():
    st.title("Financial inclusion")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Churn prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)


feature1 = st.selectbox('country:', ('Kenya', 'Rwanda', 'Tanzania', 'Uganda'))
feature2 = st.selectbox('year:', ('2018', '2016', '2017'))
feature3 = st.selectbox('location_type:', ('Rural', 'Urban'))
feature4 = st.selectbox('cellphone_access:', ('Yes', 'No'))
feature5 = st.selectbox('household_size:',
                        ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16',
                         '17', '18', '19', '20', '21'))
feature6 = st.number_input('age_of_respondent')
feature7 = st.selectbox('gender_of_respondent:', ('Female', 'Male'))
feature8 = st.selectbox('relationship_with_head:',
                        ('Spouse', 'Head of Household', 'Other relative', 'Child', 'Parent',
                         'Other non-relatives'))
feature9 = 6
feature10 = 7

"""st.selectbox('marital_status:', ('Married/Living together', 'Widowed', 'Single/Never Married',
                                                  'Divorced/Separated', 'Dont know'))
feature10 = st.selectbox('education_level:', ('Secondary education', 'No formal education',
                                                    'Vocational/Specialised training', 'Primary education',
                                                    'Tertiary education', 'Other/Dont know/RTA'))
feature11 = st.selectbox('job_type:', ('Self employed', 'Government Dependent',
                                             'Formally employed Private', 'Informally employed',
                                             'Formally employed Government', 'Farming and Fishing',
                                             'Remittance Dependent', 'Other Income',
                                             'Dont Know/Refuse to answer', 'No Income'))
"""
if st.button('Predict'):
    features = np.array(
        [[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9,
          feature10]])


    def preprocess_data_array(features):
        # Instantiate LabelEncoders for each categorical feature
        label_encoders = {
            'country': LabelEncoder(),
            'location_type': LabelEncoder(),
            'cellphone_access': LabelEncoder(),
            'gender_of_respondent': LabelEncoder(),
            'relationship_with_head': LabelEncoder(),
            'marital_status': LabelEncoder(),
            'education_level': LabelEncoder(),
            'job_type': LabelEncoder()
        }

        # Fit and transform the encoders for each feature
        features[:, 0] = label_encoders['country'].fit_transform(features[:, 0])
        features[:, 1] = label_encoders['location_type'].fit_transform(features[:, 1])
        features[:, 2] = label_encoders['cellphone_access'].fit_transform(features[:, 2])
        features[:, 3] = label_encoders['gender_of_respondent'].fit_transform(features[:, 3])
        features[:, 4] = label_encoders['relationship_with_head'].fit_transform(features[:, 4])
        features[:, 5] = label_encoders['marital_status'].fit_transform(features[:, 5])
        features[:, 6] = label_encoders['education_level'].fit_transform(features[:, 6])
        features[:, 7] = label_encoders['job_type'].fit_transform(features[:, 7])

        # Assuming feature6 is a numeric value, no encoding is needed
        # If feature6 needs to be encoded as well, you can add it here

        # Convert to integer type
        features = features.astype(int)

        return features


    # Process the features array
    features = preprocess_data_array(features)
    prediction = str2.predict(features)

    st.write(f'Prediction: {prediction}')
else:
    st.error("Click the button to predict.")
