# -*- coding: utf-8 -*-
"""
Created by Ruth on Thu 20 July 2023

To deploy classification model
"""

import numpy as np
import pickle
import streamlit as st
from PIL import Image


# Load classifier
filename = r"sb_classifier.sav"
loaded_model = pickle.load(open(filename, 'rb'))


# Mapping input to numbers for model
gender_dict = {"Male":1,"Female":0}
options_dict = {"Yes":1,"No":0}

# Mapping function
def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value


# Classification function
def classifier_func(input_data):
    
    # Changing the input_data to numpy array
    input_data_as_array = np.array(input_data)
    
    # Reshape array as we are predicting one instance
    input_data_reshaped = input_data_as_array.reshape(1, -1)
    print(input_data_reshaped)
    
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    
    if prediction[0] == 0:
        return 'The patient does not have significant bacteriuria ✅'
    else:
        return 'The patient has significant bacteriuria ❌'


# Streamlit code
def main():
    #two pages
    app_mode = st.sidebar.selectbox('Select Page',['Home','Classification']) 
    
    
    if app_mode=='Home':
        # Homepage Information
        st.title('Classification Model for Significant Bacteriuria')
        st.markdown('This will predict the presence or absence of significant bacteriuria \
                    in paediatric sickle cell patients.')
        
        image = Image.open("diagnosis.jpg")
        st.image(image)
        st.markdown('Select _Classification_ on the sidebar')
        
    
    
    elif app_mode == 'Classification':
        # Give header to webpage
        st.header('Enter Patient\'s Information for Diagnosis')
        st.markdown("**:blue[Note: Fill all the fields to get a diagnosis for your patient]**")
        
        # To get input_data from the user
        Gender = st.radio('Patient\'s Gender', (tuple(gender_dict.keys())))
        Age = st.number_input('Patient\'s Age', min_value=0, max_value=17)
        Malnourished = st.selectbox('Is the patient malnourished?',
                                    (tuple(options_dict.keys())))
        History_of_Urinary_Infection = st.selectbox('Has the patient experienced \
                                                    a urinary infection in the past?',
                                                    (tuple(options_dict.keys())))
        Fever = st.selectbox('Does the patient have a fever?',
                             (tuple(options_dict.keys())))
        Presence_of_UTI_symptoms = st.selectbox('Is the patient exhibiting symptoms \
                                                of a urinary infection?',
                                                (tuple(options_dict.keys())))
        
        # Chosen inputs in a list     
        input_list=[get_value(Gender, gender_dict),
                    Age,
                    get_value(Malnourished, options_dict),
                    get_value(History_of_Urinary_Infection, options_dict),
                    get_value(Fever, options_dict),
                    get_value(Presence_of_UTI_symptoms, options_dict)]
        
        # Classification result
        diagnosis = ''
        
        if st.button('See result'):
            diagnosis = classifier_func(input_list)
        
        st.success(diagnosis)
    

if __name__ == '__main__':
    main()
    


    
    
