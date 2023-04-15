import streamlit as st 
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Random Forest Classifier for Pinguin

This app predicts the **Palmer Pinguin** species!

The original dataset is from R [palmerpenguis library](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

st.caption("Code by [DataProfessor Channel](https://www.youtube.com/watch?v=Eai1jaZrRDs&list=PLtqF5YXg7GLmCvTswG32NqQypOuYkPRUE&index=3). Reply for [me](https://github.com/romaro-gomes)")

st.sidebar.header("User Input Features")

uploaded_file=st.sidebar.file_uploader("Upload your inout CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))

        sex=st.sidebar.selectbox('Sex',('male','femae'))

        bill_length_mm=st.sidebar.slider('Bill lenght (mm)', 32.1,59.6,43.9)

        bill_depth_mm=st.sidebar.slider("Bill depth (mm)", 13.1,2.5,17.2)

        flipper_length_mm =st.sidebar.slider('Flipper length (mm)',172.0,231.0,201.0)

        body_mass_g=st.sidebar.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)

        # The data will be the dataset for predict.
        # The colmns must be in the same order  using in the model trainning
        data = {
            'island':island,
            'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': sex       
        }

        features = pd.DataFrame(data,index=[0])
        return features
    
    input_df = user_input_features()

# Original Dataset
penguins_raw =pd.read_csv('penguins_cleaned.csv')

penguins = penguins_raw.drop(columns=['species'])

# Combine Original Dataset with input data. Remember, axis=0 is row.
df=pd.concat([input_df,penguins],axis=0)

# MOdify Dataset for model

encode = ['sex','island']
for col in encode:
    dummy =pd.get_dummies(df[col],prefix=col)
    df = pd.concat([df,dummy], axis=1) #Remeber axis=1, columns
    del df[col]

df=df[:1] # select the first row

st.subheader('User Input Features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Input the save model
load_clf = pickle.load(open('penguins_clf.pkl','rb'))

#Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader('Prediction')

penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)