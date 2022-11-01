from sklearn.datasets import load_iris
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pickle
import numpy as np

df = pd.read_csv('IRIS.csv')

st.title("Iris Flower Category Prediction")
st.header("Metadata")
with st.expander("Data used to train the ML model"):
    st.write(df)

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv = convert_df(df)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='large_df.csv',
    mime='text/csv',
)

with st.expander("See the feature distribution"):
    for feature in df.columns:
        st.line_chart(df[feature])


for column in df.columns.drop('species'):
    df[column].fillna(np.mean(df[column]))

x = df.drop('species',axis=1)
y = df['species']

rfc = RandomForestClassifier(max_depth=2,random_state=123)
model = rfc.fit(x,y)

with open('rfc_model.pkl','wb') as file:
    pickle.dump(model,file)


sepal_length_min = min(df['sepal_length'])
sepal_width_min = min(df['sepal_width'])
petal_length_min = min(df['petal_length'])
petal_width_min = min(df['petal_width'])

sepal_length_max = max(df['sepal_length'])
sepal_width_max = max(df['sepal_width'])
petal_length_max = max(df['petal_length'])
petal_width_max = max(df['petal_width'])

sepal_length_mean = np.mean(df['sepal_length'])
sepal_width_mean = np.mean(df['sepal_width'])
petal_length_mean = np.mean(df['petal_length'])
petal_width_mean = np.mean(df['petal_width'])

st.header("Please provide the inputs for prediction")

sepal_length = st.slider("Sepal Length",float(sepal_length_min),float(sepal_length_max),float(sepal_length_mean),0.1)
sepal_width = st.slider("Sepal Width",float(sepal_width_min),float(sepal_width_max),float(sepal_width_mean),0.1)
petal_length = st.slider("Petal Length",float(petal_length_min),float(petal_length_max),float(petal_length_mean),0.1)
petal_width = st.slider("Petal Width",float(petal_width_min),float(petal_width_max),float(petal_width_mean),0.1)

input_by_user = np.array([sepal_length,sepal_width,petal_length,petal_width]).reshape(1,-1)
prediction = model.predict(input_by_user)
proba_pred = model.predict_proba(input_by_user)

st.header('Predictions')

with st.expander("Checkout the predictions for your inputs"):
    st.write("Prediction:")
    st.write(prediction)
    st.write('Probability of each category of flower')
    st.write(proba_pred)

